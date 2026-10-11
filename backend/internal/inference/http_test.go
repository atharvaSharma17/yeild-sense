package inference

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"yieldsense/backend/internal/domain"
)

func model() Model {
	return Model{Version: "xbat-v1", ContractVersion: "1", Track: "B", Task: DeviceTask, Architecture: "resnet18", ArtifactS3URI: "s3://bucket/models/v1/model.tar.gz", ArtifactSHA256: strings.Repeat("a", 64), DatasetManifestSHA256: strings.Repeat("b", 64), DataVersion: "xbat-reviewed-v1", TrainingJobARN: "arn:aws:sagemaker:ap-south-1:123456789012:training-job/xbat-v1", Labels: []string{"device-a", "device-b"}, Preprocessing: Preprocessing{Width: 224, Height: 224, ColorMode: "RGB", Scale: 1.0 / 255, Mean: []float64{.485, .456, .406}, Std: []float64{.229, .224, .225}}, ConfidenceThreshold: .7, Evaluation: map[string]float64{"macro_f1": .8}}
}

func TestLocalTrainingAndBatteryThresholdProvenance(t *testing.T) {
	m := model()
	m.TrainingSource = "local"
	m.TrainingEvidence = "Training run log and held-out evaluation archived by ML developer"
	m.TrainingJobARN = ""
	threshold := .1744
	m.ConformalBatteryThreshold = &threshold
	if err := m.Validate("bucket"); err != nil {
		t.Fatal(err)
	}
	m.TrainingJobARN = model().TrainingJobARN
	if m.Validate("bucket") == nil {
		t.Fatal("local training accepted a SageMaker claim")
	}
	m.TrainingJobARN = ""
	m.TrainingEvidence = ""
	if m.Validate("bucket") == nil {
		t.Fatal("local training accepted without evidence")
	}
	m = model()
	threshold = 1.1
	m.ConformalBatteryThreshold = &threshold
	if m.Validate("bucket") == nil {
		t.Fatal("invalid battery threshold accepted")
	}
}
func requestForTest() Request {
	return Request{ContractVersion: "1", RequestID: "request", Model: model(), MassKG: 100, ImageBytes: []byte("test image bytes")}
}
func responseForTest(r Request) Response {
	return Response{ContractVersion: "1", RequestID: r.RequestID, ModelVersion: r.Model.Version, Track: r.Model.Track, Task: r.Model.Task, Probabilities: map[string]float64{"device-a": .8, "device-b": .2}, BatteryAlert: "unknown"}
}
func TestHTTPAdapterWireContract(t *testing.T) {
	in := requestForTest()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" || r.Header.Get("X-Request-ID") != in.RequestID {
			t.Error("auth/request identity missing")
		}
		var received Request
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Error(err)
		}
		if string(received.ImageBytes) != string(in.ImageBytes) || received.Model.Preprocessing.Width != 224 {
			t.Error("image/metadata not forwarded")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(responseForTest(received))
	}))
	defer server.Close()
	adapter, err := NewHTTP(server.URL, "secret")
	if err != nil {
		t.Fatal(err)
	}
	out, err := adapter.Predict(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	assessment, err := Assess(in, out)
	if err != nil || assessment.PredictedClass != "device-a" || assessment.Abstained {
		t.Fatal(assessment, err)
	}
}
func TestHTTPFailureBoundaries(t *testing.T) {
	for _, mode := range []string{"redirect", "bad_json", "oversized", "error", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch mode {
				case "redirect":
					w.Header().Set("Location", "http://example.invalid")
					w.WriteHeader(302)
				case "bad_json":
					_, _ = w.Write([]byte(`{"unknown":1}`))
				case "oversized":
					_, _ = w.Write([]byte(strings.Repeat("x", 256*1024+1)))
				case "error":
					w.WriteHeader(500)
				case "timeout":
					time.Sleep(50 * time.Millisecond)
				}
			}))
			defer server.Close()
			adapter, _ := NewHTTP(server.URL, "secret")
			ctx := context.Background()
			if mode == "timeout" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, 5*time.Millisecond)
				defer cancel()
			}
			_, err := adapter.Predict(ctx, requestForTest())
			want := ErrUnavailable
			if mode == "bad_json" || mode == "oversized" {
				want = ErrInvalidResponse
			}
			if mode == "timeout" {
				want = ErrTimeout
			}
			if !errors.Is(err, want) {
				t.Fatal(err, want)
			}
		})
	}
	if _, err := NewHTTP("https://user:password@example.com/predict", ""); err == nil {
		t.Fatal("URL credentials accepted")
	}
}
func TestPredictionValidationAndAbstention(t *testing.T) {
	req := requestForTest()
	res := responseForTest(req)
	res.Probabilities = map[string]float64{"device-a": .6, "device-b": .4}
	got, err := Assess(req, res)
	if err != nil || !got.Abstained {
		t.Fatal(got, err)
	}
	res.Probabilities = map[string]float64{"unsupported": 1}
	got, err = Assess(req, res)
	if err != nil || !got.Abstained {
		t.Fatal("unsupported labels must abstain", got, err)
	}
	res = responseForTest(req)
	res.ModelVersion = "wrong"
	if _, err := Assess(req, res); err == nil {
		t.Fatal("wrong version accepted")
	}
	res = responseForTest(req)
	res.MaterialMassKG = domain.Fractions{"FEM": 1}
	if _, err := Assess(req, res); err == nil {
		t.Fatal("device probabilities became material masses")
	}
	res = responseForTest(req)
	res.Probabilities["device-a"] = 2
	if _, err := Assess(req, res); err == nil {
		t.Fatal("invalid probabilities accepted")
	}
}
func TestParticleMassReconciliation(t *testing.T) {
	req := requestForTest()
	req.Model.Track = "A"
	req.Model.Task = CompositionTask
	req.Model.Labels = domain.Materials
	req.Model.PairedMassEvidence = "verified paired particle mass manifest"
	res := Response{ContractVersion: "1", RequestID: req.RequestID, ModelVersion: req.Model.Version, Track: "A", Task: CompositionTask, BatteryAlert: "unknown", MaterialMassKG: domain.Fractions{"FEM": 5, "NFM": 2, "PCB": 1, "PLA": 1, "OTHER": 1}, Scenarios: []domain.Fractions{{"FEM": .5, "NFM": .2, "PCB": .1, "PLA": .1, "OTHER": .1}}}
	got, err := Assess(req, res)
	if err != nil || got.ReconciliationFactor != 10 || got.MassDiscrepancyKG != -90 || got.Composition.Track != "A" {
		t.Fatal(got, err)
	}
	res.Scenarios[0]["FEM"] = .4
	res.Scenarios[0]["PLA"] = .2
	if _, err := Assess(req, res); err == nil {
		t.Fatal("mass/scenario disagreement accepted")
	}
}
func TestManifestValidation(t *testing.T) {
	m := model()
	if err := m.Validate("bucket"); err != nil {
		t.Fatal(err)
	}
	m.ArtifactS3URI = "s3://another/models/model.tar.gz"
	if err := m.Validate("bucket"); err == nil {
		t.Fatal("foreign artifact bucket accepted")
	}
	m = model()
	m.Labels = []string{"duplicate", "duplicate"}
	if err := m.Validate("bucket"); err == nil {
		t.Fatal("duplicate labels accepted")
	}
}
