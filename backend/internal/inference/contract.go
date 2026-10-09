package inference

import (
	"errors"
	"fmt"
	"math"
	"net/url"
	"regexp"
	"slices"
	"strings"

	"yieldsense/backend/internal/domain"
)

const ContractVersion = "1"
const DeviceTask = "device_classification"
const CompositionTask = "particle_composition"

var hashPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)
var versionPattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$`)
var jobPattern = regexp.MustCompile(`^arn:aws[a-z-]*:sagemaker:[a-z0-9-]+:[0-9]{12}:training-job/[a-zA-Z0-9-]+$`)

type Preprocessing struct {
	Width     int       `json:"width"`
	Height    int       `json:"height"`
	ColorMode string    `json:"color_mode"`
	Scale     float64   `json:"scale"`
	Mean      []float64 `json:"mean"`
	Std       []float64 `json:"std"`
}

// Model is immutable metadata supplied by the ML pipeline, not a training job.
type Model struct {
	Version               string             `json:"model_version"`
	CreatedAt             string             `json:"created_at,omitempty"`
	ContractVersion       string             `json:"contract_version"`
	Track                 string             `json:"track"`
	Task                  string             `json:"task"`
	Architecture          string             `json:"architecture"`
	ArtifactS3URI         string             `json:"artifact_s3_uri"`
	ArtifactSHA256        string             `json:"artifact_sha256"`
	DatasetManifestSHA256 string             `json:"dataset_manifest_sha256"`
	DataVersion           string             `json:"data_version"`
	TrainingJobARN        string             `json:"training_job_arn"`
	Labels                []string           `json:"labels"`
	Preprocessing         Preprocessing      `json:"preprocessing"`
	ConfidenceThreshold   float64            `json:"confidence_threshold"`
	Evaluation            map[string]float64 `json:"evaluation"`
	PairedMassEvidence    string             `json:"paired_mass_evidence,omitempty"`
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func (m Model) Validate(bucket string) error {
	if !versionPattern.MatchString(m.Version) || m.ContractVersion != ContractVersion {
		return errors.New("valid model_version and contract_version=1 required")
	}
	if !((m.Track == "B" && m.Task == DeviceTask) || (m.Track == "A" && m.Task == CompositionTask)) {
		return errors.New("Track B requires device_classification; Track A requires particle_composition")
	}
	u, err := url.Parse(m.ArtifactS3URI)
	if err != nil || u.Scheme != "s3" || u.Host != bucket || !strings.HasPrefix(u.Path, "/models/") || len(u.Path) <= 8 || u.RawQuery != "" || u.Fragment != "" || u.User != nil {
		return errors.New("artifact_s3_uri must reference this bucket's models/ prefix")
	}
	if !hashPattern.MatchString(m.ArtifactSHA256) || !hashPattern.MatchString(m.DatasetManifestSHA256) || !jobPattern.MatchString(m.TrainingJobARN) || m.DataVersion == "" || m.Architecture == "" {
		return errors.New("artifact/dataset hashes, training_job_arn, data_version and architecture required")
	}
	if !finite(m.ConfidenceThreshold) || m.ConfidenceThreshold < 0 || m.ConfidenceThreshold > 1 {
		return errors.New("confidence_threshold must be 0-1")
	}
	if len(m.Labels) < 1 || len(m.Labels) > 128 {
		return errors.New("provide 1-128 actual model labels in class-index order")
	}
	seen := map[string]bool{}
	for _, label := range m.Labels {
		if strings.TrimSpace(label) == "" || len(label) > 128 || seen[label] {
			return errors.New("labels must be unique nonempty strings <=128 characters")
		}
		seen[label] = true
	}
	p := m.Preprocessing
	if p.Width < 1 || p.Width > 4096 || p.Height < 1 || p.Height > 4096 || p.ColorMode != "RGB" || !finite(p.Scale) || p.Scale <= 0 || len(p.Mean) != 3 || len(p.Std) != 3 {
		return errors.New("RGB dimensions, positive scale, and 3-channel mean/std required")
	}
	for i := 0; i < 3; i++ {
		if !finite(p.Mean[i]) || !finite(p.Std[i]) || p.Std[i] <= 0 {
			return errors.New("invalid preprocessing normalization")
		}
	}
	if len(m.Evaluation) == 0 || len(m.Evaluation) > 32 {
		return errors.New("held-out evaluation metrics required")
	}
	for k, v := range m.Evaluation {
		if k == "" || !finite(v) || v < 0 {
			return errors.New("invalid evaluation metric")
		}
	}
	if m.Track == "B" {
		v, ok := m.Evaluation["macro_f1"]
		if !ok || v > 1 {
			return errors.New("Track B requires held-out macro_f1 in [0,1]")
		}
	}
	if m.Track == "A" {
		if m.PairedMassEvidence == "" {
			return errors.New("Track A requires paired_mass_evidence")
		}
		if len(m.Labels) != len(domain.Materials) {
			return errors.New("Track A requires five material labels")
		}
		for _, k := range domain.Materials {
			if !seen[k] {
				return errors.New("Track A requires FEM, NFM, PCB, PLA, OTHER labels")
			}
		}
	}
	return nil
}

type Image struct {
	ID          string `json:"id"`
	BatchID     string `json:"batch_id"`
	ObjectKey   string `json:"object_key"`
	SHA256      string `json:"sha256"`
	SizeBytes   int    `json:"size_bytes"`
	ContentType string `json:"content_type"`
	CreatedAt   string `json:"created_at"`
}

// []byte is encoded as base64 by encoding/json. Worker owns decoding,
// preprocessing and model loading; Go owns persistence and safety policy.
type Request struct {
	ContractVersion string  `json:"contract_version"`
	RequestID       string  `json:"request_id"`
	BatchID         string  `json:"batch_id"`
	MassKG          float64 `json:"mass_kg"`
	Model           Model   `json:"model"`
	Image           Image   `json:"image"`
	ImageBytes      []byte  `json:"image_base64"`
}
type Response struct {
	ContractVersion  string             `json:"contract_version"`
	RequestID        string             `json:"request_id"`
	ModelVersion     string             `json:"model_version"`
	Track            string             `json:"track"`
	Task             string             `json:"task"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	Abstained        bool               `json:"abstained"`
	AbstentionReason string             `json:"abstention_reason,omitempty"`
	BatteryAlert     string             `json:"battery_alert"`
	MaterialMassKG   domain.Fractions   `json:"material_mass_kg,omitempty"`
	Scenarios        []domain.Fractions `json:"scenarios,omitempty"`
}
type Assessment struct {
	PredictedClass       string              `json:"predicted_class,omitempty"`
	Confidence           float64             `json:"confidence"`
	Abstained            bool                `json:"abstained"`
	Reasons              []string            `json:"reasons"`
	Composition          *domain.Composition `json:"composition,omitempty"`
	EstimatedMassKG      float64             `json:"estimated_mass_kg,omitempty"`
	ReconciliationFactor float64             `json:"mass_reconciliation_factor,omitempty"`
	MassDiscrepancyKG    float64             `json:"mass_discrepancy_kg,omitempty"`
}

func Assess(req Request, res Response) (Assessment, error) {
	out := Assessment{Abstained: res.Abstained, Reasons: []string{}}
	if res.ContractVersion != ContractVersion || res.RequestID != req.RequestID || res.ModelVersion != req.Model.Version || res.Track != req.Model.Track || res.Task != req.Model.Task {
		return out, errors.New("inference response identity/contract mismatch")
	}
	if !slices.Contains([]string{"unknown", "suspected", "none"}, res.BatteryAlert) {
		return out, errors.New("invalid battery_alert; inference cannot clear batteries")
	}
	if res.Abstained {
		if strings.TrimSpace(res.AbstentionReason) == "" || len(res.AbstentionReason) > 2048 {
			return out, errors.New("abstention requires a reason <=2048 characters")
		}
		out.Reasons = append(out.Reasons, res.AbstentionReason)
	}
	if res.Task == DeviceTask {
		if len(res.MaterialMassKG) > 0 || len(res.Scenarios) > 0 {
			return out, errors.New("device classifier cannot return material composition")
		}
		if res.Abstained && len(res.Probabilities) == 0 {
			return out, nil
		}
		if len(res.Probabilities) < 1 || len(res.Probabilities) > 128 {
			return out, errors.New("invalid probabilities count")
		}
		total := 0.0
		labels := make([]string, 0, len(res.Probabilities))
		for k, v := range res.Probabilities {
			if len(k) > 128 || !finite(v) || v < 0 || v > 1 {
				return out, errors.New("invalid class probability")
			}
			total += v
			labels = append(labels, k)
		}
		if math.Abs(total-1) > 1e-6 {
			return out, errors.New("class probabilities must sum to 1")
		}
		slices.Sort(labels)
		for _, label := range labels {
			v := res.Probabilities[label]
			if !slices.Contains(req.Model.Labels, label) {
				out.Abstained = true
				out.Reasons = append(out.Reasons, "unsupported_label:"+label)
			}
			if out.PredictedClass == "" || v > out.Confidence {
				out.PredictedClass = label
				out.Confidence = v
			}
		}
		if !out.Abstained {
			for _, label := range req.Model.Labels {
				if _, ok := res.Probabilities[label]; !ok {
					return out, errors.New("response missing registered class probability")
				}
			}
		}
		if out.Confidence < req.Model.ConfidenceThreshold {
			out.Abstained = true
			out.Reasons = append(out.Reasons, "below_confidence_threshold")
		}
		return out, nil
	}
	if len(res.Probabilities) > 0 {
		return out, errors.New("particle composition cannot return device probabilities")
	}
	if res.Abstained {
		return out, nil
	}
	if err := domain.ValidateFractions(res.MaterialMassKG, false); err != nil {
		return out, err
	}
	for _, v := range res.MaterialMassKG {
		out.EstimatedMassKG += v
	}
	if out.EstimatedMassKG <= 0 || out.EstimatedMassKG > 1e9 {
		return out, errors.New("invalid estimated material mass")
	}
	c := domain.Composition{Source: "model", Track: "A", ModelVersion: req.Model.Version, ModelOutputID: req.RequestID, Evidence: fmt.Sprintf("prediction %s; paired mass evidence: %s", req.RequestID, req.Model.PairedMassEvidence), Scenarios: res.Scenarios}
	if err := c.Validate(); err != nil {
		return out, err
	}
	for _, k := range domain.Materials {
		if math.Abs(c.Scenarios[0][k]-res.MaterialMassKG[k]/out.EstimatedMassKG) > 1e-6 {
			return out, errors.New("first composition scenario must match normalized material mass estimates")
		}
	}
	out.ReconciliationFactor = req.MassKG / out.EstimatedMassKG
	out.MassDiscrepancyKG = out.EstimatedMassKG - req.MassKG
	out.Composition = &c
	return out, nil
}
