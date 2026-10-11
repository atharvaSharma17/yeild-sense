package httpapi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/labstack/echo/v5"
	"yieldsense/backend/internal/domain"
	"yieldsense/backend/internal/inference"
	"yieldsense/backend/internal/storage"
)

type Prediction struct {
	ID                  string               `json:"id"`
	BatchID             string               `json:"batch_id"`
	BatchVersion        int                  `json:"batch_version"`
	Image               inference.Image      `json:"image"`
	Model               inference.Model      `json:"model"`
	CreatedAt           string               `json:"created_at"`
	Result              inference.Response   `json:"result"`
	Assessment          inference.Assessment `json:"assessment"`
	HumanReviewRequired bool                 `json:"human_review_required"`
}

func (a *API) batchImages(c *echo.Context) error {
	var batch domain.Batch
	if err := a.Records.Get(c.Request().Context(), "batches", c.Param("id"), &batch); err != nil {
		return err
	}
	return a.list("batch_images:" + batch.ID)(c)
}

func (a *API) imageContent(c *echo.Context) error {
	var img inference.Image
	ctx := c.Request().Context()
	if err := a.Records.Get(ctx, "images", c.Param("id"), &img); err != nil {
		return err
	}
	data, err := a.Objects.Get(ctx, img.ObjectKey)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(data)
	if len(data) != img.SizeBytes || hex.EncodeToString(sum[:]) != img.SHA256 {
		return errors.New("stored image integrity mismatch")
	}
	c.Response().Header().Set("Cache-Control", "private, no-store")
	c.Response().Header().Set("X-Content-Type-Options", "nosniff")
	return c.Blob(http.StatusOK, img.ContentType, data)
}

func (a *API) registerModel(c *echo.Context) error {
	var m inference.Model
	if err := decode(c, &m); err != nil {
		return err
	}
	if err := m.Validate(a.Bucket); err != nil {
		return bad(err.Error())
	}
	m.CreatedAt = now()
	if err := a.Records.Save(c.Request().Context(), "models", m.Version, 0, m); err != nil {
		return err
	}
	return c.JSON(201, m)
}
func (a *API) validateModelComposition(ctx context.Context, batchID string, c *domain.Composition) error {
	if c == nil || c.Source != "model" {
		return nil
	}
	var p Prediction
	if err := a.Records.Get(ctx, "model_outputs", c.ModelOutputID, &p); err != nil {
		return err
	}
	if p.BatchID != batchID || p.Model.Version != c.ModelVersion || p.Model.Track != "A" || p.Assessment.Abstained || p.Assessment.Composition == nil {
		return bad("composition must reference a successful Track A prediction for this batch")
	}
	given, _ := json.Marshal(c.Scenarios)
	stored, _ := json.Marshal(p.Assessment.Composition.Scenarios)
	if string(given) != string(stored) {
		return bad("model scenarios must match stored prediction; use operator_entered for manual edits")
	}
	*c = *p.Assessment.Composition
	return nil
}
func (a *API) predict(task string) echo.HandlerFunc {
	return func(c *echo.Context) error {
		if a.Predictor == nil {
			return echo.NewHTTPError(503, "ML_NOT_CONFIGURED: configure ML_INFERENCE_URL")
		}
		var p struct {
			BatchID      string `json:"batch_id"`
			ImageID      string `json:"image_id"`
			ModelVersion string `json:"model_version"`
		}
		if err := decode(c, &p); err != nil {
			return err
		}
		if p.BatchID == "" || p.ImageID == "" {
			return bad("batch_id and registered image_id required")
		}
		if p.ModelVersion == "" {
			p.ModelVersion = a.DefaultModelVersion
		}
		if p.ModelVersion == "" {
			return bad("model_version required")
		}
		ctx := c.Request().Context()
		var b domain.Batch
		var img inference.Image
		var model inference.Model
		if err := a.Records.Get(ctx, "batches", p.BatchID, &b); err != nil {
			return err
		}
		if err := expected(c, b.Version); err != nil {
			return err
		}
		if err := a.Records.Get(ctx, "images", p.ImageID, &img); err != nil {
			return err
		}
		if img.BatchID != b.ID {
			return bad("image belongs to a different batch")
		}
		if err := a.Records.Get(ctx, "models", p.ModelVersion, &model); err != nil {
			return err
		}
		if err := model.Validate(a.Bucket); err != nil {
			return errors.New("invalid registered model metadata")
		}
		if model.Task != task {
			return bad("registered model task does not match prediction endpoint")
		}
		select {
		case a.mlSlots <- struct{}{}:
			defer func() { <-a.mlSlots }()
		default:
			return echo.NewHTTPError(503, "model inference busy; retry shortly")
		}
		data, err := a.Objects.Get(ctx, img.ObjectKey)
		if err != nil {
			return err
		}
		hash := sha256.Sum256(data)
		if hex.EncodeToString(hash[:]) != img.SHA256 || len(data) != img.SizeBytes || http.DetectContentType(data) != img.ContentType {
			return errors.New("stored image integrity check failed")
		}
		req := inference.Request{ContractVersion: inference.ContractVersion, RequestID: ID(), BatchID: b.ID, MassKG: b.MassKG, Model: model, Image: img, ImageBytes: data}
		inferCtx, cancel := context.WithTimeout(ctx, 12*time.Second)
		defer cancel()
		result, err := a.Predictor.Predict(inferCtx, req)
		if err != nil {
			if errors.Is(err, inference.ErrTimeout) {
				return echo.NewHTTPError(504, "MODEL_TIMEOUT")
			}
			if errors.Is(err, inference.ErrInvalidResponse) {
				return echo.NewHTTPError(502, "MODEL_INVALID_RESPONSE")
			}
			return echo.NewHTTPError(503, "MODEL_UNAVAILABLE")
		}
		assessment, err := inference.Assess(req, result)
		if err != nil {
			return echo.NewHTTPError(502, "MODEL_INVALID_RESPONSE: "+err.Error())
		}
		old := b.Version
		b.Version++
		b.LatestPredictionID = req.RequestID
		// ML may request review, but may never clear a battery or approve a batch.
		if assessment.Abstained || result.BatteryAlert == "suspected" {
			b.ReviewStatus = "pending"
			if result.BatteryAlert == "suspected" {
				b.BatteryStatus = "suspected"
			} else if b.BatteryStatus == "cleared" {
				b.BatteryStatus = "unknown"
			}
		}
		if assessment.Composition != nil {
			b.Composition = assessment.Composition
		}
		record := Prediction{ID: req.RequestID, BatchID: b.ID, BatchVersion: b.Version, Image: img, Model: model, CreatedAt: now(), Result: result, Assessment: assessment, HumanReviewRequired: b.ReviewStatus != "approved" || b.BatteryStatus == "unknown" || b.BatteryStatus == "suspected"}
		if err := a.Records.SaveAll(ctx, []storage.Write{{Kind: "model_outputs", ID: record.ID, Expected: 0, Value: record}, {Kind: "batches", ID: b.ID, Expected: old, Value: b}}); err != nil {
			return err
		}
		return c.JSON(201, map[string]any{"prediction": record, "batch": b})
	}
}
