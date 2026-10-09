package httpapi

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/labstack/echo/v5"
	"github.com/labstack/echo/v5/middleware"
	"yieldsense/backend/internal/domain"
	"yieldsense/backend/internal/storage"
)

type API struct {
	Records storage.Repository
	Objects storage.Blobs
	Key     string
	Origins []string
	slots   chan struct{}
}
type Run struct {
	ID          string        `json:"id"`
	ParentID    string        `json:"parent_id,omitempty"`
	CreatedAt   string        `json:"created_at"`
	InputSHA256 string        `json:"input_sha256"`
	Input       domain.Input  `json:"input"`
	Result      domain.Result `json:"result"`
	Currency    string        `json:"currency"`
	MassUnit    string        `json:"mass_unit"`
	Solver      string        `json:"solver"`
	Limitations []string      `json:"limitations"`
}
type runRecord struct {
	ID          string `json:"id"`
	CreatedAt   string `json:"created_at"`
	Status      string `json:"status"`
	ReportKey   string `json:"report_key"`
	InputSHA256 string `json:"input_sha256"`
}

func ID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b[:])
}
func now() string        { return time.Now().UTC().Format(time.RFC3339Nano) }
func bad(s string) error { return echo.NewHTTPError(422, s) }
func decode(c *echo.Context, out any) error {
	if ct := c.Request().Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		return echo.NewHTTPError(415, "Content-Type must be application/json")
	}
	d := json.NewDecoder(c.Request().Body)
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return err
		}
		return echo.NewHTTPError(400, "invalid JSON: "+err.Error())
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return echo.NewHTTPError(400, "provide one JSON object")
	}
	return nil
}
func (a *API) Echo() *echo.Echo {
	e := echo.New()
	a.slots = make(chan struct{}, 2)
	e.Use(middleware.Recover(), middleware.RequestID(), middleware.RequestLogger())
	e.Use(func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c *echo.Context) error {
			limit := int64(1024 * 1024)
			if strings.HasSuffix(c.Path(), "/images") {
				limit = 10 * 1024 * 1024
			}
			c.Request().Body = http.MaxBytesReader(c.Response(), c.Request().Body, limit)
			return next(c)
		}
	})
	if len(a.Origins) > 0 {
		e.Use(middleware.CORSWithConfig(middleware.CORSConfig{AllowOrigins: a.Origins, AllowMethods: []string{"GET", "POST", "PUT", "OPTIONS"}, AllowHeaders: []string{"Content-Type", "Authorization", "If-Match"}, ExposeHeaders: []string{"ETag", "X-Request-ID"}}))
	}
	e.HTTPErrorHandler = func(c *echo.Context, err error) {
		code := 500
		message := "internal server error"
		var he *echo.HTTPError
		if errors.As(err, &he) {
			code = he.Code
			message = fmt.Sprint(he.Message)
		} else if errors.Is(err, storage.ErrNotFound) {
			code = 404
			message = "record not found"
		} else if errors.Is(err, storage.ErrConflict) {
			code = 409
			message = "record changed; reload its version"
		}
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			code = 413
			message = "request body too large"
		}
		if code >= 500 {
			slog.Error("request failed", "error", err, "request_id", c.Response().Header().Get("X-Request-ID"))
		}
		_ = c.JSON(code, map[string]any{"error": message, "request_id": c.Response().Header().Get("X-Request-ID")})
	}
	e.GET("/health", func(c *echo.Context) error {
		return c.JSON(200, map[string]any{"ok": true, "service": "yieldsense", "solver": "go-branch-and-bound"})
	})
	v := e.Group("/v1", func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c *echo.Context) error {
			header := c.Request().Header.Get("Authorization")
			token := strings.TrimPrefix(header, "Bearer ")
			if !strings.HasPrefix(header, "Bearer ") || a.Key == "" || subtle.ConstantTimeCompare([]byte(token), []byte(a.Key)) != 1 {
				return echo.NewHTTPError(401, "valid bearer API key required")
			}
			return next(c)
		}
	})
	v.POST("/batches", a.createBatch)
	v.GET("/batches", a.list("batches"))
	v.GET("/batches/:id", a.get("batches"))
	v.PUT("/batches/:id", a.updateBatch)
	v.POST("/composition", a.composition)
	v.POST("/batches/:id/images", a.image)
	v.POST("/routes", a.createRoute)
	v.GET("/routes", a.list("routes"))
	v.GET("/routes/:id", a.get("routes"))
	v.PUT("/routes/:id", a.updateRoute)
	v.POST("/optimize", a.optimize)
	v.POST("/simulate", a.simulate)
	v.GET("/runs", a.list("runs"))
	v.GET("/runs/:id", a.report)
	v.GET("/reports/:id", a.report)
	return e
}
func (a *API) get(kind string) echo.HandlerFunc {
	return func(c *echo.Context) error {
		var v json.RawMessage
		if err := a.Records.Get(c.Request().Context(), kind, c.Param("id"), &v); err != nil {
			return err
		}
		return c.JSON(200, v)
	}
}
func (a *API) list(kind string) echo.HandlerFunc {
	return func(c *echo.Context) error {
		limit := 25
		if q := c.QueryParam("limit"); q != "" {
			n, err := strconv.Atoi(q)
			if err != nil || n < 1 || n > 100 {
				return bad("limit must be 1-100")
			}
			limit = n
		}
		cursor := c.QueryParam("cursor")
		if len(cursor) > 300 {
			return bad("invalid cursor")
		}
		if cursor != "" {
			decoded, err := base64.RawURLEncoding.DecodeString(cursor)
			if err != nil || len(decoded) == 0 || len(decoded) > 200 {
				return bad("invalid cursor")
			}
		}
		items, next, err := a.Records.List(c.Request().Context(), kind, cursor, limit)
		if err != nil {
			return err
		}
		return c.JSON(200, map[string]any{"items": items, "next_cursor": next})
	}
}
func expected(c *echo.Context, current int) error {
	v, err := strconv.Atoi(strings.Trim(c.Request().Header.Get("If-Match"), "\""))
	if err != nil || v < 1 {
		return echo.NewHTTPError(428, "If-Match header with current version required")
	}
	if v != current {
		return storage.ErrConflict
	}
	return nil
}
func (a *API) createBatch(c *echo.Context) error {
	var b domain.Batch
	if err := decode(c, &b); err != nil {
		return err
	}
	b.ID = ID()
	b.Version = 1
	b.CreatedAt = now()
	b.BatteryStatus = "unknown"
	b.ReviewStatus = "pending"
	b.ReviewEvidence = ""
	if err := b.Validate(); err != nil {
		return bad(err.Error())
	}
	if err := a.Records.Save(c.Request().Context(), "batches", b.ID, 0, b); err != nil {
		return err
	}
	c.Response().Header().Set("ETag", "\"1\"")
	return c.JSON(201, b)
}
func (a *API) updateBatch(c *echo.Context) error {
	var b, old domain.Batch
	if err := decode(c, &b); err != nil {
		return err
	}
	ctx := c.Request().Context()
	if err := a.Records.Get(ctx, "batches", c.Param("id"), &old); err != nil {
		return err
	}
	if err := expected(c, old.Version); err != nil {
		return err
	}
	b.ID = old.ID
	b.CreatedAt = old.CreatedAt
	b.Version = old.Version + 1
	if err := b.Validate(); err != nil {
		return bad(err.Error())
	}
	if err := a.Records.Save(ctx, "batches", b.ID, old.Version, b); err != nil {
		return err
	}
	c.Response().Header().Set("ETag", fmt.Sprintf("\"%d\"", b.Version))
	return c.JSON(200, b)
}
func (a *API) composition(c *echo.Context) error {
	var p struct {
		BatchID     string             `json:"batch_id"`
		Composition domain.Composition `json:"composition"`
	}
	if err := decode(c, &p); err != nil {
		return err
	}
	if err := p.Composition.Validate(); err != nil {
		return bad(err.Error())
	}
	var b domain.Batch
	ctx := c.Request().Context()
	if err := a.Records.Get(ctx, "batches", p.BatchID, &b); err != nil {
		return err
	}
	if err := expected(c, b.Version); err != nil {
		return err
	}
	old := b.Version
	b.Version++
	b.Composition = &p.Composition
	if err := a.Records.Save(ctx, "batches", b.ID, old, b); err != nil {
		return err
	}
	return c.JSON(200, b)
}
func (a *API) createRoute(c *echo.Context) error {
	var r domain.Route
	if err := decode(c, &r); err != nil {
		return err
	}
	r.ID = ID()
	r.Version = 1
	r.CreatedAt = now()
	if err := r.Validate(); err != nil {
		return bad(err.Error())
	}
	if err := a.Records.Save(c.Request().Context(), "routes", r.ID, 0, r); err != nil {
		return err
	}
	return c.JSON(201, r)
}
func (a *API) updateRoute(c *echo.Context) error {
	var r, old domain.Route
	if err := decode(c, &r); err != nil {
		return err
	}
	ctx := c.Request().Context()
	if err := a.Records.Get(ctx, "routes", c.Param("id"), &old); err != nil {
		return err
	}
	if err := expected(c, old.Version); err != nil {
		return err
	}
	r.ID = old.ID
	r.Version = old.Version + 1
	r.CreatedAt = old.CreatedAt
	if err := r.Validate(); err != nil {
		return bad(err.Error())
	}
	if err := a.Records.Save(ctx, "routes", r.ID, old.Version, r); err != nil {
		return err
	}
	return c.JSON(200, r)
}
func (a *API) image(c *echo.Context) error {
	ctx := c.Request().Context()
	var b domain.Batch
	if err := a.Records.Get(ctx, "batches", c.Param("id"), &b); err != nil {
		return err
	}
	data, err := io.ReadAll(c.Request().Body)
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return bad("image body required")
	}
	mime := http.DetectContentType(data)
	ext := ""
	switch mime {
	case "image/jpeg":
		ext = ".jpg"
	case "image/png":
		ext = ".png"
	default:
		return echo.NewHTTPError(415, "only JPEG and PNG images accepted")
	}
	id := ID()
	config, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || config.Width < 1 || config.Height < 1 || int64(config.Width)*int64(config.Height) > 16_000_000 {
		return bad("invalid image or image exceeds 16 megapixels")
	}
	if _, _, err := image.Decode(bytes.NewReader(data)); err != nil {
		return bad("invalid image data")
	}
	key := "raw/" + b.ID + "/" + id + ext
	sum := sha256.Sum256(data)
	if err := a.Objects.Put(ctx, key, data, mime); err != nil {
		return err
	}
	v := map[string]any{"id": id, "batch_id": b.ID, "object_key": key, "sha256": hex.EncodeToString(sum[:]), "size_bytes": len(data), "content_type": mime, "created_at": now()}
	if err := a.Records.Save(ctx, "images", id, 0, v); err != nil {
		return err
	}
	return c.JSON(201, v)
}
func (a *API) optimize(c *echo.Context) error {
	var p struct {
		BatchIDs     []string `json:"batch_ids"`
		RouteIDs     []string `json:"route_ids"`
		Objective    string   `json:"objective"`
		FixedRouteID string   `json:"fixed_route_id"`
		Assumptions  []string `json:"assumptions"`
	}
	if err := decode(c, &p); err != nil {
		return err
	}
	if len(p.BatchIDs) < 1 || len(p.BatchIDs) > 30 || len(p.RouteIDs) < 1 || len(p.RouteIDs) > 10 {
		return bad("provide 1-30 batch_ids and 1-10 route_ids")
	}
	if p.Objective == "" {
		p.Objective = "ROBUST"
	}
	in := domain.Input{Objective: p.Objective, FixedRouteID: p.FixedRouteID, Assumptions: p.Assumptions}
	ctx := c.Request().Context()
	for _, id := range p.BatchIDs {
		var b domain.Batch
		if err := a.Records.Get(ctx, "batches", id, &b); err != nil {
			return err
		}
		in.Batches = append(in.Batches, b)
	}
	for _, id := range p.RouteIDs {
		var r domain.Route
		if err := a.Records.Get(ctx, "routes", id, &r); err != nil {
			return err
		}
		in.Routes = append(in.Routes, r)
	}
	return a.execute(c, in, "")
}
func (a *API) execute(c *echo.Context, in domain.Input, parent string) error {
	if err := in.Validate(); err != nil {
		return bad(err.Error())
	}
	select {
	case a.slots <- struct{}{}:
		defer func() { <-a.slots }()
	default:
		return echo.NewHTTPError(503, "optimizer busy; retry shortly")
	}
	ctx, cancel := context.WithTimeout(c.Request().Context(), 3*time.Second)
	defer cancel()
	result, err := domain.Solve(ctx, in)
	if err != nil {
		return bad(err.Error())
	}
	if c.Request().Context().Err() != nil {
		return c.Request().Context().Err()
	}
	raw, err := json.Marshal(in)
	if err != nil {
		return err
	}
	hash := sha256.Sum256(raw)
	run := Run{ID: ID(), ParentID: parent, CreatedAt: now(), InputSHA256: hex.EncodeToString(hash[:]), Input: in, Result: result, Currency: "INR", MassUnit: "kg", Solver: "go-branch-and-bound-v1", Limitations: []string{"Modeled contribution, not measured savings; includes only supplied costs.", "Worst value covers supplied joint scenarios only; uncertainty is not statistically calibrated.", "No image inference or automatic safety certification; review evidence is operator supplied.", "Each run is an independent plan; capacity is not reserved across runs."}}
	data, err := json.Marshal(run)
	if err != nil {
		return err
	}
	if len(data) > 8*1024*1024 {
		return echo.NewHTTPError(413, "run report exceeds 8 MiB; reduce input evidence or batch count")
	}
	key := "reports/" + run.ID + ".json"
	requestCtx := c.Request().Context()
	if err := a.Objects.Put(requestCtx, key, data, "application/json"); err != nil {
		return err
	}
	record := runRecord{run.ID, run.CreatedAt, result.Status, key, run.InputSHA256}
	if err := a.Records.Save(requestCtx, "runs", run.ID, 0, record); err != nil {
		return err
	}
	return c.JSON(201, run)
}
func (a *API) loadRun(ctx context.Context, id string) (Run, error) {
	var r runRecord
	if err := a.Records.Get(ctx, "runs", id, &r); err != nil {
		return Run{}, err
	}
	data, err := a.Objects.Get(ctx, r.ReportKey)
	if err != nil {
		return Run{}, err
	}
	var run Run
	err = json.Unmarshal(data, &run)
	return run, err
}
func (a *API) report(c *echo.Context) error {
	run, err := a.loadRun(c.Request().Context(), c.Param("id"))
	if err != nil {
		return err
	}
	return c.JSON(200, run)
}
func (a *API) simulate(c *echo.Context) error {
	type override struct {
		RouteID  string           `json:"route_id"`
		Capacity *float64         `json:"capacity_kg,omitempty"`
		Price    domain.Fractions `json:"price,omitempty"`
	}
	var p struct {
		BaseRunID   string     `json:"base_run_id"`
		Objective   string     `json:"objective,omitempty"`
		Overrides   []override `json:"overrides"`
		Assumptions []string   `json:"assumptions"`
	}
	if err := decode(c, &p); err != nil {
		return err
	}
	if len(p.Overrides) == 0 || len(p.Overrides) > 10 || len(p.Assumptions) == 0 {
		return bad("provide route overrides and simulation assumptions")
	}
	base, err := a.loadRun(c.Request().Context(), p.BaseRunID)
	if err != nil {
		return err
	}
	in := base.Input
	if p.Objective != "" {
		in.Objective = p.Objective
	}
	in.Assumptions = append(in.Assumptions, p.Assumptions...)
	seen := map[string]bool{}
	for _, o := range p.Overrides {
		if seen[o.RouteID] || (o.Capacity == nil && o.Price == nil) {
			return bad("duplicate or empty override")
		}
		seen[o.RouteID] = true
		found := false
		for i := range in.Routes {
			if in.Routes[i].ID == o.RouteID {
				found = true
				if o.Capacity != nil {
					in.Routes[i].CapacityKG = *o.Capacity
				}
				if o.Price != nil {
					in.Routes[i].Price = o.Price
				}
			}
		}
		if !found {
			return bad("override references unknown route")
		}
	}
	return a.execute(c, in, base.ID)
}
