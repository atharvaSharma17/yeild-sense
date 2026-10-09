package domain

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

var Materials = []string{"FEM", "NFM", "PCB", "PLA", "OTHER"}
var BatteryStates = []string{"unknown", "suspected", "present", "cleared"}

type Fractions map[string]float64
type Composition struct {
	Source       string      `json:"source"`
	Evidence     string      `json:"evidence"`
	ModelVersion string      `json:"model_version,omitempty"`
	Scenarios    []Fractions `json:"scenarios"`
}
type Batch struct {
	ID             string       `json:"id"`
	Version        int          `json:"version"`
	CreatedAt      string       `json:"created_at"`
	Category       string       `json:"category"`
	MassKG         float64      `json:"mass_kg"`
	MassSource     string       `json:"mass_source"`
	BatteryStatus  string       `json:"battery_status"`
	ReviewStatus   string       `json:"review_status"`
	ReviewEvidence string       `json:"review_evidence"`
	Composition    *Composition `json:"composition,omitempty"`
}
type Route struct {
	ID                    string    `json:"id"`
	Version               int       `json:"version"`
	CreatedAt             string    `json:"created_at"`
	Name                  string    `json:"name"`
	CapacityKG            float64   `json:"capacity_kg"`
	CostPerKG             float64   `json:"cost_per_kg"`
	TransportPerLot       float64   `json:"transport_per_lot"`
	Recovery              Fractions `json:"recovery"`
	Price                 Fractions `json:"price"`
	AcceptedCategories    []string  `json:"accepted_categories"`
	AllowedBatteryStates  []string  `json:"allowed_battery_states"`
	Authorized            bool      `json:"authorized"`
	AuthorizationEvidence string    `json:"authorization_evidence"`
	ApprovedHold          bool      `json:"approved_hold"`
	EconomicsSource       string    `json:"economics_source"`
}

func validNumber(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) && x >= 0 }
func ValidateFractions(f Fractions, normalized bool) error {
	if len(f) != len(Materials) {
		return errors.New("include FEM, NFM, PCB, PLA and OTHER")
	}
	total := 0.0
	for _, k := range Materials {
		v, ok := f[k]
		if !ok || !validNumber(v) || (normalized && v > 1) {
			return fmt.Errorf("invalid material %s", k)
		}
		total += v
	}
	if normalized && math.Abs(total-1) > 1e-6 {
		return errors.New("composition fractions must sum to 1")
	}
	return nil
}
func (c Composition) Validate() error {
	if !slices.Contains([]string{"measured", "category_prior", "operator_entered", "model"}, c.Source) || c.Evidence == "" {
		return errors.New("composition requires a valid source and evidence")
	}
	if c.Source == "model" && c.ModelVersion == "" {
		return errors.New("modeled composition requires model_version")
	}
	if len(c.Scenarios) < 1 || len(c.Scenarios) > 20 {
		return errors.New("provide 1 to 20 joint scenarios")
	}
	for _, s := range c.Scenarios {
		if err := ValidateFractions(s, true); err != nil {
			return err
		}
	}
	return nil
}
func (b Batch) Validate() error {
	if b.Category == "" || !validNumber(b.MassKG) || b.MassKG <= 0 || b.MassKG > 1e9 {
		return errors.New("category and positive mass_kg <= 1e9 required")
	}
	if !slices.Contains([]string{"scale", "illustrative"}, b.MassSource) {
		return errors.New("mass_source must be scale or illustrative")
	}
	if !slices.Contains(BatteryStates, b.BatteryStatus) || !slices.Contains([]string{"pending", "approved"}, b.ReviewStatus) {
		return errors.New("invalid battery or review status")
	}
	if (b.ReviewStatus == "approved" || b.BatteryStatus == "cleared") && b.ReviewEvidence == "" {
		return errors.New("human review evidence required")
	}
	if b.Composition != nil {
		return b.Composition.Validate()
	}
	return nil
}
func (r Route) Validate() error {
	if r.Name == "" || len(r.AcceptedCategories) == 0 || len(r.AllowedBatteryStates) == 0 || r.EconomicsSource == "" {
		return errors.New("route name, categories, battery states and economics_source required")
	}
	for _, s := range r.AllowedBatteryStates {
		if !slices.Contains(BatteryStates, s) {
			return errors.New("invalid allowed battery state")
		}
	}
	for _, x := range []float64{r.CapacityKG, r.CostPerKG, r.TransportPerLot} {
		if !validNumber(x) || x > 1e9 {
			return errors.New("invalid route economics or capacity")
		}
	}
	if r.Authorized && r.AuthorizationEvidence == "" {
		return errors.New("authorization evidence required")
	}
	if err := ValidateFractions(r.Recovery, false); err != nil {
		return err
	}
	if err := ValidateFractions(r.Price, false); err != nil {
		return err
	}
	for _, v := range r.Recovery {
		if v > 1 {
			return errors.New("recovery must be between 0 and 1")
		}
	}
	for _, v := range r.Price {
		if v > 1e9 {
			return errors.New("price must be <= 1e9")
		}
	}
	return nil
}
func Eligible(b Batch, r Route) bool {
	if !r.Authorized || !slices.Contains(r.AcceptedCategories, b.Category) || !slices.Contains(r.AllowedBatteryStates, b.BatteryStatus) {
		return false
	}
	if b.ReviewStatus != "approved" || b.BatteryStatus == "unknown" || b.BatteryStatus == "suspected" {
		return r.ApprovedHold
	}
	return true
}
func Profit(b Batch, r Route, s Fractions) float64 {
	revenue := 0.0
	for _, k := range Materials {
		revenue += s[k] * r.Recovery[k] * r.Price[k]
	}
	return b.MassKG*(revenue-r.CostPerKG) - r.TransportPerLot
}
