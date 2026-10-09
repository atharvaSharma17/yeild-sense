package domain

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func fractions(fem, nfm float64) Fractions {
	return Fractions{"FEM": fem, "NFM": nfm, "PCB": 0, "PLA": 0, "OTHER": 1 - fem - nfm}
}
func batch(id string, mass float64, scenarios ...Fractions) Batch {
	return Batch{ID: id, Category: "devices", MassKG: mass, MassSource: "scale", BatteryStatus: "cleared", ReviewStatus: "approved", ReviewEvidence: "human inspection", Composition: &Composition{Source: "operator_entered", Evidence: "synthetic test", Scenarios: scenarios}}
}
func route(id string, capacity float64) Route {
	return Route{ID: id, Name: id, CapacityKG: capacity, Recovery: fractions(1, 0), Price: fractions(1, 0), AcceptedCategories: []string{"devices"}, AllowedBatteryStates: []string{"cleared"}, Authorized: true, AuthorizationEvidence: "operator evidence", EconomicsSource: "synthetic"}
}
func TestRobustUsesJointScenarios(t *testing.T) {
	b := batch("b", 1, fractions(1, 0), fractions(0, 1))
	r1, r2 := route("risky", 2), route("stable", 2)
	r1.Price["FEM"] = 10
	r2.Recovery["FEM"] = 1
	r2.Recovery["NFM"] = 1
	r2.Price["FEM"] = 4
	r2.Price["NFM"] = 4
	in := Input{Batches: []Batch{b}, Routes: []Route{r1, r2}, Objective: "ROBUST", Assumptions: []string{"synthetic"}}
	robust, err := Solve(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if robust.Status != "OPTIMAL" || robust.Assignments[0].RouteID != "stable" || robust.WorstValue != 4 {
		t.Fatalf("unexpected robust result: %+v", robust)
	}
	in.Objective = "EXPECTED"
	expected, err := Solve(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if expected.Assignments[0].RouteID != "risky" || expected.ExpectedValue != 5 {
		t.Fatalf("unexpected expected result: %+v", expected)
	}
	// Complementary scenarios must be summed BEFORE taking the worst case.
	in.Objective = "ROBUST"
	in.Routes = []Route{r1}
	in.Batches = append(in.Batches, batch("b2", 1, fractions(0, 1), fractions(1, 0)))
	res, err := Solve(context.Background(), in)
	if err != nil || res.WorstValue != 10 {
		t.Fatalf("joint scenario accounting: %+v %v", res, err)
	}
}
func TestSafetyCapacityAndTimeout(t *testing.T) {
	in := Input{Batches: []Batch{batch("b", 2, fractions(1, 0))}, Routes: []Route{route("r", 2)}, Objective: "ROBUST", Assumptions: []string{"synthetic"}}
	in.Batches[0].BatteryStatus = "unknown"
	res, _ := Solve(context.Background(), in)
	if res.Status != "INFEASIBLE" {
		t.Fatal(res)
	}
	in.Routes[0].ApprovedHold = true
	in.Routes[0].AllowedBatteryStates = []string{"unknown"}
	res, _ = Solve(context.Background(), in)
	if res.Status != "OPTIMAL" {
		t.Fatal(res)
	}
	in.Routes[0].Authorized = false
	res, _ = Solve(context.Background(), in)
	if res.Status != "INFEASIBLE" {
		t.Fatal(res)
	}
	in.Routes[0].Authorized = true
	in.Routes[0].CapacityKG = 1
	res, _ = Solve(context.Background(), in)
	if res.Status != "INFEASIBLE" {
		t.Fatal(res)
	}
	in.Routes[0].CapacityKG = 2
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res, _ = Solve(ctx, in)
	if res.Status != "TIME_LIMIT" || len(res.Assignments) != 0 {
		t.Fatal(res)
	}
}
func TestOptimizerMatchesExhaustiveOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 100; trial++ {
		in := Input{Objective: "EXPECTED", Assumptions: []string{"synthetic"}}
		if trial%2 == 0 {
			in.Objective = "ROBUST"
		}
		for i := 0; i < 5; i++ {
			sc := []Fractions{}
			for s := 0; s < 3; s++ {
				f := rng.Float64()
				sc = append(sc, fractions(f, 1-f))
			}
			in.Batches = append(in.Batches, batch(fmt.Sprint(i), float64(1+rng.Intn(4)), sc...))
		}
		for j := 0; j < 3; j++ {
			r := route(fmt.Sprint(j), float64(2+rng.Intn(10)))
			r.Recovery["NFM"] = 1
			r.Price["FEM"] = rng.Float64() * 20
			r.Price["NFM"] = rng.Float64() * 20
			r.CostPerKG = rng.Float64() * 15
			in.Routes = append(in.Routes, r)
		}
		best := math.Inf(-1)
		var enumerate func(int, []float64, []float64)
		enumerate = func(i int, used, totals []float64) {
			if i == len(in.Batches) {
				value := 0.0
				if in.Objective == "ROBUST" {
					value = math.Inf(1)
					for _, v := range totals {
						value = math.Min(value, v)
					}
				} else {
					for _, v := range totals {
						value += v / float64(len(totals))
					}
				}
				best = math.Max(best, value)
				return
			}
			b := in.Batches[i]
			for j, r := range in.Routes {
				if !Eligible(b, r) || used[j]+b.MassKG > r.CapacityKG {
					continue
				}
				nextUsed := append([]float64{}, used...)
				nextUsed[j] += b.MassKG
				nextTotals := append([]float64{}, totals...)
				for s, c := range b.Composition.Scenarios {
					nextTotals[s] += Profit(b, r, c)
				}
				enumerate(i+1, nextUsed, nextTotals)
			}
		}
		enumerate(0, make([]float64, 3), make([]float64, 3))
		got, err := Solve(context.Background(), in)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsInf(best, -1) {
			if got.Status != "INFEASIBLE" {
				t.Fatalf("trial %d: %+v", trial, got)
			}
			continue
		}
		value := got.ExpectedValue
		if in.Objective == "ROBUST" {
			value = got.WorstValue
		}
		if got.Status != "OPTIMAL" || math.Abs(best-value) > 1e-7 {
			t.Fatalf("trial %d: want %v, got %+v", trial, best, got)
		}
		used := map[string]float64{}
		for _, a := range got.Assignments {
			for _, b := range in.Batches {
				if a.BatchID == b.ID {
					used[a.RouteID] += b.MassKG
				}
			}
		}
		for _, r := range in.Routes {
			if used[r.ID] > r.CapacityKG+1e-9 {
				t.Fatal("capacity violated")
			}
		}
	}
}
func TestValidation(t *testing.T) {
	b := batch("b", 1, fractions(1, 0))
	b.Composition.Scenarios[0]["FEM"] = 0.9
	if err := b.Validate(); err == nil {
		t.Fatal("accepted unnormalized composition")
	}
	r := route("r", 1)
	r.Recovery["FEM"] = 1.1
	if err := r.Validate(); err == nil {
		t.Fatal("accepted recovery >1")
	}
	b = batch("b", 1, fractions(1, 0))
	b.ReviewEvidence = ""
	if err := b.Validate(); err == nil {
		t.Fatal("accepted cleared batch without review evidence")
	}
}
