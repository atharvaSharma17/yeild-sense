package domain

import (
 "context"
 "errors"
 "math"
 "slices"
 "time"
)

type Input struct {
 Batches []Batch `json:"batches"`
 Routes []Route `json:"routes"`
 Objective string `json:"objective"`
 FixedRouteID string `json:"fixed_route_id,omitempty"`
 Assumptions []string `json:"assumptions"`
}
type Assignment struct { BatchID string `json:"batch_id"`; RouteID string `json:"route_id"` }
type Result struct {
 Status string `json:"status"`
 Objective string `json:"objective"`
 Assignments []Assignment `json:"assignments"`
 ScenarioValues []float64 `json:"scenario_values"`
 ExpectedValue float64 `json:"expected_value"`
 WorstValue float64 `json:"worst_value"`
 Diagnostics []string `json:"diagnostics"`
 Nodes int `json:"nodes"`
 ElapsedMS int64 `json:"elapsed_ms"`
}
func (in Input) Validate() error {
 if len(in.Batches)<1 || len(in.Batches)>30 || len(in.Routes)<1 || len(in.Routes)>10 {return errors.New("provide 1-30 batches and 1-10 routes")}
 if !slices.Contains([]string{"ROBUST","EXPECTED","GREEDY","FIXED"},in.Objective) {return errors.New("objective must be ROBUST, EXPECTED, GREEDY or FIXED")}
 if len(in.Assumptions)==0 {return errors.New("explicit assumptions are required")}
 ids:=map[string]bool{}; count:=0
 for _,b:=range in.Batches {if b.ID=="" || ids[b.ID] {return errors.New("batch IDs must be unique and nonempty")}; ids[b.ID]=true; if err:=b.Validate();err!=nil{return err}; if b.Composition==nil{return errors.New("every batch needs composition")}; if count==0 {count=len(b.Composition.Scenarios)}; if count!=len(b.Composition.Scenarios) {return errors.New("all batches require the same aligned joint scenario count")}}
 ids=map[string]bool{}; for _,r:=range in.Routes {if r.ID=="" || ids[r.ID] {return errors.New("route IDs must be unique and nonempty")}; ids[r.ID]=true; if err:=r.Validate();err!=nil{return err}}
 if in.Objective=="FIXED" && !ids[in.FixedRouteID] {return errors.New("fixed_route_id must reference an input route")};return nil
}
func score(v []float64,objective string) float64 { if objective=="ROBUST" {return slices.Min(v)}; s:=0.0;for _,x:=range v{s+=x};return s/float64(len(v)) }

// Solve is bounded exact branch-and-bound. Scenario indices are joint outcomes.
func Solve(ctx context.Context,in Input) (Result,error) {
 if err:=in.Validate();err!=nil{return Result{},err}
 start:=time.Now(); n:=len(in.Batches); m:=len(in.Routes); ns:=len(in.Batches[0].Composition.Scenarios)
 res:=Result{Status:"INFEASIBLE",Objective:in.Objective,Assignments:[]Assignment{},ScenarioValues:[]float64{},Diagnostics:[]string{}}
 order:=make([]int,n); options:=make([][]int,n); values:=make([][][]float64,n)
 for i,b:=range in.Batches {order[i]=i;values[i]=make([][]float64,m);for j,r:=range in.Routes {if !Eligible(b,r) || b.MassKG>r.CapacityKG || (in.Objective=="FIXED" && r.ID!=in.FixedRouteID) {continue};options[i]=append(options[i],j);values[i][j]=make([]float64,ns);for s,c:=range b.Composition.Scenarios {values[i][j][s]=Profit(b,r,c)}};if len(options[i])==0 {res.Diagnostics=append(res.Diagnostics,"no eligible route with sufficient capacity for batch "+b.ID)}}
 if len(res.Diagnostics)>0 {res.ElapsedMS=time.Since(start).Milliseconds();return res,nil}
 slices.SortStableFunc(order,func(a,b int)int {if len(options[a])!=len(options[b]) {return len(options[a])-len(options[b])};if in.Batches[a].MassKG>in.Batches[b].MassKG{return -1};if in.Batches[a].MassKG<in.Batches[b].MassKG{return 1};return 0})
 for i:=range options {slices.SortStableFunc(options[i],func(a,b int)int {x,y:=score(values[i][a],in.Objective),score(values[i][b],in.Objective);if x>y{return -1};if x<y{return 1};return 0})}
 upper:=make([][]float64,n+1);upper[n]=make([]float64,ns)
 for d:=n-1;d>=0;d-- {upper[d]=make([]float64,ns);i:=order[d];for s:=0;s<ns;s++ {best:=math.Inf(-1);for _,j:=range options[i] {best=math.Max(best,values[i][j][s])};upper[d][s]=upper[d+1][s]+best}}
 capacities:=make([]float64,m);for j,r:=range in.Routes {capacities[j]=r.CapacityKG};totals:=make([]float64,ns);choice:=make([]int,n);best:=math.Inf(-1); stopped:=false
 save:=func() {best=score(totals,in.Objective);res.ScenarioValues=slices.Clone(totals);res.Assignments=make([]Assignment,n);for i,j:=range choice {res.Assignments[i]=Assignment{in.Batches[i].ID,in.Routes[j].ID}}}
 if in.Objective=="GREEDY" || in.Objective=="FIXED" {for _,i:=range order {found:=false;for _,j:=range options[i] {if capacities[j]+1e-9<in.Batches[i].MassKG {continue};capacities[j]-=in.Batches[i].MassKG;choice[i]=j;for s:=range totals {totals[s]+=values[i][j][s]};found=true;break};if !found {res.Status="NO_SOLUTION_FOUND";res.Diagnostics=append(res.Diagnostics,"baseline could not place all batches; this does not prove global infeasibility");res.ElapsedMS=time.Since(start).Milliseconds();return res,nil}};save();res.Status="FEASIBLE"} else {
 var search func(int);search=func(d int) {if stopped {return};res.Nodes++;if res.Nodes>2_000_000 || ctx.Err()!=nil {stopped=true;return};bound:=make([]float64,ns);for s:=range bound {bound[s]=totals[s]+upper[d][s]};if score(bound,in.Objective)<=best {return};if d==n {save();return};i:=order[d];for _,j:=range options[i] {if capacities[j]+1e-9<in.Batches[i].MassKG {continue};capacities[j]-=in.Batches[i].MassKG;choice[i]=j;before:=slices.Clone(totals);for s:=range totals {totals[s]+=values[i][j][s]};search(d+1);copy(totals,before);capacities[j]+=in.Batches[i].MassKG;if stopped{return}}};search(0)
 if stopped {res.Status="TIME_LIMIT";res.Diagnostics=append(res.Diagnostics,"search budget exhausted; optimality and infeasibility are not proven")} else if len(res.Assignments)>0 {res.Status="OPTIMAL"} else {res.Diagnostics=append(res.Diagnostics,"joint capacities cannot accommodate all batches")}
 }
 if len(res.ScenarioValues)>0 {res.ExpectedValue=score(res.ScenarioValues,"EXPECTED");res.WorstValue=score(res.ScenarioValues,"ROBUST")};res.ElapsedMS=time.Since(start).Milliseconds();return res,nil
}
