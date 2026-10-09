package httpapi

import (
 "bytes"
 "context"
 "encoding/json"
 "errors"
 "image"
 "image/png"
 "net/http"
 "net/http/httptest"
 "strings"
 "testing"

 "yieldsense/backend/internal/domain"
 "yieldsense/backend/internal/storage"
)

// Test doubles only; the running application always uses DynamoDB and S3.
type records struct {data map[string][]byte;versions map[string]int}
func (r *records)Get(_ context.Context,kind,id string,out any)error{data,ok:=r.data[kind+id];if !ok{return storage.ErrNotFound};return json.Unmarshal(data,out)}
func (r *records)Save(_ context.Context,kind,id string,expected int,v any)error{k:=kind+id;if r.versions[k]!=expected{return storage.ErrConflict};data,err:=json.Marshal(v);if err!=nil{return err};r.data[k]=data;r.versions[k]=expected+1;return nil}
func (r *records)List(_ context.Context,kind,cursor string,limit int)([]json.RawMessage,string,error){items:=[]json.RawMessage{};for k,v:=range r.data{if strings.HasPrefix(k,kind){items=append(items,json.RawMessage(v))}};return items,"",nil}
type objects struct {data map[string][]byte;fail bool}
func (o *objects)Put(_ context.Context,k string,v []byte,mime string)error{if o.fail{return errors.New("storage unavailable")};o.data[k]=append([]byte{},v...);return nil}
func (o *objects)Get(_ context.Context,k string)([]byte,error){v,ok:=o.data[k];if !ok{return nil,storage.ErrNotFound};return append([]byte{},v...),nil}
func setup()(*API,http.Handler,*records,*objects){r:=&records{map[string][]byte{},map[string]int{}};o:=&objects{data:map[string][]byte{}};a:=&API{Records:r,Objects:o,Key:strings.Repeat("a",32)};return a,a.Echo(),r,o}
func request(h http.Handler,method,path string,body []byte,version string)*httptest.ResponseRecorder{req:=httptest.NewRequest(method,path,bytes.NewReader(body));req.Header.Set("Authorization","Bearer "+strings.Repeat("a",32));req.Header.Set("Content-Type","application/json");if version!=""{req.Header.Set("If-Match",version)};rec:=httptest.NewRecorder();h.ServeHTTP(rec,req);return rec}
func marshal(v any)[]byte{data,err:=json.Marshal(v);if err!=nil{panic(err)};return data}
func testBatch()domain.Batch{return domain.Batch{Category:"devices",MassKG:10,MassSource:"scale",BatteryStatus:"cleared",ReviewStatus:"approved",ReviewEvidence:"inspection",Composition:&domain.Composition{Source:"operator_entered",Evidence:"demo",Scenarios:[]domain.Fractions{{"FEM":1,"NFM":0,"PCB":0,"PLA":0,"OTHER":0}}}}}
func testRoute()domain.Route{return domain.Route{Name:"mechanical",CapacityKG:100,Authorized:true,AuthorizationEvidence:"human authorization",AcceptedCategories:[]string{"devices"},AllowedBatteryStates:[]string{"cleared"},Recovery:domain.Fractions{"FEM":1,"NFM":0,"PCB":0,"PLA":0,"OTHER":0},Price:domain.Fractions{"FEM":10,"NFM":0,"PCB":0,"PLA":0,"OTHER":0},EconomicsSource:"synthetic"}}
func TestCreateAndReviewWithVersionConflict(t *testing.T){_,h,_,_:=setup();b:=testBatch();rec:=request(h,"POST","/v1/batches",marshal(b),"");if rec.Code!=201{t.Fatal(rec.Code,rec.Body)};if err:=json.Unmarshal(rec.Body.Bytes(),&b);err!=nil{t.Fatal(err)};if b.BatteryStatus!="unknown" || b.ReviewStatus!="pending" || b.ReviewEvidence!=""{t.Fatal("intake must start pending",b)}
 b.BatteryStatus="cleared";b.ReviewStatus="approved";b.ReviewEvidence="human inspected";path:="/v1/batches/"+b.ID
 rec=request(h,"PUT",path,marshal(b),"");if rec.Code!=428{t.Fatal(rec.Code,rec.Body)}
 rec=request(h,"PUT",path,marshal(b),"1");if rec.Code!=200{t.Fatal(rec.Code,rec.Body)}
 rec=request(h,"PUT",path,marshal(b),"1");if rec.Code!=409{t.Fatal(rec.Code,rec.Body)}
}
func TestAuthValidationAndBodyLimit(t *testing.T){_,h,_,_:=setup();rec:=httptest.NewRecorder();h.ServeHTTP(rec,httptest.NewRequest("GET","/v1/routes",nil));if rec.Code!=401{t.Fatal(rec.Code)}
 rec=request(h,"POST","/v1/batches",[]byte(`{"unexpected":1}`),"");if rec.Code!=400{t.Fatal(rec.Code)}
 rec=request(h,"POST","/v1/batches",[]byte(`{"category":"`+strings.Repeat("x",1024*1024)+`"}`),"");if rec.Code!=413{t.Fatal(rec.Code,rec.Body)}
 rec=request(h,"GET","/v1/routes?cursor=!!!",nil,"");if rec.Code!=422{t.Fatal(rec.Code,rec.Body)}
}
func TestSimulationSnapshotAndStorageFailure(t *testing.T){_,h,r,o:=setup();b:=testBatch();b.ID="batch";b.Version=1;route:=testRoute();route.ID="route";route.Version=1;_ = r.Save(context.Background(),"batches",b.ID,0,b);_ = r.Save(context.Background(),"routes",route.ID,0,route)
 payload:=map[string]any{"batch_ids":[]string{b.ID},"route_ids":[]string{route.ID},"objective":"ROBUST","assumptions":[]string{"synthetic prices"}}
 rec:=request(h,"POST","/v1/optimize",marshal(payload),"");if rec.Code!=201{t.Fatal(rec.Code,rec.Body)};var base Run;if err:=json.Unmarshal(rec.Body.Bytes(),&base);err!=nil{t.Fatal(err)};if base.Result.Status!="OPTIMAL" || base.Result.WorstValue!=100{t.Fatal(base)}
 rec=request(h,"POST","/v1/simulate",marshal(map[string]any{"base_run_id":base.ID,"overrides":[]map[string]any{{"route_id":"route","capacity_kg":0}},"assumptions":[]string{"processor shut down"}}),"");if rec.Code!=201{t.Fatal(rec.Code,rec.Body)};var derived Run;_ = json.Unmarshal(rec.Body.Bytes(),&derived);if derived.Result.Status!="INFEASIBLE" || derived.ParentID!=base.ID{t.Fatal(derived)}
 rec=request(h,"GET","/v1/reports/"+base.ID,nil,"");var reloaded Run;_ = json.Unmarshal(rec.Body.Bytes(),&reloaded);if reloaded.Input.Routes[0].CapacityKG!=100 || reloaded.Result.WorstValue!=100{t.Fatal("base run mutated")}
 var catalog domain.Route;_ = r.Get(context.Background(),"routes","route",&catalog);if catalog.CapacityKG!=100{t.Fatal("route catalog mutated")}
 o.fail=true;before:=len(r.data);rec=request(h,"POST","/v1/optimize",marshal(payload),"");if rec.Code!=500 || len(r.data)!=before{t.Fatal("failed report must not persist a run",rec.Code)}
}
func TestImageValidation(t *testing.T){_,h,r,o:=setup();b:=testBatch();b.ID="b";_ = r.Save(context.Background(),"batches","b",0,b)
 path:="/v1/batches/b/images";rec:=request(h,"POST",path,[]byte("not an image"),"");if rec.Code!=415{t.Fatal(rec.Code)}
 rec=request(h,"POST",path,[]byte("\x89PNG\r\n\x1a\ninvalid"),"");if rec.Code!=422{t.Fatal(rec.Code,rec.Body)}
 var buf bytes.Buffer;if err:=png.Encode(&buf,image.NewRGBA(image.Rect(0,0,2,2)));err!=nil{t.Fatal(err)};rec=request(h,"POST",path,buf.Bytes(),"");if rec.Code!=201 || len(o.data)!=1{t.Fatal(rec.Code,rec.Body)}
}
