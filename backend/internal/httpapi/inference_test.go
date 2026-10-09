package httpapi

import (
 "context"
 "encoding/json"
 "strings"
 "testing"

 "yieldsense/backend/internal/domain"
 "yieldsense/backend/internal/inference"
 "yieldsense/backend/internal/storage"
)
type predictorFunc func(context.Context,inference.Request)(inference.Response,error)
func(f predictorFunc)Predict(ctx context.Context,r inference.Request)(inference.Response,error){return f(ctx,r)}
func manifest()inference.Model{return inference.Model{Version:"xbat-v1",ContractVersion:"1",Track:"B",Task:inference.DeviceTask,Architecture:"resnet18",ArtifactS3URI:"s3://bucket/models/v1/model.tar.gz",ArtifactSHA256:strings.Repeat("a",64),DatasetManifestSHA256:strings.Repeat("b",64),DataVersion:"verified-xbat",TrainingJobARN:"arn:aws:sagemaker:ap-south-1:123456789012:training-job/job",Labels:[]string{"device"},Preprocessing:inference.Preprocessing{Width:224,Height:224,ColorMode:"RGB",Scale:1.0/255,Mean:[]float64{.485,.456,.406},Std:[]float64{.229,.224,.225}},ConfidenceThreshold:.7,Evaluation:map[string]float64{"macro_f1":.8}}}
func seedPrediction(t *testing.T,r *records,o *objects){t.Helper();b:=testBatch();b.ID="b";b.Version=1;_ = r.Save(context.Background(),"batches",b.ID,0,b);m:=manifest();_ = r.Save(context.Background(),"models",m.Version,0,m);data:=[]byte("\x89PNG\r\n\x1a\nfixture");img:=inference.Image{ID:"image",BatchID:"b",ObjectKey:"raw/b/image.png",SizeBytes:len(data),ContentType:"image/png"};img.SHA256=imageHash(data);_ = r.Save(context.Background(),"images",img.ID,0,img);o.data[img.ObjectKey]=data}
func imageHash(data []byte)string {return hashBytes(data)}
func TestPredictDeviceSafetyAndAudit(t *testing.T){a,_,r,o:=setup();a.Bucket="bucket";a.Predictor=predictorFunc(func(ctx context.Context,req inference.Request)(inference.Response,error){return inference.Response{ContractVersion:"1",RequestID:req.RequestID,ModelVersion:req.Model.Version,Track:"B",Task:inference.DeviceTask,Probabilities:map[string]float64{"device":1},BatteryAlert:"suspected"},nil});h:=a.Echo();seedPrediction(t,r,o)
 rec:=request(h,"POST","/v1/predict-device",marshal(map[string]string{"batch_id":"b","image_id":"image","model_version":"xbat-v1"}),"1");if rec.Code!=201{t.Fatal(rec.Code,rec.Body)};var result struct{Prediction Prediction `json:"prediction"`;Batch domain.Batch `json:"batch"`};if err:=json.Unmarshal(rec.Body.Bytes(),&result);err!=nil{t.Fatal(err)};if result.Batch.ReviewStatus!="pending"||result.Batch.BatteryStatus!="suspected"||result.Batch.Version!=2||!result.Prediction.HumanReviewRequired{t.Fatal(result)};if result.Batch.Composition.Source!="operator_entered"{t.Fatal("device prediction overwrote composition")};var p Prediction;if err:=r.Get(context.Background(),"model_outputs",result.Prediction.ID,&p);err!=nil{t.Fatal(err)}
 route:=testRoute();route.ID="r";_ = r.Save(context.Background(),"routes",route.ID,0,route);rec=request(h,"POST","/v1/optimize",marshal(map[string]any{"batch_ids":[]string{"b"},"route_ids":[]string{"r"},"assumptions":[]string{"synthetic"}}),"");var run Run;_ = json.Unmarshal(rec.Body.Bytes(),&run);if rec.Code!=201||run.Result.Status!="INFEASIBLE"||len(run.ModelOutputs)!=1{t.Fatal(rec.Code,rec.Body)}
 rec=request(h,"POST","/v1/predict-device",marshal(map[string]string{"batch_id":"b","image_id":"image","model_version":"xbat-v1"}),"1");if rec.Code!=409{t.Fatal(rec.Code,rec.Body)}
}
func TestInferenceFailureDoesNotWrite(t *testing.T){a,_,r,o:=setup();a.Bucket="bucket";seedPrediction(t,r,o);body:=marshal(map[string]string{"batch_id":"b","image_id":"image","model_version":"xbat-v1"});h:=a.Echo();rec:=request(h,"POST","/v1/predict-device",body,"1");if rec.Code!=503{t.Fatal(rec.Code)};before:=len(r.data)
 a.Predictor=predictorFunc(func(context.Context,inference.Request)(inference.Response,error){return inference.Response{},inference.ErrTimeout});rec=request(h,"POST","/v1/predict-device",body,"1");if rec.Code!=504||len(r.data)!=before{t.Fatal(rec.Code,rec.Body)};var b domain.Batch;_ = r.Get(context.Background(),"batches","b",&b);if b.Version!=1||b.ReviewStatus!="approved"{t.Fatal("failure changed batch")}
 a.Predictor=predictorFunc(func(ctx context.Context,req inference.Request)(inference.Response,error){return inference.Response{ContractVersion:"1",RequestID:req.RequestID,ModelVersion:"wrong",Track:"B",Task:inference.DeviceTask},nil});rec=request(h,"POST","/v1/predict-device",body,"1");if rec.Code!=502||len(r.data)!=before{t.Fatal(rec.Code,rec.Body)}
}
func TestAtomicPredictionConflict(t *testing.T){a,_,r,o:=setup();a.Bucket="bucket";seedPrediction(t,r,o);a.Predictor=predictorFunc(func(ctx context.Context,req inference.Request)(inference.Response,error){var b domain.Batch;_ = r.Get(ctx,"batches","b",&b);b.Version++;_ = r.Save(ctx,"batches","b",1,b);return inference.Response{ContractVersion:"1",RequestID:req.RequestID,ModelVersion:req.Model.Version,Track:"B",Task:inference.DeviceTask,Probabilities:map[string]float64{"device":1},BatteryAlert:"suspected"},nil});h:=a.Echo();rec:=request(h,"POST","/v1/predict-device",marshal(map[string]string{"batch_id":"b","image_id":"image","model_version":"xbat-v1"}),"1");if rec.Code!=409{t.Fatal(rec.Code,rec.Body)};items,_,_:=r.List(context.Background(),"model_outputs","",10);if len(items)!=0{t.Fatal("orphan prediction committed on version conflict")}
}
func TestModelRegistrationImmutable(t *testing.T){a,_,r,_:=setup();a.Bucket="bucket";h:=a.Echo();m:=manifest();rec:=request(h,"POST","/v1/models",marshal(m),"");if rec.Code!=201{t.Fatal(rec.Code,rec.Body)};rec=request(h,"POST","/v1/models",marshal(m),"");if rec.Code!=409{t.Fatal(rec.Code)};var stored inference.Model;_ = r.Get(context.Background(),"models",m.Version,&stored);if stored.CreatedAt==""{t.Fatal("model provenance missing registration time")}}
var _ storage.Repository = (*records)(nil)
