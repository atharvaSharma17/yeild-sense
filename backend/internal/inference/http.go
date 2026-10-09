package inference

import (
 "bytes"
 "context"
 "encoding/json"
 "errors"
 "io"
 "net/http"
 "net/url"
 "time"
)

var ErrUnavailable=errors.New("model inference unavailable")
var ErrTimeout=errors.New("model inference timed out")
var ErrInvalidResponse=errors.New("invalid model inference response")
type Predictor interface {Predict(context.Context,Request)(Response,error)}
type HTTP struct{URL string; Token string; Client *http.Client}
func NewHTTP(endpoint,token string)(*HTTP,error){
 u,err:=url.Parse(endpoint);if err!=nil||u.Host==""||u.User!=nil||u.RawQuery!=""||u.Fragment!=""||(u.Scheme!="http"&&u.Scheme!="https"){return nil,errors.New("ML_INFERENCE_URL must be an HTTP(S) endpoint without credentials, query or fragment")}
 return &HTTP{URL:endpoint,Token:token,Client:&http.Client{Timeout:12*time.Second,CheckRedirect:func(*http.Request,[]*http.Request)error{return http.ErrUseLastResponse}}},nil
}
func (h *HTTP)Predict(ctx context.Context,in Request)(Response,error){
 var out Response;data,err:=json.Marshal(in);if err!=nil{return out,err};req,err:=http.NewRequestWithContext(ctx,http.MethodPost,h.URL,bytes.NewReader(data));if err!=nil{return out,err};req.Header.Set("Content-Type","application/json");req.Header.Set("Accept","application/json");req.Header.Set("X-Request-ID",in.RequestID);if h.Token!=""{req.Header.Set("Authorization","Bearer "+h.Token)}
 res,err:=h.Client.Do(req);if err!=nil{if ctx.Err()!=nil||errors.Is(err,context.DeadlineExceeded){return out,ErrTimeout};return out,ErrUnavailable};defer res.Body.Close()
 if res.StatusCode<200||res.StatusCode>=300 {return out,ErrUnavailable}
 if res.Header.Get("Content-Type")!="application/json"&&res.Header.Get("Content-Type")!="application/json; charset=utf-8"{return out,ErrInvalidResponse}
 body,err:=io.ReadAll(io.LimitReader(res.Body,256*1024+1));if err!=nil{return out,ErrUnavailable};if len(body)>256*1024{return out,ErrInvalidResponse};d:=json.NewDecoder(bytes.NewReader(body));d.DisallowUnknownFields();if err:=d.Decode(&out);err!=nil{return out,ErrInvalidResponse};var extra any;if d.Decode(&extra)!=io.EOF{return out,ErrInvalidResponse};return out,nil
}
