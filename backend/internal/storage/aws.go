package storage

import (
 "bytes"
 "context"
 "encoding/base64"
 "encoding/json"
 "errors"
 "fmt"
 "io"
 "strconv"

 "github.com/aws/aws-sdk-go-v2/aws"
 "github.com/aws/aws-sdk-go-v2/service/dynamodb"
 dt "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
 "github.com/aws/aws-sdk-go-v2/service/s3"
)

var ErrNotFound=errors.New("record not found")
var ErrConflict=errors.New("version conflict")
type Repository interface {
 Get(context.Context,string,string,any) error
 Save(context.Context,string,string,int,any) error
 List(context.Context,string,string,int)([]json.RawMessage,string,error)
}
type Blobs interface {
 Put(context.Context,string,[]byte,string) error
 Get(context.Context,string)([]byte,error)
}
type Dynamo struct { Client *dynamodb.Client; Table string }
func str(s string) dt.AttributeValue {return &dt.AttributeValueMemberS{Value:s}}
func key(kind,id string) map[string]dt.AttributeValue {return map[string]dt.AttributeValue{"pk":str(kind),"sk":str(id)}}
func (d *Dynamo) Get(ctx context.Context,kind,id string,out any) error {
 r,err:=d.Client.GetItem(ctx,&dynamodb.GetItemInput{TableName:aws.String(d.Table),Key:key(kind,id),ConsistentRead:aws.Bool(true)});if err!=nil{return err};if len(r.Item)==0{return ErrNotFound};p,ok:=r.Item["payload"].(*dt.AttributeValueMemberS);if !ok{return errors.New("invalid stored record")};return json.Unmarshal([]byte(p.Value),out)
}
// expected=0 creates; otherwise compare-and-swap prevents lost updates.
func (d *Dynamo) Save(ctx context.Context,kind,id string,expected int,value any) error {
 p,err:=json.Marshal(value);if err!=nil{return err};if len(p)>300*1024{return errors.New("record exceeds 300 KiB")};item:=key(kind,id);item["payload"]=str(string(p));item["version"]=&dt.AttributeValueMemberN{Value:strconv.Itoa(expected+1)}
 in:=&dynamodb.PutItemInput{TableName:aws.String(d.Table),Item:item,ConditionExpression:aws.String("attribute_not_exists(pk)")}
 if expected>0 {in.ConditionExpression=aws.String("#v = :v");in.ExpressionAttributeNames=map[string]string{"#v":"version"};in.ExpressionAttributeValues=map[string]dt.AttributeValue{":v":&dt.AttributeValueMemberN{Value:strconv.Itoa(expected)}}}
 _,err=d.Client.PutItem(ctx,in);var conflict *dt.ConditionalCheckFailedException;if errors.As(err,&conflict){return ErrConflict};return err
}
func (d *Dynamo) List(ctx context.Context,kind,cursor string,limit int)([]json.RawMessage,string,error) {
 in:=&dynamodb.QueryInput{TableName:aws.String(d.Table),KeyConditionExpression:aws.String("pk = :pk"),ExpressionAttributeValues:map[string]dt.AttributeValue{":pk":str(kind)},Limit:aws.Int32(int32(limit)),ConsistentRead:aws.Bool(true)}
 if cursor!="" {id,err:=base64.RawURLEncoding.DecodeString(cursor);if err!=nil || len(id)==0 || len(id)>200{return nil,"",fmt.Errorf("invalid cursor")};in.ExclusiveStartKey=key(kind,string(id))}
 r,err:=d.Client.Query(ctx,in);if err!=nil{return nil,"",err};items:=[]json.RawMessage{};for _,i:=range r.Items {p,ok:=i["payload"].(*dt.AttributeValueMemberS);if !ok{return nil,"",errors.New("invalid stored record")};items=append(items,json.RawMessage(p.Value))};next:="";if v,ok:=r.LastEvaluatedKey["sk"].(*dt.AttributeValueMemberS);ok{next=base64.RawURLEncoding.EncodeToString([]byte(v.Value))};return items,next,nil
}
type S3 struct {Client *s3.Client; Bucket string}
func (s *S3) Put(ctx context.Context,key string,data []byte,mime string)error {_,err:=s.Client.PutObject(ctx,&s3.PutObjectInput{Bucket:aws.String(s.Bucket),Key:aws.String(key),Body:bytes.NewReader(data),ContentType:aws.String(mime)});return err}
func (s *S3) Get(ctx context.Context,key string)([]byte,error) {r,err:=s.Client.GetObject(ctx,&s3.GetObjectInput{Bucket:aws.String(s.Bucket),Key:aws.String(key)});if err!=nil{return nil,err};defer r.Body.Close();data,err:=io.ReadAll(io.LimitReader(r.Body,8*1024*1024+1));if len(data)>8*1024*1024{return nil,errors.New("object exceeds read limit")};return data,err}
