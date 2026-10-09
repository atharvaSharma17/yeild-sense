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
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	dt "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

var ErrNotFound = errors.New("record not found")
var ErrConflict = errors.New("version conflict")

type Repository interface {
	Get(context.Context, string, string, any) error
	Save(context.Context, string, string, int, any) error
	List(context.Context, string, string, int) ([]json.RawMessage, string, error)
	SaveAll(context.Context, []Write) error
}
type Write struct {
	Kind     string
	ID       string
	Expected int
	Value    any
}
type Blobs interface {
	Put(context.Context, string, []byte, string) error
	Get(context.Context, string) ([]byte, error)
}
type Dynamo struct {
	Client *dynamodb.Client
	Table  string
}

func str(s string) dt.AttributeValue { return &dt.AttributeValueMemberS{Value: s} }
func key(kind, id string) map[string]dt.AttributeValue {
	return map[string]dt.AttributeValue{"pk": str(kind), "sk": str(id)}
}
func (d *Dynamo) Get(ctx context.Context, kind, id string, out any) error {
	r, err := d.Client.GetItem(ctx, &dynamodb.GetItemInput{TableName: aws.String(d.Table), Key: key(kind, id), ConsistentRead: aws.Bool(true)})
	if err != nil {
		return err
	}
	if len(r.Item) == 0 {
		return ErrNotFound
	}
	p, ok := r.Item["payload"].(*dt.AttributeValueMemberS)
	if !ok {
		return errors.New("invalid stored record")
	}
	return json.Unmarshal([]byte(p.Value), out)
}

// expected=0 creates; otherwise compare-and-swap prevents lost updates.
func (d *Dynamo) Save(ctx context.Context, kind, id string, expected int, value any) error {
	p, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(p) > 300*1024 {
		return errors.New("record exceeds 300 KiB")
	}
	item := key(kind, id)
	item["payload"] = str(string(p))
	item["version"] = &dt.AttributeValueMemberN{Value: strconv.Itoa(expected + 1)}
	in := &dynamodb.PutItemInput{TableName: aws.String(d.Table), Item: item, ConditionExpression: aws.String("attribute_not_exists(pk)")}
	if expected > 0 {
		in.ConditionExpression = aws.String("#v = :v")
		in.ExpressionAttributeNames = map[string]string{"#v": "version"}
		in.ExpressionAttributeValues = map[string]dt.AttributeValue{":v": &dt.AttributeValueMemberN{Value: strconv.Itoa(expected)}}
	}
	_, err = d.Client.PutItem(ctx, in)
	var conflict *dt.ConditionalCheckFailedException
	if errors.As(err, &conflict) {
		return ErrConflict
	}
	return err
}
func (d *Dynamo) List(ctx context.Context, kind, cursor string, limit int) ([]json.RawMessage, string, error) {
	in := &dynamodb.QueryInput{TableName: aws.String(d.Table), KeyConditionExpression: aws.String("pk = :pk"), ExpressionAttributeValues: map[string]dt.AttributeValue{":pk": str(kind)}, Limit: aws.Int32(int32(limit)), ConsistentRead: aws.Bool(true)}
	if cursor != "" {
		id, err := base64.RawURLEncoding.DecodeString(cursor)
		if err != nil || len(id) == 0 || len(id) > 200 {
			return nil, "", fmt.Errorf("invalid cursor")
		}
		in.ExclusiveStartKey = key(kind, string(id))
	}
	r, err := d.Client.Query(ctx, in)
	if err != nil {
		return nil, "", err
	}
	items := []json.RawMessage{}
	for _, i := range r.Items {
		p, ok := i["payload"].(*dt.AttributeValueMemberS)
		if !ok {
			return nil, "", errors.New("invalid stored record")
		}
		items = append(items, json.RawMessage(p.Value))
	}
	next := ""
	if v, ok := r.LastEvaluatedKey["sk"].(*dt.AttributeValueMemberS); ok {
		next = base64.RawURLEncoding.EncodeToString([]byte(v.Value))
	}
	return items, next, nil
}

// SaveAll atomically commits prediction evidence and its batch safety state.
func (d *Dynamo) SaveAll(ctx context.Context, writes []Write) error {
	if len(writes) < 1 || len(writes) > 25 {
		return errors.New("provide 1-25 transactional writes")
	}
	items := make([]dt.TransactWriteItem, 0, len(writes))
	for _, w := range writes {
		p, err := json.Marshal(w.Value)
		if err != nil {
			return err
		}
		if len(p) > 300*1024 {
			return errors.New("record exceeds 300 KiB")
		}
		item := key(w.Kind, w.ID)
		item["payload"] = str(string(p))
		item["version"] = &dt.AttributeValueMemberN{Value: strconv.Itoa(w.Expected + 1)}
		put := &dt.Put{TableName: aws.String(d.Table), Item: item, ConditionExpression: aws.String("attribute_not_exists(pk)")}
		if w.Expected > 0 {
			put.ConditionExpression = aws.String("#v = :v")
			put.ExpressionAttributeNames = map[string]string{"#v": "version"}
			put.ExpressionAttributeValues = map[string]dt.AttributeValue{":v": &dt.AttributeValueMemberN{Value: strconv.Itoa(w.Expected)}}
		}
		items = append(items, dt.TransactWriteItem{Put: put})
	}
	_, err := d.Client.TransactWriteItems(ctx, &dynamodb.TransactWriteItemsInput{TransactItems: items})
	var canceled *dt.TransactionCanceledException
	if errors.As(err, &canceled) {
		for _, r := range canceled.CancellationReasons {
			if aws.ToString(r.Code) == "ConditionalCheckFailed" {
				return ErrConflict
			}
		}
	}
	return err
}

type S3 struct {
	Client *s3.Client
	Bucket string
}

func (s *S3) Put(ctx context.Context, key string, data []byte, mime string) error {
	_, err := s.Client.PutObject(ctx, &s3.PutObjectInput{Bucket: aws.String(s.Bucket), Key: aws.String(key), Body: bytes.NewReader(data), ContentType: aws.String(mime)})
	return err
}
func (s *S3) Get(ctx context.Context, key string) ([]byte, error) {
	r, err := s.Client.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(s.Bucket), Key: aws.String(key)})
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	limit := int64(8 * 1024 * 1024)
	if strings.HasPrefix(key, "raw/") {
		limit = 10 * 1024 * 1024
	}
	data, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if int64(len(data)) > limit {
		return nil, errors.New("object exceeds read limit")
	}
	return data, err
}
