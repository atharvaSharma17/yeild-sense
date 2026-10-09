package storage

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

func sdkConfig() aws.Config {
	return aws.Config{Region: "us-east-1", Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
		return aws.Credentials{AccessKeyID: "test", SecretAccessKey: "test"}, nil
	})}
}
func TestDynamoSDKRequestsAndConflicts(t *testing.T) {
	var item map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var p map[string]any
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "application/x-amz-json-1.0")
		if p["TableName"] != "records" {
			t.Error("table name not supplied")
		}
		switch r.Header.Get("X-Amz-Target") {
		case "DynamoDB_20120810.PutItem":
			condition := p["ConditionExpression"].(string)
			if item == nil {
				if condition != "attribute_not_exists(pk)" {
					t.Error("create must be conditional")
				}
			} else {
				vals, ok := p["ExpressionAttributeValues"].(map[string]any)
				expected := ""
				if ok {
					expected = vals[":v"].(map[string]any)["N"].(string)
				}
				if expected != item["version"].(map[string]any)["N"] {
					w.WriteHeader(400)
					_, _ = io.WriteString(w, `{"__type":"com.amazonaws.dynamodb.v20120810#ConditionalCheckFailedException","message":"conflict"}`)
					return
				}
			}
			item = p["Item"].(map[string]any)
			_, _ = io.WriteString(w, `{}`)
		case "DynamoDB_20120810.GetItem":
			if p["ConsistentRead"] != true {
				t.Error("strongly consistent read required")
			}
			if item == nil {
				_, _ = io.WriteString(w, `{}`)
			} else {
				_ = json.NewEncoder(w).Encode(map[string]any{"Item": item})
			}
		case "DynamoDB_20120810.Query":
			if p["KeyConditionExpression"] != "pk = :pk" || p["Limit"] != float64(1) {
				t.Error("query must use collection key and limit")
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"Items": []any{item}, "LastEvaluatedKey": map[string]any{"pk": map[string]any{"S": "batches"}, "sk": map[string]any{"S": "id"}}})
		default:
			t.Error("unexpected DynamoDB operation")
			w.WriteHeader(500)
		}
	}))
	defer server.Close()
	client := dynamodb.NewFromConfig(sdkConfig(), func(o *dynamodb.Options) { o.BaseEndpoint = aws.String(server.URL) })
	d := Dynamo{client, "records"}
	ctx := context.Background()
	var out map[string]any
	if err := d.Get(ctx, "batches", "id", &out); !errors.Is(err, ErrNotFound) {
		t.Fatal(err)
	}
	if err := d.Save(ctx, "batches", "id", 0, map[string]any{"version": 1, "name": "batch"}); err != nil {
		t.Fatal(err)
	}
	if err := d.Save(ctx, "batches", "id", 0, map[string]any{}); !errors.Is(err, ErrConflict) {
		t.Fatal("duplicate create", err)
	}
	if err := d.Save(ctx, "batches", "id", 1, map[string]any{"version": 2, "name": "reviewed"}); err != nil {
		t.Fatal(err)
	}
	if err := d.Save(ctx, "batches", "id", 1, map[string]any{}); !errors.Is(err, ErrConflict) {
		t.Fatal("stale update", err)
	}
	if err := d.Get(ctx, "batches", "id", &out); err != nil || out["name"] != "reviewed" {
		t.Fatal(out, err)
	}
	items, next, err := d.List(ctx, "batches", "", 1)
	if err != nil || len(items) != 1 || next == "" {
		t.Fatal(items, next, err)
	}
	if _, _, err := d.List(ctx, "batches", next, 1); err != nil {
		t.Fatal(err)
	}
	if _, _, err := d.List(ctx, "batches", "!!!", 1); err == nil {
		t.Fatal("invalid cursor accepted")
	}
	if err := d.Save(ctx, "batches", "huge", 0, strings.Repeat("x", 301*1024)); err == nil {
		t.Fatal("oversized record accepted")
	}
}
func TestS3SDKRoundTrip(t *testing.T) {
	var data []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/artifacts/reports/run.json" {
			t.Error("unexpected S3 key", r.URL.Path)
		}
		switch r.Method {
		case "PUT":
			if r.Header.Get("Content-Type") != "application/json" {
				t.Error("content type missing")
			}
			var err error
			data, err = io.ReadAll(r.Body)
			if err != nil {
				t.Error(err)
			}
			w.Header().Set("ETag", `"test"`)
		case "GET":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(data)
		default:
			t.Error("unexpected S3 operation", r.Method)
		}
	}))
	defer server.Close()
	client := s3.NewFromConfig(sdkConfig(), func(o *s3.Options) { o.BaseEndpoint = aws.String(server.URL); o.UsePathStyle = true })
	s := S3{client, "artifacts"}
	ctx := context.Background()
	if err := s.Put(ctx, "reports/run.json", []byte(`{"ok":true}`), "application/json"); err != nil {
		t.Fatal(err)
	}
	got, err := s.Get(ctx, "reports/run.json")
	if err != nil || string(got) != `{"ok":true}` {
		t.Fatal(string(got), err)
	}
}
func TestDynamoAtomicPredictionWrites(t *testing.T) {
	conflict := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Amz-Target") != "DynamoDB_20120810.TransactWriteItems" {
			t.Error("expected atomic write")
		}
		var p struct {
			TransactItems []struct {
				Put struct {
					TableName                 string
					ConditionExpression       string
					Item                      map[string]any
					ExpressionAttributeValues map[string]any
				}
			}
		}
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			t.Error(err)
		}
		if len(p.TransactItems) != 2 || p.TransactItems[0].Put.ConditionExpression != "attribute_not_exists(pk)" || p.TransactItems[1].Put.ConditionExpression != "#v = :v" {
			t.Error("prediction must create and batch must compare version")
		}
		if p.TransactItems[1].Put.ExpressionAttributeValues[":v"].(map[string]any)["N"] != "3" {
			t.Error("expected batch version missing")
		}
		w.Header().Set("Content-Type", "application/x-amz-json-1.0")
		if conflict {
			w.WriteHeader(400)
			_, _ = io.WriteString(w, `{"__type":"com.amazonaws.dynamodb.v20120810#TransactionCanceledException","message":"conflict","CancellationReasons":[{"Code":"None"},{"Code":"ConditionalCheckFailed"}]}`)
			return
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	defer server.Close()
	client := dynamodb.NewFromConfig(sdkConfig(), func(o *dynamodb.Options) { o.BaseEndpoint = aws.String(server.URL) })
	d := Dynamo{client, "records"}
	writes := []Write{{"model_outputs", "prediction", 0, map[string]any{"id": "prediction"}}, {"batches", "batch", 3, map[string]any{"version": 4}}}
	if err := d.SaveAll(context.Background(), writes); err != nil {
		t.Fatal(err)
	}
	conflict = true
	if err := d.SaveAll(context.Background(), writes); !errors.Is(err, ErrConflict) {
		t.Fatal("transaction conflict not mapped", err)
	}
}
