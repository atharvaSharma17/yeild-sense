package main

import (
	"context"
	"errors"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
	"yieldsense/backend/internal/httpapi"
	"yieldsense/backend/internal/storage"
)

func main() {
	if err := run(); err != nil {
		slog.Error("server stopped", "error", err)
		os.Exit(1)
	}
}
func run() error {
	table, bucket, key := os.Getenv("DYNAMODB_TABLE"), os.Getenv("S3_BUCKET"), os.Getenv("API_KEY")
	if table == "" || bucket == "" || len(key) < 32 {
		return errors.New("DYNAMODB_TABLE, S3_BUCKET and API_KEY (at least 32 characters) required")
	}
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		return err
	}
	d := dynamodb.NewFromConfig(cfg, func(o *dynamodb.Options) {
		if v := os.Getenv("DYNAMODB_ENDPOINT"); v != "" {
			o.BaseEndpoint = &v
		}
	})
	s := s3.NewFromConfig(cfg, func(o *s3.Options) {
		if v := os.Getenv("S3_ENDPOINT"); v != "" {
			o.BaseEndpoint = &v
			o.UsePathStyle = true
		}
	})
	origins := []string{}
	for _, v := range strings.Split(os.Getenv("CORS_ORIGINS"), ",") {
		if v = strings.TrimSpace(v); v != "" {
			origins = append(origins, v)
		}
	}
	a := httpapi.API{Records: &storage.Dynamo{Client: d, Table: table}, Objects: &storage.S3{Client: s, Bucket: bucket}, Key: key, Origins: origins}
	addr := os.Getenv("LISTEN_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	srv := &http.Server{Addr: addr, Handler: a.Echo(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 20 * time.Second, WriteTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 * 1024}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	done := make(chan error, 1)
	go func() { slog.Info("listening", "address", addr); done <- srv.ListenAndServe() }()
	select {
	case err := <-done:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return srv.Shutdown(shutdown)
	}
}
