package rag

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestSearchDoesNotExposeUpstreamBody 验证 RAG 上游错误正文不会透传给调用方。
func TestSearchDoesNotExposeUpstreamBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.WriteHeader(http.StatusInternalServerError)
		_, _ = writer.Write([]byte("internal secret detail"))
	}))
	defer server.Close()

	_, err := NewClient(server.URL).Search("test", 3)
	if err == nil {
		t.Fatal("Search() error = nil, want upstream status error")
	}
	if strings.Contains(err.Error(), "internal secret detail") {
		t.Fatalf("Search() leaked upstream body: %v", err)
	}
}
