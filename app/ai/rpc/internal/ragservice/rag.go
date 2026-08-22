package rag

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	baseURL string
	httpCli *http.Client
}

type SearchRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

type SearchResponse struct {
	Documents []string `json:"documents"`
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpCli: &http.Client{Timeout: 5 * time.Second},
	}
}

func (c *Client) Search(query string, topK int) ([]string, error) {
	reqBody := SearchRequest{Query: query, TopK: topK}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpCli.Post(c.baseURL+"/search", "application/json", bytes.NewReader(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rag service returned status %d", resp.StatusCode)
	}
	var result SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Documents, nil
}
