// Package httpx is the shared HTTP plumbing for llmkit providers: JSON and
// streaming POSTs, SSE and NDJSON readers, and error-envelope decoding.
package httpx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/richardwooding/llmkit/core"
)

const maxErrorBody = 64 << 10

// AuthFunc adds credentials to a request. It may perform I/O (token refresh).
type AuthFunc func(ctx context.Context, req *http.Request) error

// Client funnels every provider request through one place that applies auth,
// default headers, timeouts and error-envelope decoding.
type Client struct {
	Provider string
	BaseURL  string
	HTTP     *http.Client
	Headers  http.Header
	Timeout  time.Duration
	Auth     AuthFunc
}

// New builds a Client from a core.Config and provider defaults.
func New(provider string, cfg *core.Config, defaultBaseURL string) *Client {
	c := &Client{Provider: provider, BaseURL: defaultBaseURL, HTTP: cfg.Client(), Timeout: cfg.Timeout}
	if cfg.BaseURL != "" {
		c.BaseURL = cfg.BaseURL
	}
	c.BaseURL = strings.TrimRight(c.BaseURL, "/")
	if len(cfg.Headers) > 0 {
		c.Headers = cfg.Headers.Clone()
	}
	return c
}

// BearerAuth returns an AuthFunc setting "Authorization: Bearer <key>".
func BearerAuth(key string) AuthFunc {
	return func(_ context.Context, req *http.Request) error {
		req.Header.Set("Authorization", "Bearer "+key)
		return nil
	}
}

// HeaderAuth returns an AuthFunc setting an arbitrary header.
func HeaderAuth(name, value string) AuthFunc {
	return func(_ context.Context, req *http.Request) error {
		req.Header.Set(name, value)
		return nil
	}
}

// PostJSON sends body as JSON and decodes a 2xx response into out (which may
// be nil). The raw response body is returned for Response.Raw.
func (c *Client) PostJSON(ctx context.Context, path string, body []byte, out any) (json.RawMessage, error) {
	ctx, cancel := c.withTimeout(ctx)
	defer cancel()
	resp, err := c.do(ctx, http.MethodPost, path, body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s: read response: %w", c.Provider, err)
	}
	if out != nil {
		if err := json.Unmarshal(raw, out); err != nil {
			return nil, fmt.Errorf("%s: decode response: %w", c.Provider, err)
		}
	}
	return raw, nil
}

// PostStream sends body as JSON and returns the response body for incremental
// reading. Closing it releases the connection and the timeout.
func (c *Client) PostStream(ctx context.Context, path string, body []byte) (io.ReadCloser, error) {
	ctx, cancel := c.withTimeout(ctx)
	resp, err := c.do(ctx, http.MethodPost, path, body)
	if err != nil {
		cancel()
		return nil, err
	}
	return &cancelReader{ReadCloser: resp.Body, cancel: cancel}, nil
}

type cancelReader struct {
	io.ReadCloser
	cancel context.CancelFunc
}

// Close releases the body and the per-call timeout.
func (r *cancelReader) Close() error {
	err := r.ReadCloser.Close()
	r.cancel()
	return err
}

func (c *Client) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if c.Timeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, c.Timeout)
}

func (c *Client) do(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	url := path
	if !strings.HasPrefix(path, "http://") && !strings.HasPrefix(path, "https://") {
		url = c.BaseURL + path
	}
	var rdr io.Reader = http.NoBody
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return nil, fmt.Errorf("%s: build request: %w", c.Provider, err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json, text/event-stream")
	for k, vs := range c.Headers {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	if c.Auth != nil {
		if err := c.Auth(ctx, req); err != nil {
			return nil, fmt.Errorf("%s: auth: %w", c.Provider, err)
		}
	}
	hc := c.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.Provider, err)
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	defer func() { _ = resp.Body.Close() }()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
	return nil, ParseError(c.Provider, resp.StatusCode, resp.Header, raw)
}
