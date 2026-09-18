package core

import (
	"net/http"
	"time"
)

// Config carries connection settings for a Provider.Open call.
type Config struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Headers    http.Header
	Timeout    time.Duration

	values map[any]any
}

// Option mutates a Config.
type Option func(*Config)

// NewConfig applies opts to a fresh Config.
func NewConfig(opts ...Option) *Config {
	cfg := &Config{}
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}
	return cfg
}

// WithAPIKey overrides the provider's environment-sourced key.
func WithAPIKey(key string) Option { return func(c *Config) { c.APIKey = key } }

// WithBaseURL points the client at a different endpoint.
func WithBaseURL(u string) Option { return func(c *Config) { c.BaseURL = u } }

// WithHTTPClient replaces the default http.Client.
func WithHTTPClient(hc *http.Client) Option { return func(c *Config) { c.HTTPClient = hc } }

// WithHeader adds a header to every request.
func WithHeader(key, value string) Option {
	return func(c *Config) {
		if c.Headers == nil {
			c.Headers = http.Header{}
		}
		c.Headers.Add(key, value)
	}
}

// WithTimeout bounds each call, including a whole stream.
func WithTimeout(d time.Duration) Option { return func(c *Config) { c.Timeout = d } }

// WithValue stores a provider-specific setting; provider packages wrap it in
// typed options with unexported keys.
func WithValue(key, value any) Option {
	return func(c *Config) {
		if c.values == nil {
			c.values = map[any]any{}
		}
		c.values[key] = value
	}
}

// Value returns the setting stored by WithValue, or nil.
func (c *Config) Value(key any) any {
	if c == nil {
		return nil
	}
	return c.values[key]
}

// Client returns the configured http.Client, falling back to a fresh one.
func (c *Config) Client() *http.Client {
	if c != nil && c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{}
}
