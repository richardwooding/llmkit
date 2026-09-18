package llmkit

import (
	"fmt"

	"github.com/richardwooding/llmkit/core"
)

// Register adds a provider to the Default registry.
func Register(p core.Provider, aliases ...string) { Default.Register(p, aliases...) }

// SetFallback sets the Default registry's fallback provider.
func SetFallback(id string) { Default.SetFallback(id) }

// ParseModel resolves a model name against the Default registry.
func ParseModel(s string) (core.Provider, string, error) { return Default.ParseModel(s) }

// New opens a client for model using the Default registry.
func New(model string, opts ...core.Option) (core.Client, error) { return Default.New(model, opts...) }

// Open resolves model and asserts the client to T, typically Chatter,
// Streamer, Embedder or an interface combining them.
func Open[T any](model string, opts ...core.Option) (T, error) {
	return OpenWith[T](Default, model, opts...)
}

// OpenWith is Open against a specific Registry.
func OpenWith[T any](r *Registry, model string, opts ...core.Option) (T, error) {
	var zero T
	c, err := r.New(model, opts...)
	if err != nil {
		return zero, err
	}
	return As[T](c)
}

// As asserts c to T, returning ErrUnsupported when the provider lacks the
// capability.
func As[T any](c core.Client) (T, error) {
	if t, ok := c.(T); ok {
		return t, nil
	}
	var zero T
	return zero, fmt.Errorf("%s/%s does not implement %T: %w", c.Provider(), c.Model(), zero, core.ErrUnsupported)
}
