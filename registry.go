package llmkit

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/richardwooding/llmkit/core"
)

// DefaultProviderEnv names the environment variable consulted for the
// fallback provider when a bare model name matches nothing.
const DefaultProviderEnv = "LLMKIT_DEFAULT_PROVIDER"

// Registry resolves model names to providers. Registration order is match
// priority for bare names.
type Registry struct {
	mu       sync.RWMutex
	order    []core.Provider
	byID     map[string]core.Provider
	fallback string
}

// NewRegistry builds a Registry holding providers in the given order.
func NewRegistry(providers ...core.Provider) *Registry {
	r := &Registry{byID: map[string]core.Provider{}}
	for _, p := range providers {
		r.Register(p)
	}
	return r
}

// Register adds a provider under its ID and any aliases, replacing an earlier
// provider with the same ID.
func (r *Registry) Register(p core.Provider, aliases ...string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	id := strings.ToLower(p.ID())
	if _, exists := r.byID[id]; exists {
		for i, existing := range r.order {
			if strings.ToLower(existing.ID()) == id {
				r.order[i] = p
			}
		}
	} else {
		r.order = append(r.order, p)
	}
	r.byID[id] = p
	for _, a := range aliases {
		r.byID[strings.ToLower(a)] = p
	}
}

// SetFallback names the provider used for bare model names nothing claims.
// It takes precedence over LLMKIT_DEFAULT_PROVIDER.
func (r *Registry) SetFallback(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fallback = strings.ToLower(id)
}

// Providers returns the registered providers in priority order.
func (r *Registry) Providers() []core.Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]core.Provider, len(r.order))
	copy(out, r.order)
	return out
}

// Lookup returns the provider registered under id or an alias.
func (r *Registry) Lookup(id string) (core.Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.byID[strings.ToLower(id)]
	return p, ok
}

// ParseModel splits "<provider>/<model>" or resolves a bare "<model>" to the
// provider that claims it, then the fallback.
func (r *Registry) ParseModel(s string) (core.Provider, string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, "", fmt.Errorf("llmkit: empty model name")
	}
	if prefix, rest, ok := strings.Cut(s, "/"); ok && rest != "" {
		if p, found := r.Lookup(prefix); found {
			return p, rest, nil
		}
	}
	r.mu.RLock()
	order := r.order
	fallback := r.fallback
	r.mu.RUnlock()
	for _, p := range order {
		if p.Matches(s) {
			return p, s, nil
		}
	}
	if fallback == "" {
		fallback = strings.ToLower(os.Getenv(DefaultProviderEnv))
	}
	if fallback == "" {
		fallback = "ollama"
	}
	p, ok := r.Lookup(fallback)
	if !ok {
		return nil, "", fmt.Errorf("llmkit: no provider claims %q and fallback %q is not registered: %w", s, fallback, core.ErrUnknownProvider)
	}
	return p, s, nil
}

// New resolves model and opens a client for it.
func (r *Registry) New(model string, opts ...core.Option) (core.Client, error) {
	p, name, err := r.ParseModel(model)
	if err != nil {
		return nil, err
	}
	return p.Open(name, core.NewConfig(opts...))
}
