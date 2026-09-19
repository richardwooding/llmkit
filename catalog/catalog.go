// Package catalog is a static table of model metadata: context windows,
// output limits, list prices and capabilities, keyed by model ID. It depends
// on the standard library and core only, so it can be used without opening a
// client. The data is a snapshot (see DataAsOf); Register overrides a row and
// Lookup returns an Unknown model, rather than a guess, for anything absent.
package catalog

import (
	"sort"
	"strings"
	"sync"

	"github.com/richardwooding/llmkit/core"
)

// DataAsOf is the date the seed data was last checked against vendor pages.
const DataAsOf = "2026-09-19"

// Pricing is USD per million tokens. CacheRead and CacheWrite price the
// Usage.CachedInputTokens and Usage.CacheWriteTokens subsets of the prompt;
// providers without a write premium set CacheWrite equal to Input.
type Pricing struct {
	Input      float64
	Output     float64
	CacheRead  float64
	CacheWrite float64
}

// Capabilities lists what a model accepts or produces.
type Capabilities struct {
	Tools       bool
	Vision      bool
	Reasoning   bool
	JSONSchema  bool
	PromptCache bool
}

// Model describes one catalog row. Known is false for the Unknown placeholder
// Lookup returns when nothing matches.
type Model struct {
	ID            string
	Provider      string
	Family        string
	DisplayName   string
	ContextWindow int
	MaxOutput     int
	Pricing       Pricing
	Capabilities  Capabilities
	Aliases       []string
	Known         bool
}

// Cost prices u at m's list rates: uncached input, cache reads and cache
// writes at their own rates, plus output. It follows the core.Usage
// convention that CachedInputTokens and CacheWriteTokens are subsets of
// InputTokens. An unknown model costs zero.
func (m Model) Cost(u core.Usage) float64 {
	uncached := max(u.InputTokens-u.CachedInputTokens-u.CacheWriteTokens, 0)
	p := m.Pricing
	return (float64(uncached)*p.Input +
		float64(u.CachedInputTokens)*p.CacheRead +
		float64(u.CacheWriteTokens)*p.CacheWrite +
		float64(u.OutputTokens)*p.Output) / 1e6
}

var (
	mu      sync.RWMutex
	byID    = map[string]Model{}
	byAlias = map[string]string{}
)

func init() {
	for _, m := range seed {
		Register(m)
	}
}

// Register adds or replaces a row by ID, marking it Known. Aliases are
// case-insensitive and override any earlier owner.
func Register(m Model) {
	m.Known = true
	mu.Lock()
	defer mu.Unlock()
	id := strings.ToLower(m.ID)
	for alias, owner := range byAlias {
		if owner == id {
			delete(byAlias, alias)
		}
	}
	byID[id] = m
	for _, a := range m.Aliases {
		byAlias[strings.ToLower(a)] = id
	}
}

// All returns every row, sorted by provider then ID.
func All() []Model {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]Model, 0, len(byID))
	for _, m := range byID {
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Provider != out[j].Provider {
			return out[i].Provider < out[j].Provider
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// Unknown is the placeholder for a model the catalog has no data for.
func Unknown(model string) Model { return Model{ID: model} }

// Lookup resolves a model name to its row: exact ID, then alias, then the
// longest known ID that is a prefix followed by a dated or versioned suffix
// ("-20251101", "@20250929", ":latest" are suffixes; "-mini" is not, so a
// variant the catalog lacks stays Unknown instead of borrowing a sibling's
// prices). A leading "provider/" prefix and a Vertex "@version" suffix are
// stripped. Matching is case-insensitive; the result is Unknown(model) when
// nothing matches.
func Lookup(model string) Model {
	mu.RLock()
	defer mu.RUnlock()
	cands := candidates(model)
	for _, c := range cands {
		if m, ok := exact(c); ok {
			return m
		}
	}
	for _, c := range cands {
		if m, ok := longestPrefix(c); ok {
			return m
		}
	}
	return Unknown(model)
}

// candidates lists the name, then the name with each leading "x/" segment
// removed, each also without any "@version" suffix. Groq-style IDs that
// contain a slash themselves are tried whole first.
func candidates(model string) []string {
	var out []string
	s := strings.ToLower(strings.TrimSpace(model))
	for {
		out = append(out, s)
		if base, _, ok := strings.Cut(s, "@"); ok && base != "" {
			out = append(out, base)
		}
		_, rest, ok := strings.Cut(s, "/")
		if !ok || rest == "" {
			return out
		}
		s = rest
	}
}

func exact(name string) (Model, bool) {
	if m, ok := byID[name]; ok {
		return m, true
	}
	if id, ok := byAlias[name]; ok {
		return byID[id], true
	}
	return Model{}, false
}

func longestPrefix(name string) (Model, bool) {
	var best Model
	found := false
	for id, m := range byID {
		if len(id) >= len(name) || !strings.HasPrefix(name, id) || !versionSuffix(name[len(id):]) {
			continue
		}
		if !found || len(id) > len(best.ID) {
			best, found = m, true
		}
	}
	return best, found
}

// versionSuffix accepts a separator followed by a digit, the shape of dated
// snapshots and pinned versions.
func versionSuffix(s string) bool {
	return len(s) >= 2 && strings.ContainsRune("-@:", rune(s[0])) && s[1] >= '0' && s[1] <= '9'
}
