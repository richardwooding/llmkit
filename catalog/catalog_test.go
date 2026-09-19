package catalog_test

import (
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/richardwooding/llmkit/catalog"
	"github.com/richardwooding/llmkit/core"
)

func TestLookup(t *testing.T) {
	tests := []struct {
		name   string
		model  string
		wantID string
		known  bool
	}{
		{"exact", "claude-opus-5", "claude-opus-5", true},
		{"case-insensitive", "Claude-Opus-5", "claude-opus-5", true},
		{"alias to dated id", "claude-sonnet-4-5", "claude-sonnet-4-5-20250929", true},
		{"provider prefix", "anthropic/claude-haiku-4-5", "claude-haiku-4-5-20251001", true},
		{"vertex version suffix", "claude-opus-4-5@20251101", "claude-opus-4-5-20251101", true},
		{"vertexgrpc prefix", "vertexgrpc/gemini-2.5-flash", "gemini-2.5-flash", true},
		{"dated snapshot by prefix", "gpt-4o-2024-08-06", "gpt-4o", true},
		{"longest prefix wins", "gpt-5-mini-2025-08-07", "gpt-5-mini", true},
		{"id containing a slash", "groq/openai/gpt-oss-120b", "openai/gpt-oss-120b", true},
		{"unknown variant is not a sibling", "gpt-4o-mini", "gpt-4o-mini", false},
		{"unknown", "llama3.2:3b", "llama3.2:3b", false},
		{"empty", "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := catalog.Lookup(tt.model)
			if got.ID != tt.wantID || got.Known != tt.known {
				t.Fatalf("Lookup(%q) = %q known=%v, want %q known=%v", tt.model, got.ID, got.Known, tt.wantID, tt.known)
			}
			if !tt.known && !reflect.DeepEqual(got, catalog.Unknown(tt.model)) {
				t.Fatalf("unknown result must equal Unknown(): %+v", got)
			}
		})
	}
}

func TestCost(t *testing.T) {
	m := catalog.Lookup("claude-opus-5")
	if m.Pricing != (catalog.Pricing{Input: 5, Output: 25, CacheRead: 0.5, CacheWrite: 6.25}) {
		t.Fatalf("pricing = %+v", m.Pricing)
	}
	// 1M prompt: 600k uncached, 300k read, 100k written; 10k output.
	u := core.Usage{InputTokens: 1_000_000, CachedInputTokens: 300_000, CacheWriteTokens: 100_000, OutputTokens: 10_000}
	want := 0.6*5 + 0.3*0.5 + 0.1*6.25 + 0.01*25
	if got := m.Cost(u); math.Abs(got-want) > 1e-9 {
		t.Fatalf("Cost = %v, want %v", got, want)
	}
	// Inconsistent usage never goes negative on the uncached share.
	if got := m.Cost(core.Usage{InputTokens: 10, CachedInputTokens: 20}); math.Abs(got-20*0.5/1e6) > 1e-12 {
		t.Fatalf("Cost with oversized cache = %v", got)
	}
	if catalog.Unknown("x").Cost(u) != 0 {
		t.Fatal("unknown models must cost zero")
	}
}

func TestRegisterAndAll(t *testing.T) {
	custom := catalog.Model{
		ID: "vllm/my-finetune", Provider: "vllm", Family: "custom", DisplayName: "My finetune",
		ContextWindow: 32_000, MaxOutput: 4_000, Aliases: []string{"finetune"},
	}
	catalog.Register(custom)
	got := catalog.Lookup("FINETUNE")
	custom.Known = true
	if !reflect.DeepEqual(got, custom) {
		t.Fatalf("Lookup after Register = %+v", got)
	}
	// Re-registering replaces the row and drops its old aliases.
	custom.Aliases = []string{"ft"}
	catalog.Register(custom)
	if catalog.Lookup("finetune").Known || !catalog.Lookup("ft").Known {
		t.Fatal("stale alias survived re-registration")
	}

	all := catalog.All()
	if len(all) == 0 {
		t.Fatal("All() is empty")
	}
	if !sort.SliceIsSorted(all, func(i, j int) bool {
		if all[i].Provider != all[j].Provider {
			return all[i].Provider < all[j].Provider
		}
		return all[i].ID < all[j].ID
	}) {
		t.Fatal("All() must be sorted by provider then ID")
	}
	seen := map[string]bool{}
	for _, m := range all {
		if !m.Known || m.ID == "" || m.Provider == "" || m.DisplayName == "" {
			t.Fatalf("incomplete row: %+v", m)
		}
		if m.ID != "vllm/my-finetune" && (m.ContextWindow <= 0 || m.Pricing.Output <= 0) {
			t.Fatalf("seed row without limits or prices: %+v", m)
		}
		if seen[m.ID] {
			t.Fatalf("duplicate ID %s", m.ID)
		}
		seen[m.ID] = true
		for _, a := range m.Aliases {
			if catalog.Lookup(a).ID != m.ID {
				t.Fatalf("alias %s of %s resolves to %s", a, m.ID, catalog.Lookup(a).ID)
			}
		}
	}
	if catalog.DataAsOf == "" {
		t.Fatal("DataAsOf must be set")
	}
}
