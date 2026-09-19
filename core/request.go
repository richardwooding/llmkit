package core

import (
	"encoding/json"
	"maps"
)

// Request describes a chat completion. Pointer fields distinguish "unset" from
// a meaningful zero.
type Request struct {
	Messages   []Message
	Tools      []Tool
	ToolChoice ToolChoice

	MaxTokens   int
	Temperature *float64
	TopP        *float64
	Stop        []string
	Seed        *int64
	Format      *ResponseFormat
	Reasoning   *ReasoningConfig
	Cache       *CacheConfig

	// Extra is merged into the top level of the wire body for any provider.
	Extra map[string]any
	// ProviderOptions is keyed by provider ID; only the matching provider
	// merges its entry, after Extra.
	ProviderOptions map[string]map[string]any
}

// Tool declares a function the model may call. Parameters is a JSON Schema
// object.
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
	Strict      bool
}

// ToolChoiceMode controls whether the model may, must, or must not call tools.
type ToolChoiceMode string

// Tool choice modes.
const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNamed    ToolChoiceMode = "named"
)

// ToolChoice selects a ToolChoiceMode; Name applies to ToolChoiceNamed.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

// ResponseFormat asks for JSON output, optionally constrained by a schema.
type ResponseFormat struct {
	Type   string
	Name   string
	Schema json.RawMessage
	Strict bool
}

// Response format types.
const (
	FormatJSON       = "json"
	FormatJSONSchema = "json_schema"
)

// ReasoningConfig enables extended thinking where a provider supports it.
// Effort is low, medium or high; BudgetTokens caps thinking tokens.
//
// Summary controls how much of the thinking comes back. Anthropic
// (thinking.display) accepts "summarized", "omitted" or "updates"; OpenAI
// (reasoning.summary) accepts "auto", "concise" or "detailed". Each provider
// maps the other's "show me something" value ("auto" <-> "summarized") so one
// setting works across both; anything else is passed through verbatim.
type ReasoningConfig struct {
	Effort       string
	BudgetTokens int
	Summary      string
}

// CacheConfig asks a provider with explicit prompt caching (Anthropic) to
// mark cache breakpoints: after the system prompt, after the tool definitions
// and after the last Turns user-role messages. Providers with automatic
// caching ignore it. TTL is "5m" (default) or "1h".
type CacheConfig struct {
	System bool
	Tools  bool
	Turns  int
	TTL    string
}

// ProviderExtra returns the merged provider-specific overrides for id.
func (r *Request) ProviderExtra(id string) map[string]any {
	if r == nil {
		return nil
	}
	return mergeExtra(r.Extra, r.ProviderOptions[id])
}

func mergeExtra(extra, provider map[string]any) map[string]any {
	if len(extra) == 0 && len(provider) == 0 {
		return nil
	}
	out := make(map[string]any, len(extra)+len(provider))
	maps.Copy(out, extra)
	maps.Copy(out, provider)
	return out
}
