package openrouter

import (
	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/openaicompat"
)

var quirks = openaicompat.Quirks{
	Images:                true,
	Audio:                 true,
	Files:                 true,
	ReasoningContentField: "reasoning",
	ReasoningRequest:      reasoning,
	StreamUsage:           true,
	JSONSchema:            true,
	Seed:                  true,
	EmbedPath:             "/embeddings",
}

func reasoning(cfg *core.ReasoningConfig) map[string]any {
	r := map[string]any{}
	if cfg.Effort != "" {
		r["effort"] = cfg.Effort
	}
	if cfg.BudgetTokens > 0 {
		r["max_tokens"] = cfg.BudgetTokens
	}
	if len(r) == 0 {
		return nil
	}
	return map[string]any{"reasoning": r}
}

// WithReferer sets the HTTP-Referer header OpenRouter uses for app attribution.
func WithReferer(url string) core.Option { return core.WithHeader("HTTP-Referer", url) }

// WithTitle sets the X-Title header OpenRouter shows in its dashboard.
func WithTitle(title string) core.Option { return core.WithHeader("X-Title", title) }
