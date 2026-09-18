// Package openaicompat implements the OpenAI Chat Completions wire format that
// DeepSeek, Groq, x.ai, OpenRouter, Hugging Face and many self-hosted servers
// speak. Register your own endpoint with NewProvider.
package openaicompat

import (
	"strings"

	"github.com/richardwooding/llmkit/core"
)

// Quirks describe what a compatible endpoint accepts beyond plain text chat.
type Quirks struct {
	Images bool
	Audio  bool
	Files  bool

	// ReasoningContentField names the assistant-message field that carries
	// thinking text ("reasoning_content" for DeepSeek, "reasoning" for Groq
	// and OpenRouter). Empty means none.
	ReasoningContentField string
	// ReasoningRequest maps a ReasoningConfig onto request fields. Nil sends
	// "reasoning_effort" when Effort is set.
	ReasoningRequest func(cfg *core.ReasoningConfig) map[string]any

	StreamUsage         bool
	JSONSchema          bool
	Strict              bool
	Seed                bool
	MaxCompletionTokens bool
	DeveloperRole       bool

	// EmbedPath is the embeddings endpoint ("/embeddings"); empty disables Embed.
	EmbedPath string

	// Validate rejects requests the endpoint cannot serve for this model.
	Validate func(model string, req *core.Request) error
}

// Config describes one OpenAI-compatible endpoint.
type Config struct {
	ID          string
	BaseURL     string
	APIKeyEnv   string
	KeyOptional bool
	Headers     map[string]string
	Match       func(model string) bool
	Quirks      Quirks
}

// PrefixMatcher returns a Match func that accepts models starting with any of
// the given prefixes.
func PrefixMatcher(prefixes ...string) func(string) bool {
	return func(model string) bool {
		m := strings.ToLower(model)
		for _, p := range prefixes {
			if strings.HasPrefix(m, p) {
				return true
			}
		}
		return false
	}
}
