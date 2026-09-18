package llmkit

import (
	"github.com/richardwooding/llmkit/anthropic"
	"github.com/richardwooding/llmkit/cohere"
	"github.com/richardwooding/llmkit/deepseek"
	"github.com/richardwooding/llmkit/groq"
	"github.com/richardwooding/llmkit/huggingface"
	"github.com/richardwooding/llmkit/ollama"
	"github.com/richardwooding/llmkit/openai"
	"github.com/richardwooding/llmkit/openrouter"
	"github.com/richardwooding/llmkit/vertex"
	"github.com/richardwooding/llmkit/voyage"
	"github.com/richardwooding/llmkit/xai"
)

// Default is the registry used by the package-level functions. Providers are
// listed in bare-name match priority: strict prefixes first, then Ollama
// (tagged names and the fallback), then Hugging Face ("org/model" names).
var Default = newDefault()

func newDefault() *Registry {
	r := NewRegistry(
		openai.Provider{},
		anthropic.Provider{},
		vertex.Provider{},
		deepseek.Provider{},
		xai.Provider{},
		cohere.Provider{},
		voyage.Provider{},
		groq.Provider{},
		openrouter.Provider{},
		ollama.Provider{},
	)
	r.Register(huggingface.Provider{}, "hf")
	return r
}
