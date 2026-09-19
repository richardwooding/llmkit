package catalog

// Seed data, checked on DataAsOf against the vendors' own pages. Rows whose
// limits or prices could not be read from an official page are left out
// rather than estimated; Lookup returns Unknown for them.
//
// Sources:
//   - Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
//     (input, 5m cache write, cache read, output), https://platform.claude.com/docs/en/models/overview
//     and the per-model pages under /docs/en/models/<model>/overview (IDs,
//     aliases, context window, max output). CacheWrite is the 5-minute rate;
//     the 1-hour rate is 2x input.
//   - OpenAI: https://developers.openai.com/api/docs/pricing and the per-model
//     pages under /api/docs/models/<model> (context window, max output,
//     capabilities). Cache writes carry no premium, so CacheWrite = Input.
//   - Gemini (llmkit provider "vertex"): https://ai.google.dev/gemini-api/docs/pricing
//     (paid tier, prompts <= 200k tokens where tiered) and the per-model pages
//     under /gemini-api/docs/models/<model>. Cache writes are billed at the
//     input rate plus hourly storage, which Cost does not model.
//   - DeepSeek: https://api-docs.deepseek.com/quick_start/pricing, peak-hour
//     rates (off-peak is half). Cache writes carry no premium.
//   - xAI: https://docs.x.ai/docs/models and the per-model pages, standard tier
//     for prompts under 200k tokens; max output is not published (0).
//   - Groq: https://console.groq.com/docs/models and the per-model pages.
//
// Not seeded: Cohere (current Command models are on custom pricing), Ollama
// (local; limits depend on the pulled tag), OpenRouter and Hugging Face
// (pass-through pricing per upstream model).
var seed = []Model{
	// Anthropic
	claude("claude-fable-5-1", "fable", "Claude Fable 5.1", 1_000_000, 128_000, Pricing{Input: 10, Output: 50, CacheRead: 0.25, CacheWrite: 12.5}),
	claude("claude-fable-5", "fable", "Claude Fable 5", 1_000_000, 128_000, Pricing{Input: 10, Output: 50, CacheRead: 1, CacheWrite: 12.5}),
	claude("claude-opus-5", "opus", "Claude Opus 5", 1_000_000, 128_000, Pricing{Input: 5, Output: 25, CacheRead: 0.5, CacheWrite: 6.25}),
	claude("claude-opus-4-8", "opus", "Claude Opus 4.8", 1_000_000, 128_000, Pricing{Input: 5, Output: 25, CacheRead: 0.5, CacheWrite: 6.25}),
	claude("claude-opus-4-7", "opus", "Claude Opus 4.7", 1_000_000, 128_000, Pricing{Input: 5, Output: 25, CacheRead: 0.5, CacheWrite: 6.25}),
	claude("claude-opus-4-6", "opus", "Claude Opus 4.6", 1_000_000, 128_000, Pricing{Input: 5, Output: 25, CacheRead: 0.5, CacheWrite: 6.25}),
	claude("claude-opus-4-5-20251101", "opus", "Claude Opus 4.5", 200_000, 64_000, Pricing{Input: 5, Output: 25, CacheRead: 0.5, CacheWrite: 6.25}, "claude-opus-4-5"),
	claude("claude-sonnet-5", "sonnet", "Claude Sonnet 5", 1_000_000, 128_000, Pricing{Input: 2, Output: 10, CacheRead: 0.2, CacheWrite: 2.5}),
	claude("claude-sonnet-4-6", "sonnet", "Claude Sonnet 4.6", 1_000_000, 128_000, Pricing{Input: 3, Output: 15, CacheRead: 0.3, CacheWrite: 3.75}),
	claude("claude-sonnet-4-5-20250929", "sonnet", "Claude Sonnet 4.5", 200_000, 64_000, Pricing{Input: 3, Output: 15, CacheRead: 0.3, CacheWrite: 3.75}, "claude-sonnet-4-5"),
	claude("claude-haiku-4-5-20251001", "haiku", "Claude Haiku 4.5", 200_000, 64_000, Pricing{Input: 1, Output: 5, CacheRead: 0.1, CacheWrite: 1.25}, "claude-haiku-4-5"),

	// OpenAI (Responses API)
	gpt("gpt-5.5", "gpt-5", "GPT-5.5", 1_050_000, 128_000, 5, 0.5, 30, true),
	gpt("gpt-5.4", "gpt-5", "GPT-5.4", 1_050_000, 128_000, 2.5, 0.25, 15, true),
	gpt("gpt-5", "gpt-5", "GPT-5", 400_000, 128_000, 1.25, 0.125, 10, true),
	gpt("gpt-5-mini", "gpt-5", "GPT-5 mini", 400_000, 128_000, 0.25, 0.025, 2, true),
	gpt("gpt-5-nano", "gpt-5", "GPT-5 nano", 400_000, 128_000, 0.05, 0.005, 0.4, true),
	gpt("gpt-4.1", "gpt-4.1", "GPT-4.1", 1_047_576, 32_768, 2, 0.5, 8, false),
	gpt("gpt-4o", "gpt-4o", "GPT-4o", 128_000, 16_384, 2.5, 1.25, 10, false),
	gpt("o3", "o", "o3", 200_000, 100_000, 2, 0.5, 8, true),
	gpt("o4-mini", "o", "o4-mini", 200_000, 100_000, 1.1, 0.275, 4.4, true),

	// Gemini via Vertex AI
	gemini("gemini-3.1-pro-preview", "gemini-3", "Gemini 3.1 Pro Preview", 2, 0.2, 12),
	gemini("gemini-3.5-flash", "gemini-3", "Gemini 3.5 Flash", 1.5, 0.15, 9),
	gemini("gemini-3.5-flash-lite", "gemini-3", "Gemini 3.5 Flash-Lite", 0.3, 0.03, 2.5),
	gemini("gemini-2.5-pro", "gemini-2.5", "Gemini 2.5 Pro", 1.25, 0.125, 10),
	gemini("gemini-2.5-flash", "gemini-2.5", "Gemini 2.5 Flash", 0.3, 0.03, 2.5),
	gemini("gemini-2.5-flash-lite", "gemini-2.5", "Gemini 2.5 Flash-Lite", 0.1, 0.01, 0.4),

	// DeepSeek
	{
		ID: "deepseek-flash", Provider: "deepseek", Family: "deepseek-v4", DisplayName: "DeepSeek V4.1 Flash",
		ContextWindow: 1_000_000, MaxOutput: 384_000,
		Pricing:      Pricing{Input: 0.3, Output: 1.2, CacheRead: 0.006, CacheWrite: 0.3},
		Capabilities: Capabilities{Tools: true, Vision: true, Reasoning: true, PromptCache: true},
		Aliases:      []string{"deepseek-v4-flash"},
	},
	{
		ID: "deepseek-v4-pro", Provider: "deepseek", Family: "deepseek-v4", DisplayName: "DeepSeek V4 Pro",
		ContextWindow: 1_000_000, MaxOutput: 384_000,
		Pricing:      Pricing{Input: 1.32, Output: 3.96, CacheRead: 0.044, CacheWrite: 1.32},
		Capabilities: Capabilities{Tools: true, Reasoning: true, PromptCache: true},
	},

	// xAI
	grok("grok-4.6", "Grok 4.6", 500_000, 2, 0.5, 6),
	grok("grok-4.5", "Grok 4.5", 500_000, 2, 0.3, 6, "grok-4.5-latest", "grok-build-latest"),
	grok("grok-4.3", "Grok 4.3", 1_000_000, 1.25, 0.2, 2.5, "grok-4.3-latest"),
	grok("grok-build-0.1", "Grok Build 0.1", 256_000, 1, 0.2, 2, "grok-code-fast-1", "grok-code-fast", "grok-code-fast-1-0825"),

	// Groq
	gptOSS("openai/gpt-oss-120b", "GPT-OSS 120B", 0.15, 0.075, 0.6),
	gptOSS("openai/gpt-oss-20b", "GPT-OSS 20B", 0.075, 0.037, 0.3),
}

func claude(id, family, name string, ctx, out int, p Pricing, aliases ...string) Model {
	return Model{
		ID: id, Provider: "anthropic", Family: family, DisplayName: name,
		ContextWindow: ctx, MaxOutput: out, Pricing: p, Aliases: aliases,
		Capabilities: Capabilities{Tools: true, Vision: true, Reasoning: true, JSONSchema: true, PromptCache: true},
	}
}

func gpt(id, family, name string, ctx, out int, in, cached, output float64, reasoning bool) Model {
	return Model{
		ID: id, Provider: "openai", Family: family, DisplayName: name,
		ContextWindow: ctx, MaxOutput: out,
		Pricing:      Pricing{Input: in, Output: output, CacheRead: cached, CacheWrite: in},
		Capabilities: Capabilities{Tools: true, Vision: true, Reasoning: reasoning, JSONSchema: true, PromptCache: true},
	}
}

// gemini rows share Gemini's 1,048,576-token input and 65,536-token output limits.
func gemini(id, family, name string, in, cached, output float64) Model {
	return Model{
		ID: id, Provider: "vertex", Family: family, DisplayName: name,
		ContextWindow: 1_048_576, MaxOutput: 65_536,
		Pricing:      Pricing{Input: in, Output: output, CacheRead: cached, CacheWrite: in},
		Capabilities: Capabilities{Tools: true, Vision: true, Reasoning: true, JSONSchema: true, PromptCache: true},
	}
}

func grok(id, name string, ctx int, in, cached, output float64, aliases ...string) Model {
	return Model{
		ID: id, Provider: "xai", Family: "grok-4", DisplayName: name,
		ContextWindow: ctx, Aliases: aliases,
		Pricing:      Pricing{Input: in, Output: output, CacheRead: cached, CacheWrite: in},
		Capabilities: Capabilities{Tools: true, Vision: true, Reasoning: true, JSONSchema: true, PromptCache: true},
	}
}

func gptOSS(id, name string, in, cached, output float64) Model {
	return Model{
		ID: id, Provider: "groq", Family: "gpt-oss", DisplayName: name,
		ContextWindow: 131_072, MaxOutput: 65_536,
		Pricing:      Pricing{Input: in, Output: output, CacheRead: cached, CacheWrite: in},
		Capabilities: Capabilities{Tools: true, Reasoning: true, JSONSchema: true, PromptCache: true},
	}
}
