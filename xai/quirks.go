package xai

import "github.com/richardwooding/llmkit/openaicompat"

var quirks = openaicompat.Quirks{
	Images:                true,
	ReasoningContentField: "reasoning_content",
	StreamUsage:           true,
	JSONSchema:            true,
	Seed:                  true,
}
