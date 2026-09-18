package groq

import "github.com/richardwooding/llmkit/openaicompat"

var quirks = openaicompat.Quirks{
	Images:                true,
	ReasoningContentField: "reasoning",
	JSONSchema:            true,
	Seed:                  true,
}
