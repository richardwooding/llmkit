package deepseek

import (
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/openaicompat"
)

var quirks = openaicompat.Quirks{
	ReasoningContentField: "reasoning_content",
	ReasoningRequest:      func(*core.ReasoningConfig) map[string]any { return nil },
	StreamUsage:           true,
	Validate:              validate,
}

// deepseek-reasoner rejects tool definitions; fail before the request is sent.
func validate(model string, req *core.Request) error {
	if strings.Contains(model, "reasoner") && len(req.Tools) > 0 {
		return core.Unsupported(ID, "tools with "+model)
	}
	return nil
}
