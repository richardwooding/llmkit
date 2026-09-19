package core

import "encoding/json"

// FinishReason says why generation stopped.
type FinishReason string

// Finish reasons.
const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishContentFilter FinishReason = "content_filter"
	FinishOther         FinishReason = "other"
)

// Usage reports token consumption. Fields a provider does not report are zero.
//
// InputTokens is the whole prompt on every provider, including the tokens
// served from cache (CachedInputTokens) and written to it (CacheWriteTokens);
// both are subsets of InputTokens, so the uncached remainder is
// InputTokens - CachedInputTokens - CacheWriteTokens. TotalTokens is
// InputTokens + OutputTokens. ReasoningTokens is a subset of OutputTokens.
type Usage struct {
	InputTokens       int
	OutputTokens      int
	TotalTokens       int
	CachedInputTokens int
	CacheWriteTokens  int
	ReasoningTokens   int
}

// Add returns the element-wise sum of u and o.
func (u Usage) Add(o Usage) Usage {
	return Usage{
		InputTokens:       u.InputTokens + o.InputTokens,
		OutputTokens:      u.OutputTokens + o.OutputTokens,
		TotalTokens:       u.TotalTokens + o.TotalTokens,
		CachedInputTokens: u.CachedInputTokens + o.CachedInputTokens,
		CacheWriteTokens:  u.CacheWriteTokens + o.CacheWriteTokens,
		ReasoningTokens:   u.ReasoningTokens + o.ReasoningTokens,
	}
}

// Response is a completed assistant turn. Message.Parts are in provider order
// and may mix TextPart, ReasoningPart and ToolCall.
type Response struct {
	ID           string
	Model        string
	Message      Message
	FinishReason FinishReason
	Usage        Usage
	Raw          json.RawMessage
}

// Text returns the response's concatenated text.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	return r.Message.Text()
}

// ToolCalls returns the tool calls the model requested.
func (r *Response) ToolCalls() []ToolCall {
	if r == nil {
		return nil
	}
	return r.Message.ToolCalls()
}
