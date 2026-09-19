package anthropic

import (
	"encoding/json"

	"github.com/richardwooding/llmkit/core"
)

const (
	roleUser      = "user"
	roleAssistant = "assistant"

	blockText             = "text"
	blockImage            = "image"
	blockDocument         = "document"
	blockThinking         = "thinking"
	blockRedactedThinking = "redacted_thinking"
	blockToolUse          = "tool_use"
	blockToolResult       = "tool_result"

	sourceBase64 = "base64"
	sourceURL    = "url"
	sourceText   = "text"

	mimePDF  = "application/pdf"
	mimeText = "text/plain"

	thinkingEnabled  = "enabled"
	thinkingAdaptive = "adaptive"
	formatJSONSchema = "json_schema"

	choiceAuto = "auto"
	choiceAny  = "any"
	choiceTool = "tool"
	choiceNone = "none"

	eventMessageStart      = "message_start"
	eventContentBlockStart = "content_block_start"
	eventContentBlockDelta = "content_block_delta"
	eventMessageDelta      = "message_delta"
	eventMessageStop       = "message_stop"
	eventError             = "error"

	deltaText      = "text_delta"
	deltaThinking  = "thinking_delta"
	deltaInputJSON = "input_json_delta"
	deltaSignature = "signature_delta"

	cacheEphemeral = "ephemeral"
	cacheTTL5m     = "5m"
	cacheTTL1h     = "1h"
	// maxCacheBreakpoints is the API's per-request cache_control limit.
	maxCacheBreakpoints = 4
)

// wireRequest.System is a string, or []wireBlock when cache_control must be
// attached to it.
type wireRequest struct {
	Model         string            `json:"model"`
	MaxTokens     int               `json:"max_tokens"`
	System        any               `json:"system,omitempty"`
	Messages      []wireMessage     `json:"messages"`
	Tools         []wireTool        `json:"tools,omitempty"`
	ToolChoice    *wireToolChoice   `json:"tool_choice,omitempty"`
	Temperature   *float64          `json:"temperature,omitempty"`
	TopP          *float64          `json:"top_p,omitempty"`
	StopSequences []string          `json:"stop_sequences,omitempty"`
	Stream        bool              `json:"stream,omitempty"`
	Thinking      *wireThinking     `json:"thinking,omitempty"`
	OutputConfig  *wireOutputConfig `json:"output_config,omitempty"`
}

type wireMessage struct {
	Role    string      `json:"role"`
	Content []wireBlock `json:"content"`
}

// wireBlock is the union of every content block shape sent or received.
type wireBlock struct {
	Type         string            `json:"type"`
	Text         string            `json:"text,omitempty"`
	Source       *wireSource       `json:"source,omitempty"`
	Title        string            `json:"title,omitempty"`
	Thinking     *string           `json:"thinking,omitempty"`
	Signature    string            `json:"signature,omitempty"`
	Data         string            `json:"data,omitempty"`
	ID           string            `json:"id,omitempty"`
	Name         string            `json:"name,omitempty"`
	Input        json.RawMessage   `json:"input,omitempty"`
	ToolUseID    string            `json:"tool_use_id,omitempty"`
	Content      any               `json:"content,omitempty"`
	IsError      bool              `json:"is_error,omitempty"`
	CacheControl *wireCacheControl `json:"cache_control,omitempty"`
}

type wireCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type wireSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type wireTool struct {
	Name         string            `json:"name"`
	Description  string            `json:"description,omitempty"`
	InputSchema  json.RawMessage   `json:"input_schema"`
	Strict       bool              `json:"strict,omitempty"`
	CacheControl *wireCacheControl `json:"cache_control,omitempty"`
}

type wireToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type wireThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type wireOutputConfig struct {
	Effort string      `json:"effort,omitempty"`
	Format *wireFormat `json:"format,omitempty"`
}

type wireFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema"`
}

type wireResponse struct {
	ID         string      `json:"id"`
	Model      string      `json:"model"`
	Content    []wireBlock `json:"content"`
	StopReason string      `json:"stop_reason"`
	Usage      wireUsage   `json:"usage"`
}

type wireUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// merge overlays the non-zero fields of o, as message_delta usage repeats only
// what changed.
func (u *wireUsage) merge(o *wireUsage) {
	for _, f := range []struct{ dst, src *int }{
		{&u.InputTokens, &o.InputTokens},
		{&u.OutputTokens, &o.OutputTokens},
		{&u.CacheReadInputTokens, &o.CacheReadInputTokens},
		{&u.CacheCreationInputTokens, &o.CacheCreationInputTokens},
	} {
		if *f.src > 0 {
			*f.dst = *f.src
		}
	}
}

type wireError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type streamEvent struct {
	Type         string        `json:"type"`
	Index        int           `json:"index"`
	Message      *wireResponse `json:"message"`
	ContentBlock *wireBlock    `json:"content_block"`
	Delta        *streamDelta  `json:"delta"`
	Usage        *wireUsage    `json:"usage"`
	Error        *wireError    `json:"error"`
}

type streamDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Thinking    string `json:"thinking"`
	PartialJSON string `json:"partial_json"`
	Signature   string `json:"signature"`
	StopReason  string `json:"stop_reason"`
}

func (r *wireResponse) toResponse() *core.Response {
	resp := &core.Response{ID: r.ID, Model: r.Model, Message: core.Message{Role: core.RoleAssistant}}
	for i := range r.Content {
		if p := r.Content[i].toPart(); p != nil {
			resp.Message.Parts = append(resp.Message.Parts, p)
		}
	}
	resp.FinishReason = finishReason(r.StopReason)
	resp.Usage = r.Usage.toUsage()
	return resp
}

func (b *wireBlock) toPart() core.Part {
	switch b.Type {
	case blockText:
		return core.Text(b.Text)
	case blockThinking:
		p := core.ReasoningPart{Signature: b.Signature}
		if b.Thinking != nil {
			p.Text = *b.Thinking
		}
		return p
	case blockRedactedThinking:
		return core.ReasoningPart{Encrypted: b.Data}
	case blockToolUse:
		args := b.Input
		if len(args) == 0 || string(args) == "null" {
			args = json.RawMessage("{}")
		}
		return core.ToolCall{ID: b.ID, Name: b.Name, Arguments: args}
	default:
		return nil
	}
}

// toUsage folds the cache counters into InputTokens: the API reports
// input_tokens as only the uncached remainder, whereas core.Usage promises the
// whole prompt on every provider.
func (u *wireUsage) toUsage() core.Usage {
	input := u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	return core.Usage{
		InputTokens:       input,
		OutputTokens:      u.OutputTokens,
		TotalTokens:       input + u.OutputTokens,
		CachedInputTokens: u.CacheReadInputTokens,
		CacheWriteTokens:  u.CacheCreationInputTokens,
	}
}

func finishReason(s string) core.FinishReason {
	switch s {
	case "end_turn", "stop_sequence":
		return core.FinishStop
	case "max_tokens":
		return core.FinishLength
	case "tool_use":
		return core.FinishToolCalls
	case "refusal":
		return core.FinishContentFilter
	default:
		return core.FinishOther
	}
}
