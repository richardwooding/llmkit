package ollama

import (
	"encoding/json"
	"fmt"

	"github.com/richardwooding/llmkit/core"
)

type chatRequest struct {
	Model     string          `json:"model"`
	Messages  []wireMessage   `json:"messages"`
	Tools     []wireTool      `json:"tools,omitempty"`
	Stream    bool            `json:"stream"`
	Think     *bool           `json:"think,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	Options   map[string]any  `json:"options,omitempty"`
	KeepAlive string          `json:"keep_alive,omitempty"`
}

type wireMessage struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	Thinking  string         `json:"thinking,omitempty"`
	Images    []string       `json:"images,omitempty"`
	ToolCalls []wireToolCall `json:"tool_calls,omitempty"`
	ToolName  string         `json:"tool_name,omitempty"`
}

type wireToolCall struct {
	ID       string `json:"id,omitempty"`
	Function struct {
		Index     *int            `json:"index,omitempty"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type wireTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	} `json:"function"`
}

type chatResponse struct {
	Model           string      `json:"model"`
	Message         wireMessage `json:"message"`
	Done            bool        `json:"done"`
	DoneReason      string      `json:"done_reason"`
	PromptEvalCount int         `json:"prompt_eval_count"`
	EvalCount       int         `json:"eval_count"`
}

type embedRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
	KeepAlive  string   `json:"keep_alive,omitempty"`
}

type embedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float32 `json:"embeddings"`
	PromptEvalCount int         `json:"prompt_eval_count"`
}

func (r *chatResponse) toResponse(model string) *core.Response {
	resp := &core.Response{Model: r.Model, Message: core.Message{Role: core.RoleAssistant}}
	if resp.Model == "" {
		resp.Model = model
	}
	if r.Message.Thinking != "" {
		resp.Message.Parts = append(resp.Message.Parts, core.ReasoningPart{Text: r.Message.Thinking})
	}
	if r.Message.Content != "" {
		resp.Message.Parts = append(resp.Message.Parts, core.Text(r.Message.Content))
	}
	for i, tc := range r.Message.ToolCalls {
		resp.Message.Parts = append(resp.Message.Parts, toolCall(tc, i))
	}
	resp.FinishReason = finishReason(r.DoneReason, len(r.Message.ToolCalls) > 0)
	resp.Usage = r.usage()
	return resp
}

func (r *chatResponse) usage() core.Usage {
	return core.Usage{InputTokens: r.PromptEvalCount, OutputTokens: r.EvalCount, TotalTokens: r.PromptEvalCount + r.EvalCount}
}

func toolCall(tc wireToolCall, i int) core.ToolCall {
	id := tc.ID
	if id == "" {
		id = fmt.Sprintf("call_%d", i+1)
	}
	args := tc.Function.Arguments
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	return core.ToolCall{ID: id, Name: tc.Function.Name, Arguments: args}
}

func finishReason(s string, hasTools bool) core.FinishReason {
	switch {
	case hasTools:
		return core.FinishToolCalls
	case s == "stop" || s == "":
		return core.FinishStop
	case s == "length":
		return core.FinishLength
	default:
		return core.FinishOther
	}
}
