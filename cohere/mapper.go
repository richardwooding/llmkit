package cohere

import (
	"encoding/json"
	"fmt"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

func (c *Client) body(req *core.Request, stream bool) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("%s: nil request", ID)
	}
	msgs, err := messages(req.Messages)
	if err != nil {
		return nil, err
	}
	w := chatRequest{
		Model:         c.model,
		Messages:      msgs,
		Tools:         tools(req.Tools),
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		P:             req.TopP,
		StopSequences: req.Stop,
		Seed:          req.Seed,
		Stream:        stream,
	}
	if w.ToolChoice, err = toolChoice(req.ToolChoice); err != nil {
		return nil, err
	}
	if w.ResponseFormat, err = responseFormat(req.Format); err != nil {
		return nil, err
	}
	if req.Reasoning != nil {
		w.Thinking = &wireThinking{Type: "enabled", TokenBudget: req.Reasoning.BudgetTokens}
	}
	return httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
}

func tools(ts []core.Tool) []wireTool {
	if len(ts) == 0 {
		return nil
	}
	out := make([]wireTool, 0, len(ts))
	for _, t := range ts {
		out = append(out, wireTool{Type: typeFunction, Function: wireFunction{
			Name: t.Name, Description: t.Description, Parameters: t.Parameters,
		}})
	}
	return out
}

func toolChoice(tc core.ToolChoice) (string, error) {
	switch tc.Mode {
	case "", core.ToolChoiceAuto:
		return "", nil
	case core.ToolChoiceRequired:
		return "REQUIRED", nil
	case core.ToolChoiceNone:
		return "NONE", nil
	case core.ToolChoiceNamed:
		return "", core.Unsupported(ID, "named tool choice")
	default:
		return "", fmt.Errorf("%s: unknown tool choice %q", ID, tc.Mode)
	}
}

func responseFormat(f *core.ResponseFormat) (*wireRespFormat, error) {
	if f == nil {
		return nil, nil
	}
	switch f.Type {
	case core.FormatJSON:
		return &wireRespFormat{Type: "json_object"}, nil
	case core.FormatJSONSchema:
		if len(f.Schema) == 0 {
			return nil, fmt.Errorf("%s: json_schema format requires a schema", ID)
		}
		return &wireRespFormat{Type: "json_object", JSONSchema: f.Schema}, nil
	default:
		return nil, fmt.Errorf("%s: unknown response format %q", ID, f.Type)
	}
}

func messages(msgs []core.Message) ([]wireMessage, error) {
	out := make([]wireMessage, 0, len(msgs))
	for i := range msgs {
		m := &msgs[i]
		switch m.Role {
		case core.RoleSystem:
			out = append(out, wireMessage{Role: roleSystem, Content: m.Text()})
		case core.RoleUser:
			content, err := userContent(m.Parts)
			if err != nil {
				return nil, err
			}
			out = append(out, wireMessage{Role: roleUser, Content: content})
		case core.RoleAssistant:
			out = append(out, assistantMessage(m))
		case core.RoleTool:
			results, err := toolMessages(m)
			if err != nil {
				return nil, err
			}
			out = append(out, results...)
		default:
			return nil, fmt.Errorf("%s: unknown role %q", ID, m.Role)
		}
	}
	return out, nil
}

func userContent(parts []core.Part) (any, error) {
	if len(parts) == 1 {
		if t, ok := parts[0].(core.TextPart); ok {
			return t.Text, nil
		}
	}
	out := make([]wireContent, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case core.TextPart:
			out = append(out, wireContent{Type: typeText, Text: v.Text})
		case core.ImagePart:
			u := v.URL
			if u == "" {
				u = httpx.DataURI(v.MIME, v.Data)
			}
			out = append(out, wireContent{Type: typeImageURL, ImageURL: &wireImageURL{URL: u}})
		default:
			return nil, core.Unsupported(ID, fmt.Sprintf("%T in user message", p))
		}
	}
	return out, nil
}

func assistantMessage(m *core.Message) wireMessage {
	w := wireMessage{Role: roleAssistant}
	var content []wireContent
	for _, p := range m.Parts {
		switch v := p.(type) {
		case core.TextPart:
			content = append(content, wireContent{Type: typeText, Text: v.Text})
		case core.ReasoningPart:
			content = append(content, wireContent{Type: typeThinking, Thinking: v.Text})
		case core.ToolCall:
			args := string(v.Arguments)
			if args == "" {
				args = "{}"
			}
			w.ToolCalls = append(w.ToolCalls, wireToolCall{
				ID: v.ID, Type: typeFunction, Function: wireToolFunction{Name: v.Name, Arguments: args},
			})
		}
	}
	if len(content) > 0 {
		w.Content = content
	}
	return w
}

func toolMessages(m *core.Message) ([]wireMessage, error) {
	var out []wireMessage
	for _, p := range m.Parts {
		tr, ok := p.(core.ToolResult)
		if !ok {
			return nil, fmt.Errorf("%s: tool message may only contain ToolResult parts, got %T", ID, p)
		}
		for _, cp := range tr.Content {
			if _, isText := cp.(core.TextPart); !isText {
				return nil, core.Unsupported(ID, fmt.Sprintf("%T in tool result", cp))
			}
		}
		out = append(out, wireMessage{Role: roleTool, ToolCallID: tr.CallID, Content: []wireContent{{
			Type: typeDocument, Document: &wireDocument{Data: map[string]string{typeText: tr.Text()}},
		}}})
	}
	return out, nil
}

func (r *chatResponse) toResponse(model string) *core.Response {
	resp := &core.Response{ID: r.ID, Model: model, Message: core.Message{Role: core.RoleAssistant}}
	if r.Message.ToolPlan != "" {
		resp.Message.Parts = append(resp.Message.Parts, core.ReasoningPart{Text: r.Message.ToolPlan})
	}
	for _, block := range r.Message.Content {
		switch block.Type {
		case typeText:
			resp.Message.Parts = append(resp.Message.Parts, core.Text(block.Text))
		case typeThinking:
			resp.Message.Parts = append(resp.Message.Parts, core.ReasoningPart{Text: block.Thinking})
		}
	}
	for _, tc := range r.Message.ToolCalls {
		resp.Message.Parts = append(resp.Message.Parts, core.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: rawArgs(tc.Function.Arguments)})
	}
	resp.FinishReason = finishReason(r.FinishReason)
	if r.Usage != nil {
		resp.Usage = r.Usage.toUsage()
	}
	return resp
}

func rawArgs(s string) json.RawMessage {
	if s == "" {
		return json.RawMessage("{}")
	}
	if json.Valid([]byte(s)) {
		return json.RawMessage(s)
	}
	quoted, _ := json.Marshal(s)
	return quoted
}

func finishReason(s string) core.FinishReason {
	switch s {
	case "COMPLETE", "STOP_SEQUENCE":
		return core.FinishStop
	case "MAX_TOKENS":
		return core.FinishLength
	case "TOOL_CALL":
		return core.FinishToolCalls
	default:
		return core.FinishOther
	}
}

func (u *wireUsage) toUsage() core.Usage {
	tokens := u.Tokens
	if tokens == nil {
		tokens = u.BilledUnits
	}
	out := core.Usage{CachedInputTokens: int(u.CachedTokens)}
	if tokens != nil {
		out.InputTokens = int(tokens.InputTokens)
		out.OutputTokens = int(tokens.OutputTokens)
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}
	return out
}
