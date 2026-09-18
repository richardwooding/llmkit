package openaicompat

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

const (
	roleTool      = "tool"
	roleDeveloper = "developer"
	typeFunction  = "function"
	typeText      = "text"
)

func (c *Client) body(req *core.Request, stream bool) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("%s: nil request", c.cfg.ID)
	}
	if c.cfg.Quirks.Validate != nil {
		if err := c.cfg.Quirks.Validate(c.model, req); err != nil {
			return nil, err
		}
	}
	msgs, err := c.messages(req.Messages)
	if err != nil {
		return nil, err
	}
	w := wireRequest{
		Model:       c.model,
		Messages:    msgs,
		Tools:       c.tools(req.Tools),
		ToolChoice:  toolChoice(req.ToolChoice),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.Stop,
		Stream:      stream,
	}
	if c.cfg.Quirks.MaxCompletionTokens {
		w.MaxCompletionTokens = req.MaxTokens
	} else {
		w.MaxTokens = req.MaxTokens
	}
	if c.cfg.Quirks.Seed {
		w.Seed = req.Seed
	}
	if stream && c.cfg.Quirks.StreamUsage {
		w.StreamOptions = &wireStreamOpts{IncludeUsage: true}
	}
	if w.ResponseFormat, err = c.responseFormat(req.Format); err != nil {
		return nil, err
	}
	extra := c.reasoning(req.Reasoning)
	for k, v := range req.ProviderExtra(c.cfg.ID) {
		if extra == nil {
			extra = map[string]any{}
		}
		extra[k] = v
	}
	return httpx.MarshalWithExtra(w, extra)
}

func (c *Client) reasoning(cfg *core.ReasoningConfig) map[string]any {
	if cfg == nil {
		return nil
	}
	if c.cfg.Quirks.ReasoningRequest != nil {
		return c.cfg.Quirks.ReasoningRequest(cfg)
	}
	if cfg.Effort == "" {
		return nil
	}
	return map[string]any{"reasoning_effort": cfg.Effort}
}

func (c *Client) responseFormat(f *core.ResponseFormat) (*wireRespFormat, error) {
	if f == nil {
		return nil, nil
	}
	switch f.Type {
	case core.FormatJSON:
		return &wireRespFormat{Type: "json_object"}, nil
	case core.FormatJSONSchema:
		if !c.cfg.Quirks.JSONSchema {
			return nil, core.Unsupported(c.cfg.ID, "json_schema response format")
		}
		return &wireRespFormat{Type: "json_schema", JSONSchema: &wireJSONSchema{Name: f.Name, Schema: f.Schema, Strict: f.Strict}}, nil
	default:
		return nil, fmt.Errorf("%s: unknown response format %q", c.cfg.ID, f.Type)
	}
}

func (c *Client) tools(tools []core.Tool) []wireTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]wireTool, 0, len(tools))
	for _, t := range tools {
		fn := wireFunction{Name: t.Name, Description: t.Description, Parameters: t.Parameters}
		if t.Strict && c.cfg.Quirks.Strict {
			fn.Strict = new(true)
		}
		out = append(out, wireTool{Type: typeFunction, Function: fn})
	}
	return out
}

func toolChoice(tc core.ToolChoice) any {
	switch tc.Mode {
	case "", core.ToolChoiceAuto:
		return nil
	case core.ToolChoiceNamed:
		return map[string]any{"type": typeFunction, typeFunction: map[string]string{"name": tc.Name}}
	default:
		return string(tc.Mode)
	}
}

func (c *Client) messages(msgs []core.Message) ([]wireMessage, error) {
	out := make([]wireMessage, 0, len(msgs))
	for i := range msgs {
		m := &msgs[i]
		switch m.Role {
		case core.RoleSystem:
			role := string(core.RoleSystem)
			if c.cfg.Quirks.DeveloperRole {
				role = roleDeveloper
			}
			out = append(out, wireMessage{Role: role, Content: m.Text(), Name: m.Name})
		case core.RoleUser:
			content, err := c.userContent(m.Parts)
			if err != nil {
				return nil, err
			}
			out = append(out, wireMessage{Role: string(core.RoleUser), Content: content, Name: m.Name})
		case core.RoleAssistant:
			out = append(out, c.assistantMessage(m))
		case core.RoleTool:
			results, err := c.toolMessages(m)
			if err != nil {
				return nil, err
			}
			out = append(out, results...)
		default:
			return nil, fmt.Errorf("%s: unknown role %q", c.cfg.ID, m.Role)
		}
	}
	return out, nil
}

func (c *Client) userContent(parts []core.Part) (any, error) {
	if len(parts) == 1 {
		if t, ok := parts[0].(core.TextPart); ok {
			return t.Text, nil
		}
	}
	out := make([]wireContentPart, 0, len(parts))
	for _, p := range parts {
		w, err := c.contentPart(p)
		if err != nil {
			return nil, err
		}
		out = append(out, w)
	}
	return out, nil
}

func (c *Client) contentPart(p core.Part) (wireContentPart, error) {
	q := c.cfg.Quirks
	switch v := p.(type) {
	case core.TextPart:
		return wireContentPart{Type: typeText, Text: v.Text}, nil
	case core.ImagePart:
		if !q.Images {
			return wireContentPart{}, core.Unsupported(c.cfg.ID, "image input")
		}
		u := v.URL
		if u == "" {
			u = httpx.DataURI(v.MIME, v.Data)
		}
		return wireContentPart{Type: "image_url", ImageURL: &wireImageURL{URL: u, Detail: v.Detail}}, nil
	case core.AudioPart:
		if !q.Audio {
			return wireContentPart{}, core.Unsupported(c.cfg.ID, "audio input")
		}
		return wireContentPart{Type: "input_audio", InputAudio: &wireAudio{
			Data: base64.StdEncoding.EncodeToString(v.Data), Format: audioFormat(v.MIME),
		}}, nil
	case core.FilePart:
		if !q.Files {
			return wireContentPart{}, core.Unsupported(c.cfg.ID, "file input")
		}
		f := &wireFile{Filename: v.Name}
		if v.URL != "" {
			f.FileData = v.URL
		} else {
			f.FileData = httpx.DataURI(v.MIME, v.Data)
		}
		return wireContentPart{Type: "file", File: f}, nil
	default:
		return wireContentPart{}, core.Unsupported(c.cfg.ID, fmt.Sprintf("%T in user message", p))
	}
}

func audioFormat(mime string) string {
	_, sub, _ := strings.Cut(mime, "/")
	switch sub {
	case "wav", "x-wav", "wave":
		return "wav"
	case "mpeg", "mp3":
		return "mp3"
	default:
		return sub
	}
}

func (c *Client) assistantMessage(m *core.Message) wireMessage {
	w := wireMessage{Role: string(core.RoleAssistant), Name: m.Name}
	var text strings.Builder
	for _, p := range m.Parts {
		switch v := p.(type) {
		case core.TextPart:
			text.WriteString(v.Text)
		case core.ToolCall:
			args := string(v.Arguments)
			if args == "" {
				args = "{}"
			}
			w.ToolCalls = append(w.ToolCalls, wireToolCall{ID: v.ID, Type: typeFunction, Function: wireToolFunction{Name: v.Name, Arguments: args}})
		}
	}
	if text.Len() > 0 || len(w.ToolCalls) == 0 {
		w.Content = text.String()
	}
	return w
}

func (c *Client) toolMessages(m *core.Message) ([]wireMessage, error) {
	var out []wireMessage
	for _, p := range m.Parts {
		tr, ok := p.(core.ToolResult)
		if !ok {
			return nil, fmt.Errorf("%s: tool message may only contain ToolResult parts, got %T", c.cfg.ID, p)
		}
		for _, cp := range tr.Content {
			if _, isText := cp.(core.TextPart); !isText {
				return nil, core.Unsupported(c.cfg.ID, fmt.Sprintf("%T in tool result", cp))
			}
		}
		out = append(out, wireMessage{Role: roleTool, ToolCallID: tr.CallID, Content: tr.Text()})
	}
	return out, nil
}

func (c *Client) toResponse(out *chatResponse) (*core.Response, error) {
	resp := &core.Response{ID: out.ID, Model: out.Model, Message: core.Message{Role: core.RoleAssistant}}
	if out.Usage != nil {
		resp.Usage = usage(out.Usage)
	} else if out.XGroq != nil && out.XGroq.Usage != nil {
		resp.Usage = usage(out.XGroq.Usage)
	}
	if len(out.Choices) == 0 {
		resp.FinishReason = core.FinishOther
		return resp, nil
	}
	ch := out.Choices[0]
	var msg wireAssistant
	if ch.Message != nil {
		if err := json.Unmarshal(*ch.Message, &msg); err != nil {
			return nil, fmt.Errorf("%s: decode message: %w", c.cfg.ID, err)
		}
	}
	if r := c.reasoningText(msg.Fields); r != "" {
		resp.Message.Parts = append(resp.Message.Parts, core.ReasoningPart{Text: r})
	}
	var text string
	if len(msg.Content) > 0 && string(msg.Content) != "null" {
		if err := json.Unmarshal(msg.Content, &text); err != nil {
			text = string(msg.Content)
		}
	}
	if text != "" {
		resp.Message.Parts = append(resp.Message.Parts, core.Text(text))
	}
	if msg.Refusal != "" {
		resp.Message.Parts = append(resp.Message.Parts, core.Text(msg.Refusal))
	}
	for _, tc := range msg.ToolCalls {
		resp.Message.Parts = append(resp.Message.Parts, core.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: rawArgs(tc.Function.Arguments)})
	}
	resp.FinishReason = finishReason(ch.FinishReason, len(msg.ToolCalls) > 0)
	return resp, nil
}

func (c *Client) reasoningText(fields map[string]json.RawMessage) string {
	f := c.cfg.Quirks.ReasoningContentField
	if f == "" {
		return ""
	}
	var s string
	if raw, ok := fields[f]; ok {
		_ = json.Unmarshal(raw, &s)
	}
	return s
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

func finishReason(s string, hasTools bool) core.FinishReason {
	switch s {
	case "stop":
		if hasTools {
			return core.FinishToolCalls
		}
		return core.FinishStop
	case "length":
		return core.FinishLength
	case "tool_calls", "function_call":
		return core.FinishToolCalls
	case "content_filter":
		return core.FinishContentFilter
	case "":
		if hasTools {
			return core.FinishToolCalls
		}
		return core.FinishStop
	default:
		return core.FinishOther
	}
}

func usage(u *wireUsage) core.Usage {
	out := core.Usage{InputTokens: u.PromptTokens, OutputTokens: u.CompletionTokens, TotalTokens: u.TotalTokens}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}
	if u.PromptTokensDetails != nil {
		out.CachedInputTokens = u.PromptTokensDetails.CachedTokens
	}
	if u.CompletionTokensDetails != nil {
		out.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
	}
	return out
}
