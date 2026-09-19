package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

func (c *Client) body(req *core.Request, stream bool) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("%s: nil request", ID)
	}
	instructions, input, err := items(req.Messages)
	if err != nil {
		return nil, err
	}
	w := wireRequest{
		Model:           c.model,
		Instructions:    instructions,
		Input:           input,
		Tools:           tools(req.Tools),
		ToolChoice:      toolChoice(req.ToolChoice),
		MaxOutputTokens: req.MaxTokens,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		Stream:          stream,
	}
	if w.Text, err = textFormat(req.Format); err != nil {
		return nil, err
	}
	if req.Reasoning != nil {
		w.Include = []string{includeEncryptedReasoning}
		w.Reasoning = reasoning(req.Reasoning)
	}
	return httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
}

func reasoning(cfg *core.ReasoningConfig) *wireReasoning {
	summary := cfg.Summary
	if summary == displaySummarized {
		summary = summaryAuto
	}
	if cfg.Effort == "" && summary == "" {
		return nil
	}
	return &wireReasoning{Effort: cfg.Effort, Summary: summary}
}

func tools(ts []core.Tool) []wireTool {
	if len(ts) == 0 {
		return nil
	}
	out := make([]wireTool, 0, len(ts))
	for _, t := range ts {
		w := wireTool{Type: typeFunction, Name: t.Name, Description: t.Description, Parameters: t.Parameters}
		if t.Strict {
			w.Strict = new(true)
		}
		out = append(out, w)
	}
	return out
}

func toolChoice(tc core.ToolChoice) any {
	switch tc.Mode {
	case "", core.ToolChoiceAuto:
		return nil
	case core.ToolChoiceNamed:
		return map[string]string{"type": typeFunction, "name": tc.Name}
	default:
		return string(tc.Mode)
	}
}

func textFormat(f *core.ResponseFormat) (*wireText, error) {
	if f == nil {
		return nil, nil
	}
	switch f.Type {
	case core.FormatJSON:
		return &wireText{Format: wireFormat{Type: "json_object"}}, nil
	case core.FormatJSONSchema:
		return &wireText{Format: wireFormat{Type: "json_schema", Name: f.Name, Schema: f.Schema, Strict: f.Strict}}, nil
	default:
		return nil, fmt.Errorf("%s: unknown response format %q", ID, f.Type)
	}
}

func items(msgs []core.Message) (string, []wireItem, error) {
	var system []string
	out := make([]wireItem, 0, len(msgs))
	for i := range msgs {
		m := &msgs[i]
		switch m.Role {
		case core.RoleSystem:
			system = append(system, m.Text())
		case core.RoleUser:
			it, err := userItem(m)
			if err != nil {
				return "", nil, err
			}
			out = append(out, it)
		case core.RoleAssistant:
			out = append(out, assistantItems(m)...)
		case core.RoleTool:
			its, err := toolItems(m)
			if err != nil {
				return "", nil, err
			}
			out = append(out, its...)
		default:
			return "", nil, fmt.Errorf("%s: unknown role %q", ID, m.Role)
		}
	}
	return strings.Join(system, instructionsSeparator), out, nil
}

func userItem(m *core.Message) (wireItem, error) {
	content := make([]wireContent, 0, len(m.Parts))
	for _, p := range m.Parts {
		w, err := contentPart(p)
		if err != nil {
			return wireItem{}, err
		}
		content = append(content, w)
	}
	return wireItem{Type: typeMessage, Role: string(core.RoleUser), Content: content}, nil
}

func contentPart(p core.Part) (wireContent, error) {
	switch v := p.(type) {
	case core.TextPart:
		return wireContent{Type: typeInputText, Text: v.Text}, nil
	case core.ImagePart:
		u := v.URL
		if u == "" {
			u = httpx.DataURI(v.MIME, v.Data)
		}
		return wireContent{Type: typeInputImage, ImageURL: u, Detail: v.Detail}, nil
	case core.FilePart:
		if v.URL != "" {
			return wireContent{Type: typeInputFile, FileURL: v.URL}, nil
		}
		return wireContent{Type: typeInputFile, Filename: v.Name, FileData: httpx.DataURI(v.MIME, v.Data)}, nil
	case core.AudioPart:
		return wireContent{}, core.Unsupported(ID, "audio input")
	default:
		return wireContent{}, core.Unsupported(ID, fmt.Sprintf("%T in user message", p))
	}
}

func assistantItems(m *core.Message) []wireItem {
	var out []wireItem
	var text strings.Builder
	flush := func() {
		if text.Len() == 0 {
			return
		}
		out = append(out, wireItem{Type: typeMessage, Role: string(core.RoleAssistant), Content: []wireContent{{Type: typeOutputText, Text: text.String()}}})
		text.Reset()
	}
	for _, p := range m.Parts {
		switch v := p.(type) {
		case core.TextPart:
			text.WriteString(v.Text)
		case core.ReasoningPart:
			if v.Encrypted == "" {
				continue
			}
			flush()
			out = append(out, wireItem{Type: typeReasoning, ID: v.Signature, EncryptedContent: v.Encrypted, Summary: json.RawMessage(emptyJSONArray)})
		case core.ToolCall:
			flush()
			args := string(v.Arguments)
			if args == "" {
				args = emptyJSONObject
			}
			out = append(out, wireItem{Type: typeFunctionCall, CallID: v.ID, Name: v.Name, Arguments: args})
		}
	}
	flush()
	return out
}

func toolItems(m *core.Message) ([]wireItem, error) {
	out := make([]wireItem, 0, len(m.Parts))
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
		out = append(out, wireItem{Type: typeFunctionCallOutput, CallID: tr.CallID, Output: new(tr.Text())})
	}
	return out, nil
}

func toResponse(out *wireResponse, raw []byte) (*core.Response, error) {
	if out.Status == statusFailed {
		return nil, failure(out, raw)
	}
	resp := &core.Response{ID: out.ID, Model: out.Model, Message: core.Message{Role: core.RoleAssistant}}
	for i := range out.Output {
		resp.Message.Parts = append(resp.Message.Parts, outputParts(&out.Output[i])...)
	}
	resp.FinishReason = finishReason(out, hasFunctionCall(out))
	if out.Usage != nil {
		resp.Usage = usage(out.Usage)
	}
	return resp, nil
}

func outputParts(it *wireOutputItem) []core.Part {
	switch it.Type {
	case typeMessage:
		return messageParts(it.Content)
	case typeFunctionCall:
		return []core.Part{core.ToolCall{ID: it.CallID, Name: it.Name, Arguments: rawArgs(it.Arguments)}}
	case typeReasoning:
		if p, ok := reasoningPart(it); ok {
			return []core.Part{p}
		}
	}
	return nil
}

func messageParts(content []wireOutputContent) []core.Part {
	var parts []core.Part
	for _, c := range content {
		switch {
		case c.Type == typeOutputText && c.Text != "":
			parts = append(parts, core.Text(c.Text))
		case c.Type == typeRefusal && c.Refusal != "":
			parts = append(parts, core.Text(c.Refusal))
		}
	}
	return parts
}

func reasoningPart(it *wireOutputItem) (core.ReasoningPart, bool) {
	texts := make([]string, 0, len(it.Summary))
	for _, s := range it.Summary {
		if s.Text != "" {
			texts = append(texts, s.Text)
		}
	}
	p := core.ReasoningPart{Text: strings.Join(texts, reasoningSummarySeparator), Signature: it.ID, Encrypted: it.EncryptedContent}
	return p, p.Text != "" || p.Encrypted != ""
}

func hasFunctionCall(out *wireResponse) bool {
	for i := range out.Output {
		if out.Output[i].Type == typeFunctionCall {
			return true
		}
	}
	return false
}

func rawArgs(s string) json.RawMessage {
	if s == "" {
		return json.RawMessage(emptyJSONObject)
	}
	if json.Valid([]byte(s)) {
		return json.RawMessage(s)
	}
	quoted, _ := json.Marshal(s)
	return quoted
}

func finishReason(out *wireResponse, hasTools bool) core.FinishReason {
	if hasTools {
		return core.FinishToolCalls
	}
	if out.Status != statusIncomplete {
		return core.FinishStop
	}
	reason := ""
	if out.IncompleteDetails != nil {
		reason = out.IncompleteDetails.Reason
	}
	switch reason {
	case reasonMaxOutputTokens, reasonMaxTokens:
		return core.FinishLength
	case reasonContentFilter:
		return core.FinishContentFilter
	default:
		return core.FinishOther
	}
}

func failure(out *wireResponse, raw []byte) *core.APIError {
	e := &core.APIError{Provider: ID, Body: raw, Message: "response failed"}
	if out != nil && out.Error != nil {
		e.Code = out.Error.Code
		if out.Error.Message != "" {
			e.Message = out.Error.Message
		}
	}
	return e
}

func usage(u *wireUsage) core.Usage {
	out := core.Usage{InputTokens: u.InputTokens, OutputTokens: u.OutputTokens, TotalTokens: u.TotalTokens}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}
	if u.InputTokensDetails != nil {
		out.CachedInputTokens = u.InputTokensDetails.CachedTokens
	}
	if u.OutputTokensDetails != nil {
		out.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
	}
	return out
}
