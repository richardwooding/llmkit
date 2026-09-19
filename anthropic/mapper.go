package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

func (c *Client) body(req *core.Request, stream bool) ([]byte, error) {
	w, err := c.wire(req, stream)
	if err != nil {
		return nil, err
	}
	return httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
}

// countBody keeps only the fields /messages/count_tokens accepts; generation
// parameters such as max_tokens are rejected there.
func (c *Client) countBody(req *core.Request) ([]byte, error) {
	w, err := c.wire(req, false)
	if err != nil {
		return nil, err
	}
	return json.Marshal(wireCountRequest{
		Model: w.Model, System: w.System, Messages: w.Messages,
		Tools: w.Tools, ToolChoice: w.ToolChoice, Thinking: w.Thinking,
	})
}

func (c *Client) wire(req *core.Request, stream bool) (*wireRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("%s: nil request", ID)
	}
	system, msgs, err := messages(req.Messages)
	if err != nil {
		return nil, err
	}
	w := wireRequest{
		Model:         c.model,
		MaxTokens:     req.MaxTokens,
		Messages:      msgs,
		Tools:         tools(req.Tools),
		ToolChoice:    toolChoice(req.ToolChoice),
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.Stop,
		Stream:        stream,
	}
	if system != "" {
		w.System = system
	}
	if w.MaxTokens <= 0 {
		w.MaxTokens = defaultMaxTokens
	}
	if req.Cache != nil {
		if err := applyCache(&w, system, req.Cache); err != nil {
			return nil, err
		}
	}
	w.Thinking, w.OutputConfig = reasoning(req.Reasoning)
	if req.Format != nil {
		f, err := format(req.Format)
		if err != nil {
			return nil, err
		}
		if w.OutputConfig == nil {
			w.OutputConfig = &wireOutputConfig{}
		}
		w.OutputConfig.Format = f
	}
	return &w, nil
}

// applyCache places cache_control breakpoints in prompt order (tools, system,
// then the last cfg.Turns user-role messages) and stops at the API's limit of
// four, so the stable prefix is preferred when the caller asks for too many.
func applyCache(w *wireRequest, system string, cfg *core.CacheConfig) error {
	cc, err := cacheControl(cfg.TTL)
	if err != nil {
		return err
	}
	slots := maxCacheBreakpoints
	if cfg.Tools && len(w.Tools) > 0 {
		w.Tools[len(w.Tools)-1].CacheControl = cc
		slots--
	}
	if system != "" {
		block := wireBlock{Type: blockText, Text: system}
		if cfg.System {
			block.CacheControl = cc
			slots--
		}
		w.System = []wireBlock{block}
	}
	turns := min(cfg.Turns, slots)
	for i := len(w.Messages) - 1; i >= 0 && turns > 0; i-- {
		m := &w.Messages[i]
		if m.Role != roleUser || len(m.Content) == 0 {
			continue
		}
		m.Content[len(m.Content)-1].CacheControl = cc
		turns--
	}
	return nil
}

func cacheControl(ttl string) (*wireCacheControl, error) {
	switch ttl {
	case "", cacheTTL5m:
		return &wireCacheControl{Type: cacheEphemeral}, nil
	case cacheTTL1h:
		return &wireCacheControl{Type: cacheEphemeral, TTL: cacheTTL1h}, nil
	default:
		return nil, fmt.Errorf("%s: unsupported cache TTL %q (want 5m or 1h)", ID, ttl)
	}
}

// reasoning prefers adaptive thinking plus output_config.effort; an explicit
// budget selects the legacy enabled mode that pre-4.6 models require.
func reasoning(cfg *core.ReasoningConfig) (*wireThinking, *wireOutputConfig) {
	if cfg == nil {
		return nil, nil
	}
	th := &wireThinking{Type: thinkingAdaptive, Display: display(cfg.Summary)}
	if cfg.BudgetTokens > 0 {
		th = &wireThinking{Type: thinkingEnabled, BudgetTokens: cfg.BudgetTokens, Display: th.Display}
	}
	if cfg.Effort == "" {
		return th, nil
	}
	return th, &wireOutputConfig{Effort: cfg.Effort}
}

func display(summary string) string {
	if summary == summaryAuto {
		return displaySummarized
	}
	return summary
}

func format(f *core.ResponseFormat) (*wireFormat, error) {
	switch f.Type {
	case core.FormatJSONSchema:
		if len(f.Schema) == 0 {
			return nil, fmt.Errorf("%s: json_schema format requires a schema", ID)
		}
		return &wireFormat{Type: formatJSONSchema, Schema: f.Schema}, nil
	case core.FormatJSON:
		return nil, core.Unsupported(ID, "json response format without a schema")
	default:
		return nil, fmt.Errorf("%s: unknown response format %q", ID, f.Type)
	}
}

func tools(ts []core.Tool) []wireTool {
	if len(ts) == 0 {
		return nil
	}
	out := make([]wireTool, 0, len(ts))
	for _, t := range ts {
		schema := t.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		out = append(out, wireTool{Name: t.Name, Description: t.Description, InputSchema: schema, Strict: t.Strict})
	}
	return out
}

func toolChoice(tc core.ToolChoice) *wireToolChoice {
	switch tc.Mode {
	case core.ToolChoiceAuto:
		return &wireToolChoice{Type: choiceAuto}
	case core.ToolChoiceRequired:
		return &wireToolChoice{Type: choiceAny}
	case core.ToolChoiceNamed:
		return &wireToolChoice{Type: choiceTool, Name: tc.Name}
	case core.ToolChoiceNone:
		return &wireToolChoice{Type: choiceNone}
	default:
		return nil
	}
}

// messages hoists system text and merges adjacent same-role turns, because
// the API rejects consecutive messages with the same role.
func messages(msgs []core.Message) (string, []wireMessage, error) {
	var system []string
	out := make([]wireMessage, 0, len(msgs))
	for i := range msgs {
		m := &msgs[i]
		if m.Role == core.RoleSystem {
			system = append(system, m.Text())
			continue
		}
		role, err := wireRole(m.Role)
		if err != nil {
			return "", nil, err
		}
		content, err := blocks(m.Parts)
		if err != nil {
			return "", nil, err
		}
		if len(content) == 0 {
			continue
		}
		if n := len(out); n > 0 && out[n-1].Role == role {
			out[n-1].Content = append(out[n-1].Content, content...)
			continue
		}
		out = append(out, wireMessage{Role: role, Content: content})
	}
	return strings.Join(system, "\n\n"), out, nil
}

func wireRole(r core.Role) (string, error) {
	switch r {
	case core.RoleUser, core.RoleTool:
		return roleUser, nil
	case core.RoleAssistant:
		return roleAssistant, nil
	default:
		return "", fmt.Errorf("%s: unknown role %q", ID, r)
	}
}

func blocks(parts []core.Part) ([]wireBlock, error) {
	out := make([]wireBlock, 0, len(parts))
	for _, p := range parts {
		b, err := block(p)
		if err != nil {
			return nil, err
		}
		if b != nil {
			out = append(out, *b)
		}
	}
	return out, nil
}

func block(p core.Part) (*wireBlock, error) {
	switch v := p.(type) {
	case core.TextPart:
		return &wireBlock{Type: blockText, Text: v.Text}, nil
	case core.ImagePart:
		return &wireBlock{Type: blockImage, Source: source(v.URL, v.MIME, v.Data)}, nil
	case core.FilePart:
		return documentBlock(v), nil
	case core.ReasoningPart:
		return thinkingBlock(v), nil
	case core.ToolCall:
		return toolUseBlock(v), nil
	case core.ToolResult:
		return toolResultBlock(v)
	default:
		return nil, core.Unsupported(ID, fmt.Sprintf("%T input", p))
	}
}

func source(url, mime string, data []byte) *wireSource {
	if url != "" {
		return &wireSource{Type: sourceURL, URL: url}
	}
	return &wireSource{Type: sourceBase64, MediaType: httpx.MIMEOr(mime, data), Data: base64.StdEncoding.EncodeToString(data)}
}

func documentBlock(f core.FilePart) *wireBlock {
	b := &wireBlock{Type: blockDocument, Title: f.Name}
	switch {
	case f.URL != "":
		b.Source = &wireSource{Type: sourceURL, URL: f.URL}
	case f.MIME == mimeText:
		b.Source = &wireSource{Type: sourceText, MediaType: mimeText, Data: string(f.Data)}
	default:
		mime := f.MIME
		if mime == "" {
			mime = mimePDF
		}
		b.Source = &wireSource{Type: sourceBase64, MediaType: mime, Data: base64.StdEncoding.EncodeToString(f.Data)}
	}
	return b
}

func thinkingBlock(r core.ReasoningPart) *wireBlock {
	if r.Encrypted != "" {
		return &wireBlock{Type: blockRedactedThinking, Data: r.Encrypted}
	}
	if r.Signature == "" && r.Text == "" {
		return nil
	}
	return &wireBlock{Type: blockThinking, Thinking: new(r.Text), Signature: r.Signature}
}

func toolUseBlock(tc core.ToolCall) *wireBlock {
	input := tc.Arguments
	if len(input) == 0 || !json.Valid(input) {
		input = json.RawMessage("{}")
	}
	return &wireBlock{Type: blockToolUse, ID: tc.ID, Name: tc.Name, Input: input}
}

func toolResultBlock(tr core.ToolResult) (*wireBlock, error) {
	b := &wireBlock{Type: blockToolResult, ToolUseID: tr.CallID, IsError: tr.IsError}
	textOnly := true
	for _, p := range tr.Content {
		switch p.(type) {
		case core.TextPart:
		case core.ImagePart:
			textOnly = false
		default:
			return nil, core.Unsupported(ID, fmt.Sprintf("%T in tool result", p))
		}
	}
	if textOnly {
		if text := tr.Text(); text != "" {
			b.Content = text
		}
		return b, nil
	}
	content, err := blocks(tr.Content)
	if err != nil {
		return nil, err
	}
	b.Content = content
	return b, nil
}
