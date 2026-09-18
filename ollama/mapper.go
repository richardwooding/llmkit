package ollama

import (
	"encoding/base64"
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
	msgs, err := messages(req.Messages)
	if err != nil {
		return nil, err
	}
	w := chatRequest{Model: c.model, Messages: msgs, Stream: stream, KeepAlive: c.keepAlive, Options: options(req)}
	for _, t := range req.Tools {
		var wt wireTool
		wt.Type = "function"
		wt.Function.Name = t.Name
		wt.Function.Description = t.Description
		wt.Function.Parameters = t.Parameters
		w.Tools = append(w.Tools, wt)
	}
	if req.Reasoning != nil {
		w.Think = new(true)
	}
	if w.Format, err = format(req.Format); err != nil {
		return nil, err
	}
	if req.ToolChoice.Mode == core.ToolChoiceNone {
		w.Tools = nil
	}
	return httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
}

func options(req *core.Request) map[string]any {
	o := map[string]any{}
	if req.MaxTokens > 0 {
		o["num_predict"] = req.MaxTokens
	}
	if req.Temperature != nil {
		o["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		o["top_p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		o["stop"] = req.Stop
	}
	if req.Seed != nil {
		o["seed"] = *req.Seed
	}
	if len(o) == 0 {
		return nil
	}
	return o
}

func format(f *core.ResponseFormat) (json.RawMessage, error) {
	if f == nil {
		return nil, nil
	}
	switch f.Type {
	case core.FormatJSON:
		return json.RawMessage(`"json"`), nil
	case core.FormatJSONSchema:
		if len(f.Schema) == 0 {
			return nil, fmt.Errorf("%s: json_schema format requires a schema", ID)
		}
		return f.Schema, nil
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
			out = append(out, wireMessage{Role: "system", Content: m.Text()})
		case core.RoleUser:
			w, err := userMessage(m)
			if err != nil {
				return nil, err
			}
			out = append(out, w)
		case core.RoleAssistant:
			out = append(out, assistantMessage(m))
		case core.RoleTool:
			for _, tr := range m.ToolResults() {
				for _, p := range tr.Content {
					if _, ok := p.(core.TextPart); !ok {
						return nil, core.Unsupported(ID, fmt.Sprintf("%T in tool result", p))
					}
				}
				out = append(out, wireMessage{Role: "tool", Content: tr.Text(), ToolName: tr.Name})
			}
		default:
			return nil, fmt.Errorf("%s: unknown role %q", ID, m.Role)
		}
	}
	return out, nil
}

func userMessage(m *core.Message) (wireMessage, error) {
	w := wireMessage{Role: "user"}
	var text strings.Builder
	for _, p := range m.Parts {
		switch v := p.(type) {
		case core.TextPart:
			text.WriteString(v.Text)
		case core.ImagePart:
			if v.URL != "" {
				return w, core.Unsupported(ID, "image by URL (supply bytes)")
			}
			w.Images = append(w.Images, base64.StdEncoding.EncodeToString(v.Data))
		default:
			return w, core.Unsupported(ID, fmt.Sprintf("%T input", p))
		}
	}
	w.Content = text.String()
	return w, nil
}

func assistantMessage(m *core.Message) wireMessage {
	w := wireMessage{Role: "assistant"}
	var text strings.Builder
	for _, p := range m.Parts {
		switch v := p.(type) {
		case core.TextPart:
			text.WriteString(v.Text)
		case core.ReasoningPart:
			w.Thinking = v.Text
		case core.ToolCall:
			var tc wireToolCall
			tc.Function.Name = v.Name
			tc.Function.Arguments = v.Arguments
			if len(tc.Function.Arguments) == 0 {
				tc.Function.Arguments = json.RawMessage("{}")
			}
			w.ToolCalls = append(w.ToolCalls, tc)
		}
	}
	w.Content = text.String()
	return w
}
