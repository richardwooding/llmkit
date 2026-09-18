package core

import (
	"encoding/json"
	"fmt"
)

type wireMessage struct {
	Role  Role       `json:"role"`
	Name  string     `json:"name,omitempty"`
	Parts []wirePart `json:"parts"`
}

type wirePart struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Data      []byte          `json:"data,omitempty"`
	MIME      string          `json:"mime,omitempty"`
	URL       string          `json:"url,omitempty"`
	Detail    string          `json:"detail,omitempty"`
	Name      string          `json:"name,omitempty"`
	Signature string          `json:"signature,omitempty"`
	Encrypted string          `json:"encrypted,omitempty"`
	ID        string          `json:"id,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Content   []wirePart      `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

const (
	partText       = "text"
	partImage      = "image"
	partAudio      = "audio"
	partFile       = "file"
	partReasoning  = "reasoning"
	partToolCall   = "tool_call"
	partToolResult = "tool_result"
)

// MarshalJSON encodes the message with a "type" discriminator per part so it
// can be persisted and decoded back with UnmarshalJSON.
func (m Message) MarshalJSON() ([]byte, error) {
	parts, err := encodeParts(m.Parts)
	if err != nil {
		return nil, err
	}
	return json.Marshal(wireMessage{Role: m.Role, Name: m.Name, Parts: parts})
}

// UnmarshalJSON decodes the format produced by MarshalJSON.
func (m *Message) UnmarshalJSON(b []byte) error {
	var w wireMessage
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	parts, err := decodeParts(w.Parts)
	if err != nil {
		return err
	}
	*m = Message{Role: w.Role, Name: w.Name, Parts: parts}
	return nil
}

func encodeParts(parts []Part) ([]wirePart, error) {
	out := make([]wirePart, 0, len(parts))
	for _, p := range parts {
		w, err := encodePart(p)
		if err != nil {
			return nil, err
		}
		out = append(out, w)
	}
	return out, nil
}

func encodePart(p Part) (wirePart, error) {
	switch v := p.(type) {
	case TextPart:
		return wirePart{Type: partText, Text: v.Text}, nil
	case ImagePart:
		return wirePart{Type: partImage, Data: v.Data, MIME: v.MIME, URL: v.URL, Detail: v.Detail}, nil
	case AudioPart:
		return wirePart{Type: partAudio, Data: v.Data, MIME: v.MIME}, nil
	case FilePart:
		return wirePart{Type: partFile, Data: v.Data, MIME: v.MIME, URL: v.URL, Name: v.Name}, nil
	case ReasoningPart:
		return wirePart{Type: partReasoning, Text: v.Text, Signature: v.Signature, Encrypted: v.Encrypted}, nil
	case ToolCall:
		return wirePart{Type: partToolCall, ID: v.ID, Name: v.Name, Arguments: v.Arguments}, nil
	case ToolResult:
		content, err := encodeParts(v.Content)
		if err != nil {
			return wirePart{}, err
		}
		return wirePart{Type: partToolResult, CallID: v.CallID, Name: v.Name, Content: content, IsError: v.IsError}, nil
	default:
		return wirePart{}, fmt.Errorf("core: cannot encode part of type %T", p)
	}
}

func decodeParts(ws []wirePart) ([]Part, error) {
	out := make([]Part, 0, len(ws))
	for _, w := range ws {
		p, err := decodePart(w)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

func decodePart(w wirePart) (Part, error) {
	switch w.Type {
	case partText:
		return TextPart{Text: w.Text}, nil
	case partImage:
		return ImagePart{Data: w.Data, MIME: w.MIME, URL: w.URL, Detail: w.Detail}, nil
	case partAudio:
		return AudioPart{Data: w.Data, MIME: w.MIME}, nil
	case partFile:
		return FilePart{Data: w.Data, MIME: w.MIME, URL: w.URL, Name: w.Name}, nil
	case partReasoning:
		return ReasoningPart{Text: w.Text, Signature: w.Signature, Encrypted: w.Encrypted}, nil
	case partToolCall:
		return ToolCall{ID: w.ID, Name: w.Name, Arguments: w.Arguments}, nil
	case partToolResult:
		content, err := decodeParts(w.Content)
		if err != nil {
			return nil, err
		}
		return ToolResult{CallID: w.CallID, Name: w.Name, Content: content, IsError: w.IsError}, nil
	default:
		return nil, fmt.Errorf("core: unknown part type %q", w.Type)
	}
}
