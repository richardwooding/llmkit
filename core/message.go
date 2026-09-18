package core

import (
	"encoding/json"
	"strings"
)

// Role identifies the author of a Message.
type Role string

// Message roles.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is one turn in a conversation.
type Message struct {
	Role  Role
	Parts []Part
	Name  string
}

// Part is one piece of message content. The concrete types are TextPart,
// ImagePart, AudioPart, FilePart, ReasoningPart, ToolCall and ToolResult.
type Part interface {
	part()
}

// TextPart is plain text.
type TextPart struct {
	Text string
}

// ImagePart is an image supplied inline (Data + MIME) or by URL.
type ImagePart struct {
	Data   []byte
	MIME   string
	URL    string
	Detail string
}

// AudioPart is inline audio.
type AudioPart struct {
	Data []byte
	MIME string
}

// FilePart is a document such as a PDF, inline or by URL. Providers that
// accept video route it through the same slot.
type FilePart struct {
	Data []byte
	MIME string
	URL  string
	Name string
}

// ReasoningPart carries a model's thinking. Signature and Encrypted are opaque
// provider tokens that must be echoed back on later turns where present.
type ReasoningPart struct {
	Text      string
	Signature string
	Encrypted string
}

// ToolCall is a request from the model to invoke a tool.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolResult is the outcome of a ToolCall, sent back in a RoleTool message.
// Name is required by providers that address tools by name (Gemini).
type ToolResult struct {
	CallID  string
	Name    string
	Content []Part
	IsError bool
}

func (TextPart) part()      {}
func (ImagePart) part()     {}
func (AudioPart) part()     {}
func (FilePart) part()      {}
func (ReasoningPart) part() {}
func (ToolCall) part()      {}
func (ToolResult) part()    {}

// UnmarshalArgs decodes the call's JSON arguments into v.
func (tc ToolCall) UnmarshalArgs(v any) error {
	if len(tc.Arguments) == 0 {
		return json.Unmarshal([]byte("{}"), v)
	}
	return json.Unmarshal(tc.Arguments, v)
}

// Text returns the concatenated text of the result's TextParts.
func (tr ToolResult) Text() string {
	return joinText(tr.Content)
}

// Text returns the concatenated text of the message's TextParts.
func (m Message) Text() string {
	return joinText(m.Parts)
}

// ToolCalls returns the message's ToolCall parts in order.
func (m Message) ToolCalls() []ToolCall {
	var calls []ToolCall
	for _, p := range m.Parts {
		if tc, ok := p.(ToolCall); ok {
			calls = append(calls, tc)
		}
	}
	return calls
}

// ToolResults returns the message's ToolResult parts in order.
func (m Message) ToolResults() []ToolResult {
	var results []ToolResult
	for _, p := range m.Parts {
		if tr, ok := p.(ToolResult); ok {
			results = append(results, tr)
		}
	}
	return results
}

func joinText(parts []Part) string {
	var b strings.Builder
	for _, p := range parts {
		if t, ok := p.(TextPart); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

// Text builds a TextPart.
func Text(s string) TextPart { return TextPart{Text: s} }

// Image builds an inline ImagePart.
func Image(data []byte, mime string) ImagePart { return ImagePart{Data: data, MIME: mime} }

// ImageURL builds an ImagePart that references a URL.
func ImageURL(u string) ImagePart { return ImagePart{URL: u} }

// Audio builds an inline AudioPart.
func Audio(data []byte, mime string) AudioPart { return AudioPart{Data: data, MIME: mime} }

// File builds an inline FilePart.
func File(data []byte, mime, name string) FilePart {
	return FilePart{Data: data, MIME: mime, Name: name}
}

// FileURL builds a FilePart that references a URL.
func FileURL(u, mime string) FilePart { return FilePart{URL: u, MIME: mime} }

// ToolResultText builds a ToolResult whose content is a single text part.
func ToolResultText(callID, name, text string) ToolResult {
	return ToolResult{CallID: callID, Name: name, Content: []Part{Text(text)}}
}

// System builds a system message.
func System(text string) Message {
	return Message{Role: RoleSystem, Parts: []Part{Text(text)}}
}

// User builds a user message.
func User(parts ...Part) Message { return Message{Role: RoleUser, Parts: parts} }

// UserText builds a user message with a single text part.
func UserText(text string) Message { return User(Text(text)) }

// Assistant builds an assistant message.
func Assistant(parts ...Part) Message { return Message{Role: RoleAssistant, Parts: parts} }

// ToolResults builds a tool message carrying one or more results.
func ToolResults(results ...ToolResult) Message {
	parts := make([]Part, len(results))
	for i, r := range results {
		parts[i] = r
	}
	return Message{Role: RoleTool, Parts: parts}
}
