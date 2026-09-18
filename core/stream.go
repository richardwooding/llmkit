package core

import "encoding/json"

// ChunkKind discriminates streaming chunks.
type ChunkKind uint8

// Chunk kinds.
const (
	ChunkText ChunkKind = iota + 1
	ChunkReasoning
	ChunkToolCall
	ChunkFinish
)

// String returns the kind's name.
func (k ChunkKind) String() string {
	switch k {
	case ChunkText:
		return "text"
	case ChunkReasoning:
		return "reasoning"
	case ChunkToolCall:
		return "tool_call"
	case ChunkFinish:
		return "finish"
	default:
		return "unknown"
	}
}

// ToolCallDelta is a fragment of a streamed tool call. Index is stable for one
// call within a response; ID and Name arrive on the first fragment.
type ToolCallDelta struct {
	Index     int
	ID        string
	Name      string
	Arguments string
}

// Chunk is one streamed event. Exactly one ChunkFinish ends every successful
// stream, carrying FinishReason and, when the provider reports it, Usage.
type Chunk struct {
	Kind         ChunkKind
	Text         string
	ToolCall     *ToolCallDelta
	FinishReason FinishReason
	Usage        *Usage
	Raw          json.RawMessage
}
