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

// ReasoningDelta is a fragment of a streamed reasoning block. Index is a
// reasoning-block ordinal, stable for one block within a response and
// independent of ToolCallDelta.Index. Text accumulates across fragments;
// Signature and Encrypted are opaque provider tokens that usually arrive on
// the fragment that closes the block and must be echoed back on later turns.
type ReasoningDelta struct {
	Index     int
	Text      string
	Signature string
	Encrypted string
}

// Chunk is one streamed event. Exactly one ChunkFinish ends every successful
// stream, carrying FinishReason and, when the provider reports it, Usage.
// For ChunkReasoning, Text mirrors Reasoning.Text so simple consumers can
// print it; Reasoning carries the block index and signatures Collect needs.
type Chunk struct {
	Kind         ChunkKind
	Text         string
	Reasoning    *ReasoningDelta
	ToolCall     *ToolCallDelta
	FinishReason FinishReason
	Usage        *Usage
	Raw          json.RawMessage
}
