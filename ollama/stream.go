package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Stream performs a streaming completion over NDJSON.
func (c *Client) Stream(ctx context.Context, req *core.Request) iter.Seq2[core.Chunk, error] {
	return func(yield func(core.Chunk, error) bool) {
		body, err := c.body(req, true)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		rc, err := c.http.PostStream(ctx, "/api/chat", body)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		defer func() { _ = rc.Close() }()
		st := &streamState{}
		for line, err := range httpx.NDJSON(rc) {
			if err != nil {
				yield(core.Chunk{}, fmt.Errorf("%s: stream: %w", ID, err))
				return
			}
			chunks, err := st.apply(line)
			if err != nil {
				yield(core.Chunk{}, err)
				return
			}
			for _, ch := range chunks {
				if !yield(ch, nil) {
					return
				}
			}
		}
		yield(st.finish(), nil)
	}
}

type streamState struct {
	final chatResponse
	calls int
}

func (s *streamState) apply(line json.RawMessage) ([]core.Chunk, error) {
	var r chatResponse
	if err := json.Unmarshal(line, &r); err != nil {
		return nil, fmt.Errorf("%s: decode stream chunk: %w", ID, err)
	}
	if r.Done {
		s.final = r
		s.final.Message.ToolCalls = nil
	}
	return chunks(&r, &s.calls, line), nil
}

func (s *streamState) finish() core.Chunk {
	u := s.final.usage()
	return core.Chunk{Kind: core.ChunkFinish, FinishReason: finishReason(s.final.DoneReason, s.calls > 0), Usage: &u}
}

func chunks(r *chatResponse, calls *int, raw json.RawMessage) []core.Chunk {
	var out []core.Chunk
	if r.Message.Thinking != "" {
		out = append(out, core.Chunk{Kind: core.ChunkReasoning, Text: r.Message.Thinking, Raw: raw})
	}
	if r.Message.Content != "" {
		out = append(out, core.Chunk{Kind: core.ChunkText, Text: r.Message.Content, Raw: raw})
	}
	for _, tc := range r.Message.ToolCalls {
		call := toolCall(tc, *calls)
		out = append(out, core.Chunk{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{
			Index: *calls, ID: call.ID, Name: call.Name, Arguments: string(call.Arguments),
		}})
		*calls++
	}
	return out
}
