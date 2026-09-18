package cohere

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Stream performs a streaming completion over Cohere's SSE event stream.
func (c *Client) Stream(ctx context.Context, req *core.Request) iter.Seq2[core.Chunk, error] {
	return func(yield func(core.Chunk, error) bool) {
		body, err := c.body(req, true)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		rc, err := c.http.PostStream(ctx, "/chat", body)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		defer func() { _ = rc.Close() }()
		finish := core.Chunk{Kind: core.ChunkFinish, FinishReason: core.FinishOther}
		for ev, err := range httpx.SSE(rc) {
			if err != nil {
				yield(core.Chunk{}, fmt.Errorf("%s: stream: %w", ID, err))
				return
			}
			var e streamEvent
			if err := json.Unmarshal([]byte(ev.Data), &e); err != nil {
				yield(core.Chunk{}, fmt.Errorf("%s: decode stream event: %w", ID, err))
				return
			}
			raw := json.RawMessage(ev.Data)
			if e.Type == "message-end" {
				finish = finishChunk(&e, raw)
				continue
			}
			if ch, ok := chunk(&e, raw); ok && !yield(ch, nil) {
				return
			}
		}
		yield(finish, nil)
	}
}

func chunk(e *streamEvent, raw json.RawMessage) (core.Chunk, bool) {
	msg := &e.Delta.Message
	switch e.Type {
	case "content-delta":
		if msg.Content.Thinking != "" {
			return core.Chunk{Kind: core.ChunkReasoning, Text: msg.Content.Thinking, Raw: raw}, true
		}
		if msg.Content.Text != "" {
			return core.Chunk{Kind: core.ChunkText, Text: msg.Content.Text, Raw: raw}, true
		}
	case "tool-plan-delta":
		if msg.ToolPlan != "" {
			return core.Chunk{Kind: core.ChunkReasoning, Text: msg.ToolPlan, Raw: raw}, true
		}
	case "tool-call-start", "tool-call-delta":
		return core.Chunk{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{
			Index:     e.Index,
			ID:        msg.ToolCalls.ID,
			Name:      msg.ToolCalls.Function.Name,
			Arguments: msg.ToolCalls.Function.Arguments,
		}}, true
	}
	return core.Chunk{}, false
}

func finishChunk(e *streamEvent, raw json.RawMessage) core.Chunk {
	ch := core.Chunk{Kind: core.ChunkFinish, FinishReason: finishReason(e.Delta.FinishReason), Raw: raw}
	if e.Delta.Usage != nil {
		u := e.Delta.Usage.toUsage()
		ch.Usage = &u
	}
	return ch
}
