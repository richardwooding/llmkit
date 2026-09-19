package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Stream performs a streaming Responses API call over SSE.
func (c *Client) Stream(ctx context.Context, req *core.Request) iter.Seq2[core.Chunk, error] {
	return func(yield func(core.Chunk, error) bool) {
		body, err := c.body(req, true)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		rc, err := c.http.PostStream(ctx, pathResponses, body)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		defer func() { _ = rc.Close() }()
		var st streamState
		if st.run(rc, yield) {
			// The server closed the stream without a terminal event.
			yield(core.Chunk{Kind: core.ChunkFinish, FinishReason: core.FinishOther}, nil)
		}
	}
}

type streamState struct {
	hasTools bool
	done     bool
}

func (s *streamState) run(r io.Reader, yield func(core.Chunk, error) bool) bool {
	for ev, err := range httpx.SSE(r) {
		if err != nil {
			yield(core.Chunk{}, fmt.Errorf("%s: stream: %w", ID, err))
			return false
		}
		chunk, err := s.apply(ev.Data)
		if err != nil {
			yield(core.Chunk{}, err)
			return false
		}
		if chunk == nil {
			continue
		}
		if !yield(*chunk, nil) || s.done {
			return false
		}
	}
	return true
}

func (s *streamState) apply(data string) (*core.Chunk, error) {
	var ev wireEvent
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return nil, fmt.Errorf("%s: decode stream event: %w", ID, err)
	}
	raw := json.RawMessage(data)
	switch ev.Type {
	case eventOutputItemAdded:
		if ev.Item == nil || ev.Item.Type != typeFunctionCall {
			return nil, nil
		}
		s.hasTools = true
		return &core.Chunk{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{Index: ev.OutputIndex, ID: ev.Item.CallID, Name: ev.Item.Name}}, nil
	case eventFunctionArgsDelta:
		return &core.Chunk{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{Index: ev.OutputIndex, Arguments: ev.Delta}}, nil
	case eventOutputTextDelta, eventRefusalDelta:
		return &core.Chunk{Kind: core.ChunkText, Text: ev.Delta, Raw: raw}, nil
	case eventReasoningSummary, eventReasoningText:
		return &core.Chunk{Kind: core.ChunkReasoning, Text: ev.Delta, Raw: raw, Reasoning: &core.ReasoningDelta{
			Index: ev.OutputIndex, Text: ev.Delta,
		}}, nil
	case eventOutputItemDone:
		return reasoningDone(&ev, raw), nil
	case eventResponseCompleted, eventResponseIncomplete:
		s.done = true
		return s.finish(ev.Response, raw), nil
	case eventResponseFailed:
		return nil, failure(ev.Response, raw)
	case eventError:
		return nil, &core.APIError{Provider: ID, Code: ev.Code, Message: ev.Message, Body: raw}
	default:
		return nil, nil
	}
}

// reasoningDone surfaces the reasoning item's id and encrypted_content, which
// only exist on the completed item and must be echoed back on the next turn.
func reasoningDone(ev *wireEvent, raw json.RawMessage) *core.Chunk {
	if ev.Item == nil || ev.Item.Type != typeReasoning {
		return nil
	}
	return &core.Chunk{Kind: core.ChunkReasoning, Raw: raw, Reasoning: &core.ReasoningDelta{
		Index: ev.OutputIndex, Signature: ev.Item.ID, Encrypted: ev.Item.EncryptedContent,
	}}
}

func (s *streamState) finish(resp *wireResponse, raw json.RawMessage) *core.Chunk {
	if resp == nil {
		resp = &wireResponse{}
	}
	ch := &core.Chunk{Kind: core.ChunkFinish, Raw: raw, FinishReason: finishReason(resp, s.hasTools || hasFunctionCall(resp))}
	if resp.Usage != nil {
		u := usage(resp.Usage)
		ch.Usage = &u
	}
	return ch
}
