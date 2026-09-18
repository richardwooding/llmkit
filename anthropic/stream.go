package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Stream performs a streaming completion over SSE.
func (c *Client) Stream(ctx context.Context, req *core.Request) iter.Seq2[core.Chunk, error] {
	return func(yield func(core.Chunk, error) bool) {
		body, err := c.body(req, true)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		rc, err := c.http.PostStream(ctx, messagesPath, body)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		defer func() { _ = rc.Close() }()
		st := &streamState{tools: map[int]int{}}
		if st.run(rc, yield) && !st.done {
			yield(st.finish(), nil)
		}
	}
}

type streamState struct {
	// tools maps a content block index to the ordinal of its tool call.
	tools      map[int]int
	stopReason string
	usage      wireUsage
	done       bool
}

// run returns false once yield has declined or an error ended the sequence.
func (s *streamState) run(rc io.Reader, yield func(core.Chunk, error) bool) bool {
	for ev, err := range httpx.SSE(rc) {
		if err != nil {
			yield(core.Chunk{}, fmt.Errorf("%s: stream: %w", ID, err))
			return false
		}
		chunks, err := s.apply(ev.Data)
		if err != nil {
			yield(core.Chunk{}, err)
			return false
		}
		for _, ch := range chunks {
			if !yield(ch, nil) {
				return false
			}
		}
		if s.done {
			return true
		}
	}
	return true
}

func (s *streamState) apply(data string) ([]core.Chunk, error) {
	var ev streamEvent
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return nil, fmt.Errorf("%s: decode stream event: %w", ID, err)
	}
	raw := json.RawMessage(data)
	switch ev.Type {
	case eventMessageStart:
		if ev.Message != nil {
			s.usage = ev.Message.Usage
		}
	case eventContentBlockStart:
		return s.blockStart(&ev, raw), nil
	case eventContentBlockDelta:
		return s.blockDelta(&ev, raw), nil
	case eventMessageDelta:
		s.messageDelta(&ev)
	case eventMessageStop:
		s.done = true
		return []core.Chunk{s.finish()}, nil
	case eventError:
		return nil, streamError(ev.Error)
	}
	return nil, nil
}

func (s *streamState) blockStart(ev *streamEvent, raw json.RawMessage) []core.Chunk {
	if ev.ContentBlock == nil || ev.ContentBlock.Type != blockToolUse {
		return nil
	}
	ordinal := len(s.tools)
	s.tools[ev.Index] = ordinal
	return []core.Chunk{{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{
		Index: ordinal, ID: ev.ContentBlock.ID, Name: ev.ContentBlock.Name,
	}}}
}

func (s *streamState) blockDelta(ev *streamEvent, raw json.RawMessage) []core.Chunk {
	if ev.Delta == nil {
		return nil
	}
	switch ev.Delta.Type {
	case deltaText:
		return []core.Chunk{{Kind: core.ChunkText, Text: ev.Delta.Text, Raw: raw}}
	case deltaThinking:
		return []core.Chunk{{Kind: core.ChunkReasoning, Text: ev.Delta.Thinking, Raw: raw}}
	case deltaInputJSON:
		ordinal, ok := s.tools[ev.Index]
		if !ok {
			return nil
		}
		return []core.Chunk{{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{
			Index: ordinal, Arguments: ev.Delta.PartialJSON,
		}}}
	default:
		return nil
	}
}

func (s *streamState) messageDelta(ev *streamEvent) {
	if ev.Delta != nil && ev.Delta.StopReason != "" {
		s.stopReason = ev.Delta.StopReason
	}
	if ev.Usage == nil {
		return
	}
	if ev.Usage.OutputTokens > 0 {
		s.usage.OutputTokens = ev.Usage.OutputTokens
	}
	if ev.Usage.InputTokens > 0 {
		s.usage.InputTokens = ev.Usage.InputTokens
	}
	if ev.Usage.CacheReadInputTokens > 0 {
		s.usage.CacheReadInputTokens = ev.Usage.CacheReadInputTokens
	}
}

func (s *streamState) finish() core.Chunk {
	u := s.usage.toUsage()
	return core.Chunk{Kind: core.ChunkFinish, FinishReason: finishReason(s.stopReason), Usage: &u}
}

// streamError covers the error event the API can emit after a 200 has
// already been sent; there is no HTTP status to report.
func streamError(e *wireError) error {
	if e == nil {
		return &core.APIError{Provider: ID, Message: "stream error"}
	}
	return &core.APIError{Provider: ID, Type: e.Type, Code: e.Type, Message: e.Message}
}
