package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Stream performs a streaming completion.
func (c *Client) Stream(ctx context.Context, req *core.Request) iter.Seq2[core.Chunk, error] {
	return func(yield func(core.Chunk, error) bool) {
		body, err := c.body(req, true)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		rc, err := c.http.PostStream(ctx, "/chat/completions", body)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		defer func() { _ = rc.Close() }()
		st := &streamState{client: c}
		for ev, err := range httpx.SSE(rc) {
			if err != nil {
				yield(core.Chunk{}, fmt.Errorf("%s: stream: %w", c.cfg.ID, err))
				return
			}
			chunks, err := st.apply(ev.Data)
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
	client       *Client
	finishReason string
	usage        *core.Usage
	hasTools     bool
}

func (s *streamState) apply(data string) ([]core.Chunk, error) {
	var out chatResponse
	if err := json.Unmarshal([]byte(data), &out); err != nil {
		return nil, fmt.Errorf("%s: decode stream chunk: %w", s.client.cfg.ID, err)
	}
	if out.Usage != nil {
		u := usage(out.Usage)
		s.usage = &u
	} else if out.XGroq != nil && out.XGroq.Usage != nil {
		u := usage(out.XGroq.Usage)
		s.usage = &u
	}
	if len(out.Choices) == 0 {
		return nil, nil
	}
	ch := out.Choices[0]
	if ch.FinishReason != "" {
		s.finishReason = ch.FinishReason
	}
	if ch.Delta == nil {
		return nil, nil
	}
	var delta wireAssistant
	if err := json.Unmarshal(*ch.Delta, &delta); err != nil {
		return nil, fmt.Errorf("%s: decode delta: %w", s.client.cfg.ID, err)
	}
	raw := json.RawMessage(data)
	var chunks []core.Chunk
	if r := s.client.reasoningText(delta.Fields); r != "" {
		chunks = append(chunks, core.Chunk{Kind: core.ChunkReasoning, Text: r, Raw: raw})
	}
	var text string
	if len(delta.Content) > 0 && json.Unmarshal(delta.Content, &text) == nil && text != "" {
		chunks = append(chunks, core.Chunk{Kind: core.ChunkText, Text: text, Raw: raw})
	}
	for i, tc := range delta.ToolCalls {
		s.hasTools = true
		idx := i
		if tc.Index != nil {
			idx = *tc.Index
		}
		chunks = append(chunks, core.Chunk{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{
			Index: idx, ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments,
		}})
	}
	return chunks, nil
}

func (s *streamState) finish() core.Chunk {
	return core.Chunk{Kind: core.ChunkFinish, FinishReason: finishReason(s.finishReason, s.hasTools), Usage: s.usage}
}
