package vertex

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

// Stream performs a streamGenerateContent call over SSE.
func (c *Client) Stream(ctx context.Context, req *core.Request) iter.Seq2[core.Chunk, error] {
	return func(yield func(core.Chunk, error) bool) {
		body, err := c.body(req)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		rc, err := c.http.PostStream(ctx, c.modelPath("streamGenerateContent")+"?alt=sse", body)
		if err != nil {
			yield(core.Chunk{}, googleError(err))
			return
		}
		defer func() { _ = rc.Close() }()
		st := &streamState{}
		for ev, err := range httpx.SSE(rc) {
			if err != nil {
				yield(core.Chunk{}, fmt.Errorf("%s: stream: %w", ID, err))
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
	finishReason string
	blocked      bool
	usage        *core.Usage
	calls        int
}

func (s *streamState) apply(data string) ([]core.Chunk, error) {
	var out wireResponse
	if err := json.Unmarshal([]byte(data), &out); err != nil {
		return nil, fmt.Errorf("%s: decode stream chunk: %w", ID, err)
	}
	if len(out.Error) > 0 {
		return nil, streamError(out.Error, data)
	}
	if out.UsageMetadata != nil {
		u := usage(out.UsageMetadata)
		s.usage = &u
	}
	s.blocked = s.blocked || out.blocked()
	if len(out.Candidates) == 0 {
		return nil, nil
	}
	cand := &out.Candidates[0]
	if cand.FinishReason != "" {
		s.finishReason = cand.FinishReason
	}
	raw := json.RawMessage(data)
	chunks := make([]core.Chunk, 0, len(cand.Content.Parts))
	for i := range cand.Content.Parts {
		if ch, ok := s.chunk(&cand.Content.Parts[i], raw); ok {
			chunks = append(chunks, ch)
		}
	}
	return chunks, nil
}

func (s *streamState) chunk(p *wirePart, raw json.RawMessage) (core.Chunk, bool) {
	switch {
	case p.FunctionCall != nil:
		idx := s.calls
		s.calls++
		return core.Chunk{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{
			Index: idx, ID: callID(s.calls), Name: p.FunctionCall.Name, Arguments: string(rawArgs(p.FunctionCall.Args)),
		}}, true
	case p.Text == "":
		return core.Chunk{}, false
	case p.Thought:
		return core.Chunk{Kind: core.ChunkReasoning, Text: p.Text, Raw: raw}, true
	default:
		return core.Chunk{Kind: core.ChunkText, Text: p.Text, Raw: raw}, true
	}
}

func streamError(errObj json.RawMessage, data string) error {
	var e struct {
		Code int `json:"code"`
	}
	_ = json.Unmarshal(errObj, &e)
	return googleError(httpx.ParseError(ID, e.Code, nil, []byte(data)))
}

func (s *streamState) finish() core.Chunk {
	fr := finishReason(s.finishReason, s.calls > 0)
	if s.finishReason == "" && s.blocked {
		fr = core.FinishContentFilter
	}
	return core.Chunk{Kind: core.ChunkFinish, FinishReason: fr, Usage: s.usage}
}
