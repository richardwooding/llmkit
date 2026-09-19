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
	// reason is the current reasoning-block ordinal; a thoughtSignature closes
	// the block, so thought text streamed before it reassembles into one part.
	reason int
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
		chunks = append(chunks, s.chunks(&cand.Content.Parts[i], raw)...)
	}
	return chunks, nil
}

func (s *streamState) chunks(p *wirePart, raw json.RawMessage) []core.Chunk {
	switch {
	case p.Thought:
		d := &core.ReasoningDelta{Index: s.reason, Text: p.Text, Signature: p.ThoughtSignature}
		if p.ThoughtSignature != "" {
			s.reason++
		}
		return []core.Chunk{{Kind: core.ChunkReasoning, Text: p.Text, Raw: raw, Reasoning: d}}
	case p.FunctionCall != nil:
		idx := s.calls
		s.calls++
		return append(s.signature(p, raw), core.Chunk{Kind: core.ChunkToolCall, Raw: raw, ToolCall: &core.ToolCallDelta{
			Index: idx, ID: callID(s.calls), Name: p.FunctionCall.Name, Arguments: string(rawArgs(p.FunctionCall.Args)),
		}})
	case p.Text == "":
		return nil
	default:
		return append(s.signature(p, raw), core.Chunk{Kind: core.ChunkText, Text: p.Text, Raw: raw})
	}
}

// signature emits a bare reasoning chunk for a thoughtSignature riding on a
// text or function-call part, mirroring how the non-streaming mapper splits it.
func (s *streamState) signature(p *wirePart, raw json.RawMessage) []core.Chunk {
	if p.ThoughtSignature == "" {
		return nil
	}
	d := &core.ReasoningDelta{Index: s.reason, Signature: p.ThoughtSignature}
	s.reason++
	return []core.Chunk{{Kind: core.ChunkReasoning, Raw: raw, Reasoning: d}}
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
