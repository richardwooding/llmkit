package vertexgrpc

import (
	"context"
	"errors"
	"io"
	"iter"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"

	"github.com/richardwooding/llmkit/core"
)

// Stream performs a StreamGenerateContent call; the RPC starts when iteration begins.
func (c *Client) Stream(ctx context.Context, req *core.Request) iter.Seq2[core.Chunk, error] {
	return func(yield func(core.Chunk, error) bool) {
		greq, err := c.request(req)
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		pc, err := c.client()
		if err != nil {
			yield(core.Chunk{}, err)
			return
		}
		ctx, cancel := c.context(ctx)
		defer cancel()
		sc, err := pc.StreamGenerateContent(ctx, greq)
		if err != nil {
			yield(core.Chunk{}, apiError(err))
			return
		}
		st := &streamState{}
		if !st.pump(sc, yield) {
			return
		}
		yield(st.finish(), nil)
	}
}

type streamState struct {
	finishReason aiplatformpb.Candidate_FinishReason
	usage        *core.Usage
	calls        int
}

// pump forwards chunks until the server closes the stream; false means the
// consumer stopped or an error was yielded.
func (s *streamState) pump(sc aiplatformpb.PredictionService_StreamGenerateContentClient, yield func(core.Chunk, error) bool) bool {
	for {
		resp, err := sc.Recv()
		if errors.Is(err, io.EOF) {
			return true
		}
		if err != nil {
			yield(core.Chunk{}, apiError(err))
			return false
		}
		chunks, err := s.apply(resp)
		if err != nil {
			yield(core.Chunk{}, err)
			return false
		}
		for _, ch := range chunks {
			if !yield(ch, nil) {
				return false
			}
		}
	}
}

func (s *streamState) apply(resp *aiplatformpb.GenerateContentResponse) ([]core.Chunk, error) {
	if u := resp.GetUsageMetadata(); u != nil {
		s.usage = new(usage(u))
	}
	if len(resp.GetCandidates()) == 0 {
		return nil, nil
	}
	cand := resp.GetCandidates()[0]
	if cand.GetFinishReason() != aiplatformpb.Candidate_FINISH_REASON_UNSPECIFIED {
		s.finishReason = cand.GetFinishReason()
	}
	parts, _, err := messageParts(cand.GetContent().GetParts(), s.calls)
	if err != nil {
		return nil, err
	}
	chunks := make([]core.Chunk, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case core.TextPart:
			chunks = append(chunks, core.Chunk{Kind: core.ChunkText, Text: v.Text})
		case core.ReasoningPart:
			if v.Text != "" {
				chunks = append(chunks, core.Chunk{Kind: core.ChunkReasoning, Text: v.Text})
			}
		case core.ToolCall:
			chunks = append(chunks, core.Chunk{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{
				Index: s.calls, ID: v.ID, Name: v.Name, Arguments: string(v.Arguments),
			}})
			s.calls++
		}
	}
	return chunks, nil
}

func (s *streamState) finish() core.Chunk {
	return core.Chunk{Kind: core.ChunkFinish, FinishReason: finishReason(s.finishReason, s.calls > 0), Usage: s.usage}
}
