package llmkit_test

import (
	"encoding/json"
	"errors"
	"iter"
	"reflect"
	"testing"

	"github.com/richardwooding/llmkit"
	"github.com/richardwooding/llmkit/core"
)

func chunks(cs ...core.Chunk) iter.Seq2[core.Chunk, error] {
	return func(yield func(core.Chunk, error) bool) {
		for _, c := range cs {
			if !yield(c, nil) {
				return
			}
		}
	}
}

func TestCollect(t *testing.T) {
	seq := chunks(
		core.Chunk{Kind: core.ChunkReasoning, Text: "th"},
		core.Chunk{Kind: core.ChunkReasoning, Text: "ink"},
		core.Chunk{Kind: core.ChunkText, Text: "Hel"},
		core.Chunk{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 1, ID: "c2", Name: "g"}},
		core.Chunk{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, ID: "c1", Name: "f", Arguments: `{"a":`}},
		core.Chunk{Kind: core.ChunkText, Text: "lo"},
		core.Chunk{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, Arguments: `1}`}},
		core.Chunk{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls, Usage: &core.Usage{TotalTokens: 9}},
	)
	got, err := llmkit.Collect(seq)
	if err != nil {
		t.Fatal(err)
	}
	want := &core.Response{
		Message: core.Assistant(core.ReasoningPart{Text: "think"}, core.Text("Hello"),
			core.ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)},
			core.ToolCall{ID: "c2", Name: "g", Arguments: json.RawMessage(`{}`)}),
		FinishReason: core.FinishToolCalls,
		Usage:        core.Usage{TotalTokens: 9},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v\nwant %+v", got, want)
	}
}

func TestCollectReasoningBlocks(t *testing.T) {
	seq := chunks(
		core.Chunk{Kind: core.ChunkReasoning, Text: "hm", Reasoning: &core.ReasoningDelta{Index: 0, Text: "hm"}},
		core.Chunk{Kind: core.ChunkReasoning, Text: "m", Reasoning: &core.ReasoningDelta{Index: 0, Text: "m"}},
		core.Chunk{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 0, Signature: "sig0"}},
		core.Chunk{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 1, Encrypted: "enc"}},
		core.Chunk{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, ID: "c1", Name: "f"}},
		core.Chunk{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 2}},
		core.Chunk{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 2, Signature: "sig2"}},
		core.Chunk{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 3}},
		core.Chunk{Kind: core.ChunkText, Text: "ok"},
		core.Chunk{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls},
	)
	got, err := llmkit.Collect(seq)
	if err != nil {
		t.Fatal(err)
	}
	// Block 3 never received text or a signature, so it is dropped rather
	// than sent back as an unsigned thinking block.
	want := &core.Response{
		Message: core.Assistant(
			core.ReasoningPart{Text: "hmm", Signature: "sig0"},
			core.ReasoningPart{Encrypted: "enc"},
			core.ReasoningPart{Signature: "sig2"},
			core.Text("ok"),
			core.ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{}`)},
		),
		FinishReason: core.FinishToolCalls,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v\nwant %+v", got, want)
	}
}

func TestCollectDefaultsAndErrors(t *testing.T) {
	got, err := llmkit.Collect(chunks(core.Chunk{Kind: core.ChunkText, Text: "x"}))
	if err != nil || got.FinishReason != core.FinishStop || got.Text() != "x" {
		t.Fatalf("got %+v %v", got, err)
	}
	got, _ = llmkit.Collect(chunks(core.Chunk{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Name: "f"}}))
	if got.FinishReason != core.FinishToolCalls {
		t.Fatalf("finish = %s", got.FinishReason)
	}
	boom := errors.New("boom")
	failing := func(yield func(core.Chunk, error) bool) {
		if yield(core.Chunk{Kind: core.ChunkText, Text: "a"}, nil) {
			yield(core.Chunk{}, boom)
		}
	}
	if _, err := llmkit.Collect(failing); !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
}
