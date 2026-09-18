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
