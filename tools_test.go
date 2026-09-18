package llmkit_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/richardwooding/llmkit"
	"github.com/richardwooding/llmkit/core"
)

func TestRunTools(t *testing.T) {
	chatter := &scripted{responses: []*core.Response{
		{
			Message: core.Assistant(core.ToolCall{ID: "1", Name: "add", Arguments: json.RawMessage(`{"a":2,"b":3}`)},
				core.ToolCall{ID: "2", Name: "missing"}, core.ToolCall{ID: "3", Name: "fail"}),
			FinishReason: core.FinishToolCalls, Usage: core.Usage{InputTokens: 10},
		},
		{Message: core.Assistant(core.Text("5")), FinishReason: core.FinishStop, Usage: core.Usage{InputTokens: 20, OutputTokens: 1}},
	}}
	tools := map[string]llmkit.ToolFunc{
		"add": func(_ context.Context, args json.RawMessage) (string, error) {
			var in struct{ A, B int }
			_ = json.Unmarshal(args, &in)
			return json.Number(string(rune('0' + in.A + in.B))).String(), nil
		},
		"fail": func(context.Context, json.RawMessage) (string, error) { return "", errors.New("boom") },
	}
	req := &core.Request{Messages: []core.Message{core.UserText("2+3?")}}
	resp, err := llmkit.RunTools(context.Background(), chatter, req, tools, 5)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "5" || resp.Usage.InputTokens != 30 || resp.Usage.OutputTokens != 1 {
		t.Fatalf("resp = %+v", resp)
	}
	if len(req.Messages) != 3 || req.Messages[1].Role != core.RoleAssistant || req.Messages[2].Role != core.RoleTool {
		t.Fatalf("messages = %+v", req.Messages)
	}
	results := req.Messages[2].ToolResults()
	if len(results) != 3 || results[0].Text() != "5" || !results[1].IsError || !results[2].IsError || results[2].Text() != "boom" {
		t.Fatalf("results = %+v", results)
	}
	if len(chatter.seen[1]) != 3 {
		t.Fatal("second call must see the tool results")
	}
}

func TestRunToolsMaxIter(t *testing.T) {
	loop := &core.Response{Message: core.Assistant(core.ToolCall{ID: "1", Name: "f"})}
	chatter := &scripted{responses: []*core.Response{loop, loop, loop}}
	tools := map[string]llmkit.ToolFunc{"f": func(context.Context, json.RawMessage) (string, error) { return "", nil }}
	_, err := llmkit.RunTools(context.Background(), chatter, &core.Request{}, tools, 2)
	if !errors.Is(err, llmkit.ErrToolLoopExceeded) || chatter.calls != 2 {
		t.Fatalf("err = %v calls = %d", err, chatter.calls)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	chatter = &scripted{responses: []*core.Response{loop}}
	if _, err := llmkit.RunTools(ctx, chatter, &core.Request{}, tools, 2); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	chatter = &scripted{}
	if _, err := llmkit.RunTools(context.Background(), chatter, &core.Request{}, tools, 0); err == nil {
		t.Fatal("chat error must propagate")
	}
}
