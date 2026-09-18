package llmkit

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/richardwooding/llmkit/core"
)

// ToolFunc executes one tool call and returns its textual result.
type ToolFunc func(ctx context.Context, args json.RawMessage) (string, error)

// RunTools chats until the model stops requesting tools or maxIter calls have
// been made, appending assistant and tool messages to req.Messages as it goes.
// Tool errors and unknown tool names are fed back to the model as error
// results; context errors abort. Usage across iterations is summed.
func RunTools(ctx context.Context, c core.Chatter, req *core.Request, tools map[string]ToolFunc, maxIter int) (*core.Response, error) {
	if maxIter <= 0 {
		maxIter = 10
	}
	var total core.Usage
	for i := 0; i < maxIter; i++ {
		resp, err := c.Chat(ctx, req)
		if err != nil {
			return nil, err
		}
		total = total.Add(resp.Usage)
		calls := resp.ToolCalls()
		if len(calls) == 0 {
			resp.Usage = total
			return resp, nil
		}
		req.Messages = append(req.Messages, resp.Message)
		results := make([]core.ToolResult, 0, len(calls))
		for _, call := range calls {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			results = append(results, runOne(ctx, tools, call))
		}
		req.Messages = append(req.Messages, core.ToolResults(results...))
	}
	return nil, fmt.Errorf("after %d iterations: %w", maxIter, core.ErrToolLoopExceeded)
}

func runOne(ctx context.Context, tools map[string]ToolFunc, call core.ToolCall) core.ToolResult {
	fn, ok := tools[call.Name]
	if !ok {
		return core.ToolResult{
			CallID: call.ID, Name: call.Name, IsError: true,
			Content: []core.Part{core.Text(fmt.Sprintf("unknown tool %q", call.Name))},
		}
	}
	out, err := fn(ctx, call.Arguments)
	if err != nil {
		return core.ToolResult{CallID: call.ID, Name: call.Name, IsError: true, Content: []core.Part{core.Text(err.Error())}}
	}
	return core.ToolResultText(call.ID, call.Name, out)
}
