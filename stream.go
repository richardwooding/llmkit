package llmkit

import (
	"encoding/json"
	"iter"
	"sort"
	"strings"

	"github.com/richardwooding/llmkit/core"
)

// Collect drains a stream into a Response, concatenating text and reasoning
// deltas and reassembling tool-call arguments by index.
func Collect(seq iter.Seq2[core.Chunk, error]) (*core.Response, error) {
	var c collector
	for ch, err := range seq {
		if err != nil {
			return nil, err
		}
		c.apply(&ch)
	}
	return c.response(), nil
}

type collector struct {
	text, reasoning strings.Builder
	calls           map[int]*core.ToolCall
	order           []int
	finish          core.FinishReason
	usage           core.Usage
}

func (c *collector) apply(ch *core.Chunk) {
	switch ch.Kind {
	case core.ChunkText:
		c.text.WriteString(ch.Text)
	case core.ChunkReasoning:
		c.reasoning.WriteString(ch.Text)
	case core.ChunkToolCall:
		if ch.ToolCall != nil {
			c.toolCall(ch.ToolCall)
		}
	case core.ChunkFinish:
		c.finish = ch.FinishReason
		if ch.Usage != nil {
			c.usage = *ch.Usage
		}
	}
}

func (c *collector) toolCall(d *core.ToolCallDelta) {
	if c.calls == nil {
		c.calls = map[int]*core.ToolCall{}
	}
	tc, ok := c.calls[d.Index]
	if !ok {
		tc = &core.ToolCall{}
		c.calls[d.Index] = tc
		c.order = append(c.order, d.Index)
	}
	if d.ID != "" {
		tc.ID = d.ID
	}
	if d.Name != "" {
		tc.Name = d.Name
	}
	tc.Arguments = append(tc.Arguments, d.Arguments...)
}

func (c *collector) response() *core.Response {
	resp := &core.Response{Message: core.Message{Role: core.RoleAssistant}, FinishReason: c.finish, Usage: c.usage}
	if c.reasoning.Len() > 0 {
		resp.Message.Parts = append(resp.Message.Parts, core.ReasoningPart{Text: c.reasoning.String()})
	}
	if c.text.Len() > 0 {
		resp.Message.Parts = append(resp.Message.Parts, core.Text(c.text.String()))
	}
	sort.Ints(c.order)
	for _, idx := range c.order {
		tc := c.calls[idx]
		if len(tc.Arguments) == 0 {
			tc.Arguments = json.RawMessage("{}")
		}
		resp.Message.Parts = append(resp.Message.Parts, *tc)
	}
	if resp.FinishReason == "" {
		resp.FinishReason = core.FinishStop
		if len(c.order) > 0 {
			resp.FinishReason = core.FinishToolCalls
		}
	}
	return resp
}
