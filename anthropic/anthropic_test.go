package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/richardwooding/llmkit/anthropic"
	"github.com/richardwooding/llmkit/core"
)

type capture struct {
	body map[string]any
	path string
	hdr  http.Header
}

func newServer(t *testing.T, respond func(w http.ResponseWriter)) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.path = r.URL.Path
		cap.hdr = r.Header.Clone()
		cap.body = map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&cap.body)
		respond(w)
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func newClient(t *testing.T, url string, opts ...core.Option) *anthropic.Client {
	t.Helper()
	opts = append([]core.Option{core.WithAPIKey("sk"), core.WithBaseURL(url)}, opts...)
	c, err := anthropic.New("claude-opus-5", opts...)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func respondJSON(body string) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) { _, _ = io.WriteString(w, body) }
}

const minimalResponse = `{"id":"msg_1","model":"claude-opus-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

func obj(v any) map[string]any { return v.(map[string]any) }

func arr(v any) []any { return v.([]any) }

func TestNewMissingKey(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	_, err := anthropic.New("claude-opus-5")
	if !errors.Is(err, core.ErrMissingAPIKey) {
		t.Fatalf("err = %v", err)
	}
	t.Setenv("ANTHROPIC_API_KEY", "from-env")
	c, err := anthropic.New("claude-opus-5")
	if err != nil || c.Provider() != anthropic.ID || c.Model() != "claude-opus-5" {
		t.Fatalf("client = %v %v", c, err)
	}
}

func TestProviderMatches(t *testing.T) {
	p := anthropic.Provider{}
	if p.ID() != "anthropic" {
		t.Fatalf("ID = %s", p.ID())
	}
	for model, want := range map[string]bool{
		"claude-opus-5": true, "Claude-Sonnet-5": true, "claude": true, "claude-haiku-4-5": true,
		"gpt-4o": false, "llama3.2:3b": false, "": false,
	} {
		if got := p.Matches(model); got != want {
			t.Errorf("Matches(%q) = %v, want %v", model, got, want)
		}
	}
	c, err := p.Open("claude-opus-5", core.NewConfig(core.WithAPIKey("k")))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := c.(core.Chatter); !ok {
		t.Fatal("client should implement Chatter")
	}
	if _, ok := c.(core.Streamer); !ok {
		t.Fatal("client should implement Streamer")
	}
	if _, ok := c.(core.Embedder); ok {
		t.Fatal("client must not implement Embedder")
	}
}

func TestHeaders(t *testing.T) {
	srv, cap := newServer(t, respondJSON(minimalResponse))
	c := newClient(t, srv.URL, anthropic.WithBeta("compact-2026-01-12", "task-budgets-2026-03-13"))
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}}); err != nil {
		t.Fatal(err)
	}
	if cap.path != "/messages" {
		t.Fatalf("path = %s", cap.path)
	}
	if cap.hdr.Get("x-api-key") != "sk" || cap.hdr.Get("anthropic-version") != anthropic.DefaultVersion {
		t.Fatalf("headers = %v", cap.hdr)
	}
	if cap.hdr.Get("anthropic-beta") != "compact-2026-01-12,task-budgets-2026-03-13" {
		t.Fatalf("anthropic-beta = %q", cap.hdr.Get("anthropic-beta"))
	}
	if cap.hdr.Get("Content-Type") != "application/json" {
		t.Fatalf("content-type = %q", cap.hdr.Get("Content-Type"))
	}

	srv2, cap2 := newServer(t, respondJSON(minimalResponse))
	c2 := newClient(t, srv2.URL, anthropic.WithVersion("2030-01-01"))
	if _, err := c2.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}}); err != nil {
		t.Fatal(err)
	}
	if cap2.hdr.Get("anthropic-version") != "2030-01-01" {
		t.Fatalf("anthropic-version = %q", cap2.hdr.Get("anthropic-version"))
	}
}

func TestRequestSystemAndDefaults(t *testing.T) {
	srv, cap := newServer(t, respondJSON(minimalResponse))
	c := newClient(t, srv.URL)
	req := &core.Request{
		Messages: []core.Message{
			core.System("one"),
			core.UserText("hi"),
			core.System("two"),
		},
		Temperature:     new(0.5),
		TopP:            new(0.9),
		Stop:            []string{"END"},
		Seed:            new(int64(7)),
		Extra:           map[string]any{"top_k": 3},
		ProviderOptions: map[string]map[string]any{"anthropic": {"top_k": 4}, "other": {"z": 1}},
	}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	b := cap.body
	if b["model"] != "claude-opus-5" || b["system"] != "one\n\ntwo" || b["max_tokens"] != float64(4096) {
		t.Fatalf("body = %v", b)
	}
	if b["temperature"] != 0.5 || b["top_p"] != 0.9 || !reflect.DeepEqual(b["stop_sequences"], []any{"END"}) {
		t.Fatalf("sampling = %v", b)
	}
	if b["top_k"] != float64(4) {
		t.Fatalf("provider extra = %v", b["top_k"])
	}
	for _, k := range []string{"seed", "z", "stream", "tools", "tool_choice", "thinking", "output_config"} {
		if _, ok := b[k]; ok {
			t.Fatalf("%s must be omitted: %v", k, b)
		}
	}
	msgs := arr(b["messages"])
	if len(msgs) != 1 || obj(msgs[0])["role"] != "user" {
		t.Fatalf("messages = %v", msgs)
	}
	if got := obj(arr(obj(msgs[0])["content"])[0]); got["type"] != "text" || got["text"] != "hi" {
		t.Fatalf("content = %v", got)
	}
}

func TestRequestMergesRolesAndToolResults(t *testing.T) {
	srv, cap := newServer(t, respondJSON(minimalResponse))
	c := newClient(t, srv.URL)
	req := &core.Request{
		MaxTokens: 10,
		Messages: []core.Message{
			core.UserText("a"),
			core.UserText("b"),
			core.Assistant(
				core.ReasoningPart{Text: "hmm", Signature: "sig1"},
				core.ReasoningPart{},
				core.ReasoningPart{Encrypted: "enc"},
				core.Text("calling"),
				core.ToolCall{ID: "t1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)},
				core.ToolCall{ID: "t2", Name: "g"},
			),
			core.ToolResults(
				core.ToolResultText("t1", "f", "42"),
				core.ToolResult{CallID: "t2", IsError: true, Content: []core.Part{core.Text("boom"), core.Image([]byte{1, 2}, "image/png")}},
			),
			core.UserText("next"),
		},
	}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if cap.body["max_tokens"] != float64(10) {
		t.Fatalf("max_tokens = %v", cap.body["max_tokens"])
	}
	msgs := arr(cap.body["messages"])
	if len(msgs) != 3 {
		t.Fatalf("messages = %d: %v", len(msgs), msgs)
	}
	first := arr(obj(msgs[0])["content"])
	if obj(msgs[0])["role"] != "user" || len(first) != 2 || obj(first[0])["text"] != "a" || obj(first[1])["text"] != "b" {
		t.Fatalf("merged user = %v", msgs[0])
	}

	checkAssistantBlocks(t, obj(msgs[1]))
	checkToolResultBlocks(t, obj(msgs[2]))
}

func checkAssistantBlocks(t *testing.T, msg map[string]any) {
	t.Helper()
	asst := arr(msg["content"])
	if msg["role"] != "assistant" || len(asst) != 5 {
		t.Fatalf("assistant = %v", msg)
	}
	if th := obj(asst[0]); th["type"] != "thinking" || th["thinking"] != "hmm" || th["signature"] != "sig1" {
		t.Fatalf("thinking = %v", th)
	}
	if rd := obj(asst[1]); rd["type"] != "redacted_thinking" || rd["data"] != "enc" {
		t.Fatalf("redacted = %v", rd)
	}
	if tx := obj(asst[2]); tx["type"] != "text" || tx["text"] != "calling" {
		t.Fatalf("text = %v", tx)
	}
	if tu := obj(asst[3]); tu["type"] != "tool_use" || tu["id"] != "t1" || tu["name"] != "f" || !reflect.DeepEqual(tu["input"], map[string]any{"a": float64(1)}) {
		t.Fatalf("tool_use = %v", tu)
	}
	if tu := obj(asst[4]); !reflect.DeepEqual(tu["input"], map[string]any{}) {
		t.Fatalf("empty tool_use input = %v", tu)
	}
}

func checkToolResultBlocks(t *testing.T, msg map[string]any) {
	t.Helper()
	last := arr(msg["content"])
	if msg["role"] != "user" || len(last) != 3 {
		t.Fatalf("tool results + user = %v", msg)
	}
	if tr := obj(last[0]); tr["type"] != "tool_result" || tr["tool_use_id"] != "t1" || tr["content"] != "42" {
		t.Fatalf("tool_result = %v", tr)
	}
	tr2 := obj(last[1])
	if tr2["tool_use_id"] != "t2" || tr2["is_error"] != true {
		t.Fatalf("tool_result 2 = %v", tr2)
	}
	parts := arr(tr2["content"])
	if len(parts) != 2 || obj(parts[0])["type"] != "text" || obj(parts[1])["type"] != "image" {
		t.Fatalf("tool_result 2 content = %v", parts)
	}
	if obj(last[2])["text"] != "next" {
		t.Fatalf("trailing user = %v", last[2])
	}
}

func TestRequestMedia(t *testing.T) {
	srv, cap := newServer(t, respondJSON(minimalResponse))
	c := newClient(t, srv.URL)
	req := &core.Request{Messages: []core.Message{core.User(
		core.Image([]byte{1, 2, 3}, "image/png"),
		core.ImageURL("https://i/x.png"),
		core.File([]byte{4}, "application/pdf", "doc.pdf"),
		core.FileURL("https://d/x.pdf", "application/pdf"),
		core.File([]byte("plain"), "text/plain", "notes.txt"),
	)}}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	content := arr(obj(arr(cap.body["messages"])[0])["content"])
	if len(content) != 5 {
		t.Fatalf("content = %v", content)
	}
	img := obj(content[0])
	if img["type"] != "image" || !reflect.DeepEqual(img["source"], map[string]any{"type": "base64", "media_type": "image/png", "data": "AQID"}) {
		t.Fatalf("image = %v", img)
	}
	if src := obj(obj(content[1])["source"]); src["type"] != "url" || src["url"] != "https://i/x.png" {
		t.Fatalf("image url = %v", src)
	}
	doc := obj(content[2])
	if doc["type"] != "document" || doc["title"] != "doc.pdf" || !reflect.DeepEqual(doc["source"], map[string]any{"type": "base64", "media_type": "application/pdf", "data": "BA=="}) {
		t.Fatalf("document = %v", doc)
	}
	if src := obj(obj(content[3])["source"]); src["type"] != "url" || src["url"] != "https://d/x.pdf" {
		t.Fatalf("document url = %v", src)
	}
	if src := obj(obj(content[4])["source"]); src["type"] != "text" || src["media_type"] != "text/plain" || src["data"] != "plain" {
		t.Fatalf("text document = %v", src)
	}
}

func TestRequestTools(t *testing.T) {
	srv, cap := newServer(t, respondJSON(minimalResponse))
	c := newClient(t, srv.URL)
	req := &core.Request{
		Messages: []core.Message{core.UserText("x")},
		Tools: []core.Tool{
			{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object","properties":{"a":{"type":"integer"}}}`), Strict: true},
			{Name: "g"},
		},
		ToolChoice: core.ToolChoice{Mode: core.ToolChoiceNamed, Name: "f"},
	}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	tools := arr(cap.body["tools"])
	f := obj(tools[0])
	if f["name"] != "f" || f["description"] != "d" || f["strict"] != true || obj(f["input_schema"])["type"] != "object" {
		t.Fatalf("tool f = %v", f)
	}
	if g := obj(tools[1]); g["input_schema"] == nil {
		t.Fatalf("tool g needs a default input_schema: %v", g)
	}
	if !reflect.DeepEqual(cap.body["tool_choice"], map[string]any{"type": "tool", "name": "f"}) {
		t.Fatalf("tool_choice = %v", cap.body["tool_choice"])
	}
}

func TestToolChoiceVariants(t *testing.T) {
	srv, cap := newServer(t, respondJSON(minimalResponse))
	c := newClient(t, srv.URL)
	tests := []struct {
		mode core.ToolChoiceMode
		want any
	}{
		{"", nil},
		{core.ToolChoiceAuto, map[string]any{"type": "auto"}},
		{core.ToolChoiceRequired, map[string]any{"type": "any"}},
		{core.ToolChoiceNone, map[string]any{"type": "none"}},
	}
	for _, tc := range tests {
		req := &core.Request{Messages: []core.Message{core.UserText("x")}, ToolChoice: core.ToolChoice{Mode: tc.mode}}
		if _, err := c.Chat(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if got := cap.body["tool_choice"]; !reflect.DeepEqual(got, tc.want) {
			t.Errorf("mode %q: tool_choice = %v, want %v", tc.mode, got, tc.want)
		}
	}
}

func TestReasoningAndFormat(t *testing.T) {
	srv, cap := newServer(t, respondJSON(minimalResponse))
	c := newClient(t, srv.URL)
	send := func(req *core.Request) {
		t.Helper()
		req.Messages = []core.Message{core.UserText("x")}
		if _, err := c.Chat(context.Background(), req); err != nil {
			t.Fatal(err)
		}
	}

	send(&core.Request{Reasoning: &core.ReasoningConfig{BudgetTokens: 2048}})
	if !reflect.DeepEqual(cap.body["thinking"], map[string]any{"type": "enabled", "budget_tokens": float64(2048)}) {
		t.Fatalf("budget thinking = %v", cap.body["thinking"])
	}
	if _, ok := cap.body["output_config"]; ok {
		t.Fatal("output_config must be omitted without effort or format")
	}

	send(&core.Request{Reasoning: &core.ReasoningConfig{Effort: "low"}})
	if !reflect.DeepEqual(cap.body["thinking"], map[string]any{"type": "adaptive"}) {
		t.Fatalf("adaptive thinking = %v", cap.body["thinking"])
	}
	if !reflect.DeepEqual(cap.body["output_config"], map[string]any{"effort": "low"}) {
		t.Fatalf("output_config = %v", cap.body["output_config"])
	}

	send(&core.Request{Reasoning: &core.ReasoningConfig{}})
	if !reflect.DeepEqual(cap.body["thinking"], map[string]any{"type": "adaptive"}) {
		t.Fatalf("empty reasoning = %v", cap.body["thinking"])
	}

	send(&core.Request{Reasoning: &core.ReasoningConfig{Summary: "omitted"}})
	if !reflect.DeepEqual(cap.body["thinking"], map[string]any{"type": "adaptive", "display": "omitted"}) {
		t.Fatalf("display = %v", cap.body["thinking"])
	}
	// OpenAI's vocabulary is translated so one ReasoningConfig works on both.
	send(&core.Request{Reasoning: &core.ReasoningConfig{Summary: "auto", BudgetTokens: 1024}})
	if !reflect.DeepEqual(cap.body["thinking"], map[string]any{"type": "enabled", "budget_tokens": float64(1024), "display": "summarized"}) {
		t.Fatalf("display auto = %v", cap.body["thinking"])
	}

	schema := json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false}`)
	send(&core.Request{Format: &core.ResponseFormat{Type: core.FormatJSONSchema, Schema: schema}})
	format := obj(obj(cap.body["output_config"])["format"])
	if format["type"] != "json_schema" || obj(format["schema"])["type"] != "object" {
		t.Fatalf("format = %v", format)
	}
	if _, ok := cap.body["thinking"]; ok {
		t.Fatal("thinking must be omitted without Reasoning")
	}
}

// cacheMarks lists every wire location carrying cache_control.
func cacheMarks(body map[string]any) []string {
	var marks []string
	if tools, ok := body["tools"].([]any); ok {
		for i, tl := range tools {
			if _, ok := obj(tl)["cache_control"]; ok {
				marks = append(marks, fmt.Sprintf("tools[%d]", i))
			}
		}
	}
	if sys, ok := body["system"].([]any); ok {
		for i, b := range sys {
			if _, ok := obj(b)["cache_control"]; ok {
				marks = append(marks, fmt.Sprintf("system[%d]", i))
			}
		}
	}
	for i, m := range arr(body["messages"]) {
		for j, b := range arr(obj(m)["content"]) {
			if _, ok := obj(b)["cache_control"]; ok {
				marks = append(marks, fmt.Sprintf("messages[%d].content[%d]", i, j))
			}
		}
	}
	return marks
}

func TestRequestCacheControl(t *testing.T) {
	tests := []struct {
		name       string
		cache      *core.CacheConfig
		wantSystem any
		wantMarks  []string
		wantCC     map[string]any
		wantErr    bool
	}{
		{name: "nil keeps system a string", wantSystem: "sys"},
		{
			name: "system", cache: &core.CacheConfig{System: true},
			wantSystem: []any{map[string]any{"type": "text", "text": "sys", "cache_control": map[string]any{"type": "ephemeral"}}},
			wantMarks:  []string{"system[0]"}, wantCC: map[string]any{"type": "ephemeral"},
		},
		{
			name: "tools with 1h ttl", cache: &core.CacheConfig{Tools: true, TTL: "1h"},
			wantSystem: []any{map[string]any{"type": "text", "text": "sys"}},
			wantMarks:  []string{"tools[1]"}, wantCC: map[string]any{"type": "ephemeral", "ttl": "1h"},
		},
		{
			name: "turns mark the last block of the last user-role messages", cache: &core.CacheConfig{Turns: 2},
			wantSystem: []any{map[string]any{"type": "text", "text": "sys"}},
			wantMarks:  []string{"messages[2].content[1]", "messages[4].content[0]"}, wantCC: map[string]any{"type": "ephemeral"},
		},
		{
			name: "capped at four breakpoints", cache: &core.CacheConfig{System: true, Tools: true, Turns: 5, TTL: "5m"},
			wantSystem: []any{map[string]any{"type": "text", "text": "sys", "cache_control": map[string]any{"type": "ephemeral"}}},
			wantMarks:  []string{"tools[1]", "system[0]", "messages[2].content[1]", "messages[4].content[0]"},
			wantCC:     map[string]any{"type": "ephemeral"},
		},
		{name: "bad ttl", cache: &core.CacheConfig{System: true, TTL: "2h"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, cap := newServer(t, respondJSON(minimalResponse))
			c := newClient(t, srv.URL)
			req := &core.Request{
				Cache: tt.cache,
				Tools: []core.Tool{{Name: "f"}, {Name: "g"}},
				Messages: []core.Message{
					core.System("sys"),
					core.UserText("a"),
					core.Assistant(core.Text("x"), core.ToolCall{ID: "t1", Name: "f"}),
					core.ToolResults(core.ToolResultText("t1", "f", "1")),
					core.UserText("b"),
					core.Assistant(core.Text("y")),
					core.UserText("c"),
				},
			}
			_, err := c.Chat(context.Background(), req)
			if tt.wantErr {
				if err == nil || cap.body != nil {
					t.Fatalf("want error before I/O, got err=%v body=%v", err, cap.body)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(cap.body["system"], tt.wantSystem) {
				t.Fatalf("system = %#v, want %#v", cap.body["system"], tt.wantSystem)
			}
			if got := cacheMarks(cap.body); !reflect.DeepEqual(got, tt.wantMarks) {
				t.Fatalf("cache_control at %v, want %v", got, tt.wantMarks)
			}
			if len(tt.wantMarks) > 0 {
				last := obj(arr(obj(arr(cap.body["messages"])[4])["content"])[0])
				if cc, ok := last["cache_control"]; ok && !reflect.DeepEqual(cc, tt.wantCC) {
					t.Fatalf("cache_control = %v, want %v", cc, tt.wantCC)
				}
			}
		})
	}
}

func TestUnsupportedFailsBeforeIO(t *testing.T) {
	hit := false
	srv, _ := newServer(t, func(http.ResponseWriter) { hit = true })
	c := newClient(t, srv.URL)
	reqs := []*core.Request{
		{Messages: []core.Message{core.User(core.Audio([]byte{1}, "audio/wav"))}},
		{Messages: []core.Message{core.ToolResults(core.ToolResult{CallID: "c", Content: []core.Part{core.Audio([]byte{1}, "audio/wav")}})}},
		{Messages: []core.Message{core.UserText("x")}, Format: &core.ResponseFormat{Type: core.FormatJSON}},
	}
	for i, req := range reqs {
		_, err := c.Chat(context.Background(), req)
		if !errors.Is(err, core.ErrUnsupported) {
			t.Fatalf("%d: err = %v", i, err)
		}
		n := 0
		for _, err := range c.Stream(context.Background(), req) {
			n++
			if !errors.Is(err, core.ErrUnsupported) {
				t.Fatalf("%d: stream err = %v", i, err)
			}
		}
		if n != 1 {
			t.Fatalf("%d: stream yielded %d items", i, n)
		}
	}
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Format: &core.ResponseFormat{Type: core.FormatJSONSchema}}); err == nil {
		t.Fatal("json_schema without schema must fail")
	}
	if hit {
		t.Fatal("server must not be contacted")
	}
}

func TestResponseDecode(t *testing.T) {
	srv, _ := newServer(t, respondJSON(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5",
	  "content":[
	    {"type":"thinking","thinking":"let me see","signature":"sig"},
	    {"type":"redacted_thinking","data":"enc"},
	    {"type":"text","text":"Checking."},
	    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}},
	    {"type":"tool_use","id":"toolu_2","name":"noop","input":{}}
	  ],
	  "stop_reason":"tool_use","stop_sequence":null,
	  "usage":{"input_tokens":30,"output_tokens":12,"cache_read_input_tokens":20,"cache_creation_input_tokens":8}}`))
	c := newClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Raw) == 0 {
		t.Fatal("Raw must be set")
	}
	resp.Raw = nil
	want := &core.Response{
		ID: "msg_1", Model: "claude-opus-5", FinishReason: core.FinishToolCalls,
		Message: core.Assistant(
			core.ReasoningPart{Text: "let me see", Signature: "sig"},
			core.ReasoningPart{Encrypted: "enc"},
			core.Text("Checking."),
			core.ToolCall{ID: "toolu_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`)},
			core.ToolCall{ID: "toolu_2", Name: "noop", Arguments: json.RawMessage(`{}`)},
		),
		// InputTokens is the whole prompt: 30 uncached + 20 read + 8 written.
		Usage: core.Usage{InputTokens: 58, OutputTokens: 12, TotalTokens: 70, CachedInputTokens: 20, CacheWriteTokens: 8},
	}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("resp:\n got %+v\nwant %+v", resp, want)
	}
}

func TestCountTokens(t *testing.T) {
	srv, cap := newServer(t, respondJSON(`{"input_tokens":403}`))
	c := newClient(t, srv.URL)
	var counter core.TokenCounter = c
	req := &core.Request{
		Messages:    []core.Message{core.System("sys"), core.UserText("hi")},
		Tools:       []core.Tool{{Name: "f"}},
		ToolChoice:  core.ToolChoice{Mode: core.ToolChoiceAuto},
		Reasoning:   &core.ReasoningConfig{Effort: "low"},
		Cache:       &core.CacheConfig{System: true},
		MaxTokens:   10,
		Temperature: new(0.5),
		Stop:        []string{"x"},
		Extra:       map[string]any{"top_k": 3},
	}
	n, err := counter.CountTokens(context.Background(), req)
	if err != nil || n != 403 {
		t.Fatalf("count = %d, %v", n, err)
	}
	if cap.path != "/messages/count_tokens" {
		t.Fatalf("path = %s", cap.path)
	}
	b := cap.body
	if b["model"] != "claude-opus-5" || len(arr(b["messages"])) != 1 || len(arr(b["tools"])) != 1 || obj(b["tool_choice"])["type"] != "auto" || obj(b["thinking"])["type"] != "adaptive" {
		t.Fatalf("body = %v", b)
	}
	if _, ok := obj(arr(b["system"])[0])["cache_control"]; !ok {
		t.Fatalf("system = %v", b["system"])
	}
	for _, k := range []string{"max_tokens", "temperature", "stop_sequences", "stream", "output_config", "top_k"} {
		if _, ok := b[k]; ok {
			t.Fatalf("%s must be omitted from count_tokens: %v", k, b)
		}
	}
	if _, err := counter.CountTokens(context.Background(), &core.Request{Messages: []core.Message{core.User(core.Audio(nil, "audio/wav"))}}); !errors.Is(err, core.ErrUnsupported) {
		t.Fatalf("err = %v", err)
	}
}

func TestStopReasons(t *testing.T) {
	tests := map[string]core.FinishReason{
		"end_turn":      core.FinishStop,
		"stop_sequence": core.FinishStop,
		"max_tokens":    core.FinishLength,
		"tool_use":      core.FinishToolCalls,
		"refusal":       core.FinishContentFilter,
		"pause_turn":    core.FinishOther,
	}
	for reason, want := range tests {
		srv, _ := newServer(t, respondJSON(`{"id":"m","model":"x","content":[],"stop_reason":"`+reason+`","usage":{"input_tokens":1,"output_tokens":1}}`))
		c := newClient(t, srv.URL)
		resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != want {
			t.Errorf("%s: finish = %s, want %s", reason, resp.FinishReason, want)
		}
	}
}

func TestChatHTTPError(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		w.Header().Set("Retry-After", "2")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
	})
	c := newClient(t, srv.URL)
	_, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	var apiErr *core.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 429 || apiErr.Type != "rate_limit_error" || apiErr.Message != "slow down" {
		t.Fatalf("err = %#v", err)
	}
	if !errors.Is(err, core.ErrRateLimited) {
		t.Fatalf("err should match ErrRateLimited: %v", err)
	}
}

const streamFixture = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1,"cache_read_input_tokens":5,"cache_creation_input_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":25,"cache_read_input_tokens":5,"cache_creation_input_tokens":10,"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}

`

func sseServer(t *testing.T, fixture string) (*httptest.Server, *capture) {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, fixture)
	})
}

func TestStream(t *testing.T) {
	srv, cap := sseServer(t, streamFixture)
	c := newClient(t, srv.URL)
	var got []core.Chunk
	for ch, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		if ch.Kind != core.ChunkFinish && len(ch.Raw) == 0 {
			t.Fatalf("chunk without Raw: %+v", ch)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	if cap.body["stream"] != true {
		t.Fatalf("stream flag = %v", cap.body["stream"])
	}
	want := []core.Chunk{
		{Kind: core.ChunkText, Text: "Hel"},
		{Kind: core.ChunkText, Text: "lo"},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, ID: "toolu_1", Name: "f"}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, Arguments: `{"a":`}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, Arguments: `1}`}},
		{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls, Usage: &core.Usage{InputTokens: 40, OutputTokens: 7, TotalTokens: 47, CachedInputTokens: 5, CacheWriteTokens: 10}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks:\n got %+v\nwant %+v", got, want)
	}
}

func TestStreamThinking(t *testing.T) {
	fixture := `event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":3,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"enc"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"sig2"}}

event: content_block_start
data: {"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"done"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}

event: message_stop
data: {"type":"message_stop"}

`
	srv, _ := sseServer(t, fixture)
	c := newClient(t, srv.URL)
	var got []core.Chunk
	for ch, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	want := []core.Chunk{
		{Kind: core.ChunkReasoning, Text: "hmm", Reasoning: &core.ReasoningDelta{Index: 0, Text: "hmm"}},
		{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 0, Signature: "sig"}},
		{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 1, Encrypted: "enc"}},
		{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 2}},
		{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 2, Signature: "sig2"}},
		{Kind: core.ChunkText, Text: "done"},
		{Kind: core.ChunkFinish, FinishReason: core.FinishStop, Usage: &core.Usage{InputTokens: 3, OutputTokens: 4, TotalTokens: 7}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks:\n got %+v\nwant %+v", got, want)
	}
}

func TestStreamHTTPError(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`)
	})
	c := newClient(t, srv.URL)
	var n int
	var gotErr error
	for _, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		n++
		gotErr = err
	}
	var apiErr *core.APIError
	if n != 1 || !errors.As(gotErr, &apiErr) || apiErr.Status != 401 || apiErr.Type != "authentication_error" {
		t.Fatalf("n=%d err=%#v", n, gotErr)
	}
}

func TestStreamErrorEvent(t *testing.T) {
	fixture := `event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":3,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}

event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}

`
	srv, _ := sseServer(t, fixture)
	c := newClient(t, srv.URL)
	var chunks []core.Chunk
	var gotErr error
	for ch, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			gotErr = err
			continue
		}
		chunks = append(chunks, ch)
	}
	var apiErr *core.APIError
	if !errors.As(gotErr, &apiErr) || apiErr.Type != "overloaded_error" || apiErr.Message != "Overloaded" || apiErr.Status != 0 {
		t.Fatalf("err = %#v", gotErr)
	}
	if len(chunks) != 1 || chunks[0].Kind != core.ChunkText || chunks[0].Text != "partial" {
		t.Fatalf("chunks before error = %+v", chunks)
	}
}

func TestStreamBreak(t *testing.T) {
	srv, _ := sseServer(t, streamFixture)
	c := newClient(t, srv.URL)
	n := 0
	for _, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		n++
		if n == 2 {
			break
		}
	}
	if n != 2 {
		t.Fatalf("break did not stop iteration: %d", n)
	}
}
