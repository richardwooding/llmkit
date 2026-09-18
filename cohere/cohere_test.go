package cohere_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/richardwooding/llmkit/cohere"
	"github.com/richardwooding/llmkit/core"
)

var (
	_ core.Chatter  = (*cohere.Client)(nil)
	_ core.Streamer = (*cohere.Client)(nil)
	_ core.Embedder = (*cohere.Client)(nil)
)

type capture struct {
	body map[string]any
	path string
	auth string
	hits int
}

func newServer(t *testing.T, respond func(w http.ResponseWriter)) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.hits++
		cap.path = r.URL.Path
		cap.auth = r.Header.Get("Authorization")
		cap.body = map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&cap.body)
		respond(w)
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func jsonServer(t *testing.T, response string) (*httptest.Server, *capture) {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter) { _, _ = io.WriteString(w, response) })
}

func newClient(t *testing.T, url string) *cohere.Client {
	t.Helper()
	c, err := cohere.New("command-a-03-2025", core.WithAPIKey("ck"), core.WithBaseURL(url))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func obj(t *testing.T, v any) map[string]any {
	t.Helper()
	m, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("expected object, got %T: %v", v, v)
	}
	return m
}

func list(t *testing.T, v any) []any {
	t.Helper()
	l, ok := v.([]any)
	if !ok {
		t.Fatalf("expected array, got %T: %v", v, v)
	}
	return l
}

const chatFixture = `{"id":"r1","finish_reason":"COMPLETE","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"hello"}]},
  "usage":{"billed_units":{"input_tokens":30,"output_tokens":20},"tokens":{"input_tokens":3,"output_tokens":2},"cached_tokens":1}}`

func chatRoundTrip(t *testing.T) (map[string]any, *core.Response) {
	t.Helper()
	srv, cap := jsonServer(t, chatFixture)
	c := newClient(t, srv.URL)
	req := &core.Request{
		Messages: []core.Message{
			core.System("sys"),
			core.User(core.Text("look"), core.Image([]byte{1, 2}, "image/png"), core.ImageURL("https://i/x.png")),
			core.Assistant(core.ReasoningPart{Text: "plan"}, core.Text("calling"), core.ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)}),
			core.Assistant(core.ToolCall{ID: "c2", Name: "g"}),
			core.ToolResults(core.ToolResultText("c1", "f", "42"), core.ToolResultText("c2", "g", "43")),
			core.UserText("go"),
		},
		Tools:           []core.Tool{{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:      core.ToolChoice{Mode: core.ToolChoiceRequired},
		MaxTokens:       10,
		Temperature:     new(0.5),
		TopP:            new(0.9),
		Seed:            new(int64(7)),
		Stop:            []string{"x"},
		Format:          &core.ResponseFormat{Type: core.FormatJSONSchema, Schema: json.RawMessage(`{"type":"object"}`)},
		Reasoning:       &core.ReasoningConfig{BudgetTokens: 512},
		Extra:           map[string]any{"k": 3},
		ProviderOptions: map[string]map[string]any{"cohere": {"k": 4}, "other": {"z": 1}},
	}
	resp, err := c.Chat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if cap.path != "/chat" || cap.auth != "Bearer ck" {
		t.Fatalf("path=%s auth=%s", cap.path, cap.auth)
	}
	return cap.body, resp
}

func TestChatRequestParams(t *testing.T) {
	b, _ := chatRoundTrip(t)
	if b["model"] != "command-a-03-2025" || b["max_tokens"] != float64(10) || b["temperature"] != 0.5 || b["p"] != 0.9 || b["seed"] != float64(7) || b["k"] != float64(4) {
		t.Fatalf("body = %v", b)
	}
	if _, ok := b["top_p"]; ok {
		t.Fatal("top_p must be sent as p")
	}
	if _, ok := b["z"]; ok {
		t.Fatal("other provider's options leaked")
	}
	if _, ok := b["stream"]; ok {
		t.Fatal("stream must be omitted for Chat")
	}
	if !reflect.DeepEqual(b["stop_sequences"], []any{"x"}) || b["tool_choice"] != "REQUIRED" {
		t.Fatalf("body = %v", b)
	}
	thinking := obj(t, b["thinking"])
	if thinking["type"] != "enabled" || thinking["token_budget"] != float64(512) {
		t.Fatalf("thinking = %v", thinking)
	}
	rf := obj(t, b["response_format"])
	if rf["type"] != "json_object" || obj(t, rf["json_schema"])["type"] != "object" {
		t.Fatalf("response_format = %v", rf)
	}
	tool := obj(t, obj(t, list(t, b["tools"])[0])["function"])
	if tool["name"] != "f" || tool["description"] != "d" || obj(t, tool["parameters"])["type"] != "object" {
		t.Fatalf("tools = %v", tool)
	}
}

func chatMessages(t *testing.T) []any {
	t.Helper()
	b, _ := chatRoundTrip(t)
	msgs := list(t, b["messages"])
	if len(msgs) != 7 {
		t.Fatalf("messages = %d: %v", len(msgs), msgs)
	}
	return msgs
}

func TestChatRequestUserMessages(t *testing.T) {
	msgs := chatMessages(t)
	sys := obj(t, msgs[0])
	if sys["role"] != "system" || sys["content"] != "sys" {
		t.Fatalf("system = %v", sys)
	}
	user := list(t, obj(t, msgs[1])["content"])
	if len(user) != 3 || obj(t, user[0])["type"] != "text" || obj(t, user[0])["text"] != "look" {
		t.Fatalf("user = %v", user)
	}
	img := obj(t, obj(t, user[1])["image_url"])["url"].(string)
	if obj(t, user[1])["type"] != "image_url" || !strings.HasPrefix(img, "data:image/png;base64,") {
		t.Fatalf("image = %v", user[1])
	}
	if obj(t, obj(t, user[2])["image_url"])["url"] != "https://i/x.png" {
		t.Fatalf("image url = %v", user[2])
	}
	if obj(t, msgs[6])["content"] != "go" {
		t.Fatal("single text part should be a plain string")
	}
}

func TestChatRequestAssistantAndToolMessages(t *testing.T) {
	msgs := chatMessages(t)
	asst := obj(t, msgs[2])
	content := list(t, asst["content"])
	if asst["role"] != "assistant" || len(content) != 2 || obj(t, content[0])["type"] != "thinking" || obj(t, content[0])["thinking"] != "plan" || obj(t, content[1])["text"] != "calling" {
		t.Fatalf("assistant = %v", asst)
	}
	tc := obj(t, list(t, asst["tool_calls"])[0])
	if tc["id"] != "c1" || tc["type"] != "function" || obj(t, tc["function"])["name"] != "f" || obj(t, tc["function"])["arguments"] != `{"a":1}` {
		t.Fatalf("tool_calls = %v", tc)
	}
	asst2 := obj(t, msgs[3])
	if _, hasContent := asst2["content"]; hasContent {
		t.Fatalf("assistant with only tool calls must omit content: %v", asst2)
	}
	if obj(t, obj(t, list(t, asst2["tool_calls"])[0])["function"])["arguments"] != "{}" {
		t.Fatalf("empty arguments must become {}: %v", asst2)
	}
	for i, want := range []struct{ id, text string }{{"c1", "42"}, {"c2", "43"}} {
		tm := obj(t, msgs[4+i])
		doc := obj(t, list(t, tm["content"])[0])
		if tm["role"] != "tool" || tm["tool_call_id"] != want.id || doc["type"] != "document" || obj(t, obj(t, doc["document"])["data"])["text"] != want.text {
			t.Fatalf("tool msg %d = %v", i, tm)
		}
	}
}

func TestChatResponseDecode(t *testing.T) {
	_, resp := chatRoundTrip(t)
	want := &core.Response{
		ID: "r1", Model: "command-a-03-2025", FinishReason: core.FinishStop,
		Message: core.Assistant(core.ReasoningPart{Text: "hmm"}, core.Text("hello")),
		Usage:   core.Usage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5, CachedInputTokens: 1},
	}
	if len(resp.Raw) == 0 {
		t.Fatal("Raw must be set")
	}
	resp.Raw = nil
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("resp = %+v", resp)
	}
}

func TestChatOptionalFieldsOmitted(t *testing.T) {
	srv, cap := jsonServer(t, `{"id":"r2","finish_reason":"COMPLETE","message":{"content":[]}}`)
	c := newClient(t, srv.URL)
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Format: &core.ResponseFormat{Type: core.FormatJSON}}); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"tools", "tool_choice", "max_tokens", "temperature", "p", "seed", "stop_sequences", "thinking"} {
		if _, ok := cap.body[k]; ok {
			t.Errorf("%s must be omitted when unset: %v", k, cap.body[k])
		}
	}
	if obj(t, cap.body["response_format"])["type"] != "json_object" {
		t.Fatalf("response_format = %v", cap.body["response_format"])
	}
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, ToolChoice: core.ToolChoice{Mode: core.ToolChoiceNone}}); err != nil {
		t.Fatal(err)
	}
	if cap.body["tool_choice"] != "NONE" {
		t.Fatalf("tool_choice = %v", cap.body["tool_choice"])
	}
}

func TestChatToolCallsResponse(t *testing.T) {
	srv, _ := jsonServer(t, `{"id":"r3","finish_reason":"TOOL_CALL","message":{"role":"assistant","tool_plan":"I will call f.","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}},{"id":"c2","type":"function","function":{"name":"g","arguments":"not json"}},{"id":"c3","type":"function","function":{"name":"h"}}]},
	  "usage":{"billed_units":{"input_tokens":4,"output_tokens":6}}}`)
	c := newClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	want := core.Assistant(
		core.ReasoningPart{Text: "I will call f."},
		core.ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)},
		core.ToolCall{ID: "c2", Name: "g", Arguments: json.RawMessage(`"not json"`)},
		core.ToolCall{ID: "c3", Name: "h", Arguments: json.RawMessage(`{}`)},
	)
	if !reflect.DeepEqual(resp.Message, want) {
		t.Fatalf("message = %+v", resp.Message)
	}
	if resp.FinishReason != core.FinishToolCalls || resp.Text() != "" {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Usage != (core.Usage{InputTokens: 4, OutputTokens: 6, TotalTokens: 10}) {
		t.Fatalf("usage fallback to billed_units failed: %+v", resp.Usage)
	}
}

func TestFinishReasonMapping(t *testing.T) {
	tests := map[string]core.FinishReason{
		"COMPLETE":      core.FinishStop,
		"STOP_SEQUENCE": core.FinishStop,
		"MAX_TOKENS":    core.FinishLength,
		"TOOL_CALL":     core.FinishToolCalls,
		"ERROR":         core.FinishOther,
		"TIMEOUT":       core.FinishOther,
		"SOMETHING_NEW": core.FinishOther,
		"":              core.FinishOther,
	}
	for reason, want := range tests {
		srv, _ := jsonServer(t, `{"id":"r","finish_reason":"`+reason+`","message":{"content":[{"type":"text","text":"x"}]}}`)
		resp, err := newClient(t, srv.URL).Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != want {
			t.Errorf("finish_reason %q = %s, want %s", reason, resp.FinishReason, want)
		}
	}
}

func TestUnsupportedFailsBeforeIO(t *testing.T) {
	srv, cap := jsonServer(t, chatFixture)
	c := newClient(t, srv.URL)
	tests := []*core.Request{
		{Messages: []core.Message{core.User(core.Audio([]byte{1}, "audio/wav"))}},
		{Messages: []core.Message{core.User(core.File([]byte{1}, "application/pdf", "x.pdf"))}},
		{Messages: []core.Message{core.ToolResults(core.ToolResult{CallID: "c", Content: []core.Part{core.Image([]byte{1}, "image/png")}})}},
		{Messages: []core.Message{core.UserText("x")}, ToolChoice: core.ToolChoice{Mode: core.ToolChoiceNamed, Name: "f"}},
	}
	for i, req := range tests {
		if _, err := c.Chat(context.Background(), req); !errors.Is(err, core.ErrUnsupported) {
			t.Fatalf("case %d: err = %v", i, err)
		}
		n := 0
		for _, err := range c.Stream(context.Background(), req) {
			n++
			if !errors.Is(err, core.ErrUnsupported) {
				t.Fatalf("case %d: stream err = %v", i, err)
			}
		}
		if n != 1 {
			t.Fatalf("case %d: stream yielded %d items", i, n)
		}
	}
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Format: &core.ResponseFormat{Type: "yaml"}}); err == nil {
		t.Fatal("unknown format must error")
	}
	if _, err := c.Chat(context.Background(), nil); err == nil {
		t.Fatal("nil request must error")
	}
	if cap.hits != 0 {
		t.Fatalf("server must not be contacted, got %d hits", cap.hits)
	}
}

const streamFixture = `event: message-start
data: {"type":"message-start","id":"s1","delta":{"message":{"role":"assistant"}}}

event: content-start
data: {"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}

event: content-delta
data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hel"}}}}

event: content-delta
data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"lo"}}}}

event: content-end
data: {"type":"content-end","index":0}

event: tool-plan-delta
data: {"type":"tool-plan-delta","delta":{"message":{"tool_plan":"I will call f."}}}

event: tool-call-start
data: {"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"c1","type":"function","function":{"name":"f","arguments":""}}}}}

event: tool-call-delta
data: {"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"{\"a\":"}}}}}

event: tool-call-delta
data: {"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"1}"}}}}}

event: tool-call-end
data: {"type":"tool-call-end","index":0}

event: message-end
data: {"type":"message-end","id":"s1","delta":{"finish_reason":"TOOL_CALL","usage":{"billed_units":{"input_tokens":50,"output_tokens":70},"tokens":{"input_tokens":5,"output_tokens":7}}}}

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
	seq := c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	if cap.hits != 0 {
		t.Fatal("request must not be sent before iteration starts")
	}
	var got []core.Chunk
	for ch, err := range seq {
		if err != nil {
			t.Fatal(err)
		}
		if len(ch.Raw) == 0 {
			t.Fatalf("chunk %+v has no Raw", ch)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	if cap.body["stream"] != true {
		t.Fatalf("body = %v", cap.body)
	}
	want := []core.Chunk{
		{Kind: core.ChunkText, Text: "Hel"},
		{Kind: core.ChunkText, Text: "lo"},
		{Kind: core.ChunkReasoning, Text: "I will call f."},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, ID: "c1", Name: "f"}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, Arguments: `{"a":`}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, Arguments: `1}`}},
		{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls, Usage: &core.Usage{InputTokens: 5, OutputTokens: 7, TotalTokens: 12}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks:\n got %+v\nwant %+v", got, want)
	}
}

func TestStreamThinking(t *testing.T) {
	srv, _ := sseServer(t, `event: content-start
data: {"type":"content-start","index":0,"delta":{"message":{"content":{"type":"thinking","thinking":""}}}}

event: content-delta
data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"thinking":"let me"}}}}

event: content-delta
data: {"type":"content-delta","index":1,"delta":{"message":{"content":{"text":"ok"}}}}

event: message-end
data: {"type":"message-end","delta":{"finish_reason":"MAX_TOKENS"}}

`)
	var got []core.Chunk
	for ch, err := range newClient(t, srv.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	want := []core.Chunk{
		{Kind: core.ChunkReasoning, Text: "let me"},
		{Kind: core.ChunkText, Text: "ok"},
		{Kind: core.ChunkFinish, FinishReason: core.FinishLength},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks:\n got %+v\nwant %+v", got, want)
	}
}

func TestStreamTruncatedStillFinishes(t *testing.T) {
	srv, _ := sseServer(t, `event: content-delta
data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"partial"}}}}

`)
	var got []core.Chunk
	for ch, err := range newClient(t, srv.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, ch)
	}
	if len(got) != 2 || got[1].Kind != core.ChunkFinish || got[1].FinishReason != core.FinishOther || got[1].Usage != nil {
		t.Fatalf("chunks = %+v", got)
	}
}

func TestStreamErrorsAndBreak(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"message":"invalid api token"}`)
	})
	c := newClient(t, srv.URL)
	var n int
	var gotErr error
	for _, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		n++
		gotErr = err
	}
	var apiErr *core.APIError
	if n != 1 || !errors.As(gotErr, &apiErr) || apiErr.Status != 401 || apiErr.Message != "invalid api token" {
		t.Fatalf("n=%d err=%v", n, gotErr)
	}

	srv2, _ := sseServer(t, streamFixture)
	n = 0
	for _, err := range newClient(t, srv2.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
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

	srv3, _ := sseServer(t, "event: content-delta\ndata: {not json\n\n")
	n = 0
	for _, err := range newClient(t, srv3.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		n++
		gotErr = err
	}
	if n != 1 || gotErr == nil {
		t.Fatalf("bad event: n=%d err=%v", n, gotErr)
	}
}

func TestEmbed(t *testing.T) {
	srv, cap := jsonServer(t, `{"id":"e1","embeddings":{"float":[[0.1,0.2],[0.3,0.4]]},"texts":["a","b"],"meta":{"api_version":{"version":"2"},"billed_units":{"input_tokens":9},"tokens":{"input_tokens":11}}}`)
	c, err := cohere.New("embed-v4.0", core.WithAPIKey("ck"), core.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{
		Inputs:          []string{"a", "b"},
		Dimensions:      256,
		InputType:       core.EmbedQuery,
		ProviderOptions: map[string]map[string]any{"cohere": {"truncate": "START"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	b := cap.body
	if cap.path != "/embed" || b["model"] != "embed-v4.0" || b["input_type"] != "search_query" || b["output_dimension"] != float64(256) || b["truncate"] != "START" {
		t.Fatalf("req = %s %v", cap.path, b)
	}
	if !reflect.DeepEqual(b["texts"], []any{"a", "b"}) || !reflect.DeepEqual(b["embedding_types"], []any{"float"}) {
		t.Fatalf("body = %v", b)
	}
	resp.Raw = nil
	want := &core.EmbedResponse{Embeddings: [][]float32{{0.1, 0.2}, {0.3, 0.4}}, Model: "embed-v4.0", Usage: core.Usage{InputTokens: 9, TotalTokens: 9}}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("resp = %+v", resp)
	}

	if _, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a"}, InputType: core.EmbedDocument}); err != nil {
		t.Fatal(err)
	}
	if cap.body["input_type"] != "search_document" || cap.body["truncate"] != "END" {
		t.Fatalf("body = %v", cap.body)
	}
	if _, ok := cap.body["output_dimension"]; ok {
		t.Fatal("output_dimension must be omitted when Dimensions is zero")
	}
	if _, err := c.Embed(context.Background(), &core.EmbedRequest{}); err != nil {
		if cap.hits != 2 {
			t.Fatal("empty inputs must fail before I/O")
		}
	} else {
		t.Fatal("empty inputs must error")
	}
}

func TestMissingKey(t *testing.T) {
	t.Setenv("COHERE_API_KEY", "")
	_, err := cohere.New("command-r")
	if !errors.Is(err, core.ErrMissingAPIKey) || !strings.Contains(err.Error(), "COHERE_API_KEY") {
		t.Fatalf("err = %v", err)
	}
	t.Setenv("COHERE_API_KEY", "from-env")
	c, err := cohere.New("command-r")
	if err != nil || c.Provider() != "cohere" || c.Model() != "command-r" {
		t.Fatalf("client = %v %v", c, err)
	}
}

func TestProviderMatches(t *testing.T) {
	p := cohere.Provider{}
	if p.ID() != "cohere" {
		t.Fatalf("id = %s", p.ID())
	}
	tests := map[string]bool{
		"command-a-03-2025":      true,
		"Command-R-Plus":         true,
		"command":                true,
		"embed-v4.0":             true,
		"embed-english-v3.0":     true,
		"rerank-v3.5":            true,
		"c4ai-aya-expanse-8b":    true,
		"aya-vision-32b":         true,
		"gpt-4o":                 false,
		"text-embedding-3-small": false,
		"voyage-4":               false,
		"my-command":             false,
		"":                       false,
	}
	for model, want := range tests {
		if got := p.Matches(model); got != want {
			t.Errorf("Matches(%q) = %v, want %v", model, got, want)
		}
	}
	t.Setenv("COHERE_API_KEY", "k")
	c, err := p.Open("command-r", nil)
	if err != nil || c.Model() != "command-r" || c.Provider() != "cohere" {
		t.Fatalf("Open = %v %v", c, err)
	}
	if _, ok := c.(core.Chatter); !ok {
		t.Fatal("client should implement Chatter")
	}
}
