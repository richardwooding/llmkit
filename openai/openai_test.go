package openai_test

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

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/openai"
)

type capture struct {
	body map[string]any
	path string
	hdr  http.Header
	hits int
}

func newServer(t *testing.T, respond func(w http.ResponseWriter)) (*httptest.Server, *capture) {
	t.Helper()
	c := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c.hits++
		c.path = r.URL.Path
		c.hdr = r.Header.Clone()
		c.body = map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&c.body)
		respond(w)
	}))
	t.Cleanup(srv.Close)
	return srv, c
}

func newClient(t *testing.T, url string, opts ...core.Option) *openai.Client {
	t.Helper()
	c, err := openai.New("gpt-4.1", append([]core.Option{core.WithAPIKey("sk"), core.WithBaseURL(url)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func serve(s string) func(http.ResponseWriter) {
	return func(w http.ResponseWriter) { _, _ = io.WriteString(w, s) }
}

func serveSSE(s string) func(http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, s)
	}
}

func obj(t *testing.T, v any) map[string]any {
	t.Helper()
	m, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("expected object, got %T: %v", v, v)
	}
	return m
}

func arr(t *testing.T, v any) []any {
	t.Helper()
	a, ok := v.([]any)
	if !ok {
		t.Fatalf("expected array, got %T: %v", v, v)
	}
	return a
}

func TestMatches(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gpt-4.1", true},
		{"GPT-5", true},
		{"gpt5", true},
		{"chatgpt-4o-latest", true},
		{"o1-mini", true},
		{"o3", true},
		{"o4-mini", true},
		{"text-embedding-3-small", true},
		{"text-embedding-ada-002", true},
		{"claude-3", false},
		{"llama3.2:3b", false},
		{"deepseek-chat", false},
		{"ollama", false},
	}
	for _, tt := range tests {
		if got := (openai.Provider{}).Matches(tt.model); got != tt.want {
			t.Errorf("Matches(%q) = %v, want %v", tt.model, got, tt.want)
		}
	}
	if (openai.Provider{}).ID() != "openai" {
		t.Fatal("ID")
	}
}

func TestNewMissingKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	_, err := openai.New("gpt-4.1")
	if !errors.Is(err, core.ErrMissingAPIKey) {
		t.Fatalf("err = %v", err)
	}
	t.Setenv("OPENAI_API_KEY", "env-key")
	c, err := openai.New("gpt-4.1")
	if err != nil || c.Provider() != "openai" || c.Model() != "gpt-4.1" {
		t.Fatalf("client = %v %v", c, err)
	}
	cl, err := openai.Provider{}.Open("o3", core.NewConfig())
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cl.(core.Chatter); !ok {
		t.Fatal("Chatter")
	}
	if _, ok := cl.(core.Streamer); !ok {
		t.Fatal("Streamer")
	}
	if _, ok := cl.(core.Embedder); !ok {
		t.Fatal("Embedder")
	}
}

func TestChatRequestMapping(t *testing.T) {
	srv, cap := newServer(t, serve(`{"id":"resp_1","model":"gpt-4.1","status":"completed","output":[]}`))
	c := newClient(t, srv.URL, openai.WithOrganization("org-1"), openai.WithProject("proj-1"))
	req := &core.Request{
		Messages: []core.Message{
			core.System("be terse"),
			core.System("be kind"),
			core.User(core.Text("look"), core.ImagePart{Data: []byte{1}, MIME: "image/png", Detail: "low"}, core.ImageURL("https://i/x.png"),
				core.File([]byte{3}, "application/pdf", "d.pdf"), core.FileURL("https://f/d.pdf", "application/pdf")),
			core.Assistant(core.ReasoningPart{Text: "summary", Signature: "rs_1", Encrypted: "enc"}, core.ReasoningPart{Text: "plain only"},
				core.Text("calling"), core.ToolCall{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)}, core.ToolCall{ID: "call_2", Name: "g"}),
			core.ToolResults(core.ToolResultText("call_1", "f", "42"), core.ToolResultText("call_2", "g", "")),
			core.UserText("go"),
		},
		Tools:           []core.Tool{{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`), Strict: true}, {Name: "g"}},
		ToolChoice:      core.ToolChoice{Mode: core.ToolChoiceNamed, Name: "f"},
		MaxTokens:       10,
		Temperature:     new(0.5),
		TopP:            new(0.9),
		Seed:            new(int64(7)),
		Stop:            []string{"x"},
		Format:          &core.ResponseFormat{Type: core.FormatJSONSchema, Name: "s", Schema: json.RawMessage(`{"type":"object"}`), Strict: true},
		Reasoning:       &core.ReasoningConfig{Effort: "high"},
		Extra:           map[string]any{"store": false},
		ProviderOptions: map[string]map[string]any{"openai": {"previous_response_id": "resp_0"}, "other": {"z": 1}},
	}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if cap.path != "/responses" || cap.hdr.Get("Authorization") != "Bearer sk" || cap.hdr.Get("OpenAI-Organization") != "org-1" || cap.hdr.Get("OpenAI-Project") != "proj-1" {
		t.Fatalf("path=%s hdr=%v", cap.path, cap.hdr)
	}
	b := cap.body
	if b["model"] != "gpt-4.1" || b["instructions"] != "be terse\n\nbe kind" || b["max_output_tokens"] != float64(10) || b["temperature"] != 0.5 || b["top_p"] != 0.9 {
		t.Fatalf("body = %v", b)
	}
	for _, k := range []string{"seed", "stop", "stream", "max_tokens", "messages", "z"} {
		if _, ok := b[k]; ok {
			t.Fatalf("%s must not be sent", k)
		}
	}
	if b["store"] != false || b["previous_response_id"] != "resp_0" {
		t.Fatalf("extra not merged: %v", b)
	}
	if obj(t, b["reasoning"])["effort"] != "high" || !reflect.DeepEqual(b["include"], []any{"reasoning.encrypted_content"}) {
		t.Fatalf("reasoning = %v include = %v", b["reasoning"], b["include"])
	}
	format := obj(t, obj(t, b["text"])["format"])
	if format["type"] != "json_schema" || format["name"] != "s" || format["strict"] != true || obj(t, format["schema"])["type"] != "object" {
		t.Fatalf("text.format = %v", format)
	}
	tools := arr(t, b["tools"])
	if len(tools) != 2 || obj(t, tools[0])["type"] != "function" || obj(t, tools[0])["name"] != "f" || obj(t, tools[0])["strict"] != true || obj(t, tools[0])["description"] != "d" {
		t.Fatalf("tools = %v", tools)
	}
	if _, nested := obj(t, tools[0])["function"]; nested {
		t.Fatal("tools must be flat")
	}
	if _, ok := obj(t, tools[1])["strict"]; ok {
		t.Fatal("strict must be omitted when false")
	}
	if tc := obj(t, b["tool_choice"]); tc["type"] != "function" || tc["name"] != "f" {
		t.Fatalf("tool_choice = %v", tc)
	}

	input := arr(t, b["input"])
	if len(input) != 8 {
		t.Fatalf("input = %d items: %v", len(input), input)
	}
	user := obj(t, input[0])
	if user["role"] != "user" || user["type"] != "message" {
		t.Fatalf("user = %v", user)
	}
	parts := arr(t, user["content"])
	types := make([]string, 0, len(parts))
	for _, p := range parts {
		types = append(types, obj(t, p)["type"].(string))
	}
	if !reflect.DeepEqual(types, []string{"input_text", "input_image", "input_image", "input_file", "input_file"}) {
		t.Fatalf("part types = %v", types)
	}
	if obj(t, parts[0])["text"] != "look" {
		t.Fatalf("text = %v", parts[0])
	}
	if img := obj(t, parts[1]); !strings.HasPrefix(img["image_url"].(string), "data:image/png;base64,") || img["detail"] != "low" {
		t.Fatalf("image = %v", img)
	}
	if obj(t, parts[2])["image_url"] != "https://i/x.png" {
		t.Fatalf("image url = %v", parts[2])
	}
	if f := obj(t, parts[3]); f["filename"] != "d.pdf" || !strings.HasPrefix(f["file_data"].(string), "data:application/pdf;base64,") {
		t.Fatalf("file = %v", f)
	}
	if f := obj(t, parts[4]); f["file_url"] != "https://f/d.pdf" || f["file_data"] != nil {
		t.Fatalf("file url = %v", f)
	}

	reasoning := obj(t, input[1])
	if reasoning["type"] != "reasoning" || reasoning["id"] != "rs_1" || reasoning["encrypted_content"] != "enc" || !reflect.DeepEqual(reasoning["summary"], []any{}) {
		t.Fatalf("reasoning item = %v", reasoning)
	}
	asst := obj(t, input[2])
	if asst["role"] != "assistant" || obj(t, arr(t, asst["content"])[0])["type"] != "output_text" || obj(t, arr(t, asst["content"])[0])["text"] != "calling" {
		t.Fatalf("assistant = %v", asst)
	}
	call := obj(t, input[3])
	if call["type"] != "function_call" || call["call_id"] != "call_1" || call["name"] != "f" || call["arguments"] != `{"a":1}` {
		t.Fatalf("function_call = %v", call)
	}
	if obj(t, input[4])["arguments"] != "{}" {
		t.Fatalf("empty arguments must be {}: %v", input[4])
	}
	result := obj(t, input[5])
	if result["type"] != "function_call_output" || result["call_id"] != "call_1" || result["output"] != "42" {
		t.Fatalf("function_call_output = %v", result)
	}
	if obj(t, input[6])["output"] != "" {
		t.Fatalf("empty output must be sent: %v", input[6])
	}
	if obj(t, input[7])["role"] != "user" || obj(t, arr(t, obj(t, input[7])["content"])[0])["text"] != "go" {
		t.Fatalf("last user = %v", input[7])
	}
}

func TestChatRequestDefaults(t *testing.T) {
	srv, cap := newServer(t, serve(`{"id":"r","status":"completed","output":[]}`))
	c := newClient(t, srv.URL)
	req := &core.Request{Messages: []core.Message{core.UserText("hi")}, Format: &core.ResponseFormat{Type: core.FormatJSON}, ToolChoice: core.ToolChoice{Mode: core.ToolChoiceRequired}, Reasoning: &core.ReasoningConfig{}}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	b := cap.body
	for _, k := range []string{"instructions", "tools", "reasoning", "temperature", "max_output_tokens"} {
		if _, ok := b[k]; ok {
			t.Fatalf("%s must be omitted", k)
		}
	}
	if obj(t, obj(t, b["text"])["format"])["type"] != "json_object" || b["tool_choice"] != "required" || !reflect.DeepEqual(b["include"], []any{"reasoning.encrypted_content"}) {
		t.Fatalf("body = %v", b)
	}
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Format: &core.ResponseFormat{Type: "yaml"}}); err == nil {
		t.Fatal("unknown format must error")
	}
	if _, err := c.Chat(context.Background(), nil); err == nil {
		t.Fatal("nil request must error")
	}
}

const chatResponse = `{"id":"resp_1","object":"response","model":"o3","status":"completed","error":null,"incomplete_details":null,
"output":[
 {"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"think"},{"type":"summary_text","text":"more"}],"encrypted_content":"enc"},
 {"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]},{"type":"refusal","refusal":"nope"}]},
 {"id":"fc_1","type":"function_call","call_id":"call_1","name":"f","arguments":"{\"a\":1}","status":"completed"},
 {"id":"fc_2","type":"function_call","call_id":"call_2","name":"g","arguments":"not json"},
 {"id":"fc_3","type":"function_call","call_id":"call_3","name":"h","arguments":""}
],
"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"cached_tokens":4},"output_tokens_details":{"reasoning_tokens":3}}}`

func TestChatResponseDecode(t *testing.T) {
	srv, _ := newServer(t, serve(chatResponse))
	c := newClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	if string(resp.Raw) == "" {
		t.Fatal("Raw must be set")
	}
	resp.Raw = nil
	want := &core.Response{
		ID: "resp_1", Model: "o3", FinishReason: core.FinishToolCalls,
		Message: core.Assistant(
			core.ReasoningPart{Text: "think\n\nmore", Signature: "rs_1", Encrypted: "enc"},
			core.Text("hello"), core.Text("nope"),
			core.ToolCall{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)},
			core.ToolCall{ID: "call_2", Name: "g", Arguments: json.RawMessage(`"not json"`)},
			core.ToolCall{ID: "call_3", Name: "h", Arguments: json.RawMessage(`{}`)},
		),
		Usage: core.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15, CachedInputTokens: 4, ReasoningTokens: 3},
	}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("resp:\n got %+v\nwant %+v", resp, want)
	}
}

func TestChatFinishReasons(t *testing.T) {
	tests := []struct {
		name string
		body string
		want core.FinishReason
	}{
		{"stop", `{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"a"}]}]}`, core.FinishStop},
		{"length", `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}`, core.FinishLength},
		{"filter", `{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}`, core.FinishContentFilter},
		{"other", `{"status":"incomplete","incomplete_details":{"reason":"max_messages"},"output":[]}`, core.FinishOther},
		{"usage without total", `{"status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":3}}`, core.FinishStop},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := newServer(t, serve(tt.body))
			resp, err := newClient(t, srv.URL).Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
			if err != nil {
				t.Fatal(err)
			}
			if resp.FinishReason != tt.want {
				t.Fatalf("finish = %s, want %s", resp.FinishReason, tt.want)
			}
			if tt.name == "usage without total" && resp.Usage.TotalTokens != 5 {
				t.Fatalf("usage = %+v", resp.Usage)
			}
		})
	}
}

func TestChatFailedAndHTTPError(t *testing.T) {
	srv, _ := newServer(t, serve(`{"id":"resp_1","status":"failed","error":{"code":"server_error","message":"boom"},"output":[]}`))
	_, err := newClient(t, srv.URL).Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	var apiErr *core.APIError
	if !errors.As(err, &apiErr) || apiErr.Code != "server_error" || apiErr.Message != "boom" || apiErr.Provider != "openai" {
		t.Fatalf("err = %#v", err)
	}
	srv2, _ := newServer(t, func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"slow","type":"rate_limit_error","code":"rate_limit_exceeded"}}`)
	})
	_, err = newClient(t, srv2.URL).Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	if !errors.As(err, &apiErr) || apiErr.Status != 429 || !errors.Is(err, core.ErrRateLimited) {
		t.Fatalf("err = %#v", err)
	}
}

const streamFixture = `event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","status":"in_progress","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","sequence_number":2,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"hmm"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"hmm"}],"encrypted_content":"enc"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":3,"output_index":1,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":4,"item_id":"msg_1","output_index":1,"content_index":0,"part":{"type":"output_text","text":""}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":5,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hel"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":6,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"lo"}

event: response.refusal.delta
data: {"type":"response.refusal.delta","sequence_number":7,"item_id":"msg_1","output_index":1,"content_index":1,"delta":"no"}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":8,"output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"f","arguments":"","status":"in_progress"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":9,"item_id":"fc_1","output_index":2,"delta":"{\"a\":"}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":10,"item_id":"fc_1","output_index":2,"delta":"1}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":11,"item_id":"fc_1","output_index":2,"arguments":"{\"a\":1}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":11,"output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"f","arguments":"{\"a\":1}"}}

event: response.completed
data: {"type":"response.completed","sequence_number":12,"response":{"id":"resp_1","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"f","arguments":"{\"a\":1}"}],"usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12,"input_tokens_details":{"cached_tokens":1},"output_tokens_details":{"reasoning_tokens":2}}}}

`

func TestStream(t *testing.T) {
	srv, cap := newServer(t, serveSSE(streamFixture))
	c := newClient(t, srv.URL)
	var got []core.Chunk
	for ch, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		if len(ch.Raw) == 0 {
			t.Fatalf("Raw must be set on %v", ch)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	if cap.body["stream"] != true || cap.path != "/responses" {
		t.Fatalf("body = %v path = %s", cap.body, cap.path)
	}
	want := []core.Chunk{
		{Kind: core.ChunkReasoning, Text: "hmm", Reasoning: &core.ReasoningDelta{Index: 0, Text: "hmm"}},
		{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 0, Signature: "rs_1", Encrypted: "enc"}},
		{Kind: core.ChunkText, Text: "Hel"},
		{Kind: core.ChunkText, Text: "lo"},
		{Kind: core.ChunkText, Text: "no"},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 2, ID: "call_1", Name: "f"}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 2, Arguments: `{"a":`}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 2, Arguments: `1}`}},
		{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls, Usage: &core.Usage{InputTokens: 5, OutputTokens: 7, TotalTokens: 12, CachedInputTokens: 1, ReasoningTokens: 2}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks:\n got %+v\nwant %+v", got, want)
	}
}

func TestStreamIncompleteAndTruncated(t *testing.T) {
	incomplete := "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"a\"}\n\n" +
		"event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_tokens\"},\"output\":[]}}\n\n" +
		"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ignored after finish\"}\n\n"
	srv, _ := newServer(t, serveSSE(incomplete))
	var got []core.Chunk
	for ch, err := range newClient(t, srv.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	want := []core.Chunk{{Kind: core.ChunkText, Text: "a"}, {Kind: core.ChunkFinish, FinishReason: core.FinishLength}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks:\n got %+v\nwant %+v", got, want)
	}

	srv2, _ := newServer(t, serveSSE("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"a\"}\n\n"))
	got = nil
	for ch, err := range newClient(t, srv2.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, ch)
	}
	if len(got) != 2 || got[1].Kind != core.ChunkFinish || got[1].FinishReason != core.FinishOther {
		t.Fatalf("truncated stream chunks = %+v", got)
	}
}

func TestStreamErrorsAndBreak(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"bad key","code":"invalid_api_key"}}`)
	})
	c := newClient(t, srv.URL)
	var n int
	var gotErr error
	for _, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		n++
		gotErr = err
	}
	var apiErr *core.APIError
	if n != 1 || !errors.As(gotErr, &apiErr) || apiErr.Status != 401 || apiErr.Code != "invalid_api_key" {
		t.Fatalf("n=%d err=%v", n, gotErr)
	}

	failed := "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"a\"}\n\n" +
		"event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_1\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"The model failed\"},\"output\":[]}}\n\n"
	srv2, _ := newServer(t, serveSSE(failed))
	n, gotErr = 0, nil
	for ch, err := range newClient(t, srv2.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		n++
		if err != nil {
			gotErr = err
			continue
		}
		if ch.Kind != core.ChunkText {
			t.Fatalf("chunk = %+v", ch)
		}
	}
	if n != 2 || !errors.As(gotErr, &apiErr) || apiErr.Code != "server_error" || apiErr.Message != "The model failed" {
		t.Fatalf("n=%d err=%#v", n, gotErr)
	}

	srv3, _ := newServer(t, serveSSE("event: error\ndata: {\"type\":\"error\",\"code\":\"ERR_X\",\"message\":\"Something went wrong\",\"param\":null,\"sequence_number\":1}\n\n"))
	gotErr = nil
	for _, err := range newClient(t, srv3.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		gotErr = err
	}
	if !errors.As(gotErr, &apiErr) || apiErr.Code != "ERR_X" || apiErr.Message != "Something went wrong" {
		t.Fatalf("err=%#v", gotErr)
	}

	srv4, _ := newServer(t, serveSSE(streamFixture))
	n = 0
	for _, err := range newClient(t, srv4.URL).Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
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

func TestUnsupportedPartsFailBeforeIO(t *testing.T) {
	srv, cap := newServer(t, serve(`{}`))
	c := newClient(t, srv.URL)
	tests := []core.Message{
		core.User(core.Audio([]byte{1}, "audio/wav")),
		core.ToolResults(core.ToolResult{CallID: "c", Content: []core.Part{core.Image([]byte{1}, "image/png")}}),
		core.User(core.ToolCall{ID: "x"}),
	}
	for _, m := range tests {
		req := &core.Request{Messages: []core.Message{m}}
		if _, err := c.Chat(context.Background(), req); !errors.Is(err, core.ErrUnsupported) {
			t.Fatalf("%T: Chat err = %v", m.Parts[0], err)
		}
		var streamErr error
		for _, err := range c.Stream(context.Background(), req) {
			streamErr = err
		}
		if !errors.Is(streamErr, core.ErrUnsupported) {
			t.Fatalf("%T: Stream err = %v", m.Parts[0], streamErr)
		}
	}
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{{Role: "weird"}}}); err == nil {
		t.Fatal("unknown role must error")
	}
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{{Role: core.RoleTool, Parts: []core.Part{core.Text("x")}}}}); err == nil {
		t.Fatal("non-result part in tool message must error")
	}
	if cap.hits != 0 {
		t.Fatal("server must not be contacted")
	}
}

func TestEmbed(t *testing.T) {
	srv, cap := newServer(t, serve(`{"object":"list","data":[{"index":1,"embedding":[0.3]},{"index":0,"embedding":[0.1,0.2]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":4,"total_tokens":4}}`))
	c, err := openai.New("text-embedding-3-small", core.WithAPIKey("sk"), core.WithBaseURL(srv.URL), openai.WithProject("p"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a", "b"}, Dimensions: 2})
	if err != nil {
		t.Fatal(err)
	}
	if cap.path != "/embeddings" || cap.hdr.Get("Authorization") != "Bearer sk" || cap.hdr.Get("OpenAI-Project") != "p" {
		t.Fatalf("path=%s hdr=%v", cap.path, cap.hdr)
	}
	if cap.body["model"] != "text-embedding-3-small" || cap.body["dimensions"] != float64(2) || !reflect.DeepEqual(cap.body["input"], []any{"a", "b"}) {
		t.Fatalf("body = %v", cap.body)
	}
	if !reflect.DeepEqual(resp.Embeddings, [][]float32{{0.1, 0.2}, {0.3}}) || resp.Model != "text-embedding-3-small" || resp.Usage.InputTokens != 4 {
		t.Fatalf("resp = %+v", resp)
	}
}
