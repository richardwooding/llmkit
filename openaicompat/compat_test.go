package openaicompat_test

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
	"github.com/richardwooding/llmkit/openaicompat"
)

type capture struct {
	body map[string]any
	path string
	auth string
	hdr  http.Header
}

func newServer(t *testing.T, respond func(w http.ResponseWriter, body map[string]any)) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.path = r.URL.Path
		cap.auth = r.Header.Get("Authorization")
		cap.hdr = r.Header.Clone()
		cap.body = map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&cap.body)
		respond(w, cap.body)
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func fullQuirks() openaicompat.Quirks {
	return openaicompat.Quirks{
		Images: true, Audio: true, Files: true, ReasoningContentField: "reasoning_content",
		StreamUsage: true, JSONSchema: true, Strict: true, Seed: true, EmbedPath: "/embeddings",
	}
}

func newClient(t *testing.T, url string, q openaicompat.Quirks) *openaicompat.Client {
	t.Helper()
	c, err := openaicompat.NewClient("m1", openaicompat.Config{ID: "test", BaseURL: url, APIKeyEnv: "TEST_KEY", Quirks: q, Headers: map[string]string{"X-Default": "d"}},
		core.NewConfig(core.WithAPIKey("sk")))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestNewClientMissingKey(t *testing.T) {
	t.Setenv("TEST_KEY", "")
	_, err := openaicompat.NewClient("m", openaicompat.Config{ID: "test", APIKeyEnv: "TEST_KEY"}, nil)
	if !errors.Is(err, core.ErrMissingAPIKey) {
		t.Fatalf("err = %v", err)
	}
	t.Setenv("TEST_KEY", "from-env")
	c, err := openaicompat.NewClient("m", openaicompat.Config{ID: "test", APIKeyEnv: "TEST_KEY"}, nil)
	if err != nil || c.Provider() != "test" || c.Model() != "m" {
		t.Fatalf("client = %v %v", c, err)
	}
	if _, err := openaicompat.NewClient("m", openaicompat.Config{ID: "local", KeyOptional: true}, nil); err != nil {
		t.Fatal(err)
	}
}

func TestChatRequestMapping(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"id":"r1","model":"m1","choices":[{"index":0,"message":{"role":"assistant","content":"hello","reasoning_content":"think"},"finish_reason":"stop"}],
		  "usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":1}}}`)
	})
	c := newClient(t, srv.URL, fullQuirks())
	req := &core.Request{
		Messages: []core.Message{
			core.System("sys"),
			core.User(core.Text("look"), core.Image([]byte{1}, "image/png"), core.ImageURL("https://i/x.png"),
				core.Audio([]byte{2}, "audio/wav"), core.File([]byte{3}, "application/pdf", "d.pdf")),
			core.Assistant(core.ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)}),
			core.ToolResults(core.ToolResultText("c1", "f", "42")),
			core.UserText("go"),
		},
		Tools:           []core.Tool{{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`), Strict: true}},
		ToolChoice:      core.ToolChoice{Mode: core.ToolChoiceNamed, Name: "f"},
		MaxTokens:       10,
		Temperature:     new(0.5),
		Seed:            new(int64(7)),
		Stop:            []string{"x"},
		Format:          &core.ResponseFormat{Type: core.FormatJSONSchema, Name: "s", Schema: json.RawMessage(`{"type":"object"}`), Strict: true},
		Reasoning:       &core.ReasoningConfig{Effort: "high"},
		Extra:           map[string]any{"top_k": 3},
		ProviderOptions: map[string]map[string]any{"test": {"top_k": 4}, "other": {"z": 1}},
	}
	resp, err := c.Chat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if cap.path != "/chat/completions" || cap.auth != "Bearer sk" || cap.hdr.Get("X-Default") != "d" {
		t.Fatalf("path=%s auth=%s hdr=%v", cap.path, cap.auth, cap.hdr)
	}
	b := cap.body
	if b["model"] != "m1" || b["max_tokens"] != float64(10) || b["temperature"] != 0.5 || b["seed"] != float64(7) || b["top_k"] != float64(4) || b["reasoning_effort"] != "high" {
		t.Fatalf("body = %v", b)
	}
	if _, ok := b["z"]; ok {
		t.Fatal("other provider's options leaked")
	}
	msgs := b["messages"].([]any)
	if len(msgs) != 5 {
		t.Fatalf("messages = %d", len(msgs))
	}
	user := msgs[1].(map[string]any)["content"].([]any)
	types := make([]string, 0, len(user))
	for _, p := range user {
		types = append(types, p.(map[string]any)["type"].(string))
	}
	if !reflect.DeepEqual(types, []string{"text", "image_url", "image_url", "input_audio", "file"}) {
		t.Fatalf("part types = %v", types)
	}
	img := user[1].(map[string]any)["image_url"].(map[string]any)["url"].(string)
	if !strings.HasPrefix(img, "data:image/png;base64,") {
		t.Fatalf("image url = %s", img)
	}
	if user[3].(map[string]any)["input_audio"].(map[string]any)["format"] != "wav" {
		t.Fatalf("audio = %v", user[3])
	}
	asst := msgs[2].(map[string]any)
	if _, hasContent := asst["content"]; hasContent {
		t.Fatalf("assistant with only tool calls must omit content: %v", asst)
	}
	tc := asst["tool_calls"].([]any)[0].(map[string]any)
	if tc["id"] != "c1" || tc["function"].(map[string]any)["arguments"] != `{"a":1}` {
		t.Fatalf("tool_calls = %v", tc)
	}
	tool := msgs[3].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "c1" || tool["content"] != "42" {
		t.Fatalf("tool msg = %v", tool)
	}
	if msgs[4].(map[string]any)["content"] != "go" {
		t.Fatal("single text part should be a plain string")
	}
	tools := b["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if tools["strict"] != true || tools["name"] != "f" {
		t.Fatalf("tools = %v", tools)
	}
	if b["tool_choice"].(map[string]any)["function"].(map[string]any)["name"] != "f" {
		t.Fatalf("tool_choice = %v", b["tool_choice"])
	}
	if b["response_format"].(map[string]any)["type"] != "json_schema" {
		t.Fatalf("response_format = %v", b["response_format"])
	}
	if _, ok := b["stream"]; ok {
		t.Fatal("stream must be omitted for Chat")
	}

	want := &core.Response{
		ID: "r1", Model: "m1", FinishReason: core.FinishStop,
		Message: core.Assistant(core.ReasoningPart{Text: "think"}, core.Text("hello")),
		Usage:   core.Usage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5, CachedInputTokens: 1, ReasoningTokens: 1},
	}
	resp.Raw = nil
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("resp = %+v", resp)
	}
}

func TestChatToolCallsResponse(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}},{"id":"c2","type":"function","function":{"name":"g","arguments":"not json"}}]},"finish_reason":"tool_calls"}],"x_groq":{"usage":{"prompt_tokens":1,"completion_tokens":1}}}`)
	})
	c := newClient(t, srv.URL, openaicompat.Quirks{})
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 2 || calls[0].Name != "f" || string(calls[0].Arguments) != `{"a":1}` || string(calls[1].Arguments) != `"not json"` {
		t.Fatalf("calls = %+v", calls)
	}
	if resp.FinishReason != core.FinishToolCalls || resp.Usage.TotalTokens != 2 || resp.Text() != "" {
		t.Fatalf("resp = %+v", resp)
	}
}

func TestUnsupportedPartsFailBeforeIO(t *testing.T) {
	hit := false
	srv, _ := newServer(t, func(w http.ResponseWriter, _ map[string]any) { hit = true })
	c := newClient(t, srv.URL, openaicompat.Quirks{})
	tests := []core.Message{
		core.User(core.Image([]byte{1}, "image/png")),
		core.User(core.Audio([]byte{1}, "audio/wav")),
		core.User(core.File([]byte{1}, "application/pdf", "x")),
		core.ToolResults(core.ToolResult{CallID: "c", Content: []core.Part{core.Image([]byte{1}, "image/png")}}),
	}
	for _, m := range tests {
		_, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{m}})
		if !errors.Is(err, core.ErrUnsupported) {
			t.Fatalf("%T: err = %v", m.Parts[0], err)
		}
	}
	_, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Format: &core.ResponseFormat{Type: core.FormatJSONSchema}})
	if !errors.Is(err, core.ErrUnsupported) {
		t.Fatalf("json_schema err = %v", err)
	}
	if hit {
		t.Fatal("server must not be contacted")
	}
}

func TestQuirkValidateAndMaxCompletionTokens(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter, _ map[string]any) { _, _ = io.WriteString(w, `{"choices":[]}`) })
	q := openaicompat.Quirks{MaxCompletionTokens: true, DeveloperRole: true, Validate: func(model string, req *core.Request) error {
		if len(req.Tools) > 0 {
			return core.Unsupported("test", "tools on "+model)
		}
		return nil
	}}
	c := newClient(t, srv.URL, q)
	_, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Tools: []core.Tool{{Name: "f"}}})
	if !errors.Is(err, core.ErrUnsupported) {
		t.Fatalf("err = %v", err)
	}
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.System("s")}, MaxTokens: 5}); err != nil {
		t.Fatal(err)
	}
	if cap.body["max_completion_tokens"] != float64(5) || cap.body["messages"].([]any)[0].(map[string]any)["role"] != "developer" {
		t.Fatalf("body = %v", cap.body)
	}
}

const streamFixture = `data: {"id":"s1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"hmm"},"finish_reason":null}]}

data: {"id":"s1","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}

data: {"id":"s1","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}

data: {"id":"s1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}

data: {"id":"s1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":"}}]},"finish_reason":null}]}

data: {"id":"s1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}

data: {"id":"s1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"s1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}

data: [DONE]

`

func TestStream(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, streamFixture)
	})
	c := newClient(t, srv.URL, fullQuirks())
	var got []core.Chunk
	for ch, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	if cap.body["stream"] != true || cap.body["stream_options"].(map[string]any)["include_usage"] != true {
		t.Fatalf("body = %v", cap.body)
	}
	want := []core.Chunk{
		{Kind: core.ChunkReasoning, Text: "hmm"},
		{Kind: core.ChunkText, Text: "Hel"},
		{Kind: core.ChunkText, Text: "lo"},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, ID: "c1", Name: "f"}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, Arguments: `{"a":`}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, Arguments: `1}`}},
		{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls, Usage: &core.Usage{InputTokens: 5, OutputTokens: 7, TotalTokens: 12}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks:\n got %+v\nwant %+v", got, want)
	}
}

func TestStreamErrorsAndBreak(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"bad key","code":"invalid_api_key"}}`)
	})
	c := newClient(t, srv.URL, openaicompat.Quirks{})
	var n int
	var gotErr error
	for _, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		n++
		gotErr = err
	}
	var apiErr *core.APIError
	if n != 1 || !errors.As(gotErr, &apiErr) || apiErr.Status != 401 {
		t.Fatalf("n=%d err=%v", n, gotErr)
	}

	srv2, _ := newServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, streamFixture)
	})
	c2 := newClient(t, srv2.URL, fullQuirks())
	n = 0
	for _, err := range c2.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
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

func TestEmbed(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"data":[{"index":1,"embedding":[0.3]},{"index":0,"embedding":[0.1,0.2]}],"model":"e1","usage":{"prompt_tokens":4,"total_tokens":4}}`)
	})
	c := newClient(t, srv.URL, fullQuirks())
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a", "b"}, Dimensions: 2})
	if err != nil {
		t.Fatal(err)
	}
	if cap.path != "/embeddings" || cap.body["dimensions"] != float64(2) || cap.body["encoding_format"] != "float" {
		t.Fatalf("req = %s %v", cap.path, cap.body)
	}
	if !reflect.DeepEqual(resp.Embeddings, [][]float32{{0.1, 0.2}, {0.3}}) || resp.Model != "e1" || resp.Usage.InputTokens != 4 {
		t.Fatalf("resp = %+v", resp)
	}
	noEmbed := newClient(t, srv.URL, openaicompat.Quirks{})
	if _, err := noEmbed.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"a"}}); !errors.Is(err, core.ErrUnsupported) {
		t.Fatalf("err = %v", err)
	}
}

func TestProvider(t *testing.T) {
	p := openaicompat.NewProvider(openaicompat.Config{ID: "vllm", BaseURL: "http://x", KeyOptional: true, Match: openaicompat.PrefixMatcher("my-")})
	if p.ID() != "vllm" || !p.Matches("MY-model") || p.Matches("gpt-4") {
		t.Fatal("provider identity/match wrong")
	}
	c, err := p.Open("my-model", core.NewConfig())
	if err != nil || c.Model() != "my-model" {
		t.Fatalf("Open = %v %v", c, err)
	}
	if _, ok := c.(core.Chatter); !ok {
		t.Fatal("client should implement Chatter")
	}
}
