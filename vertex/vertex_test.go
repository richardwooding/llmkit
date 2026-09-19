package vertex_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"golang.org/x/oauth2"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/vertex"
)

const okResponse = `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],
  "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"modelVersion":"gemini-2.5-flash-001","responseId":"r1"}`

const okEmbed = `{"predictions":[{"embeddings":{"values":[0.1,0.2],"statistics":{"token_count":3,"truncated":false}}},
  {"embeddings":{"values":[0.3,0.4],"statistics":{"token_count":4,"truncated":false}}}]}`

type capture struct {
	path  string
	query string
	auth  string
	body  map[string]any
	hits  atomic.Int32
}

func newServer(t *testing.T, respond func(w http.ResponseWriter, r *http.Request)) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.hits.Add(1)
		cap.path = r.URL.Path
		cap.query = r.URL.RawQuery
		cap.auth = r.Header.Get("Authorization")
		cap.body = map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&cap.body)
		respond(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func jsonServer(t *testing.T, body string) (*httptest.Server, *capture) {
	t.Helper()
	return newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, body) })
}

func clearEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{"GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"} {
		t.Setenv(k, "")
	}
}

func newClient(t *testing.T, srvURL string, opts ...core.Option) *vertex.Client {
	t.Helper()
	clearEnv(t)
	all := append([]core.Option{vertex.WithProject("proj"), core.WithBaseURL(srvURL), vertex.WithAccessToken("t")}, opts...)
	c, err := vertex.New("gemini-2.5-flash", all...)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestHostAndPathTable(t *testing.T) {
	tests := []struct {
		name    string
		env     map[string]string
		opts    []core.Option
		wantURL string
	}{
		{
			name: "defaults", env: map[string]string{"GOOGLE_CLOUD_PROJECT": "envproj"},
			wantURL: "https://us-central1-aiplatform.googleapis.com/v1/projects/envproj/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent",
		},
		{
			name: "gcloud project env", env: map[string]string{"GCLOUD_PROJECT": "legacy"},
			wantURL: "https://us-central1-aiplatform.googleapis.com/v1/projects/legacy/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent",
		},
		{
			name: "location env", env: map[string]string{"GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_CLOUD_LOCATION": "asia-east1"},
			wantURL: "https://asia-east1-aiplatform.googleapis.com/v1/projects/p/locations/asia-east1/publishers/google/models/gemini-2.5-flash:generateContent",
		},
		{
			name: "options beat env", env: map[string]string{"GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_CLOUD_LOCATION": "asia-east1"},
			opts:    []core.Option{vertex.WithProject("opt"), vertex.WithLocation("europe-west4")},
			wantURL: "https://europe-west4-aiplatform.googleapis.com/v1/projects/opt/locations/europe-west4/publishers/google/models/gemini-2.5-flash:generateContent",
		},
		{
			name: "global", opts: []core.Option{vertex.WithProject("p"), vertex.WithLocation("global")},
			wantURL: "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-flash:generateContent",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearEnv(t)
			for k, v := range tt.env {
				t.Setenv(k, v)
			}
			var got string
			hc := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				got = r.URL.String()
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(okResponse)), Header: http.Header{}}, nil
			})}
			opts := append([]core.Option{core.WithHTTPClient(hc), vertex.WithAccessToken("t")}, tt.opts...)
			c, err := vertex.New("gemini-2.5-flash", opts...)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}); err != nil {
				t.Fatal(err)
			}
			if got != tt.wantURL {
				t.Fatalf("url = %s\nwant %s", got, tt.wantURL)
			}
		})
	}
}

func TestPathsAndAuth(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, ":predict"):
			_, _ = io.WriteString(w, okEmbed)
		case strings.HasSuffix(r.URL.Path, ":streamGenerateContent"):
			_, _ = io.WriteString(w, "data: "+strings.ReplaceAll(okResponse, "\n", "")+"\n\n")
		default:
			_, _ = io.WriteString(w, okResponse)
		}
	})
	c := newClient(t, srv.URL)
	ctx := context.Background()
	req := &core.Request{Messages: []core.Message{core.UserText("x")}}
	const prefix = "/v1/projects/proj/locations/us-central1/publishers/google/models/gemini-2.5-flash:"

	if _, err := c.Chat(ctx, req); err != nil {
		t.Fatal(err)
	}
	if cap.path != prefix+"generateContent" || cap.auth != "Bearer t" {
		t.Fatalf("chat path=%s auth=%s", cap.path, cap.auth)
	}
	for _, err := range c.Stream(ctx, req) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if cap.path != prefix+"streamGenerateContent" || cap.query != "alt=sse" || cap.auth != "Bearer t" {
		t.Fatalf("stream path=%s query=%s auth=%s", cap.path, cap.query, cap.auth)
	}
	if _, err := c.Embed(ctx, &core.EmbedRequest{Inputs: []string{"a", "b"}}); err != nil {
		t.Fatal(err)
	}
	if cap.path != prefix+"predict" || cap.auth != "Bearer t" {
		t.Fatalf("embed path=%s auth=%s", cap.path, cap.auth)
	}
}

func TestAuthSources(t *testing.T) {
	srv, cap := jsonServer(t, okResponse)
	req := &core.Request{Messages: []core.Message{core.UserText("x")}}
	tests := []struct {
		name string
		opt  core.Option
		want string
	}{
		{"token source", vertex.WithTokenSource(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "from-ts"})), "Bearer from-ts"},
		{"access token", vertex.WithAccessToken("static"), "Bearer static"},
		{"api key", core.WithAPIKey("key"), "Bearer key"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearEnv(t)
			c, err := vertex.New("gemini-2.5-flash", vertex.WithProject("p"), core.WithBaseURL(srv.URL), tt.opt)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Chat(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			if cap.auth != tt.want {
				t.Fatalf("auth = %q, want %q", cap.auth, tt.want)
			}
		})
	}
}

func TestMissingProject(t *testing.T) {
	clearEnv(t)
	_, err := vertex.New("gemini-2.5-flash", vertex.WithAccessToken("t"))
	if !errors.Is(err, core.ErrMissingAPIKey) || !strings.Contains(err.Error(), "GOOGLE_CLOUD_PROJECT") {
		t.Fatalf("err = %v", err)
	}
	c, err := vertex.Provider{}.Open("gemini-2.5-flash", core.NewConfig(vertex.WithProject("p")))
	if err != nil || c.Provider() != vertex.ID || c.Model() != "gemini-2.5-flash" {
		t.Fatalf("open = %v %v", c, err)
	}
}

func TestMatches(t *testing.T) {
	p := vertex.Provider{}
	if p.ID() != "vertex" {
		t.Fatal(p.ID())
	}
	for model, want := range map[string]bool{
		"gemini-2.5-pro": true, "Gemini-2.0-flash": true, "imagen-4.0-generate-001": true,
		"text-embedding-005": true, "text-embedding-004": true, "text-multilingual-embedding-002": true,
		"gemini-embedding-001": true, "gpt-4o": false, "claude-sonnet-4": false, "text-embedding-3-small": false,
	} {
		if p.Matches(model) != want {
			t.Errorf("Matches(%q) = %v", model, !want)
		}
	}
}

func TestChatRequestMapping(t *testing.T) {
	srv, cap := jsonServer(t, okResponse)
	c := newClient(t, srv.URL)
	req := &core.Request{
		Messages: []core.Message{
			core.System("sys1"),
			core.System("sys2"),
			core.User(core.Text("look"), core.Image([]byte("png"), "image/png"), core.ImageURL("gs://b/x.jpg"),
				core.Audio([]byte("wav"), "audio/wav"), core.File([]byte("pdf"), "application/pdf", "d.pdf"),
				core.FileURL("https://h/doc", "application/pdf")),
			core.Assistant(core.Text("calling"), core.ToolCall{ID: "call_1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)}),
			core.ToolResults(core.ToolResultText("call_1", "f", `{"ok":true}`), core.ToolResult{CallID: "c2", Name: "g", Content: []core.Part{core.Text("plain")}, IsError: true}),
			core.UserText("go"),
		},
		Tools:           []core.Tool{{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:      core.ToolChoice{Mode: core.ToolChoiceNamed, Name: "f"},
		MaxTokens:       10,
		Temperature:     new(0.5),
		TopP:            new(0.9),
		Seed:            new(int64(7)),
		Stop:            []string{"x"},
		Format:          &core.ResponseFormat{Type: core.FormatJSONSchema, Schema: json.RawMessage(`{"type":"object"}`)},
		Reasoning:       &core.ReasoningConfig{BudgetTokens: 1024},
		Extra:           map[string]any{"labels": map[string]any{"a": "b"}},
		ProviderOptions: map[string]map[string]any{"vertex": {"labels": map[string]any{"a": "c"}}, "other": {"z": 1}},
	}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	b := cap.body
	if _, leaked := b["z"]; leaked {
		t.Fatal("other provider's options leaked")
	}
	if !reflect.DeepEqual(b["labels"], map[string]any{"a": "c"}) {
		t.Fatalf("labels = %v", b["labels"])
	}
	sys := b["systemInstruction"].(map[string]any)["parts"].([]any)
	if len(sys) != 1 || sys[0].(map[string]any)["text"] != "sys1\nsys2" {
		t.Fatalf("systemInstruction = %v", sys)
	}
	contents := b["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %d: %v", len(contents), contents)
	}
	user := contents[0].(map[string]any)
	parts := user["parts"].([]any)
	if user["role"] != "user" || len(parts) != 6 {
		t.Fatalf("user content = %v", user)
	}
	if parts[0].(map[string]any)["text"] != "look" {
		t.Fatalf("part0 = %v", parts[0])
	}
	inline := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if inline["mimeType"] != "image/png" || inline["data"] != "cG5n" {
		t.Fatalf("inlineData = %v", inline)
	}
	file := parts[2].(map[string]any)["fileData"].(map[string]any)
	if file["fileUri"] != "gs://b/x.jpg" || file["mimeType"] != "image/jpeg" {
		t.Fatalf("fileData = %v", file)
	}
	if parts[3].(map[string]any)["inlineData"].(map[string]any)["mimeType"] != "audio/wav" {
		t.Fatalf("audio = %v", parts[3])
	}
	if parts[4].(map[string]any)["inlineData"].(map[string]any)["mimeType"] != "application/pdf" {
		t.Fatalf("file = %v", parts[4])
	}
	if parts[5].(map[string]any)["fileData"].(map[string]any)["mimeType"] != "application/pdf" {
		t.Fatalf("file url = %v", parts[5])
	}
	model := contents[1].(map[string]any)
	mparts := model["parts"].([]any)
	if model["role"] != "model" || mparts[0].(map[string]any)["text"] != "calling" {
		t.Fatalf("model content = %v", model)
	}
	fc := mparts[1].(map[string]any)["functionCall"].(map[string]any)
	if fc["name"] != "f" || !reflect.DeepEqual(fc["args"], map[string]any{"a": float64(1)}) {
		t.Fatalf("functionCall = %v", fc)
	}
	tool := contents[2].(map[string]any)
	tparts := tool["parts"].([]any)
	if tool["role"] != "user" || len(tparts) != 3 {
		t.Fatalf("tool content should merge with the following user turn: %v", tool)
	}
	fr := tparts[0].(map[string]any)["functionResponse"].(map[string]any)
	if fr["name"] != "f" || !reflect.DeepEqual(fr["response"], map[string]any{"result": map[string]any{"ok": true}}) {
		t.Fatalf("functionResponse = %v", fr)
	}
	fr2 := tparts[1].(map[string]any)["functionResponse"].(map[string]any)
	if fr2["name"] != "g" || !reflect.DeepEqual(fr2["response"], map[string]any{"error": "plain"}) {
		t.Fatalf("error functionResponse = %v", fr2)
	}
	if tparts[2].(map[string]any)["text"] != "go" {
		t.Fatalf("trailing user text = %v", tparts[2])
	}
	decl := b["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	if decl["name"] != "f" || decl["description"] != "d" || decl["parameters"].(map[string]any)["type"] != "object" {
		t.Fatalf("functionDeclarations = %v", decl)
	}
	fcc := b["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" || !reflect.DeepEqual(fcc["allowedFunctionNames"], []any{"f"}) {
		t.Fatalf("functionCallingConfig = %v", fcc)
	}
	gen := b["generationConfig"].(map[string]any)
	want := map[string]any{
		"temperature": 0.5, "topP": 0.9, "maxOutputTokens": float64(10), "stopSequences": []any{"x"}, "seed": float64(7),
		"responseMimeType": "application/json", "responseJsonSchema": map[string]any{"type": "object"},
		"thinkingConfig": map[string]any{"includeThoughts": true, "thinkingBudget": float64(1024)},
	}
	if !reflect.DeepEqual(gen, want) {
		t.Fatalf("generationConfig = %v\nwant %v", gen, want)
	}
}

func TestMinimalRequestOmitsOptionalObjects(t *testing.T) {
	srv, cap := jsonServer(t, okResponse)
	c := newClient(t, srv.URL)
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}}); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"systemInstruction", "tools", "toolConfig", "generationConfig"} {
		if _, ok := cap.body[k]; ok {
			t.Errorf("%s should be omitted: %v", k, cap.body[k])
		}
	}
}

func TestToolConfigModes(t *testing.T) {
	srv, cap := jsonServer(t, okResponse)
	c := newClient(t, srv.URL)
	tests := []struct {
		choice core.ToolChoice
		want   any
	}{
		{core.ToolChoice{}, nil},
		{core.ToolChoice{Mode: core.ToolChoiceAuto}, nil},
		{core.ToolChoice{Mode: core.ToolChoiceNone}, map[string]any{"functionCallingConfig": map[string]any{"mode": "NONE"}}},
		{core.ToolChoice{Mode: core.ToolChoiceRequired}, map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY"}}},
		{core.ToolChoice{Mode: core.ToolChoiceNamed, Name: "f"}, map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY", "allowedFunctionNames": []any{"f"}}}},
	}
	for _, tt := range tests {
		req := &core.Request{Messages: []core.Message{core.UserText("x")}, Tools: []core.Tool{{Name: "f"}}, ToolChoice: tt.choice}
		if _, err := c.Chat(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(cap.body["toolConfig"], tt.want) {
			t.Errorf("%+v: toolConfig = %v, want %v", tt.choice, cap.body["toolConfig"], tt.want)
		}
	}
}

func TestReasoningConfig(t *testing.T) {
	srv, cap := jsonServer(t, okResponse)
	c := newClient(t, srv.URL)
	for _, tt := range []struct {
		cfg  *core.ReasoningConfig
		want any
	}{
		{nil, nil},
		{&core.ReasoningConfig{Effort: "high"}, map[string]any{"includeThoughts": true, "thinkingBudget": float64(-1)}},
		{&core.ReasoningConfig{BudgetTokens: 512}, map[string]any{"includeThoughts": true, "thinkingBudget": float64(512)}},
	} {
		if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, Reasoning: tt.cfg}); err != nil {
			t.Fatal(err)
		}
		var got any
		if gen, ok := cap.body["generationConfig"].(map[string]any); ok {
			got = gen["thinkingConfig"]
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%+v: thinkingConfig = %v, want %v", tt.cfg, got, tt.want)
		}
	}
}

func TestReasoningEcho(t *testing.T) {
	srv, cap := jsonServer(t, okResponse)
	c := newClient(t, srv.URL)
	req := &core.Request{Messages: []core.Message{
		core.UserText("q"),
		core.Assistant(
			core.ReasoningPart{Text: "unsigned summary"},
			core.ReasoningPart{Text: "signed thought", Signature: "sig1"},
			core.ReasoningPart{Signature: "sig2"},
			core.ToolCall{Name: "f"},
		),
		core.ToolResults(core.ToolResultText("call_1", "f", "42")),
	}}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	parts := cap.body["contents"].([]any)[1].(map[string]any)["parts"].([]any)
	want := []any{
		map[string]any{"text": "signed thought", "thought": true, "thoughtSignature": "sig1"},
		map[string]any{"functionCall": map[string]any{"name": "f", "args": map[string]any{}}, "thoughtSignature": "sig2"},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("parts = %v\nwant %v", parts, want)
	}
	fr := cap.body["contents"].([]any)[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if !reflect.DeepEqual(fr["response"], map[string]any{"result": float64(42)}) {
		t.Fatalf("numeric result should be parsed JSON: %v", fr)
	}
}

func TestRequestErrorsBeforeHTTP(t *testing.T) {
	srv, cap := jsonServer(t, okResponse)
	c := newClient(t, srv.URL)
	tests := []struct {
		name string
		req  *core.Request
		want error
		msg  string
	}{
		{"tool result without name", &core.Request{Messages: []core.Message{core.ToolResults(core.ToolResultText("c1", "", "x"))}}, nil, "vertex: ToolResult.Name is required"},
		{"image in tool result", &core.Request{Messages: []core.Message{core.ToolResults(core.ToolResult{Name: "f", Content: []core.Part{core.Image(nil, "image/png")}})}}, core.ErrUnsupported, ""},
		{"unknown role", &core.Request{Messages: []core.Message{{Role: "bogus", Parts: []core.Part{core.Text("x")}}}}, nil, "unknown role"},
		{"bad format", &core.Request{Messages: []core.Message{core.UserText("x")}, Format: &core.ResponseFormat{Type: "yaml"}}, nil, "unknown response format"},
		{"schema without schema", &core.Request{Messages: []core.Message{core.UserText("x")}, Format: &core.ResponseFormat{Type: core.FormatJSONSchema}}, nil, "requires a schema"},
		{"invalid tool call args", &core.Request{Messages: []core.Message{core.Assistant(core.ToolCall{Name: "f", Arguments: json.RawMessage(`{`)})}}, nil, "not valid JSON"},
		{"nil request", nil, nil, "nil request"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := c.Chat(context.Background(), tt.req)
			if err == nil {
				t.Fatal("expected error")
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
			if tt.msg != "" && !strings.Contains(err.Error(), tt.msg) {
				t.Fatalf("err = %v, want containing %q", err, tt.msg)
			}
			var n int
			for _, serr := range c.Stream(context.Background(), tt.req) {
				n++
				if serr == nil {
					t.Fatal("stream should yield the setup error")
				}
			}
			if n != 1 {
				t.Fatalf("stream yielded %d items", n)
			}
		})
	}
	if cap.hits.Load() != 0 {
		t.Fatalf("server hit %d times", cap.hits.Load())
	}
}

func TestResponseDecode(t *testing.T) {
	srv, _ := jsonServer(t, `{"candidates":[{"content":{"role":"model","parts":[
	  {"text":"plan","thought":true,"thoughtSignature":"s0"},
	  {"text":"Sure."},
	  {"functionCall":{"name":"a","args":{"x":1}},"thoughtSignature":"s1"},
	  {"functionCall":{"name":"b"}}
	]},"finishReason":"STOP"}],
	"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":18,"cachedContentTokenCount":4,"thoughtsTokenCount":3},
	"modelVersion":"gemini-2.5-flash-001","responseId":"resp-1"}`)
	c := newClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	wantParts := []core.Part{
		core.ReasoningPart{Text: "plan", Signature: "s0"},
		core.Text("Sure."),
		core.ReasoningPart{Signature: "s1"},
		core.ToolCall{ID: "call_1", Name: "a", Arguments: json.RawMessage(`{"x":1}`)},
		core.ToolCall{ID: "call_2", Name: "b", Arguments: json.RawMessage(`{}`)},
	}
	if !reflect.DeepEqual(resp.Message.Parts, wantParts) {
		t.Fatalf("parts = %#v", resp.Message.Parts)
	}
	if resp.Message.Role != core.RoleAssistant || resp.FinishReason != core.FinishToolCalls || resp.ID != "resp-1" || resp.Model != "gemini-2.5-flash-001" {
		t.Fatalf("resp = %+v", resp)
	}
	wantUsage := core.Usage{InputTokens: 10, OutputTokens: 8, TotalTokens: 18, CachedInputTokens: 4, ReasoningTokens: 3}
	if resp.Usage != wantUsage {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if len(resp.Raw) == 0 || resp.Text() != "Sure." || len(resp.ToolCalls()) != 2 {
		t.Fatalf("raw/text/calls = %d %q %d", len(resp.Raw), resp.Text(), len(resp.ToolCalls()))
	}
}

func TestFinishReasonTable(t *testing.T) {
	tests := []struct {
		body string
		want core.FinishReason
	}{
		{`{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"STOP"}]}`, core.FinishStop},
		{`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f"}}]},"finishReason":"STOP"}]}`, core.FinishToolCalls},
		{`{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"MAX_TOKENS"}]}`, core.FinishLength},
		{`{"candidates":[{"finishReason":"SAFETY"}]}`, core.FinishContentFilter},
		{`{"candidates":[{"finishReason":"RECITATION"}]}`, core.FinishContentFilter},
		{`{"candidates":[{"finishReason":"BLOCKLIST"}]}`, core.FinishContentFilter},
		{`{"candidates":[{"finishReason":"PROHIBITED_CONTENT"}]}`, core.FinishContentFilter},
		{`{"candidates":[{"finishReason":"SPII"}]}`, core.FinishContentFilter},
		{`{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}`, core.FinishOther},
		{`{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":2,"totalTokenCount":2}}`, core.FinishContentFilter},
		{`{}`, core.FinishOther},
	}
	for _, tt := range tests {
		srv, _ := jsonServer(t, tt.body)
		c := newClient(t, srv.URL)
		resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != tt.want {
			t.Errorf("%s: finish = %s, want %s", tt.body, resp.FinishReason, tt.want)
		}
		if resp.Model != "gemini-2.5-flash" {
			t.Errorf("model should fall back to the requested name, got %q", resp.Model)
		}
	}
}

func TestChatAPIError(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}`)
	})
	c := newClient(t, srv.URL)
	_, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	var apiErr *core.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 429 || apiErr.Message != "Quota exceeded" || apiErr.Code != "RESOURCE_EXHAUSTED" || !errors.Is(err, core.ErrRateLimited) {
		t.Fatalf("err = %#v", err)
	}
}

const streamFixture = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"think","thought":true}]}}],"usageMetadata":{"promptTokenCount":3,"totalTokenCount":3}}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ing","thought":true,"thoughtSignature":"s0"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}},"thoughtSignature":"s1"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":9,"thoughtsTokenCount":2},"modelVersion":"gemini-2.5-flash"}

`

func TestStream(t *testing.T) {
	srv, cap := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, streamFixture)
	})
	c := newClient(t, srv.URL)
	seq := c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}})
	if cap.hits.Load() != 0 {
		t.Fatal("request must not be sent before iteration")
	}
	var got []core.Chunk
	for ch, err := range seq {
		if err != nil {
			t.Fatal(err)
		}
		if len(ch.Raw) == 0 && ch.Kind != core.ChunkFinish {
			t.Fatalf("chunk without Raw: %+v", ch)
		}
		ch.Raw = nil
		got = append(got, ch)
	}
	want := []core.Chunk{
		{Kind: core.ChunkReasoning, Text: "think", Reasoning: &core.ReasoningDelta{Index: 0, Text: "think"}},
		{Kind: core.ChunkReasoning, Text: "ing", Reasoning: &core.ReasoningDelta{Index: 0, Text: "ing", Signature: "s0"}},
		{Kind: core.ChunkText, Text: "Hel"},
		{Kind: core.ChunkText, Text: "lo"},
		{Kind: core.ChunkReasoning, Reasoning: &core.ReasoningDelta{Index: 1, Signature: "s1"}},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, ID: "call_1", Name: "f", Arguments: `{"a":1}`}},
		{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls, Usage: &core.Usage{InputTokens: 3, OutputTokens: 6, TotalTokens: 9, ReasoningTokens: 2}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks = %+v\nwant %+v", got, want)
	}
	if cap.hits.Load() != 1 {
		t.Fatalf("hits = %d", cap.hits.Load())
	}
}

func TestStreamHTTPError(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `{"error":{"code":403,"message":"Permission denied","status":"PERMISSION_DENIED"}}`)
	})
	c := newClient(t, srv.URL)
	var n int
	for _, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		n++
		var apiErr *core.APIError
		if !errors.As(err, &apiErr) || apiErr.Status != 403 || apiErr.Message != "Permission denied" {
			t.Fatalf("err = %#v", err)
		}
	}
	if n != 1 {
		t.Fatalf("yielded %d items", n)
	}
}

func TestStreamMidStreamError(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]}}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"error\":{\"code\":500,\"message\":\"boom\",\"status\":\"INTERNAL\"}}\n\n")
	})
	c := newClient(t, srv.URL)
	var kinds []core.ChunkKind
	var last error
	for ch, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			last = err
			continue
		}
		kinds = append(kinds, ch.Kind)
	}
	var apiErr *core.APIError
	if !errors.As(last, &apiErr) || apiErr.Status != 500 || apiErr.Message != "boom" || apiErr.Code != "INTERNAL" {
		t.Fatalf("err = %#v", last)
	}
	if !reflect.DeepEqual(kinds, []core.ChunkKind{core.ChunkText}) {
		t.Fatalf("kinds = %v", kinds)
	}
}

func TestStreamBreakClosesBody(t *testing.T) {
	done := make(chan struct{})
	srv, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		defer close(done)
		flusher := w.(http.Flusher)
		for range 3 {
			_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]}}]}\n\n")
			flusher.Flush()
		}
		<-r.Context().Done()
	})
	c := newClient(t, srv.URL)
	var n int
	for _, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}) {
		if err != nil {
			t.Fatal(err)
		}
		n++
		if n == 2 {
			break
		}
	}
	<-done
	if n != 2 {
		t.Fatalf("n = %d", n)
	}
}

func TestEmbed(t *testing.T) {
	srv, cap := jsonServer(t, okEmbed)
	clearEnv(t)
	c, err := vertex.New("text-embedding-005", vertex.WithProject("proj"), core.WithBaseURL(srv.URL), vertex.WithAccessToken("t"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{
		Inputs: []string{"a", "b"}, Dimensions: 256, InputType: core.EmbedQuery,
		ProviderOptions: map[string]map[string]any{"vertex": {"custom": true}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"instances":  []any{map[string]any{"content": "a", "task_type": "RETRIEVAL_QUERY"}, map[string]any{"content": "b", "task_type": "RETRIEVAL_QUERY"}},
		"parameters": map[string]any{"outputDimensionality": float64(256), "autoTruncate": true},
		"custom":     true,
	}
	if !reflect.DeepEqual(cap.body, want) {
		t.Fatalf("body = %v\nwant %v", cap.body, want)
	}
	if !reflect.DeepEqual(resp.Embeddings, [][]float32{{0.1, 0.2}, {0.3, 0.4}}) || resp.Model != "text-embedding-005" || len(resp.Raw) == 0 {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Usage != (core.Usage{InputTokens: 7, TotalTokens: 7}) {
		t.Fatalf("usage = %+v", resp.Usage)
	}

	if _, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"doc"}, InputType: core.EmbedDocument}); err != nil {
		t.Fatal(err)
	}
	inst := cap.body["instances"].([]any)[0].(map[string]any)
	params := cap.body["parameters"].(map[string]any)
	if inst["task_type"] != "RETRIEVAL_DOCUMENT" {
		t.Fatalf("task_type = %v", inst["task_type"])
	}
	if _, ok := params["outputDimensionality"]; ok {
		t.Fatalf("outputDimensionality should be omitted: %v", params)
	}
	if _, err := c.Embed(context.Background(), &core.EmbedRequest{Inputs: []string{"x"}}); err != nil {
		t.Fatal(err)
	}
	if _, ok := cap.body["instances"].([]any)[0].(map[string]any)["task_type"]; ok {
		t.Fatal("task_type should be omitted without an input type")
	}
	if _, err := c.Embed(context.Background(), &core.EmbedRequest{}); err == nil {
		t.Fatal("empty inputs should fail")
	}
}
