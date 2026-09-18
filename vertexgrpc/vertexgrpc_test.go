package vertexgrpc_test

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"reflect"
	"strings"
	"sync"
	"testing"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/vertexgrpc"
)

const wantModel = "projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash"

type fakeServer struct {
	aiplatformpb.UnimplementedPredictionServiceServer
	mu       sync.Mutex
	genReq   *aiplatformpb.GenerateContentRequest
	predReq  *aiplatformpb.PredictRequest
	genResp  *aiplatformpb.GenerateContentResponse
	stream   []*aiplatformpb.GenerateContentResponse
	predResp *aiplatformpb.PredictResponse
	err      error
}

func (f *fakeServer) GenerateContent(_ context.Context, req *aiplatformpb.GenerateContentRequest) (*aiplatformpb.GenerateContentResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.genReq = req
	if f.err != nil {
		return nil, f.err
	}
	return f.genResp, nil
}

func (f *fakeServer) StreamGenerateContent(req *aiplatformpb.GenerateContentRequest, srv aiplatformpb.PredictionService_StreamGenerateContentServer) error {
	f.mu.Lock()
	f.genReq = req
	f.mu.Unlock()
	for _, r := range f.stream {
		if err := srv.Send(r); err != nil {
			return err
		}
	}
	return nil
}

func (f *fakeServer) Predict(_ context.Context, req *aiplatformpb.PredictRequest) (*aiplatformpb.PredictResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.predReq = req
	return f.predResp, nil
}

func newClient(t *testing.T, fake *fakeServer, model string) *vertexgrpc.Client {
	t.Helper()
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	aiplatformpb.RegisterPredictionServiceServer(srv, fake)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	conn, err := grpc.NewClient("passthrough:///bufconn",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	c, err := vertexgrpc.New(model,
		vertexgrpc.WithProject("p"),
		vertexgrpc.WithLocation("us-central1"),
		vertexgrpc.WithClientOptions(option.WithGRPCConn(conn), option.WithoutAuthentication()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func text(s string) *aiplatformpb.Part {
	return &aiplatformpb.Part{Data: &aiplatformpb.Part_Text{Text: s}}
}

func mustStruct(t *testing.T, m map[string]any) *structpb.Struct {
	t.Helper()
	st, err := structpb.NewStruct(m)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func mustValue(t *testing.T, v any) *structpb.Value {
	t.Helper()
	pv, err := structpb.NewValue(v)
	if err != nil {
		t.Fatal(err)
	}
	return pv
}

func TestProvider(t *testing.T) {
	if got := (vertexgrpc.Provider{}).ID(); got != "vertexgrpc" {
		t.Fatalf("ID = %q", got)
	}
	if (vertexgrpc.Provider{}).Matches("gemini-x") {
		t.Fatal("Matches should be false for bare names")
	}
	c, err := vertexgrpc.Provider{}.Open("gemini-2.5-flash", core.NewConfig(vertexgrpc.WithProject("p")))
	if err != nil {
		t.Fatal(err)
	}
	if c.Provider() != "vertexgrpc" || c.Model() != "gemini-2.5-flash" {
		t.Fatalf("Provider/Model = %q/%q", c.Provider(), c.Model())
	}
}

func TestMissingProject(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	_, err := vertexgrpc.New("gemini-2.5-flash")
	if !errors.Is(err, core.ErrMissingAPIKey) {
		t.Fatalf("err = %v, want ErrMissingAPIKey", err)
	}
	if !strings.Contains(err.Error(), "vertexgrpc.WithProject") {
		t.Fatalf("err = %v", err)
	}
}

func TestChatRequestMapping(t *testing.T) {
	fake := &fakeServer{genResp: &aiplatformpb.GenerateContentResponse{
		ResponseId: "r1",
		Candidates: []*aiplatformpb.Candidate{{
			FinishReason: aiplatformpb.Candidate_STOP,
			Content:      &aiplatformpb.Content{Role: "model", Parts: []*aiplatformpb.Part{text("hi")}},
		}},
	}}
	c := newClient(t, fake, "gemini-2.5-flash")
	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string","description":"City"}},"required":["city"]}`)
	req := &core.Request{
		Messages: []core.Message{
			core.System("be brief"),
			core.User(core.Text("look"), core.Image([]byte{1, 2, 3}, "image/png"), core.FileURL("gs://b/doc.pdf", "application/pdf")),
			core.Assistant(
				core.ReasoningPart{Text: "thinking", Signature: "c2ln"},
				core.ToolCall{ID: "call_1", Name: "weather", Arguments: json.RawMessage(`{"city":"Cape Town"}`)},
				core.ToolCall{ID: "call_2", Name: "time", Arguments: nil},
			),
			core.ToolResults(core.ToolResultText("call_1", "weather", `{"temp":21}`)),
			core.ToolResults(core.ToolResultText("call_2", "time", "noon")),
			core.UserText("thanks"),
		},
		Tools:       []core.Tool{{Name: "weather", Description: "Get weather", Parameters: schema}},
		ToolChoice:  core.ToolChoice{Mode: core.ToolChoiceNamed, Name: "weather"},
		MaxTokens:   100,
		Temperature: new(0.5),
		TopP:        new(0.9),
		Stop:        []string{"END"},
		Seed:        new(int64(7)),
		Format:      &core.ResponseFormat{Type: core.FormatJSONSchema, Schema: json.RawMessage(`{"type":"object"}`)},
		Reasoning:   &core.ReasoningConfig{BudgetTokens: 1024},
		ProviderOptions: map[string]map[string]any{
			"vertexgrpc": {"labels": map[string]string{"team": "x"}},
		},
	}
	resp, err := c.Chat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "hi" || resp.ID != "r1" || resp.FinishReason != core.FinishStop {
		t.Fatalf("resp = %+v", resp)
	}
	got := fake.genReq
	if got.GetModel() != wantModel {
		t.Fatalf("model = %q", got.GetModel())
	}
	if got.GetSystemInstruction().GetParts()[0].GetText() != "be brief" {
		t.Fatalf("system = %v", got.GetSystemInstruction())
	}
	if got.GetLabels()["team"] != "x" {
		t.Fatalf("labels = %v", got.GetLabels())
	}
	assertContents(t, got.GetContents())
	assertTools(t, got)
	assertGenerationConfig(t, got.GetGenerationConfig())
}

func assertContents(t *testing.T, contents []*aiplatformpb.Content) {
	t.Helper()
	roles := make([]string, len(contents))
	for i, ct := range contents {
		roles[i] = ct.GetRole()
	}
	if want := []string{"user", "model", "user"}; !reflect.DeepEqual(roles, want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	user := contents[0].GetParts()
	if len(user) != 3 || user[0].GetText() != "look" {
		t.Fatalf("user parts = %v", user)
	}
	if blob := user[1].GetInlineData(); blob.GetMimeType() != "image/png" || !reflect.DeepEqual(blob.GetData(), []byte{1, 2, 3}) {
		t.Fatalf("inline data = %v", blob)
	}
	if fd := user[2].GetFileData(); fd.GetFileUri() != "gs://b/doc.pdf" || fd.GetMimeType() != "application/pdf" {
		t.Fatalf("file data = %v", fd)
	}
	model := contents[1].GetParts()
	if len(model) != 3 || !model[0].GetThought() || model[0].GetText() != "thinking" || string(model[0].GetThoughtSignature()) != "sig" {
		t.Fatalf("model parts = %v", model)
	}
	if fc := model[1].GetFunctionCall(); fc.GetName() != "weather" || fc.GetArgs().GetFields()["city"].GetStringValue() != "Cape Town" {
		t.Fatalf("function call = %v", fc)
	}
	if fc := model[2].GetFunctionCall(); fc.GetName() != "time" || len(fc.GetArgs().GetFields()) != 0 {
		t.Fatalf("function call = %v", fc)
	}
	results := contents[2].GetParts()
	if len(results) != 3 {
		t.Fatalf("expected 2 grouped function responses + text, got %v", results)
	}
	if fr := results[0].GetFunctionResponse(); fr.GetName() != "weather" ||
		fr.GetResponse().GetFields()["result"].GetStructValue().GetFields()["temp"].GetNumberValue() != 21 {
		t.Fatalf("function response = %v", fr)
	}
	if fr := results[1].GetFunctionResponse(); fr.GetName() != "time" || fr.GetResponse().GetFields()["result"].GetStringValue() != "noon" {
		t.Fatalf("function response = %v", fr)
	}
	if results[2].GetText() != "thanks" {
		t.Fatalf("merged text = %v", results[2])
	}
}

func assertTools(t *testing.T, got *aiplatformpb.GenerateContentRequest) {
	t.Helper()
	decls := got.GetTools()[0].GetFunctionDeclarations()
	if len(decls) != 1 || decls[0].GetName() != "weather" || decls[0].GetDescription() != "Get weather" {
		t.Fatalf("declarations = %v", decls)
	}
	params := decls[0].GetParametersJsonSchema().GetStructValue().GetFields()
	if params["type"].GetStringValue() != "object" || params["required"].GetListValue().GetValues()[0].GetStringValue() != "city" {
		t.Fatalf("parameters = %v", params)
	}
	city := params["properties"].GetStructValue().GetFields()["city"].GetStructValue().GetFields()
	if city["type"].GetStringValue() != "string" || city["description"].GetStringValue() != "City" {
		t.Fatalf("city schema = %v", city)
	}
	fcc := got.GetToolConfig().GetFunctionCallingConfig()
	if fcc.GetMode() != aiplatformpb.FunctionCallingConfig_ANY || !reflect.DeepEqual(fcc.GetAllowedFunctionNames(), []string{"weather"}) {
		t.Fatalf("tool config = %v", fcc)
	}
}

func assertGenerationConfig(t *testing.T, gen *aiplatformpb.GenerationConfig) {
	t.Helper()
	if gen.GetTemperature() != 0.5 || gen.GetTopP() != 0.9 || gen.GetMaxOutputTokens() != 100 || gen.GetSeed() != 7 {
		t.Fatalf("generation config = %v", gen)
	}
	if !reflect.DeepEqual(gen.GetStopSequences(), []string{"END"}) {
		t.Fatalf("stop = %v", gen.GetStopSequences())
	}
	if gen.GetResponseMimeType() != "application/json" || gen.GetResponseJsonSchema().GetStructValue().GetFields()["type"].GetStringValue() != "object" {
		t.Fatalf("response format = %v", gen)
	}
	if tc := gen.GetThinkingConfig(); tc.GetThinkingBudget() != 1024 || !tc.GetIncludeThoughts() {
		t.Fatalf("thinking config = %v", tc)
	}
}

func TestToolChoiceModes(t *testing.T) {
	fake := &fakeServer{genResp: &aiplatformpb.GenerateContentResponse{}}
	c := newClient(t, fake, "gemini-2.5-flash")
	for mode, want := range map[core.ToolChoiceMode]aiplatformpb.FunctionCallingConfig_Mode{
		core.ToolChoiceNone:     aiplatformpb.FunctionCallingConfig_NONE,
		core.ToolChoiceRequired: aiplatformpb.FunctionCallingConfig_ANY,
	} {
		if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}, ToolChoice: core.ToolChoice{Mode: mode}}); err != nil {
			t.Fatal(err)
		}
		if got := fake.genReq.GetToolConfig().GetFunctionCallingConfig().GetMode(); got != want {
			t.Fatalf("%s: mode = %v, want %v", mode, got, want)
		}
	}
	if _, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("x")}}); err != nil {
		t.Fatal(err)
	}
	if fake.genReq.GetToolConfig() != nil {
		t.Fatalf("auto should send no tool config, got %v", fake.genReq.GetToolConfig())
	}
}

func TestToolResultRequiresName(t *testing.T) {
	c := newClient(t, &fakeServer{}, "gemini-2.5-flash")
	_, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{
		core.ToolResults(core.ToolResultText("call_1", "", "x")),
	}})
	if err == nil || !strings.Contains(err.Error(), "needs Name") {
		t.Fatalf("err = %v", err)
	}
}

func TestChatResponseMapping(t *testing.T) {
	fake := &fakeServer{genResp: &aiplatformpb.GenerateContentResponse{
		ModelVersion: "gemini-2.5-flash-001",
		Candidates: []*aiplatformpb.Candidate{{
			FinishReason: aiplatformpb.Candidate_STOP,
			Content: &aiplatformpb.Content{Role: "model", Parts: []*aiplatformpb.Part{
				{Data: &aiplatformpb.Part_Text{Text: "plan"}, Thought: true, ThoughtSignature: []byte("sig")},
				text("calling"),
				{Data: &aiplatformpb.Part_FunctionCall{FunctionCall: &aiplatformpb.FunctionCall{Name: "weather", Args: mustStruct(t, map[string]any{"city": "CT"})}}},
				{Data: &aiplatformpb.Part_FunctionCall{FunctionCall: &aiplatformpb.FunctionCall{Name: "time"}}},
			}},
		}},
		UsageMetadata: &aiplatformpb.GenerateContentResponse_UsageMetadata{
			PromptTokenCount: 10, CandidatesTokenCount: 5, TotalTokenCount: 20, CachedContentTokenCount: 2, ThoughtsTokenCount: 5,
		},
	}}
	c := newClient(t, fake, "gemini-2.5-flash")
	resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	want := []core.Part{
		core.ReasoningPart{Text: "plan", Signature: "c2ln"},
		core.TextPart{Text: "calling"},
		core.ToolCall{ID: "call_1", Name: "weather", Arguments: json.RawMessage(`{"city":"CT"}`)},
		core.ToolCall{ID: "call_2", Name: "time", Arguments: json.RawMessage(`{}`)},
	}
	if !reflect.DeepEqual(resp.Message.Parts, want) {
		t.Fatalf("parts = %#v\nwant %#v", resp.Message.Parts, want)
	}
	if resp.FinishReason != core.FinishToolCalls || resp.Model != "gemini-2.5-flash-001" {
		t.Fatalf("finish/model = %v/%v", resp.FinishReason, resp.Model)
	}
	wantUsage := core.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 20, CachedInputTokens: 2, ReasoningTokens: 5}
	if resp.Usage != wantUsage {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if !json.Valid(resp.Raw) || !strings.Contains(string(resp.Raw), "gemini-2.5-flash-001") {
		t.Fatalf("raw = %s", resp.Raw)
	}
}

func TestFinishReasons(t *testing.T) {
	cases := map[aiplatformpb.Candidate_FinishReason]core.FinishReason{
		aiplatformpb.Candidate_STOP:                    core.FinishStop,
		aiplatformpb.Candidate_MAX_TOKENS:              core.FinishLength,
		aiplatformpb.Candidate_SAFETY:                  core.FinishContentFilter,
		aiplatformpb.Candidate_RECITATION:              core.FinishContentFilter,
		aiplatformpb.Candidate_BLOCKLIST:               core.FinishContentFilter,
		aiplatformpb.Candidate_PROHIBITED_CONTENT:      core.FinishContentFilter,
		aiplatformpb.Candidate_SPII:                    core.FinishContentFilter,
		aiplatformpb.Candidate_MALFORMED_FUNCTION_CALL: core.FinishOther,
	}
	fake := &fakeServer{}
	c := newClient(t, fake, "gemini-2.5-flash")
	for reason, want := range cases {
		fake.genResp = &aiplatformpb.GenerateContentResponse{Candidates: []*aiplatformpb.Candidate{{
			FinishReason: reason, Content: &aiplatformpb.Content{Parts: []*aiplatformpb.Part{text("x")}},
		}}}
		resp, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}})
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != want {
			t.Errorf("%v: finish = %v, want %v", reason, resp.FinishReason, want)
		}
	}
}

func TestStream(t *testing.T) {
	fake := &fakeServer{stream: []*aiplatformpb.GenerateContentResponse{
		{Candidates: []*aiplatformpb.Candidate{{Content: &aiplatformpb.Content{Role: "model", Parts: []*aiplatformpb.Part{
			{Data: &aiplatformpb.Part_Text{Text: "hmm"}, Thought: true},
			text("Hello"),
		}}}}},
		{Candidates: []*aiplatformpb.Candidate{{Content: &aiplatformpb.Content{Role: "model", Parts: []*aiplatformpb.Part{
			{Data: &aiplatformpb.Part_FunctionCall{FunctionCall: &aiplatformpb.FunctionCall{Name: "weather", Args: mustStruct(t, map[string]any{"city": "CT"})}}},
		}}}}},
		{
			Candidates:    []*aiplatformpb.Candidate{{FinishReason: aiplatformpb.Candidate_STOP, Content: &aiplatformpb.Content{Role: "model"}}},
			UsageMetadata: &aiplatformpb.GenerateContentResponse_UsageMetadata{PromptTokenCount: 3, CandidatesTokenCount: 4, TotalTokenCount: 7},
		},
	}}
	c := newClient(t, fake, "gemini-2.5-flash")
	var got []core.Chunk
	for ch, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}}) {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, ch)
	}
	want := []core.Chunk{
		{Kind: core.ChunkReasoning, Text: "hmm"},
		{Kind: core.ChunkText, Text: "Hello"},
		{Kind: core.ChunkToolCall, ToolCall: &core.ToolCallDelta{Index: 0, ID: "call_1", Name: "weather", Arguments: `{"city":"CT"}`}},
		{Kind: core.ChunkFinish, FinishReason: core.FinishToolCalls, Usage: &core.Usage{InputTokens: 3, OutputTokens: 4, TotalTokens: 7}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks = %#v\nwant %#v", got, want)
	}
	if fake.genReq.GetModel() != wantModel {
		t.Fatalf("model = %q", fake.genReq.GetModel())
	}
}

func TestStreamBreakEarly(t *testing.T) {
	fake := &fakeServer{stream: []*aiplatformpb.GenerateContentResponse{
		{Candidates: []*aiplatformpb.Candidate{{Content: &aiplatformpb.Content{Parts: []*aiplatformpb.Part{text("a")}}}}},
		{Candidates: []*aiplatformpb.Candidate{{Content: &aiplatformpb.Content{Parts: []*aiplatformpb.Part{text("b")}}}}},
	}}
	c := newClient(t, fake, "gemini-2.5-flash")
	n := 0
	for _, err := range c.Stream(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}}) {
		if err != nil {
			t.Fatal(err)
		}
		n++
		break
	}
	if n != 1 {
		t.Fatalf("yielded %d chunks", n)
	}
}

func TestStreamSetupError(t *testing.T) {
	c := newClient(t, &fakeServer{}, "gemini-2.5-flash")
	var errs []error
	for _, err := range c.Stream(context.Background(), nil) {
		errs = append(errs, err)
	}
	if len(errs) != 1 || errs[0] == nil {
		t.Fatalf("errs = %v", errs)
	}
}

func TestRateLimited(t *testing.T) {
	fake := &fakeServer{err: status.Error(codes.ResourceExhausted, "quota exceeded")}
	c := newClient(t, fake, "gemini-2.5-flash")
	_, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}})
	if !errors.Is(err, core.ErrRateLimited) {
		t.Fatalf("err = %v, want ErrRateLimited", err)
	}
	var apiErr *core.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 429 || apiErr.Code != "ResourceExhausted" || apiErr.Provider != "vertexgrpc" {
		t.Fatalf("apiErr = %+v", apiErr)
	}
	if !strings.Contains(apiErr.Message, "quota exceeded") {
		t.Fatalf("message = %q", apiErr.Message)
	}
}

func TestErrorStatusMapping(t *testing.T) {
	cases := map[codes.Code]int{
		codes.InvalidArgument:  400,
		codes.Unauthenticated:  401,
		codes.PermissionDenied: 403,
		codes.NotFound:         404,
		codes.Unavailable:      503,
		codes.Internal:         500,
		codes.DataLoss:         500,
	}
	fake := &fakeServer{}
	c := newClient(t, fake, "gemini-2.5-flash")
	for code, want := range cases {
		fake.err = status.Error(code, code.String())
		_, err := c.Chat(context.Background(), &core.Request{Messages: []core.Message{core.UserText("hi")}})
		var apiErr *core.APIError
		if !errors.As(err, &apiErr) || apiErr.Status != want {
			t.Errorf("%v: err = %v, want status %d", code, err, want)
		}
	}
}

func TestEmbed(t *testing.T) {
	prediction := func(vals []any, tokens float64) *structpb.Value {
		return mustValue(t, map[string]any{"embeddings": map[string]any{
			"values":     vals,
			"statistics": map[string]any{"token_count": tokens, "truncated": false},
		}})
	}
	fake := &fakeServer{predResp: &aiplatformpb.PredictResponse{Predictions: []*structpb.Value{
		prediction([]any{0.1, 0.2}, 3),
		prediction([]any{0.3, 0.4}, 4),
	}}}
	c := newClient(t, fake, "text-embedding-005")
	resp, err := c.Embed(context.Background(), &core.EmbedRequest{
		Inputs: []string{"a", "b"}, Dimensions: 2, InputType: core.EmbedQuery,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{0.1, 0.2}, {0.3, 0.4}}
	if !reflect.DeepEqual(resp.Embeddings, want) {
		t.Fatalf("embeddings = %v", resp.Embeddings)
	}
	if resp.Usage.InputTokens != 7 || resp.Usage.TotalTokens != 7 || resp.Model != "text-embedding-005" {
		t.Fatalf("usage/model = %+v/%q", resp.Usage, resp.Model)
	}
	got := fake.predReq
	if got.GetEndpoint() != "projects/p/locations/us-central1/publishers/google/models/text-embedding-005" {
		t.Fatalf("endpoint = %q", got.GetEndpoint())
	}
	if len(got.GetInstances()) != 2 {
		t.Fatalf("instances = %v", got.GetInstances())
	}
	inst := got.GetInstances()[1].GetStructValue().GetFields()
	if inst["content"].GetStringValue() != "b" || inst["task_type"].GetStringValue() != "RETRIEVAL_QUERY" {
		t.Fatalf("instance = %v", inst)
	}
	params := got.GetParameters().GetStructValue().GetFields()
	if params["outputDimensionality"].GetNumberValue() != 2 || !params["autoTruncate"].GetBoolValue() {
		t.Fatalf("parameters = %v", params)
	}
}

func TestEmbedNoInputs(t *testing.T) {
	c := newClient(t, &fakeServer{}, "text-embedding-005")
	if _, err := c.Embed(context.Background(), &core.EmbedRequest{}); err == nil {
		t.Fatal("expected error for empty inputs")
	}
}

func TestCloseWithoutUse(t *testing.T) {
	c, err := vertexgrpc.New("gemini-2.5-flash", vertexgrpc.WithProject("p"))
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
}
