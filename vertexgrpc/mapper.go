package vertexgrpc

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/richardwooding/llmkit/core"
)

const (
	roleUser  = "user"
	roleModel = "model"
	mimeJSON  = "application/json"
)

func (c *Client) request(req *core.Request) (*aiplatformpb.GenerateContentRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("%s: nil request", ID)
	}
	contents, system, err := contents(req.Messages)
	if err != nil {
		return nil, err
	}
	tools, err := tools(req.Tools)
	if err != nil {
		return nil, err
	}
	gen, err := generationConfig(req)
	if err != nil {
		return nil, err
	}
	out := &aiplatformpb.GenerateContentRequest{
		Model:             c.modelName(),
		Contents:          contents,
		SystemInstruction: system,
		Tools:             tools,
		ToolConfig:        toolConfig(req.ToolChoice),
		GenerationConfig:  gen,
	}
	if labels, ok := req.ProviderOptions[ID]["labels"].(map[string]string); ok {
		out.Labels = labels
	}
	return out, nil
}

func contents(msgs []core.Message) (turns []*aiplatformpb.Content, system *aiplatformpb.Content, err error) {
	turns = make([]*aiplatformpb.Content, 0, len(msgs))
	for i := range msgs {
		m := &msgs[i]
		if m.Role == core.RoleSystem {
			if system == nil {
				system = &aiplatformpb.Content{}
			}
			system.Parts = append(system.Parts, &aiplatformpb.Part{Data: &aiplatformpb.Part_Text{Text: m.Text()}})
			continue
		}
		r, err := role(m.Role)
		if err != nil {
			return nil, nil, err
		}
		ps, err := parts(m.Parts)
		if err != nil {
			return nil, nil, err
		}
		if n := len(turns); n > 0 && turns[n-1].Role == r {
			turns[n-1].Parts = append(turns[n-1].Parts, ps...)
			continue
		}
		turns = append(turns, &aiplatformpb.Content{Role: r, Parts: ps})
	}
	return turns, system, nil
}

func role(r core.Role) (string, error) {
	switch r {
	case core.RoleUser, core.RoleTool:
		return roleUser, nil
	case core.RoleAssistant:
		return roleModel, nil
	default:
		return "", fmt.Errorf("%s: unknown role %q", ID, r)
	}
}

// parts converts message parts; a signature-only ReasoningPart is attached
// to the part that follows it, mirroring how Gemini signs function calls.
func parts(in []core.Part) ([]*aiplatformpb.Part, error) {
	out := make([]*aiplatformpb.Part, 0, len(in))
	var pending []byte
	for _, p := range in {
		if r, ok := p.(core.ReasoningPart); ok && r.Text == "" {
			pending = signature(r.Signature)
			continue
		}
		part, err := part(p)
		if err != nil {
			return nil, err
		}
		if pending != nil {
			part.ThoughtSignature = pending
			pending = nil
		}
		out = append(out, part)
	}
	return out, nil
}

func part(p core.Part) (*aiplatformpb.Part, error) {
	switch v := p.(type) {
	case core.TextPart:
		return &aiplatformpb.Part{Data: &aiplatformpb.Part_Text{Text: v.Text}}, nil
	case core.ImagePart:
		return media(v.Data, v.URL, v.MIME), nil
	case core.AudioPart:
		return media(v.Data, "", v.MIME), nil
	case core.FilePart:
		return media(v.Data, v.URL, v.MIME), nil
	case core.ReasoningPart:
		return &aiplatformpb.Part{
			Data: &aiplatformpb.Part_Text{Text: v.Text}, Thought: true, ThoughtSignature: signature(v.Signature),
		}, nil
	case core.ToolCall:
		return functionCall(v)
	case core.ToolResult:
		return functionResponse(v)
	default:
		return nil, fmt.Errorf("%s: unsupported part %T", ID, p)
	}
}

func media(data []byte, url, mime string) *aiplatformpb.Part {
	if url != "" {
		return &aiplatformpb.Part{Data: &aiplatformpb.Part_FileData{FileData: &aiplatformpb.FileData{MimeType: mime, FileUri: url}}}
	}
	return &aiplatformpb.Part{Data: &aiplatformpb.Part_InlineData{InlineData: &aiplatformpb.Blob{MimeType: mime, Data: data}}}
}

func signature(s string) []byte {
	if s == "" {
		return nil
	}
	if b, err := base64.StdEncoding.DecodeString(s); err == nil {
		return b
	}
	return []byte(s)
}

func functionCall(tc core.ToolCall) (*aiplatformpb.Part, error) {
	args := map[string]any{}
	if len(tc.Arguments) > 0 {
		if err := json.Unmarshal(tc.Arguments, &args); err != nil {
			return nil, fmt.Errorf("%s: tool call %q arguments: %w", ID, tc.Name, err)
		}
	}
	st, err := structpb.NewStruct(args)
	if err != nil {
		return nil, fmt.Errorf("%s: tool call %q arguments: %w", ID, tc.Name, err)
	}
	return &aiplatformpb.Part{Data: &aiplatformpb.Part_FunctionCall{FunctionCall: &aiplatformpb.FunctionCall{Name: tc.Name, Args: st}}}, nil
}

func functionResponse(tr core.ToolResult) (*aiplatformpb.Part, error) {
	if tr.Name == "" {
		return nil, fmt.Errorf("%s: tool result for call %q needs Name", ID, tr.CallID)
	}
	text := tr.Text()
	var result any = text
	var parsed any
	if json.Unmarshal([]byte(text), &parsed) == nil {
		result = parsed
	}
	st, err := structpb.NewStruct(map[string]any{"result": result})
	if err != nil {
		return nil, fmt.Errorf("%s: tool result %q: %w", ID, tr.Name, err)
	}
	return &aiplatformpb.Part{Data: &aiplatformpb.Part_FunctionResponse{
		FunctionResponse: &aiplatformpb.FunctionResponse{Name: tr.Name, Response: st},
	}}, nil
}

func tools(in []core.Tool) ([]*aiplatformpb.Tool, error) {
	if len(in) == 0 {
		return nil, nil
	}
	decls := make([]*aiplatformpb.FunctionDeclaration, 0, len(in))
	for _, t := range in {
		params, err := jsonValue(t.Parameters)
		if err != nil {
			return nil, fmt.Errorf("%s: tool %q parameters: %w", ID, t.Name, err)
		}
		decls = append(decls, &aiplatformpb.FunctionDeclaration{
			Name: t.Name, Description: t.Description, ParametersJsonSchema: params,
		})
	}
	return []*aiplatformpb.Tool{{FunctionDeclarations: decls}}, nil
}

func jsonValue(raw json.RawMessage) (*structpb.Value, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, err
	}
	return structpb.NewValue(v)
}

func toolConfig(tc core.ToolChoice) *aiplatformpb.ToolConfig {
	fc := &aiplatformpb.FunctionCallingConfig{}
	switch tc.Mode {
	case "", core.ToolChoiceAuto:
		return nil
	case core.ToolChoiceNone:
		fc.Mode = aiplatformpb.FunctionCallingConfig_NONE
	case core.ToolChoiceRequired:
		fc.Mode = aiplatformpb.FunctionCallingConfig_ANY
	case core.ToolChoiceNamed:
		fc.Mode = aiplatformpb.FunctionCallingConfig_ANY
		fc.AllowedFunctionNames = []string{tc.Name}
	}
	return &aiplatformpb.ToolConfig{FunctionCallingConfig: fc}
}

func generationConfig(req *core.Request) (*aiplatformpb.GenerationConfig, error) {
	gen := &aiplatformpb.GenerationConfig{StopSequences: req.Stop}
	if req.Temperature != nil {
		gen.Temperature = new(float32(*req.Temperature))
	}
	if req.TopP != nil {
		gen.TopP = new(float32(*req.TopP))
	}
	if req.MaxTokens > 0 {
		gen.MaxOutputTokens = new(clamp32(int64(req.MaxTokens)))
	}
	if req.Seed != nil {
		gen.Seed = new(clamp32(*req.Seed))
	}
	if err := responseFormat(gen, req.Format); err != nil {
		return nil, err
	}
	gen.ThinkingConfig = thinkingConfig(req.Reasoning)
	return gen, nil
}

func responseFormat(gen *aiplatformpb.GenerationConfig, f *core.ResponseFormat) error {
	if f == nil {
		return nil
	}
	switch f.Type {
	case core.FormatJSON:
		gen.ResponseMimeType = mimeJSON
	case core.FormatJSONSchema:
		schema, err := jsonValue(f.Schema)
		if err != nil {
			return fmt.Errorf("%s: response schema: %w", ID, err)
		}
		gen.ResponseMimeType = mimeJSON
		gen.ResponseJsonSchema = schema
	default:
		return fmt.Errorf("%s: unknown response format %q", ID, f.Type)
	}
	return nil
}

func clamp32(n int64) int32 {
	if n > math.MaxInt32 {
		return math.MaxInt32
	}
	if n < math.MinInt32 {
		return math.MinInt32
	}
	return int32(n)
}

func thinkingConfig(r *core.ReasoningConfig) *aiplatformpb.GenerationConfig_ThinkingConfig {
	if r == nil {
		return nil
	}
	tc := &aiplatformpb.GenerationConfig_ThinkingConfig{IncludeThoughts: new(true)}
	if r.BudgetTokens > 0 {
		tc.ThinkingBudget = new(clamp32(int64(r.BudgetTokens)))
	}
	levels := map[string]aiplatformpb.GenerationConfig_ThinkingConfig_ThinkingLevel{
		"minimal": aiplatformpb.GenerationConfig_ThinkingConfig_MINIMAL,
		"low":     aiplatformpb.GenerationConfig_ThinkingConfig_LOW,
		"medium":  aiplatformpb.GenerationConfig_ThinkingConfig_MEDIUM,
		"high":    aiplatformpb.GenerationConfig_ThinkingConfig_HIGH,
	}
	if lvl, ok := levels[r.Effort]; ok && tc.ThinkingBudget == nil {
		tc.ThinkingLevel = new(lvl)
	}
	return tc
}
