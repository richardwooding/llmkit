package vertex

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime"
	"net/url"
	"path"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

const (
	roleUser  = "user"
	roleModel = "model"
	modeAny   = "ANY"
	mimeJSON  = "application/json"
)

func (c *Client) body(req *core.Request) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("%s: nil request", ID)
	}
	w := wireRequest{Tools: tools(req.Tools), ToolConfig: toolConfig(req.ToolChoice)}
	var err error
	if w.SystemInstruction, w.Contents, err = contents(req.Messages); err != nil {
		return nil, err
	}
	if w.GenerationConfig, err = generationConfig(req); err != nil {
		return nil, err
	}
	return httpx.MarshalWithExtra(&w, req.ProviderExtra(ID))
}

func contents(msgs []core.Message) (system *wireContent, turns []wireContent, err error) {
	var sysText []string
	turns = make([]wireContent, 0, len(msgs))
	for i := range msgs {
		m := &msgs[i]
		if m.Role == core.RoleSystem {
			sysText = append(sysText, m.Text())
			continue
		}
		role, err := wireRole(m.Role)
		if err != nil {
			return nil, nil, err
		}
		parts, err := wireParts(m.Parts)
		if err != nil {
			return nil, nil, err
		}
		if len(parts) == 0 {
			continue
		}
		if n := len(turns); n > 0 && turns[n-1].Role == role {
			turns[n-1].Parts = append(turns[n-1].Parts, parts...)
			continue
		}
		turns = append(turns, wireContent{Role: role, Parts: parts})
	}
	if len(sysText) > 0 {
		system = &wireContent{Parts: []wirePart{{Text: strings.Join(sysText, "\n")}}}
	}
	return system, turns, nil
}

func wireRole(r core.Role) (string, error) {
	switch r {
	case core.RoleUser, core.RoleTool:
		return roleUser, nil
	case core.RoleAssistant:
		return roleModel, nil
	default:
		return "", fmt.Errorf("%s: unknown role %q", ID, r)
	}
}

func wireParts(in []core.Part) ([]wirePart, error) {
	out := make([]wirePart, 0, len(in))
	var pending string
	for _, p := range in {
		// A text-less ReasoningPart is a bare thoughtSignature that rides on the next part.
		if r, ok := p.(core.ReasoningPart); ok && r.Text == "" {
			pending = r.Signature
			continue
		}
		w, err := wirePartOf(p)
		if err != nil {
			return nil, err
		}
		if w == nil {
			continue
		}
		if w.ThoughtSignature == "" {
			w.ThoughtSignature = pending
		}
		pending = ""
		out = append(out, *w)
	}
	return out, nil
}

func wirePartOf(p core.Part) (*wirePart, error) {
	switch v := p.(type) {
	case core.TextPart:
		if v.Text == "" {
			return nil, nil
		}
		return &wirePart{Text: v.Text}, nil
	case core.ImagePart:
		return media(v.Data, v.MIME, v.URL), nil
	case core.AudioPart:
		return media(v.Data, v.MIME, ""), nil
	case core.FilePart:
		return media(v.Data, v.MIME, v.URL), nil
	case core.ReasoningPart:
		if v.Signature == "" {
			return nil, nil
		}
		return &wirePart{Text: v.Text, Thought: true, ThoughtSignature: v.Signature}, nil
	case core.ToolCall:
		return functionCall(&v)
	case core.ToolResult:
		return functionResponse(&v)
	default:
		return nil, core.Unsupported(ID, fmt.Sprintf("%T", p))
	}
}

func media(data []byte, mimeType, uri string) *wirePart {
	if uri != "" {
		if mimeType == "" {
			mimeType = mimeFromURL(uri)
		}
		return &wirePart{FileData: &wireFileData{MIMEType: mimeType, FileURI: uri}}
	}
	return &wirePart{InlineData: &wireBlob{
		MIMEType: httpx.MIMEOr(mimeType, data), Data: base64.StdEncoding.EncodeToString(data),
	}}
}

func mimeFromURL(uri string) string {
	p := uri
	if u, err := url.Parse(uri); err == nil {
		p = u.Path
	}
	m, _, _ := strings.Cut(mime.TypeByExtension(path.Ext(p)), ";")
	return m
}

func functionCall(tc *core.ToolCall) (*wirePart, error) {
	if len(tc.Arguments) > 0 && !json.Valid(tc.Arguments) {
		return nil, fmt.Errorf("%s: tool call %q: arguments are not valid JSON", ID, tc.Name)
	}
	return &wirePart{FunctionCall: &wireFunctionCall{Name: tc.Name, Args: rawArgs(tc.Arguments)}}, nil
}

func functionResponse(tr *core.ToolResult) (*wirePart, error) {
	if tr.Name == "" {
		return nil, fmt.Errorf("%s: ToolResult.Name is required", ID)
	}
	for _, cp := range tr.Content {
		if _, ok := cp.(core.TextPart); !ok {
			return nil, core.Unsupported(ID, fmt.Sprintf("%T in tool result", cp))
		}
	}
	text := tr.Text()
	var value any = text
	if json.Valid([]byte(text)) {
		value = json.RawMessage(text)
	}
	key := "result"
	if tr.IsError {
		key = "error"
	}
	return &wirePart{FunctionResponse: &wireFunctionResp{Name: tr.Name, Response: map[string]any{key: value}}}, nil
}

func tools(ts []core.Tool) []wireTool {
	if len(ts) == 0 {
		return nil
	}
	decls := make([]wireFunctionDecl, 0, len(ts))
	for i := range ts {
		t := &ts[i]
		decls = append(decls, wireFunctionDecl{Name: t.Name, Description: t.Description, Parameters: t.Parameters})
	}
	return []wireTool{{FunctionDeclarations: decls}}
}

func toolConfig(tc core.ToolChoice) *wireToolConfig {
	var fc wireFunctionCalling
	switch tc.Mode {
	case core.ToolChoiceNone:
		fc.Mode = "NONE"
	case core.ToolChoiceRequired:
		fc.Mode = modeAny
	case core.ToolChoiceNamed:
		fc.Mode = modeAny
		fc.AllowedFunctionNames = []string{tc.Name}
	default:
		return nil
	}
	return &wireToolConfig{FunctionCallingConfig: fc}
}

func generationConfig(req *core.Request) (*wireGenConfig, error) {
	g := &wireGenConfig{
		Temperature: req.Temperature, TopP: req.TopP, MaxOutputTokens: req.MaxTokens,
		StopSequences: req.Stop, Seed: req.Seed,
	}
	if err := g.responseFormat(req.Format); err != nil {
		return nil, err
	}
	if r := req.Reasoning; r != nil {
		budget := -1 // dynamic thinking when no explicit budget is given
		if r.BudgetTokens > 0 {
			budget = r.BudgetTokens
		}
		g.ThinkingConfig = &wireThinkingConfig{IncludeThoughts: true, ThinkingBudget: budget}
	}
	if g.empty() {
		return nil, nil
	}
	return g, nil
}

func (g *wireGenConfig) responseFormat(f *core.ResponseFormat) error {
	if f == nil {
		return nil
	}
	switch f.Type {
	case core.FormatJSON:
		g.ResponseMIMEType = mimeJSON
	case core.FormatJSONSchema:
		if len(f.Schema) == 0 {
			return fmt.Errorf("%s: json_schema format requires a schema", ID)
		}
		g.ResponseMIMEType = mimeJSON
		g.ResponseJSONSchema = f.Schema
	default:
		return fmt.Errorf("%s: unknown response format %q", ID, f.Type)
	}
	return nil
}

func (g *wireGenConfig) empty() bool {
	return g.Temperature == nil && g.TopP == nil && g.MaxOutputTokens == 0 && len(g.StopSequences) == 0 &&
		g.Seed == nil && g.ResponseMIMEType == "" && g.ThinkingConfig == nil
}

func (c *Client) toResponse(out *wireResponse) *core.Response {
	resp := &core.Response{ID: out.ResponseID, Model: out.ModelVersion, Message: core.Message{Role: core.RoleAssistant}}
	if resp.Model == "" {
		resp.Model = c.model
	}
	if out.UsageMetadata != nil {
		resp.Usage = usage(out.UsageMetadata)
	}
	if len(out.Candidates) == 0 {
		resp.FinishReason = core.FinishOther
		if out.blocked() {
			resp.FinishReason = core.FinishContentFilter
		}
		return resp
	}
	cand := &out.Candidates[0]
	var calls int
	for i := range cand.Content.Parts {
		resp.Message.Parts = append(resp.Message.Parts, coreParts(&cand.Content.Parts[i], &calls)...)
	}
	resp.FinishReason = finishReason(cand.FinishReason, calls > 0)
	return resp
}

func (r *wireResponse) blocked() bool {
	return r.PromptFeedback != nil && r.PromptFeedback.BlockReason != ""
}

func coreParts(p *wirePart, calls *int) []core.Part {
	var out []core.Part
	switch {
	case p.Thought:
		return []core.Part{core.ReasoningPart{Text: p.Text, Signature: p.ThoughtSignature}}
	case p.FunctionCall != nil:
		*calls++
		if p.ThoughtSignature != "" {
			out = append(out, core.ReasoningPart{Signature: p.ThoughtSignature})
		}
		return append(out, core.ToolCall{ID: callID(*calls), Name: p.FunctionCall.Name, Arguments: rawArgs(p.FunctionCall.Args)})
	case p.Text != "":
		if p.ThoughtSignature != "" {
			out = append(out, core.ReasoningPart{Signature: p.ThoughtSignature})
		}
		return append(out, core.Text(p.Text))
	default:
		return nil
	}
}

// Gemini function calls carry no id, so one is synthesized from the call's position.
func callID(n int) string { return fmt.Sprintf("call_%d", n) }

func rawArgs(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage(`{}`)
	}
	return raw
}

func finishReason(s string, hasTools bool) core.FinishReason {
	switch s {
	case "STOP":
		if hasTools {
			return core.FinishToolCalls
		}
		return core.FinishStop
	case "MAX_TOKENS":
		return core.FinishLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY":
		return core.FinishContentFilter
	default:
		return core.FinishOther
	}
}

func usage(u *wireUsage) core.Usage {
	out := core.Usage{
		InputTokens:       u.PromptTokenCount,
		OutputTokens:      u.CandidatesTokenCount + u.ThoughtsTokenCount,
		TotalTokens:       u.TotalTokenCount,
		CachedInputTokens: u.CachedContentTokenCount,
		ReasoningTokens:   u.ThoughtsTokenCount,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}
	return out
}
