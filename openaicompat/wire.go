package openaicompat

import "encoding/json"

type wireRequest struct {
	Model               string          `json:"model"`
	Messages            []wireMessage   `json:"messages"`
	Tools               []wireTool      `json:"tools,omitempty"`
	ToolChoice          any             `json:"tool_choice,omitempty"`
	MaxTokens           int             `json:"max_tokens,omitempty"`
	MaxCompletionTokens int             `json:"max_completion_tokens,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	Stop                []string        `json:"stop,omitempty"`
	Seed                *int64          `json:"seed,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *wireStreamOpts `json:"stream_options,omitempty"`
	ResponseFormat      *wireRespFormat `json:"response_format,omitempty"`
	Extra               map[string]any  `json:"-"`
}

type wireStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type wireRespFormat struct {
	Type       string          `json:"type"`
	JSONSchema *wireJSONSchema `json:"json_schema,omitempty"`
}

type wireJSONSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema,omitempty"`
	Strict bool            `json:"strict,omitempty"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type wireContentPart struct {
	Type       string        `json:"type"`
	Text       string        `json:"text,omitempty"`
	ImageURL   *wireImageURL `json:"image_url,omitempty"`
	InputAudio *wireAudio    `json:"input_audio,omitempty"`
	File       *wireFile     `json:"file,omitempty"`
}

type wireImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type wireAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type wireFile struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
}

type wireTool struct {
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

type wireFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

type wireToolCall struct {
	Index    *int             `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function wireToolFunction `json:"function"`
}

type wireToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Model   string       `json:"model"`
	Choices []wireChoice `json:"choices"`
	Usage   *wireUsage   `json:"usage"`
	XGroq   *struct {
		Usage *wireUsage `json:"usage"`
	} `json:"x_groq,omitempty"`
}

type wireChoice struct {
	Index        int              `json:"index"`
	Message      *json.RawMessage `json:"message,omitempty"`
	Delta        *json.RawMessage `json:"delta,omitempty"`
	FinishReason string           `json:"finish_reason"`
}

type wireAssistant struct {
	Content   json.RawMessage `json:"content"`
	ToolCalls []wireToolCall  `json:"tool_calls"`
	Refusal   string          `json:"refusal"`
	Fields    map[string]json.RawMessage
}

// UnmarshalJSON keeps every raw field so provider-specific reasoning keys stay reachable.
func (a *wireAssistant) UnmarshalJSON(b []byte) error {
	type plain wireAssistant
	var p plain
	if err := json.Unmarshal(b, &p); err != nil {
		return err
	}
	*a = wireAssistant(p)
	return json.Unmarshal(b, &a.Fields)
}

type wireUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

type embedRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	Dimensions     int      `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

type embedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string     `json:"model"`
	Usage *wireUsage `json:"usage"`
}
