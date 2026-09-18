package cohere

import "encoding/json"

const (
	roleSystem    = "system"
	roleUser      = "user"
	roleAssistant = "assistant"
	roleTool      = "tool"

	typeText     = "text"
	typeImageURL = "image_url"
	typeThinking = "thinking"
	typeDocument = "document"
	typeFunction = "function"
)

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []wireMessage   `json:"messages"`
	Tools          []wireTool      `json:"tools,omitempty"`
	ToolChoice     string          `json:"tool_choice,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	P              *float64        `json:"p,omitempty"`
	StopSequences  []string        `json:"stop_sequences,omitempty"`
	Seed           *int64          `json:"seed,omitempty"`
	ResponseFormat *wireRespFormat `json:"response_format,omitempty"`
	Thinking       *wireThinking   `json:"thinking,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
}

type wireRespFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

type wireThinking struct {
	Type        string `json:"type"`
	TokenBudget int    `json:"token_budget,omitempty"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type wireContent struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	Thinking string        `json:"thinking,omitempty"`
	ImageURL *wireImageURL `json:"image_url,omitempty"`
	Document *wireDocument `json:"document,omitempty"`
}

type wireImageURL struct {
	URL string `json:"url"`
}

type wireDocument struct {
	ID   string            `json:"id,omitempty"`
	Data map[string]string `json:"data"`
}

type wireTool struct {
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

type wireFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type wireToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function wireToolFunction `json:"function"`
}

type wireToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type chatResponse struct {
	ID           string     `json:"id"`
	FinishReason string     `json:"finish_reason"`
	Message      wireOutput `json:"message"`
	Usage        *wireUsage `json:"usage"`
}

type wireOutput struct {
	Content   []wireContent  `json:"content"`
	ToolCalls []wireToolCall `json:"tool_calls"`
	ToolPlan  string         `json:"tool_plan"`
}

type wireUsage struct {
	BilledUnits  *wireTokens `json:"billed_units"`
	Tokens       *wireTokens `json:"tokens"`
	CachedTokens float64     `json:"cached_tokens"`
}

// Cohere reports token counts as JSON doubles.
type wireTokens struct {
	InputTokens  float64 `json:"input_tokens"`
	OutputTokens float64 `json:"output_tokens"`
}

type streamEvent struct {
	Type  string      `json:"type"`
	ID    string      `json:"id"`
	Index int         `json:"index"`
	Delta streamDelta `json:"delta"`
}

type streamDelta struct {
	Message      streamMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
	Usage        *wireUsage    `json:"usage"`
}

type streamMessage struct {
	Content   streamContent `json:"content"`
	ToolPlan  string        `json:"tool_plan"`
	ToolCalls wireToolCall  `json:"tool_calls"`
}

type streamContent struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking"`
}

type embedRequest struct {
	Model           string   `json:"model"`
	Texts           []string `json:"texts"`
	InputType       string   `json:"input_type"`
	EmbeddingTypes  []string `json:"embedding_types"`
	OutputDimension int      `json:"output_dimension,omitempty"`
	Truncate        string   `json:"truncate,omitempty"`
}

type embedResponse struct {
	ID         string `json:"id"`
	Embeddings struct {
		Float [][]float32 `json:"float"`
	} `json:"embeddings"`
	Meta struct {
		BilledUnits *wireTokens `json:"billed_units"`
		Tokens      *wireTokens `json:"tokens"`
	} `json:"meta"`
}
