package vertex

import "encoding/json"

type wireRequest struct {
	Contents          []wireContent   `json:"contents"`
	SystemInstruction *wireContent    `json:"systemInstruction,omitempty"`
	Tools             []wireTool      `json:"tools,omitempty"`
	ToolConfig        *wireToolConfig `json:"toolConfig,omitempty"`
	GenerationConfig  *wireGenConfig  `json:"generationConfig,omitempty"`
}

type wireContent struct {
	Role  string     `json:"role,omitempty"`
	Parts []wirePart `json:"parts"`
}

type wirePart struct {
	Text             string            `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
	InlineData       *wireBlob         `json:"inlineData,omitempty"`
	FileData         *wireFileData     `json:"fileData,omitempty"`
	FunctionCall     *wireFunctionCall `json:"functionCall,omitempty"`
	FunctionResponse *wireFunctionResp `json:"functionResponse,omitempty"`
}

type wireBlob struct {
	MIMEType string `json:"mimeType"`
	Data     string `json:"data"`
}

type wireFileData struct {
	MIMEType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type wireFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type wireFunctionResp struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type wireTool struct {
	FunctionDeclarations []wireFunctionDecl `json:"functionDeclarations"`
}

type wireFunctionDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type wireToolConfig struct {
	FunctionCallingConfig wireFunctionCalling `json:"functionCallingConfig"`
}

type wireFunctionCalling struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type wireGenConfig struct {
	Temperature        *float64            `json:"temperature,omitempty"`
	TopP               *float64            `json:"topP,omitempty"`
	MaxOutputTokens    int                 `json:"maxOutputTokens,omitempty"`
	StopSequences      []string            `json:"stopSequences,omitempty"`
	Seed               *int64              `json:"seed,omitempty"`
	ResponseMIMEType   string              `json:"responseMimeType,omitempty"`
	ResponseJSONSchema json.RawMessage     `json:"responseJsonSchema,omitempty"`
	ThinkingConfig     *wireThinkingConfig `json:"thinkingConfig,omitempty"`
}

type wireThinkingConfig struct {
	IncludeThoughts bool `json:"includeThoughts"`
	ThinkingBudget  int  `json:"thinkingBudget"`
}

type wireResponse struct {
	Candidates     []wireCandidate     `json:"candidates"`
	PromptFeedback *wirePromptFeedback `json:"promptFeedback"`
	UsageMetadata  *wireUsage          `json:"usageMetadata"`
	ModelVersion   string              `json:"modelVersion"`
	ResponseID     string              `json:"responseId"`
	Error          json.RawMessage     `json:"error"`
}

type wireCandidate struct {
	Content      wireContent `json:"content"`
	FinishReason string      `json:"finishReason"`
}

type wirePromptFeedback struct {
	BlockReason string `json:"blockReason"`
}

type wireUsage struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
}

type embedRequest struct {
	Instances  []embedInstance `json:"instances"`
	Parameters embedParameters `json:"parameters"`
}

type embedInstance struct {
	Content  string `json:"content"`
	TaskType string `json:"task_type,omitempty"`
}

type embedParameters struct {
	OutputDimensionality int  `json:"outputDimensionality,omitempty"`
	AutoTruncate         bool `json:"autoTruncate"`
}

type embedResponse struct {
	Predictions []struct {
		Embeddings struct {
			Values     []float32 `json:"values"`
			Statistics struct {
				TokenCount int `json:"token_count"`
			} `json:"statistics"`
		} `json:"embeddings"`
	} `json:"predictions"`
}
