package openai

import "encoding/json"

const (
	typeMessage            = "message"
	typeFunction           = "function"
	typeFunctionCall       = "function_call"
	typeFunctionCallOutput = "function_call_output"
	typeReasoning          = "reasoning"
	typeOutputText         = "output_text"
	typeRefusal            = "refusal"
	typeInputText          = "input_text"
	typeInputImage         = "input_image"
	typeInputFile          = "input_file"

	statusFailed     = "failed"
	statusIncomplete = "incomplete"

	includeEncryptedReasoning = "reasoning.encrypted_content"
	// summaryAuto is reasoning.summary's "return a summary" value; Anthropic's
	// thinking.display "summarized" maps onto it.
	summaryAuto       = "auto"
	displaySummarized = "summarized"

	eventOutputItemAdded      = "response.output_item.added"
	eventOutputItemDone       = "response.output_item.done"
	eventOutputTextDelta      = "response.output_text.delta"
	eventRefusalDelta         = "response.refusal.delta"
	eventFunctionArgsDelta    = "response.function_call_arguments.delta"
	eventReasoningSummary     = "response.reasoning_summary_text.delta"
	eventReasoningText        = "response.reasoning_text.delta"
	eventResponseCompleted    = "response.completed"
	eventResponseIncomplete   = "response.incomplete"
	eventResponseFailed       = "response.failed"
	eventError                = "error"
	reasonMaxOutputTokens     = "max_output_tokens"
	reasonMaxTokens           = "max_tokens"
	reasonContentFilter       = "content_filter"
	emptyJSONArray            = "[]"
	emptyJSONObject           = "{}"
	instructionsSeparator     = "\n\n"
	reasoningSummarySeparator = "\n\n"
)

type wireRequest struct {
	Model           string         `json:"model"`
	Instructions    string         `json:"instructions,omitempty"`
	Input           []wireItem     `json:"input"`
	Tools           []wireTool     `json:"tools,omitempty"`
	ToolChoice      any            `json:"tool_choice,omitempty"`
	MaxOutputTokens int            `json:"max_output_tokens,omitempty"`
	Temperature     *float64       `json:"temperature,omitempty"`
	TopP            *float64       `json:"top_p,omitempty"`
	Text            *wireText      `json:"text,omitempty"`
	Reasoning       *wireReasoning `json:"reasoning,omitempty"`
	Include         []string       `json:"include,omitempty"`
	Stream          bool           `json:"stream,omitempty"`
}

type wireItem struct {
	Type             string          `json:"type,omitempty"`
	Role             string          `json:"role,omitempty"`
	Content          []wireContent   `json:"content,omitempty"`
	ID               string          `json:"id,omitempty"`
	CallID           string          `json:"call_id,omitempty"`
	Name             string          `json:"name,omitempty"`
	Arguments        string          `json:"arguments,omitempty"`
	Output           *string         `json:"output,omitempty"`
	EncryptedContent string          `json:"encrypted_content,omitempty"`
	Summary          json.RawMessage `json:"summary,omitempty"`
}

type wireContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
}

type wireTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

type wireText struct {
	Format wireFormat `json:"format"`
}

type wireFormat struct {
	Type   string          `json:"type"`
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
	Strict bool            `json:"strict,omitempty"`
}

type wireReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type wireResponse struct {
	ID                string     `json:"id"`
	Model             string     `json:"model"`
	Status            string     `json:"status"`
	Error             *wireError `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Output []wireOutputItem `json:"output"`
	Usage  *wireUsage       `json:"usage"`
}

type wireError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type wireOutputItem struct {
	Type             string              `json:"type"`
	ID               string              `json:"id"`
	Role             string              `json:"role"`
	Content          []wireOutputContent `json:"content"`
	CallID           string              `json:"call_id"`
	Name             string              `json:"name"`
	Arguments        string              `json:"arguments"`
	Summary          []wireSummary       `json:"summary"`
	EncryptedContent string              `json:"encrypted_content"`
}

type wireOutputContent struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	Refusal string `json:"refusal"`
}

type wireSummary struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type wireUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	TotalTokens        int `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

type wireEvent struct {
	Type        string          `json:"type"`
	OutputIndex int             `json:"output_index"`
	Item        *wireOutputItem `json:"item"`
	Delta       string          `json:"delta"`
	Response    *wireResponse   `json:"response"`
	Code        string          `json:"code"`
	Message     string          `json:"message"`
}
