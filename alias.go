package llmkit

import "github.com/richardwooding/llmkit/core"

type (
	// Chatter is an alias of core.Chatter.
	Chatter = core.Chatter
	// Streamer is an alias of core.Streamer.
	Streamer = core.Streamer
	// Embedder is an alias of core.Embedder.
	Embedder = core.Embedder
	// Reranker is an alias of core.Reranker.
	Reranker = core.Reranker
	// MultimodalEmbedder is an alias of core.MultimodalEmbedder.
	MultimodalEmbedder = core.MultimodalEmbedder
	// TokenCounter is an alias of core.TokenCounter.
	TokenCounter = core.TokenCounter
	// Client is an alias of core.Client.
	Client = core.Client
	// Provider is an alias of core.Provider.
	Provider = core.Provider
)

type (
	// Role is an alias of core.Role.
	Role = core.Role
	// Message is an alias of core.Message.
	Message = core.Message
	// Part is an alias of core.Part.
	Part = core.Part
	// TextPart is an alias of core.TextPart.
	TextPart = core.TextPart
	// ImagePart is an alias of core.ImagePart.
	ImagePart = core.ImagePart
	// AudioPart is an alias of core.AudioPart.
	AudioPart = core.AudioPart
	// FilePart is an alias of core.FilePart.
	FilePart = core.FilePart
	// ReasoningPart is an alias of core.ReasoningPart.
	ReasoningPart = core.ReasoningPart
	// ToolCall is an alias of core.ToolCall.
	ToolCall = core.ToolCall
	// ToolResult is an alias of core.ToolResult.
	ToolResult = core.ToolResult
)

type (
	// Request is an alias of core.Request.
	Request = core.Request
	// Tool is an alias of core.Tool.
	Tool = core.Tool
	// ToolChoice is an alias of core.ToolChoice.
	ToolChoice = core.ToolChoice
	// ToolChoiceMode is an alias of core.ToolChoiceMode.
	ToolChoiceMode = core.ToolChoiceMode
	// ResponseFormat is an alias of core.ResponseFormat.
	ResponseFormat = core.ResponseFormat
	// ReasoningConfig is an alias of core.ReasoningConfig.
	ReasoningConfig = core.ReasoningConfig
	// CacheConfig is an alias of core.CacheConfig.
	CacheConfig = core.CacheConfig
	// Response is an alias of core.Response.
	Response = core.Response
	// FinishReason is an alias of core.FinishReason.
	FinishReason = core.FinishReason
	// Usage is an alias of core.Usage.
	Usage = core.Usage
	// Chunk is an alias of core.Chunk.
	Chunk = core.Chunk
	// ChunkKind is an alias of core.ChunkKind.
	ChunkKind = core.ChunkKind
	// ToolCallDelta is an alias of core.ToolCallDelta.
	ToolCallDelta = core.ToolCallDelta
	// ReasoningDelta is an alias of core.ReasoningDelta.
	ReasoningDelta = core.ReasoningDelta
	// EmbedRequest is an alias of core.EmbedRequest.
	EmbedRequest = core.EmbedRequest
	// EmbedResponse is an alias of core.EmbedResponse.
	EmbedResponse = core.EmbedResponse
	// EmbedInputType is an alias of core.EmbedInputType.
	EmbedInputType = core.EmbedInputType
	// RerankRequest is an alias of core.RerankRequest.
	RerankRequest = core.RerankRequest
	// RerankResult is an alias of core.RerankResult.
	RerankResult = core.RerankResult
	// RerankResponse is an alias of core.RerankResponse.
	RerankResponse = core.RerankResponse
	// MultimodalEmbedRequest is an alias of core.MultimodalEmbedRequest.
	MultimodalEmbedRequest = core.MultimodalEmbedRequest
	// Config is an alias of core.Config.
	Config = core.Config
	// Option is an alias of core.Option.
	Option = core.Option
	// APIError is an alias of core.APIError.
	APIError = core.APIError
)

// Roles.
const (
	RoleSystem    = core.RoleSystem
	RoleUser      = core.RoleUser
	RoleAssistant = core.RoleAssistant
	RoleTool      = core.RoleTool
)

// Tool choice modes.
const (
	ToolChoiceAuto     = core.ToolChoiceAuto
	ToolChoiceNone     = core.ToolChoiceNone
	ToolChoiceRequired = core.ToolChoiceRequired
	ToolChoiceNamed    = core.ToolChoiceNamed
)

// Response formats.
const (
	FormatJSON       = core.FormatJSON
	FormatJSONSchema = core.FormatJSONSchema
)

// Finish reasons.
const (
	FinishStop          = core.FinishStop
	FinishLength        = core.FinishLength
	FinishToolCalls     = core.FinishToolCalls
	FinishContentFilter = core.FinishContentFilter
	FinishOther         = core.FinishOther
)

// Chunk kinds.
const (
	ChunkText      = core.ChunkText
	ChunkReasoning = core.ChunkReasoning
	ChunkToolCall  = core.ChunkToolCall
	ChunkFinish    = core.ChunkFinish
)

// Embedding input types.
const (
	EmbedQuery    = core.EmbedQuery
	EmbedDocument = core.EmbedDocument
)

// Sentinel errors.
var (
	ErrUnsupported      = core.ErrUnsupported
	ErrUnknownProvider  = core.ErrUnknownProvider
	ErrMissingAPIKey    = core.ErrMissingAPIKey
	ErrRateLimited      = core.ErrRateLimited
	ErrContextLength    = core.ErrContextLength
	ErrToolLoopExceeded = core.ErrToolLoopExceeded
)

// Message constructors.
var (
	Text           = core.Text
	Image          = core.Image
	ImageURL       = core.ImageURL
	Audio          = core.Audio
	File           = core.File
	FileURL        = core.FileURL
	ToolResultText = core.ToolResultText
	System         = core.System
	User           = core.User
	UserText       = core.UserText
	Assistant      = core.Assistant
	ToolResults    = core.ToolResults
)

// Options.
var (
	WithAPIKey     = core.WithAPIKey
	WithBaseURL    = core.WithBaseURL
	WithHTTPClient = core.WithHTTPClient
	WithHeader     = core.WithHeader
	WithTimeout    = core.WithTimeout
	WithValue      = core.WithValue
	NewConfig      = core.NewConfig
)
