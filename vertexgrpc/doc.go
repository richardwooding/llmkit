// Package vertexgrpc talks to Vertex AI Gemini models over gRPC through the
// cloud.google.com/go/aiplatform PredictionService. It authenticates with
// Application Default Credentials and lives in its own module so the root
// llmkit module stays free of the Google Cloud dependency tree.
//
// Register it to resolve "vertexgrpc/<model>" names:
//
//	llmkit.Register(vertexgrpc.Provider{})
//	chat, err := llmkit.Open[llmkit.Chatter]("vertexgrpc/gemini-2.5-flash")
//
// Or build a client directly:
//
//	c, err := vertexgrpc.New("gemini-2.5-flash", vertexgrpc.WithProject("my-project"))
//
// Request.Extra cannot be merged into a protobuf request and is ignored;
// ProviderOptions["vertexgrpc"]["labels"] (map[string]string) becomes the
// request's billing labels.
package vertexgrpc
