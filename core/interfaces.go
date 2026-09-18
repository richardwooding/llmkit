package core

import (
	"context"
	"iter"
)

// Chatter produces a single assistant response for a conversation.
type Chatter interface {
	Chat(ctx context.Context, req *Request) (*Response, error)
}

// Streamer produces an assistant response incrementally. The request is sent
// when iteration starts; breaking out of the loop closes the connection. The
// final element is always a Chunk of kind ChunkFinish unless an error ends
// the sequence.
type Streamer interface {
	Stream(ctx context.Context, req *Request) iter.Seq2[Chunk, error]
}

// Embedder turns text into vectors.
type Embedder interface {
	Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error)
}

// Client is the handle a Provider returns. Assert it to Chatter, Streamer or
// Embedder to use it; a provider only implements what its API supports.
type Client interface {
	Provider() string
	Model() string
}

// Provider is implemented by each backend package and registered with a
// Registry. Open must not perform network I/O; reading environment variables
// is allowed.
type Provider interface {
	ID() string
	Matches(bareModel string) bool
	Open(model string, cfg *Config) (Client, error)
}
