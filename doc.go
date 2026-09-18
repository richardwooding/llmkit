// Package llmkit is a multi-vendor AI client. One factory resolves a model
// name such as "gpt-5", "claude-sonnet-4-5", "llama3.2:3b" or
// "openrouter/openai/gpt-4o" to a provider, and callers program against small
// single-method interfaces:
//
//	chat, err := llmkit.Open[llmkit.Chatter]("gpt-5")
//	resp, err := chat.Chat(ctx, &llmkit.Request{Messages: []llmkit.Message{llmkit.UserText("hi")}})
//
// Every type in this package is an alias of the same name in package core,
// so provider packages and applications share one set of types.
package llmkit
