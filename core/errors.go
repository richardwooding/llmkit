package core

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Sentinel errors. Match them with errors.Is.
var (
	ErrUnsupported      = errors.New("llmkit: unsupported")
	ErrUnknownProvider  = errors.New("llmkit: unknown provider")
	ErrMissingAPIKey    = errors.New("llmkit: missing API key")
	ErrRateLimited      = errors.New("llmkit: rate limited")
	ErrContextLength    = errors.New("llmkit: context length exceeded")
	ErrToolLoopExceeded = errors.New("llmkit: tool loop exceeded max iterations")
)

// APIError is a non-success response from a provider.
type APIError struct {
	Provider   string
	Status     int
	Code       string
	Type       string
	Message    string
	RetryAfter time.Duration
	Body       []byte
}

// Error formats the provider, status and message.
func (e *APIError) Error() string {
	var b strings.Builder
	b.WriteString(e.Provider)
	b.WriteString(": ")
	if e.Status != 0 {
		fmt.Fprintf(&b, "HTTP %d", e.Status)
	}
	if e.Code != "" {
		fmt.Fprintf(&b, " [%s]", e.Code)
	}
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	} else if e.Status == 0 {
		b.WriteString("request failed")
	}
	return b.String()
}

// Is lets errors.Is match ErrRateLimited and ErrContextLength without
// additional wrapping.
func (e *APIError) Is(target error) bool {
	switch target {
	case ErrRateLimited:
		return e.Status == http.StatusTooManyRequests || strings.Contains(strings.ToLower(e.Code), "rate_limit")
	case ErrContextLength:
		return e.isContextLength()
	default:
		return false
	}
}

func (e *APIError) isContextLength() bool {
	code := strings.ToLower(e.Code)
	if strings.Contains(code, "context_length") || strings.Contains(code, "context_window") {
		return true
	}
	msg := strings.ToLower(e.Message)
	for _, needle := range []string{
		"context length",
		"context window",
		"prompt is too long",
		"too many tokens",
		"maximum context",
		"exceeds the maximum number of tokens",
		"input token count",
	} {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}

// Unsupported builds an ErrUnsupported error naming the provider and feature.
func Unsupported(provider, what string) error {
	return fmt.Errorf("%s: %s: %w", provider, what, ErrUnsupported)
}
