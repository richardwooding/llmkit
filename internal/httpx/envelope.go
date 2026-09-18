package httpx

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/richardwooding/llmkit/core"
)

type errorEnvelope struct {
	Error   json.RawMessage `json:"error"`
	Message string          `json:"message"`
	Detail  json.RawMessage `json:"detail"`
	Type    string          `json:"type"`
	Code    json.RawMessage `json:"code"`
}

type errorObject struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Code    json.RawMessage `json:"code"`
	Status  string          `json:"status"`
	Param   string          `json:"param"`
}

// ParseError decodes the error envelopes used by the supported providers into
// a *core.APIError. Unknown shapes fall back to the raw body as the message.
func ParseError(provider string, status int, hdr http.Header, body []byte) *core.APIError {
	e := &core.APIError{Provider: provider, Status: status, Body: body}
	if hdr != nil {
		e.RetryAfter = retryAfter(hdr.Get("Retry-After"))
	}
	fillFromBody(e, body)
	if e.Message == "" {
		e.Message = strings.TrimSpace(string(body))
		if len(e.Message) > 512 {
			e.Message = e.Message[:512]
		}
	}
	return e
}

func fillFromBody(e *core.APIError, body []byte) {
	var env errorEnvelope
	if json.Unmarshal(body, &env) != nil {
		return
	}
	e.Type = env.Type
	e.Code = rawString(env.Code)
	if env.Message != "" {
		e.Message = env.Message
	}
	if len(env.Detail) > 0 {
		var s string
		if json.Unmarshal(env.Detail, &s) == nil {
			e.Message = s
		}
	}
	if len(env.Error) == 0 {
		return
	}
	var s string
	if json.Unmarshal(env.Error, &s) == nil {
		e.Message = s
		return
	}
	var obj errorObject
	if json.Unmarshal(env.Error, &obj) != nil {
		return
	}
	if obj.Message != "" {
		e.Message = obj.Message
	}
	if obj.Type != "" {
		e.Type = obj.Type
	}
	if code := rawString(obj.Code); code != "" {
		e.Code = code
	} else if obj.Status != "" {
		e.Code = obj.Status
	}
}

func rawString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var n json.Number
	if json.Unmarshal(raw, &n) == nil {
		return n.String()
	}
	return ""
}

func retryAfter(v string) time.Duration {
	if v == "" {
		return 0
	}
	if secs, err := strconv.ParseFloat(v, 64); err == nil {
		return time.Duration(secs * float64(time.Second))
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}
