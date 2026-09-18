package vertexgrpc

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/richardwooding/llmkit/core"
)

func (c *Client) response(resp *aiplatformpb.GenerateContentResponse) (*core.Response, error) {
	out := &core.Response{ID: resp.GetResponseId(), Model: c.model, Message: core.Message{Role: core.RoleAssistant}}
	if v := resp.GetModelVersion(); v != "" {
		out.Model = v
	}
	if raw, err := protojson.Marshal(resp); err == nil {
		out.Raw = raw
	}
	out.Usage = usage(resp.GetUsageMetadata())
	if len(resp.GetCandidates()) == 0 {
		out.FinishReason = core.FinishOther
		if resp.GetPromptFeedback().GetBlockReason() != 0 {
			out.FinishReason = core.FinishContentFilter
		}
		return out, nil
	}
	cand := resp.GetCandidates()[0]
	var calls int
	var err error
	if out.Message.Parts, calls, err = messageParts(cand.GetContent().GetParts(), 0); err != nil {
		return nil, err
	}
	out.FinishReason = finishReason(cand.GetFinishReason(), calls > 0)
	return out, nil
}

// messageParts converts candidate parts, numbering tool calls from nextCall.
func messageParts(in []*aiplatformpb.Part, nextCall int) ([]core.Part, int, error) {
	out := make([]core.Part, 0, len(in))
	calls := 0
	for _, p := range in {
		if sig := p.GetThoughtSignature(); len(sig) > 0 && !p.GetThought() {
			out = append(out, core.ReasoningPart{Signature: base64.StdEncoding.EncodeToString(sig)})
		}
		switch {
		case p.GetThought():
			out = append(out, core.ReasoningPart{Text: p.GetText(), Signature: base64.StdEncoding.EncodeToString(p.GetThoughtSignature())})
		case p.GetFunctionCall() != nil:
			args, err := protojson.Marshal(p.GetFunctionCall().GetArgs())
			if err != nil {
				return nil, 0, fmt.Errorf("%s: function call args: %w", ID, err)
			}
			calls++
			out = append(out, core.ToolCall{ID: callID(nextCall + calls), Name: p.GetFunctionCall().GetName(), Arguments: json.RawMessage(args)})
		case p.GetText() != "":
			out = append(out, core.TextPart{Text: p.GetText()})
		}
	}
	return out, calls, nil
}

func callID(n int) string { return fmt.Sprintf("call_%d", n) }

func finishReason(r aiplatformpb.Candidate_FinishReason, hasCalls bool) core.FinishReason {
	switch r {
	case aiplatformpb.Candidate_STOP, aiplatformpb.Candidate_FINISH_REASON_UNSPECIFIED:
		if hasCalls {
			return core.FinishToolCalls
		}
		return core.FinishStop
	case aiplatformpb.Candidate_MAX_TOKENS:
		return core.FinishLength
	case aiplatformpb.Candidate_SAFETY, aiplatformpb.Candidate_RECITATION, aiplatformpb.Candidate_BLOCKLIST,
		aiplatformpb.Candidate_PROHIBITED_CONTENT, aiplatformpb.Candidate_SPII, aiplatformpb.Candidate_MODEL_ARMOR:
		return core.FinishContentFilter
	default:
		return core.FinishOther
	}
}

func usage(u *aiplatformpb.GenerateContentResponse_UsageMetadata) core.Usage {
	return core.Usage{
		InputTokens:       int(u.GetPromptTokenCount()),
		OutputTokens:      int(u.GetCandidatesTokenCount()),
		TotalTokens:       int(u.GetTotalTokenCount()),
		CachedInputTokens: int(u.GetCachedContentTokenCount()),
		ReasoningTokens:   int(u.GetThoughtsTokenCount()),
	}
}

var httpStatus = map[codes.Code]int{
	codes.InvalidArgument:    http.StatusBadRequest,
	codes.FailedPrecondition: http.StatusBadRequest,
	codes.OutOfRange:         http.StatusBadRequest,
	codes.Unauthenticated:    http.StatusUnauthorized,
	codes.PermissionDenied:   http.StatusForbidden,
	codes.NotFound:           http.StatusNotFound,
	codes.AlreadyExists:      http.StatusConflict,
	codes.Aborted:            http.StatusConflict,
	codes.ResourceExhausted:  http.StatusTooManyRequests,
	codes.Unimplemented:      http.StatusNotImplemented,
	codes.Unavailable:        http.StatusServiceUnavailable,
	codes.DeadlineExceeded:   http.StatusGatewayTimeout,
}

func apiError(err error) error {
	st, ok := status.FromError(err)
	if !ok || st.Code() == codes.Canceled {
		return fmt.Errorf("%s: %w", ID, err)
	}
	code := st.Code()
	httpCode, known := httpStatus[code]
	if !known {
		httpCode = http.StatusInternalServerError
	}
	return &core.APIError{Provider: ID, Status: httpCode, Code: code.String(), Message: st.Message()}
}
