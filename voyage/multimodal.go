package voyage

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/richardwooding/llmkit/core"
	"github.com/richardwooding/llmkit/internal/httpx"
)

type multimodalRequest struct {
	Model      string            `json:"model"`
	Inputs     []multimodalInput `json:"inputs"`
	InputType  string            `json:"input_type,omitempty"`
	Truncation bool              `json:"truncation"`
}

type multimodalInput struct {
	Content []multimodalContent `json:"content"`
}

type multimodalContent struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	ImageBase64 string `json:"image_base64,omitempty"`
	ImageURL    string `json:"image_url,omitempty"`
	VideoBase64 string `json:"video_base64,omitempty"`
	VideoURL    string `json:"video_url,omitempty"`
}

type multimodalResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TextTokens  int `json:"text_tokens"`
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// EmbedMultimodal calls /multimodalembeddings with text, image and video parts.
func (c *Client) EmbedMultimodal(ctx context.Context, req *core.MultimodalEmbedRequest) (*core.EmbedResponse, error) {
	if req == nil || len(req.Inputs) == 0 {
		return nil, fmt.Errorf("%s: no inputs", ID)
	}
	w := multimodalRequest{Model: c.model, InputType: string(req.InputType), Truncation: true, Inputs: make([]multimodalInput, 0, len(req.Inputs))}
	for _, parts := range req.Inputs {
		in, err := multimodalParts(parts)
		if err != nil {
			return nil, err
		}
		w.Inputs = append(w.Inputs, in)
	}
	body, err := httpx.MarshalWithExtra(w, req.ProviderExtra(ID))
	if err != nil {
		return nil, err
	}
	var out multimodalResponse
	raw, err := c.http.PostJSON(ctx, "/multimodalembeddings", body, &out)
	if err != nil {
		return nil, err
	}
	sort.Slice(out.Data, func(i, j int) bool { return out.Data[i].Index < out.Data[j].Index })
	resp := &core.EmbedResponse{
		Model: out.Model, Raw: raw, Embeddings: make([][]float32, 0, len(out.Data)),
		Usage: core.Usage{InputTokens: out.Usage.TotalTokens, TotalTokens: out.Usage.TotalTokens},
	}
	for _, d := range out.Data {
		resp.Embeddings = append(resp.Embeddings, d.Embedding)
	}
	if resp.Model == "" {
		resp.Model = c.model
	}
	return resp, nil
}

func multimodalParts(parts []core.Part) (multimodalInput, error) {
	in := multimodalInput{Content: make([]multimodalContent, 0, len(parts))}
	for _, p := range parts {
		switch v := p.(type) {
		case core.TextPart:
			in.Content = append(in.Content, multimodalContent{Type: "text", Text: v.Text})
		case core.ImagePart:
			if v.URL != "" {
				in.Content = append(in.Content, multimodalContent{Type: "image_url", ImageURL: v.URL})
			} else {
				in.Content = append(in.Content, multimodalContent{Type: "image_base64", ImageBase64: httpx.DataURI(v.MIME, v.Data)})
			}
		case core.FilePart:
			if !strings.HasPrefix(httpx.MIMEOr(v.MIME, v.Data), "video/") {
				return in, core.Unsupported(ID, "file input other than video")
			}
			if v.URL != "" {
				in.Content = append(in.Content, multimodalContent{Type: "video_url", VideoURL: v.URL})
			} else {
				in.Content = append(in.Content, multimodalContent{Type: "video_base64", VideoBase64: httpx.DataURI(v.MIME, v.Data)})
			}
		default:
			return in, core.Unsupported(ID, fmt.Sprintf("%T in multimodal embedding", p))
		}
	}
	return in, nil
}
