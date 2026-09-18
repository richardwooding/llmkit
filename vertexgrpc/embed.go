package vertexgrpc

import (
	"context"
	"fmt"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/richardwooding/llmkit/core"
)

// Embed calls Predict on a text-embedding model such as "text-embedding-005".
func (c *Client) Embed(ctx context.Context, req *core.EmbedRequest) (*core.EmbedResponse, error) {
	if req == nil || len(req.Inputs) == 0 {
		return nil, fmt.Errorf("%s: no inputs", ID)
	}
	preq, err := c.predictRequest(req)
	if err != nil {
		return nil, err
	}
	pc, err := c.client()
	if err != nil {
		return nil, err
	}
	ctx, cancel := c.context(ctx)
	defer cancel()
	resp, err := pc.Predict(ctx, preq)
	if err != nil {
		return nil, apiError(err)
	}
	return c.embedResponse(resp)
}

func (c *Client) predictRequest(req *core.EmbedRequest) (*aiplatformpb.PredictRequest, error) {
	taskType := map[core.EmbedInputType]string{core.EmbedQuery: "RETRIEVAL_QUERY", core.EmbedDocument: "RETRIEVAL_DOCUMENT"}[req.InputType]
	instances := make([]*structpb.Value, 0, len(req.Inputs))
	for _, in := range req.Inputs {
		fields := map[string]any{"content": in}
		if taskType != "" {
			fields["task_type"] = taskType
		}
		v, err := structpb.NewValue(fields)
		if err != nil {
			return nil, fmt.Errorf("%s: embed instance: %w", ID, err)
		}
		instances = append(instances, v)
	}
	params := map[string]any{"autoTruncate": true}
	if req.Dimensions > 0 {
		params["outputDimensionality"] = req.Dimensions
	}
	pv, err := structpb.NewValue(params)
	if err != nil {
		return nil, fmt.Errorf("%s: embed parameters: %w", ID, err)
	}
	return &aiplatformpb.PredictRequest{Endpoint: c.modelName(), Instances: instances, Parameters: pv}, nil
}

func (c *Client) embedResponse(resp *aiplatformpb.PredictResponse) (*core.EmbedResponse, error) {
	out := &core.EmbedResponse{Model: c.model, Embeddings: make([][]float32, 0, len(resp.GetPredictions()))}
	if raw, err := protojson.Marshal(resp); err == nil {
		out.Raw = raw
	}
	for i, p := range resp.GetPredictions() {
		emb := p.GetStructValue().GetFields()["embeddings"].GetStructValue()
		values := emb.GetFields()["values"].GetListValue().GetValues()
		if values == nil {
			return nil, fmt.Errorf("%s: prediction %d has no embeddings.values", ID, i)
		}
		vec := make([]float32, len(values))
		for j, v := range values {
			vec[j] = float32(v.GetNumberValue())
		}
		out.Embeddings = append(out.Embeddings, vec)
		tokens := int(emb.GetFields()["statistics"].GetStructValue().GetFields()["token_count"].GetNumberValue())
		out.Usage.InputTokens += tokens
		out.Usage.TotalTokens += tokens
	}
	return out, nil
}
