package httpx

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
)

// MarshalWithExtra encodes body and overlays extra onto its top-level keys.
// With no extra it is a plain json.Marshal.
func MarshalWithExtra(body any, extra map[string]any) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	if len(extra) == 0 {
		return raw, nil
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("merge extra: %w", err)
	}
	maps.Copy(m, extra)
	return json.Marshal(m)
}

// DataURI encodes data as a base64 data: URI with the given MIME type.
func DataURI(mime string, data []byte) string {
	return "data:" + MIMEOr(mime, data) + ";base64," + base64.StdEncoding.EncodeToString(data)
}

// MIMEOr returns mime, or sniffs one from data when mime is empty.
func MIMEOr(mime string, data []byte) string {
	if mime != "" {
		return mime
	}
	return http.DetectContentType(data)
}
