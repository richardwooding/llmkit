package httpx

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"iter"
)

// NDJSON yields each non-empty line of a newline-delimited JSON body.
func NDJSON(r io.Reader) iter.Seq2[json.RawMessage, error] {
	return func(yield func(json.RawMessage, error) bool) {
		br := bufio.NewReader(r)
		for {
			line, err := br.ReadBytes('\n')
			if err != nil && !errors.Is(err, io.EOF) {
				yield(nil, err)
				return
			}
			line = bytes.TrimSpace(line)
			if len(line) > 0 && !yield(json.RawMessage(line), nil) {
				return
			}
			if errors.Is(err, io.EOF) {
				return
			}
		}
	}
}
