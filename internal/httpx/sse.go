package httpx

import (
	"bufio"
	"errors"
	"io"
	"iter"
	"strings"
)

// Event is one server-sent event. Data has multi-line payloads joined with
// newlines; comment lines are dropped.
type Event struct {
	Name string
	Data string
	ID   string
}

// SSE parses a text/event-stream body. A "[DONE]" data payload ends the
// sequence. Lines have no length limit.
func SSE(r io.Reader) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		br := bufio.NewReader(r)
		var p sseParser
		for {
			line, err := br.ReadString('\n')
			if err != nil && !errors.Is(err, io.EOF) {
				yield(Event{}, err)
				return
			}
			if !p.line(strings.TrimRight(line, "\r\n"), yield) {
				return
			}
			if errors.Is(err, io.EOF) {
				p.flush(yield)
				return
			}
		}
	}
}

type sseParser struct {
	ev   Event
	data []string
}

func (p *sseParser) line(line string, yield func(Event, error) bool) bool {
	if line == "" {
		return p.flush(yield)
	}
	field, value, _ := strings.Cut(line, ":")
	value = strings.TrimPrefix(value, " ")
	switch field {
	case "event":
		p.ev.Name = value
	case "data":
		p.data = append(p.data, value)
	case "id":
		p.ev.ID = value
	}
	return true
}

// flush emits the buffered event; it returns false when iteration must stop,
// either because the consumer broke out or a [DONE] sentinel arrived.
func (p *sseParser) flush(yield func(Event, error) bool) bool {
	if len(p.data) == 0 && p.ev.Name == "" && p.ev.ID == "" {
		return true
	}
	ev := p.ev
	ev.Data = strings.Join(p.data, "\n")
	p.ev, p.data = Event{}, nil
	if ev.Data == "[DONE]" {
		return false
	}
	return yield(ev, nil)
}
