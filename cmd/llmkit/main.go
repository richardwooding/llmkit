// Command llmkit is a small CLI over the llmkit library: chat with any
// supported model, stream the reply, or embed text.
//
//	llmkit chat -m claude-sonnet-4-5 "Explain iter.Seq2"
//	echo "prompt" | llmkit chat -m llama3.2 -stream
//	llmkit embed -m voyage-3-large "first" "second"
//	llmkit resolve gpt-5 openrouter/openai/gpt-4o
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/richardwooding/llmkit"
)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "llmkit:", err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout io.Writer) error {
	if len(args) == 0 {
		return usage()
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	switch args[0] {
	case "chat":
		return chat(ctx, args[1:], stdin, stdout)
	case "embed":
		return embed(ctx, args[1:], stdin, stdout)
	case "resolve":
		return resolve(args[1:], stdout)
	case "-h", "--help", "help":
		return usage()
	default:
		return fmt.Errorf("unknown command %q\n\n%w", args[0], usage())
	}
}

func usage() error {
	return errors.New(strings.TrimSpace(`
usage:
  llmkit chat    -m <model> [-system text] [-stream] [-max-tokens n] [-temperature t] [-json] [prompt...]
  llmkit embed   -m <model> [-json] [text...]
  llmkit resolve <model>...

Prompts and texts default to stdin when no arguments are given.
Model names are "<model>" or "<provider>/<model>"; keys come from the usual environment variables.`))
}

func chat(ctx context.Context, args []string, stdin io.Reader, stdout io.Writer) error {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	model := fs.String("m", "", "model name")
	system := fs.String("system", "", "system prompt")
	stream := fs.Bool("stream", false, "stream the reply")
	maxTokens := fs.Int("max-tokens", 0, "maximum output tokens")
	temperature := fs.Float64("temperature", -1, "sampling temperature")
	asJSON := fs.Bool("json", false, "print the full response as JSON")
	timeout := fs.Duration("timeout", 5*time.Minute, "request timeout")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *model == "" {
		return errors.New("chat: -m <model> is required")
	}
	prompt, err := readInput(fs.Args(), stdin, " ")
	if err != nil {
		return err
	}
	req := &llmkit.Request{Messages: []llmkit.Message{llmkit.UserText(prompt)}, MaxTokens: *maxTokens}
	if *system != "" {
		req.Messages = append([]llmkit.Message{llmkit.System(*system)}, req.Messages...)
	}
	if *temperature >= 0 {
		req.Temperature = new(*temperature)
	}
	opts := []llmkit.Option{llmkit.WithTimeout(*timeout)}
	if *stream && !*asJSON {
		s, err := llmkit.Open[llmkit.Streamer](*model, opts...)
		if err != nil {
			return err
		}
		return streamTo(ctx, s, req, stdout)
	}
	c, err := llmkit.Open[llmkit.Chatter](*model, opts...)
	if err != nil {
		return err
	}
	resp, err := c.Chat(ctx, req)
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(stdout, resp)
	}
	_, err = fmt.Fprintln(stdout, resp.Text())
	return err
}

func streamTo(ctx context.Context, s llmkit.Streamer, req *llmkit.Request, stdout io.Writer) error {
	for chunk, err := range s.Stream(ctx, req) {
		if err != nil {
			return err
		}
		if chunk.Kind == llmkit.ChunkText {
			if _, err := io.WriteString(stdout, chunk.Text); err != nil {
				return err
			}
		}
	}
	_, err := fmt.Fprintln(stdout)
	return err
}

func embed(ctx context.Context, args []string, stdin io.Reader, stdout io.Writer) error {
	fs := flag.NewFlagSet("embed", flag.ContinueOnError)
	model := fs.String("m", "", "model name")
	asJSON := fs.Bool("json", false, "print the full response as JSON")
	timeout := fs.Duration("timeout", 2*time.Minute, "request timeout")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *model == "" {
		return errors.New("embed: -m <model> is required")
	}
	inputs := fs.Args()
	if len(inputs) == 0 {
		text, err := readInput(nil, stdin, "")
		if err != nil {
			return err
		}
		inputs = strings.Split(strings.TrimRight(text, "\n"), "\n")
	}
	e, err := llmkit.Open[llmkit.Embedder](*model, llmkit.WithTimeout(*timeout))
	if err != nil {
		return err
	}
	resp, err := e.Embed(ctx, &llmkit.EmbedRequest{Inputs: inputs})
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(stdout, resp)
	}
	for _, v := range resp.Embeddings {
		line, _ := json.Marshal(v)
		if _, err := fmt.Fprintln(stdout, string(line)); err != nil {
			return err
		}
	}
	return nil
}

func resolve(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return errors.New("resolve: at least one model name is required")
	}
	for _, name := range args {
		p, model, err := llmkit.ParseModel(name)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(stdout, "%s\t%s\t%s\n", name, p.ID(), model); err != nil {
			return err
		}
	}
	return nil
}

func readInput(args []string, stdin io.Reader, sep string) (string, error) {
	if len(args) > 0 {
		return strings.Join(args, sep), nil
	}
	b, err := io.ReadAll(stdin)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(string(b)) == "" {
		return "", errors.New("no input: pass text as arguments or on stdin")
	}
	return string(b), nil
}

func printJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
