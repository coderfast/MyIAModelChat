## 🧪 API Compatibility (Ollama/OpenAI style)

This project exposes endpoints compatible with Ollama/OpenAI v1-style clients.

Examples (assumes server running on `http://127.0.0.1:11434`):

- Create completion (v1/completions):

```bash
curl.exe -X POST "http://127.0.0.1:11434/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hola, ¿puedes responder?","max_tokens":50,"stream":false}'
```

- Chat completion (v1/chat/completions):

```bash
curl.exe -X POST "http://127.0.0.1:11434/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hola, ¿puedes responder?"}],"max_tokens":50,"stream":false}'
```

- Ollama-style endpoint already provided at `/api/chat/completions` (keeps backward compatibility):

```bash
curl.exe -X POST "http://127.0.0.1:11434/api/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hola, ¿puedes responder?"}],"max_tokens":50,"stream":false}'
```

If you want streaming responses set `"stream": true` and consume NDJSON stream chunks from the response.
> Note: in Windows PowerShell, use `curl.exe` instead of the built-in `curl` alias to send raw JSON payloads correctly.
One-line examples:
- Windows CMD:
  `curl.exe -X POST "http://127.0.0.1:11434/v1/chat/completions" -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hola\"}],\"max_tokens\":50,\"stream\":false}"
`
- Linux CLI:
  `curl -X POST http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hola"}],"max_tokens":50,"stream":false}'`
