# Local Ollama context guard

Opt-in ceiling for chat requests Odysseus sends to a configured local Ollama.
It is not a default and it is not `OLLAMA_CONTEXT_LENGTH`.

Ollama 0.40.14 uses `OLLAMA_CONTEXT_LENGTH` and a Modelfile `num_ctx` only when
the request does not set `options.num_ctx`. A native request can replace both.
The OpenAI-compatible `/v1/chat/completions` schema on that build has no
`num_ctx` field, so adding one does not change the loaded window. On a GPU
under 23 GiB the server default when nothing is set is 4096, and Odysseus's
known-model table will ask for 131072 for a `qwen2.5` name on the native API.

## Enable

```text
ODYSSEUS_OLLAMA_MAX_NUM_CTX=2048
```

Unset or blank leaves every provider unchanged. The only enabling value is
`2048`. Any other non-empty value is invalid. Invalid configuration refuses
local Ollama chat before the HTTP call. It does not fall open.

While enforcing, a discovered or requested window above 2048 is sent as
`options.num_ctx` 2048. A smaller positive window is sent as that smaller
value and is not raised. Streaming, non-streaming, and fallback candidates each
build their own request, and each local Ollama candidate is capped. Retries
reuse that capped body.

## Which URLs

The guard applies to:

- Native local Ollama (`http://127.0.0.1:11434`, `.../api`, or another loopback
  port with that shape).
- OpenAI-compatible Ollama on port **11434** (`http://127.0.0.1:11434/v1`).
  Those calls are posted to `/api/chat` on the same host, because that is the
  dialect that actually honors `options.num_ctx`.
- Any extra origin listed in `ODYSSEUS_OLLAMA_CONTEXT_BASES` (comma-separated
  absolute `http` or `https` URLs). A non-11434 `/v1` base is not assumed to
  be Ollama.

It does not apply to Ollama Cloud (`ollama.com`) or to other loopback OpenAI
servers such as LM Studio on port 1234. Those keep their existing request
shape.

Admin model probe, gallery auto-tag, and the email-task classifier use the
same cap. They post native `/api/chat` with `num_ctx` at most 2048, or they
fail before any completion request when the setting is invalid. The legacy
`llm_call_with_fallback` and `llm_call_async_with_fallback` helpers do not
send the prompt to another candidate after that configuration rejection.

`/v1` matching ignores path case, so `http://127.0.0.1:11434/V1` is covered.
A non-11434 `/v1` base is still not assumed to be Ollama unless it is listed
in `ODYSSEUS_OLLAMA_CONTEXT_BASES`. A malformed entry in that list refuses
loopback calls; it does not by itself stop an unrelated non-loopback URL.

Prompt trimming still uses the model's advertised window from
`get_context_length`. The loaded context is the `num_ctx` on the request.
A long transcript can be rejected or truncated by Ollama; it must not cause
a larger loaded window. This is not a VRAM limiter.

This does not sandbox the process or change tool permissions. The running
Ollama still has to accept the native `num_ctx` field. Confirm that on the
target machine before pointing Odysseus at a GPU model.
