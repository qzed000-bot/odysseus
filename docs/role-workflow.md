# Sequential text workflow

Opt-in Writer, then Critic, then Organiser. This is one saved text workflow.
It does not make images, publish, browse, schedule work, or remember chats.

The workflow is off unless `ODYSSEUS_ROLE_WORKFLOW` is `1` after trimming
whitespace. Any other value, including unset, leaves the routes disabled. A
step also refuses to call the model unless `ODYSSEUS_OLLAMA_MAX_NUM_CTX` is
`2048` after trimming whitespace.

## What one click does

`Run next step` runs the first step that is not already completed. That is one
model completion. It does not run the other two roles in the same request.
The server builds the step input from the saved brief and any saved earlier
outputs, then stores the reply before the next step can start. Critic and
Organiser do not run until `Run next step` is pressed again.

A completed step is not repeated. A failed step stays failed until the next
explicit run, which retries that same step. Earlier saved outputs stay in
place.

`Recover held step` is for a step left in `running` after a crash or restart.
It marks that step `interrupted` and does not call the model. The next
`Run next step` tries it once. This is not an exactly-once guarantee: if the
process died after the model replied but before that reply was saved, the
retry sends a new completion.

`Cancel in-flight step` asks the running request in this process to discard an
unsaved reply. It does not delete a reply that was already saved. Closing the
page does not cancel a request that is already in flight. Two Odysseus
processes do not share the in-process lock; run one server.

Saved model text is untrusted text. It is not executed, and a mocked critic
reply is not a reviewed judgement.

## What it will call

The pinned endpoint must be `http` on `127.0.0.1`, `localhost`, or `::1`,
port `11434`, with an Ollama-shaped path such as `/v1`. The intended model
allowlist is `qwen2.5:7b-instruct`. On every run the server checks that the
endpoint still exists, is enabled, belongs to the signed-in user or is shared,
and still has that exact URL and an allowlisted model. Otherwise the step
fails and no completion is sent.

A guarded `/v1` base is posted to loopback `/api/chat` with `options.num_ctx`
2048 and `options.num_predict` 512. Redirects are not followed. There is no
fallback model, endpoint, or provider. `ODYSSEUS_OLLAMA_CONTEXT_BASES` does
not add hosts or ports for this workflow.

The prompt must fit a conservative budget of 1536 input tokens plus the
512-token completion. The bound is the larger of the shared character
estimator and a fixed rule: one token per two Latin characters, and two
tokens per non-ASCII character. Saved text is not shortened to make it fit.
If it does not fit, the step fails before the call.

That bound is not a tokenizer. A character that becomes more tokens than the
rule allows can still exceed a 2048 window. No tokenizer or embedding model
is downloaded to close that gap.

## What it will not call

The runner does not call the agent loop, pipelines, session send, groups, or
the scheduler. It does not enable tools, files, shell, MCP, web, Chroma, or
embeddings. It does not grant `can_use_agent` or admin. Project text is stored
and shown as text.

Owner on every record is the authenticated user. A client-supplied owner is
ignored.
