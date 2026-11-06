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
page does not cancel a request that is already in flight. Cancel in another
process does not clear a step that is still running.

Saved model text is untrusted text. It is not executed, and a mocked critic
reply is not a reviewed judgement.

## What it will call

The pinned endpoint must be `http` on numeric loopback port `11434`, with an
Ollama-shaped path such as `/v1`. The hostname `localhost` is stored as
`127.0.0.1` and is not resolved. `::1` is accepted as written. The intended
model allowlist is `qwen2.5:7b-instruct`. On every create and run the server
checks three gates together: the workflow allowlist, the signed-in account's
current model list (`block_all_models` and `allowed_models`), and the
endpoint. The endpoint must still exist, be enabled, belong to that user or
have no owner, and still be that numeric URL. Otherwise the step fails and
no completion is sent. An endpoint owned by another account stays hidden.
This workflow does not mark endpoints shared.

A guarded `/v1` base is posted to numeric loopback `/api/chat` with
`options.num_ctx` 2048 and `options.num_predict` 512. The post does not use
`HTTP_PROXY`, `ALL_PROXY`, or their lowercase forms, and it does not follow
redirects. There is no fallback model, endpoint, or provider.
`ODYSSEUS_OLLAMA_CONTEXT_BASES` does not add hosts or ports for this workflow.

The prompt must fit 1536 input tokens beside the 512-token completion. Each
message counts as its UTF-8 byte length plus 8 tokens of framing. Newlines,
emoji, and dense text are counted by encoded size. Saved text is not
shortened to make it fit. If it does not fit, the step fails before the call.
No tokenizer or embedding model is downloaded. A real tokenizer may have
accepted some prompts this bound rejects. That is intentional. There is no
setting that raises 2048 or 512.

One model call runs at a time for a shared SQLite file. The lock is a file
beside that database (`<database>.role-workflow.lock`), held from the moment
a step is accepted until its result is saved, and the operating system
releases it when the process exits. Another process that shares the file gets
a conflict on Run and on Recover while the lock is held, and it does not
clear the active step or start a second call. An in-memory database cannot
be shared across processes; only threads in that one process are serialized.
This does not coordinate two database servers. A reply that was generated
and never saved can still be generated again after an explicit Recover.

## What it will not call

The runner does not call the agent loop, pipelines, session send, groups, or
the scheduler. It does not enable tools, files, shell, MCP, web, Chroma, or
embeddings. It does not grant `can_use_agent` or admin. Project text is stored
and shown as text.

Owner on every record is the authenticated user. A client-supplied owner is
ignored.
