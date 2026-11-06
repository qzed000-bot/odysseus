# Role-workflow dependency closures

These files are for the isolated role-workflow pilot. They are not a
replacement for `requirements.txt`, and they do not install the embedding,
audio, browser, or MCP stacks.

`requirements-bootstrap.txt` stays the hashed pip and certifi pin. Install it
first with `--require-hashes --only-binary=:all: --no-deps`. The certifi hash
in the two files below is the same hash.

## Safety closure

`requirements-role-safety.txt` is the import set for the socket-blocked safety
harness. The versions are the ones already installed when the 171-test suite
passed, plus `opentelemetry-api==1.45.2`.

`fastapi==0.162.0` imports `opentelemetry-api` on startup. That API package
has no exporter. The SDK and the OTLP exporter are not in this file. Do not
set `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_TRACES_EXPORTER`, or
`OTEL_METRICS_EXPORTER`.

```text
python -m pip install --require-hashes --only-binary=:all: --no-deps -r requirements-bootstrap.txt
python -m pip install --require-hashes --only-binary=:all: --no-deps -r requirements-role-safety.txt
python -m pip check
```

The hashes cover CPython 3.11 Windows x64 wheels and CPython 3.12 manylinux
x64 wheels, plus the shared pure-Python wheels. A platform without one of
those wheels must stop. Do not fall back to a source build.

## App-process closure

`requirements-role-runtime.txt` includes the safety closure and the further
packages a process import reached: `beautifulsoup4`, `soupsieve`, `numpy`,
`pydantic-settings`, `python-dateutil`, `tzdata`, `python-multipart`,
`python-dotenv`, `click`, and `uvicorn`. `numpy==2.4.6` is pinned because
`2.5.4` has no CPython 3.11 Windows wheel. `fastembed`, `chromadb`,
`faster-whisper`, `yt-dlp`, and `mcp` are not in the file. A missing module
after this install is a stop, not a reason to run an unpinned install.

```text
python -m pip install --require-hashes --only-binary=:all: --no-deps -r requirements-bootstrap.txt
python -m pip install --require-hashes --only-binary=:all: --no-deps -r requirements-role-runtime.txt
python -m pip check
```
