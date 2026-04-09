## How to install

```
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sfL https://direnv.net/install.sh | bash
direnv allow
```

if not using direnv rememeber to set CUDA Toolkit's NVCC for build vars:
```
CMAKE_ARGS="-DGGML_CUDA=on" \
  uv add llama-cpp-python \
    --reinstall \
    --no-cache-dir \
    --verbose \
    --package server
```

test server with no cli:
```bash
uv run server
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" -d '{"prompt": "The capital of France is", "max_tokens": 32}'
```

pytest:
```bash
uv run pytest
uv run --package cli pytest packages/cli/tests
uv run --package server pytest packages/server/tests
```