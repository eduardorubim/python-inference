CMAKE_ARGS="-DGGML_CUDA=on" \
  uv add llama-cpp-python \
    --reinstall \
    --no-cache-dir \
    --verbose \
    --package server

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" -d '{"prompt": "The capital of France is", "max_tokens": 32}'