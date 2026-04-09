from pathlib import Path
import os

from llama_cpp import Llama
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from domain import CompletionRequest, CompletionResponse, CompletionChunk

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getcwd()) / "models/gemma-4-E2B-it/gemma-4-E2B-it-BF16.gguf"
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=131072,
    n_gpu_layers=-1,
    verbose=True,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LLM Server")

@app.post("/v1/completions", response_model=CompletionResponse)
def complete(req: CompletionRequest) -> CompletionResponse:
    raw = llm.create_completion(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=req.stop,
    )
    choice = raw["choices"][0]
    usage  = raw["usage"]
    return CompletionResponse(
        text=choice["text"],
        finish_reason=choice["finish_reason"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
    )

@app.post("/v1/completions/stream")
def complete_stream(req: CompletionRequest):
    def _generate() -> Iterator[str]:
        raw_iter = llm.create_completion(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=req.stop,
            stream=True,
        )
        for raw in raw_iter:
            choice = raw["choices"][0]
            chunk = CompletionChunk(
                text=choice["text"],
                finish_reason=choice["finish_reason"],
                index=choice["index"],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
