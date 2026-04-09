from pathlib import Path
import os
from pydantic import BaseModel, Field
from typing import List

from llama_cpp import Llama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
# Schemas
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The input text prompt that the language model will complete or respond to."
    )
    max_tokens: int = Field(
        128,
        ge=1,
        description="The maximum number of tokens to generate in the completion. Must be at least 1."
    )
    temperature: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="Controls randomness in the output. Lower values make responses more deterministic; higher values increase creativity."
    )
    stop: List[str] = Field(
        default_factory=list,
        description="A list of stop sequences where the model will stop generating further tokens."
    )


class CompletionResponse(BaseModel):
    text: str = Field(
        ...,
        description="The generated completion text produced by the language model."
    )
    finish_reason: str = Field(
        ...,
        description="The reason why the generation stopped (e.g., 'stop', 'length', or other conditions)."
    )
    prompt_tokens: int = Field(
        ...,
        description="The number of tokens consumed by the input prompt."
    )
    completion_tokens: int = Field(
        ...,
        description="The number of tokens generated in the completion."
    )
    total_tokens: int = Field(
        ...,
        description="The total number of tokens used (prompt_tokens + completion_tokens)."
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

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
