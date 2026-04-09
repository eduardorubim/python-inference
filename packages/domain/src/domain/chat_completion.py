from typing import List
from pydantic import BaseModel, Field

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


class CompletionChunk(BaseModel):
    text: str = Field(
        ...,
        description="Partial token text emitted during streaming."
    )
    finish_reason: str | None = Field(
        None,
        description="None while streaming; 'stop' or 'length' on the final chunk."
    )
    index: int = Field(
        0,
        description="Choice index (always 0 for single completion)."
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