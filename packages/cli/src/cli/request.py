"""HTTP client for the LLM server — stdlib only (http.client)."""
import http.client
import json
from collections.abc import Iterator

from domain import CompletionRequest, CompletionResponse, CompletionChunk

SERVER_HOST = "localhost"
SERVER_PORT = 8000

_HEADERS = {"Content-Type": "application/json"}


def post_completion(req: CompletionRequest) -> CompletionResponse:
    """Blocking POST /v1/completions → validated CompletionResponse."""
    body = req.model_dump_json()
    conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT)
    try:
        conn.request("POST", "/v1/completions", body, _HEADERS)
        raw = json.loads(conn.getresponse().read())
    finally:
        conn.close()
    return CompletionResponse.model_validate(raw)


def post_stream(req: CompletionRequest) -> Iterator[CompletionChunk]:
    """Streaming POST /v1/completions/stream → yields validated CompletionChunks.

    The server emits Server-Sent Events:
        data: {json}\n\n
        data: [DONE]\n\n
    """
    body = req.model_dump_json()

    conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT)
    try:
        conn.request("POST", "/v1/completions/stream", body, _HEADERS)
        resp = conn.getresponse()
        while True:
            line = resp.readline()
            if not line:            # EOF
                break
            line = line.decode().rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                break
            yield CompletionChunk.model_validate_json(payload)
    finally:
        conn.close()
