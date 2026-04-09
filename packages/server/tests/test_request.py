"""Tests for server endpoints

Para para evitar invocar o construtor do modelo do outro lado (`llama_cpp.Llama`)
injetamos no em cada handler do servidor o `Depends(get_llm)` 
Assim, o TestClient do FastAPI age como um middleware no `get_llm` usando `app.dependency_overrides` 
para injetar o MagicMock substituto antes de despachar qualquer request 
"""
import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from server.main import app, get_llm
from domain import CompletionChunk


def _make_raw_chunk(text: str, finish_reason: str | None = None) -> dict:
    """Reproduz o shape exato que llama-cpp retorna por chunk no modo stream.

    Cada item do iterador de `create_completion(stream=True)` tem essa
    estrutura. Manter o shape real aqui garante que o handler vai parsear
    os campos nos lugares certos, mesmo que a lib mude internamente.
    """
    return {"choices": [{"text": text, "finish_reason": finish_reason, "index": 0}]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """MagicMock puro aceita qualquer chamada.
    """
    return MagicMock()


@pytest.fixture
def client(mock_llm):
    """TestClient com o modelo substituído via dependency_overrides.

    `app.dependency_overrides` é um dict do FastAPI: 
        chave: função original (`get_llm`) de dependência 
        valor: é um callable que retorna o substituto. 
    O `lambda: mock_llm` garante que cada request receba uma instância idêntica

    O `yield` separa setup de teardown: `clear()` remove o override após
    cada teste, evitando que um teste contamine o próximo.
    """
    app.dependency_overrides[get_llm] = lambda: mock_llm
    yield TestClient(app, raise_server_exceptions=True)
    app.dependency_overrides.clear()  # teardown a cada teste isolado


# ---------------------------------------------------------------------------
# POST /v1/completions  (blocking)
# ---------------------------------------------------------------------------

def test_completion_returns_200_with_valid_body(client, mock_llm):
    """Verifica o contrato completo do endpoint de completion.

    O mock retorna o shape bruto do llama-cpp (com `choices` e `usage`).
    Estamos testando o handler sem o modelo:
      1. chama `create_completion` com os parâmetros do request
      2. extrai os campos certos do dict bruto
      3. os serializa para o schema `CompletionResponse`
      4. devolve status 200 com o JSON correto
    
    TODO DEPENDS ON `CompletionResponse`
    """
    mock_llm.create_completion.return_value = {
        "choices": [{"text": "Paris", "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    resp = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4})

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "Paris"
    assert body["finish_reason"] == "stop"
    assert body["total_tokens"] == 8


# ---------------------------------------------------------------------------
# POST /v1/completions/stream  (SSE)
# ---------------------------------------------------------------------------

def test_stream_returns_event_stream_content_type(client, mock_llm):
    """Garante que o endpoint anuncia SSE no Content-Type.

    Teste trivial: streamings não usam o `application/jsonl` tradicional
    """
    mock_llm.create_completion.return_value = iter([
        _make_raw_chunk("hi", finish_reason="stop"),
    ])
    resp = client.post("/v1/completions/stream", json={"prompt": "hi", "max_tokens": 4})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_stream_yields_valid_chunks_then_done(client, mock_llm):
    """Valida o protocolo SSE ponta a ponta: formato, schema e sentinel.

    O body de uma SSE stream é texto puro com linhas no formato:
        data: {json}\n\n
        data: [DONE]\n\n

    Parseamos as linhas exatamente como um cliente real faria (filtrando
    pelo prefixo `data: ` e ignorando linhas vazias) e então validamos
    cada payload contra `CompletionChunk` via Pydantic.

    Isso testa:
      - que o handler itera todos os chunks do llama-cpp
      - que cada chunk é serializado no formato correto
      - que `finish_reason` é propagado no chunk final
      - que `[DONE]` aparece após todos os chunks de dados
    """
    mock_llm.create_completion.return_value = iter([
        _make_raw_chunk("hel"),
        _make_raw_chunk("lo", finish_reason="stop"),
    ])
    resp = client.post("/v1/completions/stream", json={"prompt": "hi", "max_tokens": 4})

    # extrai apenas as linhas de dados, removendo o prefixo SSE
    data_lines = [
        line.removeprefix("data: ")
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]

    assert data_lines[-1] == "[DONE]"  # sentinel sempre no final
    chunks = [CompletionChunk.model_validate_json(d) for d in data_lines[:-1]]
    assert chunks[0].text == "hel"
    assert chunks[1].finish_reason == "stop"


def test_stream_ends_with_done_sentinel(client, mock_llm):
    """Garante que a stream sempre termina com o sentinel [DONE].

    É pelo sentinal [DONE] que o cliente de streaming sabe se a conexão terminou
    normalmente ou foi interrompida. 
    Se o handler lançar uma exceção depois de emitir chunks parciais e o
    [DONE] nunca chegar, o cliente fica bloqueado esperando mais dados.
    TODO timeout tests
    """
    mock_llm.create_completion.return_value = iter([
        _make_raw_chunk("hi", finish_reason="stop"),
    ])
    resp = client.post("/v1/completions/stream", json={"prompt": "hi", "max_tokens": 4})

    assert resp.text.strip().endswith("data: [DONE]")
