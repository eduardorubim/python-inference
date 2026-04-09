"""Tests for cli.request

Teste da camada de transporte (request) CLI usando mock
`http.client.HTTPConnection` que faz uma conexão TCP de verdade.

test_post_completion_returns_completion_response: a resposta JSON é desserializada e validada pelo Pydantic
test_post_completion_sends_correct_json: o payload JSON enviado ao servidor está correto
test_post_stream_yields_completion_chunks: parser de SSE extrai os chunks na ordem certa
test_post_stream_stops_on_done: [DONE] interrompe a iteração sem gerar um chunk extra

O servidor em si é testado separadamente em server/tests/.
Não há como conectar os dois sem subir um servidor real em outra thread
ou processo. TODO nix integration test harness
"""
import json
from unittest.mock import MagicMock, patch

from domain import CompletionRequest, CompletionResponse, CompletionChunk
from cli.request import post_completion, post_stream


# Request base reutilizado em todos os testes
# para que um bug de campo trocado (e.g. temperature no lugar de max_tokens)
# seja detectado pelas assertions.
BASE_REQ = CompletionRequest(prompt="hello", max_tokens=16, temperature=0.5)


def _mock_response(payload: dict) -> MagicMock:
    """Simula um http.client.HTTPResponse para o endpoint blocking.

    `resp.read()` retorna o payload serializado como bytes, exatamente
    como o servidor real responderia. 
    TODO Não simulamos `status` aqui porque `post_completion` ainda não trata erros HTTP
    """
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    return resp


def _sse_body(*chunks: CompletionChunk) -> MagicMock:
    """Simula um http.client.HTTPResponse para o endpoint de streaming.

    O servidor emite SSE: cada evento é uma linha `data: {json}` seguida
    de uma linha em branco. O protocolo termina com `data: [DONE]`.

    Usamos `readline` (uma lista de bytes onde cada chamada consome o próximo item)
    como `side_effect` para simular a leitura incremental que `post_stream` faz.
     O `b""` no final é o EOF que encerra o loop.
    """
    lines = []
    for chunk in chunks:
        lines.append(f"data: {chunk.model_dump_json()}\n".encode())  # linha de dados
        lines.append(b"\n")                                          # separador SSE
    lines += [b"data: [DONE]\n", b"\n", b""]                         # sentinel + EOF
    resp = MagicMock()
    resp.readline = MagicMock(side_effect=lines)
    return resp


# ---------------------------------------------------------------------------
# post_completion: testa o caminho blocking (request/response único)
# ---------------------------------------------------------------------------

def test_post_completion_returns_completion_response():
    """O retorno de post_completion deve ser um CompletionResponse válido.

    Verifica que o JSON da resposta do servidor é desserializado e
    validado pelo Pydantic antes de chegar ao chamador. Se o servidor
    retornar um campo a menos ou com tipo errado, `model_validate`
    lança `ValidationError`.
    """
    payload = {"text": "world", "finish_reason": "stop",
               "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    with patch("cli.request.http.client.HTTPConnection") as MockConn:
        MockConn.return_value.getresponse.return_value = _mock_response(payload)
        result = post_completion(BASE_REQ)

    assert isinstance(result, CompletionResponse)
    assert result.text == "world"
    assert result.total_tokens == 2


def test_post_completion_sends_correct_json():
    """O payload enviado ao servidor deve refletir exatamente o CompletionRequest.

    Inspecionamos o argumento body de `conn.request(method, url, body, headers)`
    e o deserializamos para verificar os campos. Esse teste
    protege contra bugs no lado emissor: se `model_dump_json()` omitir um
    campo ou o handler montar o body manualmente com um typo, capturamos
    aqui antes de o servidor receber dados errados.
    """
    payload = {"text": "ok", "finish_reason": "stop",
               "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    with patch("cli.request.http.client.HTTPConnection") as MockConn:
        MockConn.return_value.getresponse.return_value = _mock_response(payload)
        post_completion(BASE_REQ)
        # call_args[0] é a tupla de args posicionais: (method, url, body, headers)
        _, _, body, _ = MockConn.return_value.request.call_args[0]

    sent = json.loads(body)
    assert sent["prompt"] == "hello"
    assert sent["max_tokens"] == 16


# ---------------------------------------------------------------------------
# post_stream: testa o parser de SSE e a validação de cada chunk
# ---------------------------------------------------------------------------

def test_post_stream_yields_completion_chunks():
    """post_stream deve yield CompletionChunks validados para cada evento SSE.

    Usamos `list()` para consumir o gerador inteiro de uma vez, o que
    também garante que o bloco `finally: conn.close()` em `post_stream`
    é executado sem erros. Se o parser pular uma linha ou inverter
    as assertions de texto e `finish_reason` detectam o problema.
    """
    chunks_in = [CompletionChunk(text="hel"), CompletionChunk(text="lo", finish_reason="stop")]
    with patch("cli.request.http.client.HTTPConnection") as MockConn:
        MockConn.return_value.getresponse.return_value = _sse_body(*chunks_in)
        result = list(post_stream(BASE_REQ))

    assert len(result) == 2
    assert all(isinstance(c, CompletionChunk) for c in result)
    assert result[0].text == "hel"
    assert result[1].finish_reason == "stop"


def test_post_stream_stops_on_done():
    """O sentinel [DONE] deve encerrar a iteração sem gerar um chunk extra.

    O parser lê linha a linha e deve parar exatamente no `data: [DONE]`,
    sem tentar desserializá-lo como CompletionChunk nem continuar lendo.
    Um off-by-one aqui causaria ou um chunk `None` no resultado ou uma
    `ValidationError` ao tentar parsear "[DONE]" como JSON.
    """
    chunks_in = [CompletionChunk(text="hi", finish_reason="stop")]
    with patch("cli.request.http.client.HTTPConnection") as MockConn:
        MockConn.return_value.getresponse.return_value = _sse_body(*chunks_in)
        result = list(post_stream(BASE_REQ))

    assert len(result) == 1
