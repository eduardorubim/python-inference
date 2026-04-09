"""CLI entry-point — subcommands: completion | stream."""
import argparse
import sys

from domain import CompletionRequest
from cli.request import post_completion, post_stream


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli",
        description="LLM inference CLI (gemma-4 via local server)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- completion (blocking) ---
    p_comp = sub.add_parser("completion", help="Single blocking completion")
    p_comp.add_argument("prompt", help="Input prompt")
    p_comp.add_argument("--max-tokens", type=int, default=128)
    p_comp.add_argument("--temperature", type=float, default=1.0)

    # --- stream (SSE) ---
    p_stream = sub.add_parser("stream", help="Token-by-token streaming completion")
    p_stream.add_argument("prompt", help="Input prompt")
    p_stream.add_argument("--max-tokens", type=int, default=128)
    p_stream.add_argument("--temperature", type=float, default=1.0)

    return parser


def _req_from_args(args: argparse.Namespace) -> CompletionRequest:
    return CompletionRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    req = _req_from_args(args)

    if args.command == "completion":
        response = post_completion(req)
        print(response.text)

    elif args.command == "stream":
        for chunk in post_stream(req):
            print(chunk.text, end="", flush=True)
        print()  # newline after stream ends


if __name__ == "__main__":
    main()