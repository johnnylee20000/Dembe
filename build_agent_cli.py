"""CLI for building an AI agent blueprint from task context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_agent import AgentBuilder, EnvironmentInfo


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Input payload must be a JSON object")
    return payload


def build_blueprint(payload: dict[str, Any]) -> str:
    user_info = payload.get("user_info")
    user_query = payload.get("user_query", "")
    if not isinstance(user_info, dict):
        raise ValueError("Payload requires an object field named 'user_info'")
    if not isinstance(user_query, str):
        raise ValueError("Payload field 'user_query' must be a string")

    env = EnvironmentInfo.from_mapping(user_info)
    builder = AgentBuilder(env)
    blueprint = builder.build(user_query)
    return blueprint.to_json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an AI agent blueprint from user context."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON payload containing user_info and user_query.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. If omitted, writes to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    payload = load_payload(input_path)
    result = build_blueprint(payload)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
