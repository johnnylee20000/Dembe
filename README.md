# Dembe

Build a task-focused AI agent blueprint from runtime context.

## What this project includes

- `ai_agent.py`: core data model and agent blueprint builder.
- `build_agent_cli.py`: command line entry point for generating blueprints.
- `sample_task.json`: example input based on the provided environment and request.

## Input format

The CLI expects a JSON object with:

- `user_info` (object):
  - `OS Version`
  - `Shell`
  - `Workspace Path`
  - `Today's date`
  - optional: `Is directory a git repo`, `Git repo`, `Terminals folder`
- `user_query` (string): objective for the generated agent.

## Usage

Generate to stdout:

```bash
python3 build_agent_cli.py --input sample_task.json
```

Write to a file:

```bash
python3 build_agent_cli.py --input sample_task.json --output agent_blueprint.json
```

The output includes:

- startup checks tailored to the environment
- execution plan steps
- a reusable system prompt for autonomous execution

## English-law legal AI benchmarking artifact

This repo now also contains:

- `legal_ai_english_law_report.md`

It provides:

- a top-5 list of legal AI platforms relevant to English-law workflows
- a feature matrix showing the shared product structure
- analysis of how each platform chunks legal information
- a reusable chunk schema and pipeline you can apply to your own agent
