"""Build a task-focused AI agent specification from workspace context."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class EnvironmentInfo:
    """Runtime information supplied by the user/session."""

    os_version: str
    shell: str
    workspace_path: str
    is_git_repo: bool
    git_repo_path: str | None
    current_date: str
    terminals_folder: str | None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EnvironmentInfo":
        required = ("OS Version", "Shell", "Workspace Path", "Today's date")
        missing: list[str] = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing required user_info fields: {', '.join(missing)}")

        return cls(
            os_version=str(data["OS Version"]),
            shell=str(data["Shell"]),
            workspace_path=str(data["Workspace Path"]),
            is_git_repo=bool(data.get("Is directory a git repo", False)),
            git_repo_path=str(data.get("Git repo", "") or "") or None,
            current_date=str(data["Today's date"]),
            terminals_folder=str(data.get("Terminals folder", "") or "") or None,
        )

    def validate(self) -> None:
        if not self.workspace_path:
            raise ValueError("Workspace Path cannot be empty")
        if not Path(self.workspace_path).is_absolute():
            raise ValueError("Workspace Path must be absolute")
        if self.git_repo_path and not Path(self.git_repo_path).is_absolute():
            raise ValueError("Git repo path must be absolute when provided")
        if self.terminals_folder and not Path(self.terminals_folder).is_absolute():
            raise ValueError("Terminals folder must be absolute when provided")


@dataclass(frozen=True)
class AgentBlueprint:
    """Structured output for a generated coding agent."""

    agent_name: str
    objective: str
    environment: dict[str, Any]
    startup_checks: list[str]
    execution_plan: list[str]
    system_prompt: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class AgentBuilder:
    """Creates an AI coding agent blueprint from context."""

    def __init__(self, environment: EnvironmentInfo) -> None:
        self.environment = environment
        self.environment.validate()

    def build(self, user_query: str) -> AgentBlueprint:
        objective = user_query.strip() or "Build and improve software in the workspace"
        startup_checks = self._build_startup_checks()
        execution_plan = self._build_execution_plan(objective)
        system_prompt = self._render_system_prompt(objective, startup_checks, execution_plan)

        return AgentBlueprint(
            agent_name="Workspace Execution Agent",
            objective=objective,
            environment={
                "os_version": self.environment.os_version,
                "shell": self.environment.shell,
                "workspace_path": self.environment.workspace_path,
                "is_git_repo": self.environment.is_git_repo,
                "git_repo_path": self.environment.git_repo_path,
                "current_date": self.environment.current_date,
                "terminals_folder": self.environment.terminals_folder,
            },
            startup_checks=startup_checks,
            execution_plan=execution_plan,
            system_prompt=system_prompt,
        )

    def _build_startup_checks(self) -> list[str]:
        checks = [
            f"Confirm workspace exists: {self.environment.workspace_path}",
            f"Use shell `{self.environment.shell}` compatible commands.",
            "Read repository documentation and detect existing tooling.",
        ]
        if self.environment.is_git_repo and self.environment.git_repo_path:
            checks.append(f"Verify git status in repo: {self.environment.git_repo_path}")
        if self.environment.terminals_folder:
            checks.append(
                f"Inspect active terminal session snapshots in: {self.environment.terminals_folder}"
            )
        return checks

    def _build_execution_plan(self, objective: str) -> list[str]:
        plan = [
            f"Restate and scope objective: {objective}",
            "Discover project files and dependencies.",
            "Implement code changes with focused, small edits.",
            "Run available checks/tests and capture output.",
            "Summarize what changed, risks, and next steps.",
        ]
        if self.environment.is_git_repo:
            plan.insert(
                3,
                "Create a feature branch, commit logically grouped changes, and push.",
            )
        return plan

    def _render_system_prompt(
        self,
        objective: str,
        startup_checks: list[str],
        execution_plan: list[str],
    ) -> str:
        checks = "\n".join(f"- {check}" for check in startup_checks)
        plan = "\n".join(f"{idx}. {step}" for idx, step in enumerate(execution_plan, start=1))
        git_rule = (
            "Use git for traceability: branch, commit, and push after significant milestones."
            if self.environment.is_git_repo
            else "No git workflow required unless repository is initialized."
        )
        return (
            "You are an autonomous coding agent.\n"
            f"Objective: {objective}\n\n"
            "Environment facts:\n"
            f"- OS: {self.environment.os_version}\n"
            f"- Shell: {self.environment.shell}\n"
            f"- Workspace: {self.environment.workspace_path}\n"
            f"- Date: {self.environment.current_date}\n\n"
            "Startup checklist:\n"
            f"{checks}\n\n"
            "Execution runbook:\n"
            f"{plan}\n\n"
            "Operating constraints:\n"
            f"- {git_rule}\n"
            "- Prefer deterministic scripts and reproducible commands.\n"
            "- Always explain completed work and unresolved risks."
        )
