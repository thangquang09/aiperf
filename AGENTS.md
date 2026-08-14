<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# AIPerf

Python 3.10+ async AI benchmarking tool for measuring LLM inference server performance. 10 services communicate via ZMQ message bus.

**Reference documentation:**
- [`docs/architecture.md`](docs/architecture.md) - Three-plane architecture, core components, credit system, data flow, communication patterns
- [`docs/dev/patterns.md`](docs/dev/patterns.md) - Code examples for CLI commands, services, models, messages, plugins, error handling, logging, testing
- [`docs/cli-options.md`](docs/cli-options.md) - Complete CLI command and option reference
- [`docs/environment-variables.md`](docs/environment-variables.md) - All `AIPERF_*` environment variables by subsystem
- [`docs/metrics-reference.md`](docs/metrics-reference.md) - Metric definitions, formulas, and requirements
- [`docs/plugins/plugin-system.md`](docs/plugins/plugin-system.md) - Plugin architecture, categories, creation guide
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - Development setup, available commands, pre-commit hooks, DCO

## Coding Standards

- async/await for ALL I/O - no `time.sleep`, no blocking calls.
- `Field(description="...")` on EVERY Pydantic field. Docstrings on dataclass fields.
- Type hints on ALL functions (params and return).
- KISS + DRY: minimal code, optimize for reader.
- `AIPerfBaseModel` for data, `BaseConfig` for configuration. `@dataclass(slots=True)` for hot-path inner models created at high volume (e.g. SSE chunks, parsed responses) where Pydantic overhead matters. Use `__pydantic_config__ = ConfigDict(extra="forbid")` on dataclasses that participate in Pydantic union discrimination.
- `BaseComponentService` for services, `BaseService` for SystemController only.
- Message bus for inter-service communication - no shared mutable state.
- CLI commands: one file per command in `cli_commands/`, lazily loaded via import strings in `cli.py`. See `docs/dev/patterns.md`.
- YAML plugin registry for extensible features (`src/aiperf/plugin/plugins.yaml`).
- Lambda for expensive logs: `self.debug(lambda: f"{self._x()}")`. Direct string for cheap ones.
- Always `orjson.loads(s)`, `orjson.dumps(d)` for JSON.
- No `Optional[X]` or `Union[X, Y]` - use `X | Y`.
- Comments only for "why?" not "what".
- Enums are string-based - use `MessageType.X` directly, never `.value`.
- Dependencies: always use `uv` (never pip) - `uv add package`, `uv run pytest`.
- Use mermaid diagrams instead of ASCII art in markdown files.
- Do not create markdown files to document code changes or decisions.
- Do not over-comment code. Removing code is fine without adding comments to explain why.
- No emojis in code or comments.
- Hide a metric from the console table with `console_group = MetricConsoleGroup.NONE`; group it into a separate section with `MetricConsoleGroup.{USAGE,CACHE,PREDICTION,AUDIO,REASONING}`. Default is `DEFAULT`. See `docs/metrics-reference.md` "Metric Console Group Reference".

## Build and Test Commands

```bash
make first-time-setup                                      # Initial environment setup
make install                                               # Install project + mock server
uv run pytest tests/unit/ -n auto                          # Unit tests (fast, isolated)
uv run pytest -m integration -n auto                       # Integration tests (real services, multiprocess)
uv run pytest -m component_integration -n auto             # Component integration tests (single process)
ruff format . && ruff check --fix .                        # Format and lint
make validate-plugin-schemas                               # Validate plugin registry
pre-commit run                                             # Pre-commit on staged files
pre-commit run --all-files                                 # Pre-commit on all files
make generate-all-docs                                     # Regenerate CLI + env var docs
make generate-all-plugin-files                             # Regenerate plugin enums, overloads, schemas
```

## Pre-Commit Hooks

Run pre-commit after every code change, even before creating commits:

```bash
pre-commit run              # Staged files only
pre-commit run --all-files  # All files (recommended after significant changes)
```

Hooks: `check-ast`, `debug-statements`, `detect-private-key`, `check-added-large-files`, `check-case-conflict`, `check-merge-conflict`, `check-executables-have-shebangs`, `check-shebang-scripts-are-executable`, `check-json`, `check-toml`, `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, `mixed-line-ending`, `no-commit-to-branch`, `requirements-txt-fixer`, `codespell`, `add-license`, `generate-cli-docs`, `generate-env-vars-docs`, `generate-plugin-artifacts`, `validate-plugin-schemas`, `test-imports`, `ruff`, `ruff-format`.

## Adding a New Service

1. Create class extending `BaseComponentService` with `@on_message` handlers
2. Register in `src/aiperf/plugin/plugins.yaml` under `service` category with `class`, `description`, `metadata`
3. Add message type to `src/aiperf/common/enums/enums.py` if new messages needed
4. Create message class in `src/aiperf/common/messages/` with `message_type` field
5. Validate with `make validate-plugin-schemas`

## Adding a New Message

1. Add enum value to `MessageType` in `src/aiperf/common/enums/enums.py`
2. Create message class in `src/aiperf/common/messages/` inheriting from `Message` with `message_type` field set
3. Add `@on_message(MessageType.X)` handler in the receiving service
4. Auto-subscription happens during `@on_init` phase

## Adding a New Plugin

1. Create plugin class implementing the appropriate base
2. Add entry to `src/aiperf/plugin/plugins.yaml` with `class`, `description`, `metadata`
3. Validate with `make validate-plugin-schemas`
4. Use via `plugins.get_class(PluginType.X, 'name')`

## Testing Conventions

- `@pytest.mark.asyncio` for async tests, `@pytest.mark.parametrize` for data-driven
- `from tests.harness import mock_plugin` for plugin mocking
- Name: `test_<function>_<scenario>_<expected>` e.g. `test_parse_config_missing_field_raises_error`
- Imports at file top, fixtures for setup, one focus per test
- Auto-fixtures (always active): asyncio.sleep runs instantly, RNG=42, singletons reset between tests

## Git Workflow

Feature branches use `<username>/feature-name` format, forked from `main`. One PR = one concern.

## Tips

- SystemController uses `BaseService` (not `BaseComponentService`) - it's the orchestrator.
- Worker/TimingManager disable GC for latency - see `service_metadata.disable_gc`.
- macOS child processes close terminal FDs to prevent Textual UI corruption.
- Plugin priority resolves conflicts: higher wins, external beats built-in at equal priority.
- Decorators: `@on_init`, `@on_start`, `@on_stop`, `@on_message`, `@on_command`, `@background_task`, `@on_pull_message`, `@on_request`.
- Communication: `publish()` for broadcast, `@on_message` to subscribe, `send_command_and_wait_for_response()` for sync.
- `AIPerfLifecycleMixin` for standalone components: `CREATED` -> `INITIALIZING` -> `INITIALIZED` -> `STARTING` -> `RUNNING` -> `STOPPING` -> `STOPPED`; `FAILED` terminal.
- `dag_jsonl` input type: conversation DAG benchmarks (fork + spawn modes); see `docs/benchmark-modes/dag.md`.

## Pre-Commit Checklist

1. Review diff: all lines required?
2. `ruff format . && ruff check --fix .`
3. `uv run pytest tests/unit/ -n auto`
4. Type hints on all functions
5. `Field(description=...)` on all Pydantic fields
6. `git commit -s`

## Three-File Sync Rule

`CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` must contain identical content (only headers/frontmatter differ). When updating one, update all three. Always diff them after editing to confirm sync.

## Documentation Updates

When making changes, update the appropriate documentation files. When adding a new tutorial, also add it to `README.md`'s tutorial index.

| Change type | Files to update |
|---|---|
| Architecture, components, data flow, communication | `docs/architecture.md` |
| Coding standards, build commands, new patterns | `CLAUDE.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` |
| Code patterns, examples, base classes | `docs/dev/patterns.md` |
| CLI arguments or commands | `docs/cli-options.md` (auto-generated via `make generate-cli-docs`) |
| Environment variables | `docs/environment-variables.md` (auto-generated via `make generate-env-vars-docs`) |
| Metrics definitions or formulas | `docs/metrics-reference.md` |
| Plugin system, categories, creation | `docs/plugins/plugin-system.md` |
| Accuracy benchmarks, graders | `docs/accuracy/` |
| Server metrics, schemas | `docs/server-metrics/` |
| Benchmark modes, timing, traces | `docs/benchmark-modes/` |
| Tokenizer, reference docs | `docs/reference/` |
| Dataset synthesis API | `docs/api/synthesis.md` |
| Dev setup, make targets, pre-commit | `CONTRIBUTING.md` |
| Contribution process, DCO | `CONTRIBUTING.md` |
| New services, message types, plugin types | `docs/architecture.md` + `docs/dev/patterns.md` |
| Tutorials and feature guides | `docs/tutorials/` + `README.md` tutorial index |

**A feature is incomplete until documentation is updated.**

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **aiperf** (48445 symbols, 97194 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/aiperf/context` | Codebase overview, check index freshness |
| `gitnexus://repo/aiperf/clusters` | All functional areas |
| `gitnexus://repo/aiperf/processes` | All execution flows |
| `gitnexus://repo/aiperf/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
