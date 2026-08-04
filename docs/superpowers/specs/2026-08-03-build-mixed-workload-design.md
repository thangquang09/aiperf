# Design: build_mixed_workload.py — single-file DAG mix of chat / RAG / agentic

**Date:** 2026-08-03
**Author:** greenneode / AIPerf mix-corpus work
**Status:** Approved (inline, before write)

## Motivation

Goal: one AIPerf run that mixes production-realistic workloads — chat, coding/RAG,
and agentic — against an OpenAI-compatible endpoint (GLM-5.2 open-source). AIPerf
accepts exactly one dataset source per run (`--public-dataset` is mutually
exclusive with `--custom-dataset-type`), so mixing requires pre-merging the
sources into a single file. `dag_jsonl` is the unifying format: it expresses
single-turn requests, multi-turn conversations, FORK children (shared prefix +
sticky routing for prefix-cache hits), and SPAWN children (fresh-context
sub-agents), all in one file.

Two constraints drive the design:

1. **Do not modify AIPerf source.** The converter lives outside `src/aiperf/`
   (in `tools/`) and reuses AIPerf's own loaders via the plugin registry.
2. **Do not break the format.** The converter must emit `dag_jsonl` that
   `DagJsonlLoader` itself accepts — validated by re-running the loader on the
   output.

## Non-goals

- Adding multi-loader support to `DatasetManager` (Level B). The user explicitly
  does not want `--dataset sharegpt + weka + rag`; they control dataset creation.
- Preserving Weka's LCP/hash-id prefix-cache *simulation* semantics. Weka's
  `hash_ids`-driven prompt synthesis is replaced by FORK mode (shared real text
  prefix + sticky routing), which the user accepted. Absolute per-request
  timestamps / fixed-schedule replay are also out of scope; only relative
  think-time `delay` is carried over.
- Making the converter load raw HF datasets itself. It calls AIPerf loaders.

## Sources (public, AIPerf-supported)

| Slice | Loader (`--public-dataset` key) | Weight | Notes |
|---|---|---|---|
| chat | `sharegpt` | 30% | single-turn (loader takes first prompt/completion pair), `max_tokens` = original completion length, random `delay` 500–3000 ms |
| rag | `speed_bench_rag` | 30% | single-turn, `max_tokens` = original, random `delay` 500–3000 ms |
| agentic | `semianalysis_cc_traces_weka_062126_256k` | 40% | HF `semianalysisai/cc-traces-weka-062126-256k`; smoke uses 2 traces via `num_dataset_entries=2` |

## 1. Tool — `tools/build_mixed_workload.py`

**Run:** `uv run python tools/build_mixed_workload.py --config config.yml`

Standalone Python script inside the aiperf repo (runs with the project venv so
it can import AIPerf). Does not touch `src/aiperf/`.

### 1.1 Config (YAML) + CLI overrides

```yaml
# config.yml — user-controlled
sources:
  chat:
    loader: sharegpt
    weight: 30
    chat_delay_ms: [500, 3000]     # random think-time between conversations
  rag:
    loader: speed_bench_rag
    weight: 30
    chat_delay_ms: [500, 3000]
  agentic:
    loader: semianalysis_cc_traces_weka_062126_256k
    weight: 40
    num_traces: 2
total_conversations: 30
out_file: data/merged_workload.dag.jsonl
tokenizer: builtin
```

CLI overrides: `--total-conversations`, `--out-file`, `--weka-num-traces`.
Weights must sum to 100.

### 1.2 Data flow

```
build_mixed_workload.py
  ├─ read config
  ├─ Tokenizer.load("builtin")            # o200k_base, zero network; 1x shared
  ├─ load chat:     sharegpt.load_dataset()        → slice ~30% of total
  ├─ load rag:      speed_bench_rag.load_dataset() → slice ~30% of total
  ├─ load agentic:  weka loader (num_dataset_entries=2)
  │                 → convert_to_conversations()   → [Conversation] root+subagent
  ├─ normalize: each Conversation → one dag_jsonl line
  ├─ write merged_workload.dag.jsonl
  └─ re-validate: DagJsonlLoader(out_file).load()  → fail ⇒ delete file, exit 2
```

Slice sizes: `count_slice = round(total_conversations * weight / 100)` per
source; agentic slice is seeded by `num_traces` (each trace expands to one root
+ subagent conversations, so it can exceed the slice count — accepted).

### 1.3 Conversation → DagConversation mapping

| `Conversation` field | `dag_jsonl` field | Notes |
|---|---|---|
| `session_id` | `session_id` | prefixed per source (`sharegpt-0001`, `rag-0001`, `weka-...`) to avoid collisions |
| `turns[].raw_messages` | `turns[].messages` | kept as-is (delta-encoded); dag uses the same pure-append `DELTAS_WITHOUT_RESPONSES` mode |
| `turns[].max_tokens` | `turns[].max_tokens` | omitted when None |
| `turns[].delay` | `turns[].delay` | ms, copied straight through |
| `conversation.branches` (FORK/SPAWN) | `turns[].forks` / `turns[].spawns` | branch child ids → field lists |
| `turns[].prerequisites` (SPAWN_JOIN) | `spawns` object `{children, join_at}` | `join_at` rebuilt from the gated turn index |
| `conversation.is_root` | — | root = any session not referenced as a child |

Serialization pitfalls handled explicitly:

- **Duplicate `session_id` across sources** → per-source prefix.
- **Weka `system` message on a non-root turn** → loader rejects it; serializer
  demotes it to a `user` message (dag.md:206-217 rule: system only on the
  accumulator-seeding turn).
- **`raw_messages` kept delta** — must NOT be accumulated during serialization.
- **sharegpt/rag single-turn**: one `user` message per turn; `max_tokens` from
  the loader (original completion length); random `delay` from
  `chat_delay_ms` range.

### 1.4 Error handling

- Per-source try/except: on failure, log `source <name> failed: <err>` and exit
  2 — never write a partial file.
- Per-line validation via `DagConversation.model_validate` before writing;
  report `session_id` + field + message on failure.
- Post-write re-validation via `DagJsonlLoader(out_file).load()` +
  `validate_for_orchestrator_v1(...)`; on failure delete the file and exit 2.
  This is the final "format barrier".

## 2. Testing

### 2a. Converter unit tests

- Serialize each source shape (sharegpt single-turn, rag single-turn, weka
  root + subagent with branches/prereqs) → each line passes
  `DagConversation.model_validate`.
- Branch mapping: weka SPAWN/SPAWN_JOIN → correct `spawns`/`join_at`.
- Session-id prefixing and system-demotion edge cases.

### 2b. Golden test

Build from 5 sharegpt + 5 rag + 2 weka traces → `DagJsonlLoader(...).load()`
passes; snapshot asserts session/turn/branch counts.

### 2c. Mock-server smoke

`aiperf profile --custom-dataset-type dag_jsonl --input-file merged_workload.dag.jsonl`
against the in-repo mock server → completes, 0 errors (matches the 6 scenarios
already validated).

### 2d. Real-LLM smoke (GLM-5.2 via `.env`)

Same command against the real endpoint using url/api-key/model from `aiperf/.env`
(`--tokenizer builtin`); small scale (10–20 requests). Success criteria:
prefix-cache hit rate > 0 on the FORK weka slice, `delay` think-time respected
(no burst), no server-side schema errors.

## 3. Files

- New: `tools/build_mixed_workload.py`
- New (test): unit + golden tests under `tests/tools/`
- New (example): `data/config.mix.yml`
- Generated: `data/merged_workload.dag.jsonl` (not committed)

## 4. Open items (post-approval, during implementation)

- Confirm exact `PublicDatasetLoader` plugin API shape when calling loaders
  programmatically (may need a minimal `UserConfig` to satisfy loader
  constructors).
- Confirm whether `sharegpt` loader requires HF tokenizer (it calls
  `self.tokenizer.encode`); fall back to the shared builtin tokenizer if the
  loader accepts an injected tokenizer, else route sharegpt through the same
  `PromptGenerator` path.
