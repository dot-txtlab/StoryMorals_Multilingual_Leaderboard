# Multilingual Story-Morals Leaderboard

### 🏆 [**View the live leaderboard →**](https://dot-txtlab.github.io/StoryMorals_Multilingual_Leaderboard/)

[![leaderboard](https://img.shields.io/badge/leaderboard-live-3ddc97)](https://dot-txtlab.github.io/StoryMorals_Multilingual_Leaderboard/)

This project introduces story moral generation as a new benchmark for evaluating the cultural alignment of large language models. Rather than treating culture as factual knowledge to be memorized, it measures whether AI systems can reproduce the diverse ways people from different linguistic and cultural communities interpret stories.

The project is detailed more fully in [**Wu & Piper, *Lessons Without Borders? Evaluating Cultural Alignment of LLMs Using Multilingual Story Moral Generation***](https://arxiv.org/abs/2604.08797)

Here we present an automated leaderboard that replicates the evaluation from
the paper and turns it into a public, periodically-updated dashboard.

The underlying dataset consists of 588 human written morals for a collection of 14 stories translated into 14 languages. Each story is represented as a story summary from Wikipedia. Human annotators were recruited through the Prolific platform using demographic filters to ensure fluency and geographic location in specific regions associated with each language.

Models generate story morals every story in each language. We then ask
two questions:

1. **Within-language alignment.** Are a model's morals as similar to
   human morals as humans are to each other? → it sits inside the *human band of
   variance*. **Higher is better.**
2. **Cross-language diversity.** Across languages, are a model's morals
   *more* similar to each other than humans' are? That means it restates one
   moral in many languages instead of reflecting cultural variation
   ("flattening"). **Lower is better.**

A model **succeeds** when it is high on (1) and low on (2).

---

## What changes vs. what stays fixed

| Stays fixed | Changes each refresh |
|---|---|
| `data/human_morals.csv` — human-written morals | `config/models.yaml` — the list of models to test |
| `data/passages.csv` — the 196 story summaries | API keys in `.env` |
| `data/prompt_template.txt` — the socio-demographic prompt | |


---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # create your private keys file, then paste keys in (see "API keys" below)

# 1. Generate morals for every enabled model (calls the APIs)
python scripts/run.py generate

# 2. Score everything against the human baselines
python scripts/run.py evaluate

# 3. Build the dashboard -> docs/index.html
python scripts/run.py dashboard

# ...or all three at once:
python scripts/run.py all
```

---

## Running at scale & troubleshooting

A full run is 13 models × 196 passages ≈ 2,500 API calls, so things *will* go
wrong (rate limits, refusals, a mistyped model id). The pipeline is built to
fail safe and resume.

**Always smoke-test first** — one passage validates your key + model id in seconds:

```bash
python scripts/run.py generate --model claude-opus-4-8 --limit 1
python scripts/run.py generate --model gpt-5.5 --story brazil   # one story, all 14 languages
```

**Check progress / find failures** anytime (also works mid-run from another shell):

```bash
python scripts/run.py status
# OK  Claude Opus 4.8      done=196  errors=0    missing=0
# !!  GPT-5.5              done=190  errors=6    missing=0
#       e.g. MODEL_ERROR: gpt-5.5: failed after 5 attempts: rate_limit ...
```

**How failures are handled:**

| Situation | Behavior |
|---|---|
| Rate limit / 5xx / timeout (transient) | Retried with exponential backoff (`max_retries`, default 5). |
| A call hangs | Cut off after `request_timeout` (default 120s) and retried — never blocks forever. |
| Refusal / empty response | Treated as a retryable error; if it persists, recorded as `MODEL_ERROR` (not stored as a fake moral). |
| Bad API key / unknown model id (fatal) | **Fails fast** — that model aborts immediately instead of burning 196 calls. Fix `models.yaml`/`.env` and re-run. |
| A passage ultimately fails | Written as a `MODEL_ERROR: …` row — never silently dropped. |

**Outputs accumulate live.** Each passage is written to
`output/model_morals.csv` the moment it returns — so you can `cat` it (or even
run `evaluate`) on partial results mid-run, and a crash loses at most the single
call in flight.

**Resume is automatic and safe.** Re-running `generate` skips passages already
done and **retries only the failed/missing ones** (it replaces their error rows).
So the recovery loop is just: `generate` → `status` → fix anything fatal →
`generate` again.

**Long runs — close the terminal.** `generate` runs in the foreground, so
closing the window stops it. For a full run, detach it with `--background`:

```bash
python scripts/run.py generate --background
# STARTED in background (pid 12345). Safe to close this window.
#   log:      tail -f run.log
#   progress: python scripts/run.py status
#   stop it:  kill 12345
```

It survives logout/window-close, logs to `run.log`, and checkpoints as usual —
monitor it anytime with `status` (from any terminal) or `tail -f run.log`.

Tuning knobs live in `config/models.yaml` → `defaults`: `concurrency` (parallel
calls per model — lower it if you hit rate limits), `max_retries`,
`request_timeout`.

`MODEL_ERROR` rows are ignored by `evaluate`, so a model that partially failed
just won't appear (or appears only if it has complete data) — it never corrupts
the leaderboard.

---

## Adding / changing models

Edit [`config/models.yaml`](config/models.yaml):

```yaml
models:
  - {id: gpt-5.5,          provider: openai,    display: "GPT-5.5",         enabled: true}
  - {id: claude-opus-4-8,  provider: anthropic, display: "Claude Opus 4.8", enabled: true}
  - {id: my-new-model,     provider: deepseek,  display: "My New Model",    enabled: true}
```

- **`id`** must be the provider's exact API model string.
- **`provider`** is one of the routes defined at the bottom of the file
  (`openai`, `google`, `alibaba`, `anthropic`, `deepseek`, `lambda`). All except
  Anthropic are OpenAI-compatible; adding a new provider is one entry with a
  `base_url` and an API-key env var.
- Set `enabled: false` to keep a model in the file without running it.

Re-running `generate` **skips passages already finished** and retries new/failed
ones (results are checkpointed in `output/model_morals.csv`). See
[Running at scale & troubleshooting](#running-at-scale--troubleshooting) for how
errors, refusals, and hangs are handled.

### API keys (`.env`)

The pipeline reads your API keys from a file called `.env`. You create it once.

**1. Make your private copy of the template** (run once, in the project folder):

```bash
cp .env.example .env
```

`cp` copies `.env.example` (the blank template that ships with the repo) to a new
file `.env`. The `.env` file is your private copy — it's git-ignored, so your keys
are never committed or pushed.

**2. Open `.env` and paste your keys** after each `=` (no spaces, no quotes):

```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

Open it with `open -e .env` (macOS TextEdit), `code .env` (VS Code), or any editor.
**You only need keys for the providers used by your *enabled* models** in
`config/models.yaml` — leave the rest blank. (E.g. if only Claude models are
enabled, only `ANTHROPIC_API_KEY` is required.)

| Provider | Env var in `.env` | Get a key from |
|---|---|---|
| OpenAI (GPT) | `OPENAI_API_KEY` | platform.openai.com → API keys |
| Google (Gemini) | `GEMINI_API_KEY` | aistudio.google.com → Get API key |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | console.anthropic.com → API keys |
| Alibaba (Qwen) | `DASHSCOPE_API_KEY` | DashScope console (international) |
| DeepSeek | `DEEPSEEK_API_KEY` | platform.deepseek.com → API keys |
| Lambda (open-source, phase 2) | `LAMBDA_API_KEY` | cloud.lambda.ai → API keys |

**3. Verify it works** without spending much — one call, fails fast if the key is
missing/wrong:

```bash
python scripts/run.py generate --model claude-opus-4-8 --limit 1
```

`scripts/run.py` loads `.env` automatically on startup (via `load_dotenv()`), so
you never pass keys on the command line.

---

## How scoring works

For each moral we compute cosine similarity over **multilingual sentence
embeddings** (default: **BGE-M3** + **LaBSE** — a current SOTA multilingual
encoder plus a classic anchor; the original-language moral is embedded directly,
so no translation step is needed). We then fit, per model,
a **mixed-effects regression** of similarity with random intercepts for story,
language(-pair), and embedding model — exactly as in the paper:

- **Test 1 (H3) / alignment:** human–model vs. human–human pairs, same story & language.
- **Test 2 (H4) / diversity:** model–model vs. human–human pairs, same story, *different*
  languages. A positive coefficient = flattening.

Pick embedders with `python scripts/run.py evaluate --embedders bge-m3 labse`.
Available: `bge-m3`, `labse`, `minilm`, `mpnet-en` (English-only, needs
translations), `tfidf` (fast dev-only encoder — not valid for real numbers).
For apples-to-apples with the published Figs 3 & 4, use the paper preset:
`--embedders labse minilm mpnet-en`.

Outputs land in `output/` (`model_morals.csv`, `leaderboard.json`) and the
dashboard in `docs/index.html`.

---

## Repository layout

```
config/        models.yaml (edit me) · languages.yaml (fixed)
data/          human_morals.csv · passages.csv · prompt_template.txt
src/storymorals/
  config.py    load YAML configs
  providers.py provider routing (OpenAI-compatible + Anthropic)
  generate.py  call APIs -> output/model_morals.csv
  datasets.py  merge human + model morals (long format)
  embed.py     multilingual sentence embeddings (cached)
  metrics.py   H3 / H4 mixed-effects gaps (Figs 3 & 4)
  leaderboard.py  ranking + verdicts -> output/leaderboard.json
  dashboard.py    -> docs/index.html
scripts/run.py CLI: generate | evaluate | dashboard | all
Lambda_Example/  open-source-via-Lambda recipe (phase 2)
```

## Roadmap

- **Phase 2:** open-source models via the Lambda Inference API (`provider: lambda`),
  reusing the same pipeline. The `Lambda_Example/` folder documents the GPU-instance
  approach used in prior work.
