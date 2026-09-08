# PLAN: Legal Contract Clause Classification — LoRA Fine-Tuning Pipeline

Build plan derived from [legal-clause-classifier-prd.md](legal-clause-classifier-prd.md). Ten phases, each a git branch/commit boundary. Sub-steps within a phase are execution detail with their own verify checks. Git is manual-only per CLAUDE.md — the commands at the end of each phase are for the human to run; nothing here is executed automatically.

**Repo layout** (created in Phase 0):

```
LORA/
  etl/            # downloader, extraction/labeling function, Spark job
  indexing/       # ES schema + bulk indexing
  curation/       # sampling, CIK-based splits, leakage test
  training/       # LoRA fine-tuning + sweep configs
  evaluation/     # eval harness, GPT-4o adjudicator, catastrophic-forgetting suite
  serving/        # vLLM A/B endpoint
  tests/          # mirrors the above, per PRD Testing Decisions
  docker/         # per-service Dockerfiles referenced by docker-compose.yml
  docker-compose.yml
  pyproject.toml  # uv-managed
  PLAN.md
  REPORT.md       # final experiment report (Phase 9)
```

---

## Phase 0 — Environment & Repo Scaffolding

**Goal**: a working, empty-but-runnable skeleton everything else builds into.

1. `git init`; `uv init` (Python 3.11 pinned in `pyproject.toml`).
   Verify: `uv run python --version` reports 3.11.
2. Create the directory tree above (empty `__init__.py`/`.gitkeep` placeholders as needed).
   Verify: tree matches the layout.
3. `.env.example` listing required secrets (`SEC_EDGAR_USER_AGENT`, `OPENAI_API_KEY`, `WANDB_API_KEY`, `ES_HOST`); `.gitignore` excludes `.env`, model weights, raw downloaded filings, Parquet output.
   Verify: `.env` is git-ignored; `.env.example` has no real values.
4. `docker-compose.yml` skeleton with an Elasticsearch service stub (other services added in their own phases).
   Verify: `docker compose config` validates without error.
5. `README.md` stub: project summary, setup instructions (`uv sync`, `docker compose up`), links to `PLAN.md` and `REPORT.md`.

**Deliverable**: skeleton repo, no pipeline logic yet.

```bash
git checkout -b phase-0-scaffolding
git add -A
git commit -m "Phase 0: environment and repo scaffolding"
# to run: uv sync && docker compose config
```

---

## Phase 1 — De-Risking Pilot (download + extraction + label-frequency check)

**Goal**: validate the taxonomy and extraction approach hold up on real data *before* committing to the full 50,000-filing run. This directly addresses the PRD's "known risk to monitor early" note.

1. Implement the SEC EDGAR bulk downloader in `etl/`: respects the ~10 req/sec fair-access limit, sends a proper identifying User-Agent header, and checkpoints progress (e.g. a local SQLite/JSON file tracking completed CIK+accession numbers) so a restart resumes rather than re-downloading.
   Verify: run it, kill it mid-run, restart — confirm it resumes instead of re-fetching completed filings.
2. Run the downloader against the **first ~300 filings** and stop.
   Verify: 300 raw filings on disk, checkpoint file reflects them.
3. Hand-pick 8-10 filings from the pilot batch (or EDGAR generally) covering: a few clean/standard-header filings, a few with inconsistent formatting (numbered vs. named headers, casing variance), and at least one genuinely ambiguous header. Manually record expected clause boundaries + labels as test fixtures in `tests/fixtures/`.
4. Implement the extraction+labeling function in `etl/` as a **pure Python function** operating on a single document's raw text (no Spark dependency): detects clause boundaries primarily via section headers, labels each clause against the fixed 10-category taxonomy.
   Verify: unit tests against the 8-10 hand-verified fixtures pass.
5. Run the extraction+labeling function across the full 300-filing pilot batch; tabulate clause counts per category.
   Verify: all 10 taxonomy categories appear with reasonable frequency. If any category is critically rare or absent, stop and resolve before continuing (revisit header-matching rules, or flag/document the gap) — this is the decision gate the PRD calls for.
6. Resume the checkpointed downloader (same job, not a new one) to pull the remaining ~49,700 filings in the background while Phase 2 development proceeds.

**Deliverable**: working downloader (paused/resumable), validated extraction/labeling function with passing unit tests, label-frequency report, background download running.

```bash
git checkout -b phase-1-pilot
git add etl/ tests/fixtures/ reports/
git commit -m "Phase 1: EDGAR downloader, extraction/labeling function, label-frequency pilot"
# to run pilot: uv run python -m etl.download --limit 300
# to run frequency check: uv run python -m etl.label_frequency_report
```

---

## Phase 2 — Full-Scale Spark ETL

**Goal**: apply the validated Phase 1 function across the full corpus in parallel.

*Depends on*: Phase 1's background download reaching completion (or a substantial fraction — Spark job can be run/re-run incrementally as more filings land).

1. Wrap the Phase 1 pure function for Spark via `mapPartitions` (thin adapter only — the function itself doesn't change).
2. Build the Spark job (local mode, multiple partitions): read raw filings, extract+label clauses per partition, write structured records (Parquet) with metadata — company/CIK, filing date, industry/SIC code, clause type, source accession number.
3. Handle unparseable/malformed filings by skipping + logging, not crashing the job.
4. Run against the full downloaded corpus.
   Verify: job completes; output row count is sane relative to filing count; spot-check a sample of output records against known filings.

```bash
git checkout -b phase-2-spark-etl
git add etl/
git commit -m "Phase 2: full-scale Spark clause extraction and labeling"
# to run: uv run python -m etl.spark_job --input data/raw --output data/clauses.parquet
```

---

## Phase 3 — Elasticsearch Indexing

**Goal**: index extracted clauses for stratified sampling/filtering.

1. Define the ES index mapping: clause text, label, company/CIK, filing date, SIC code, source accession, clause ID.
2. Add the Elasticsearch service (single-node, Dockerized) to `docker-compose.yml`.
3. Build a bulk-indexing script (`indexing/`) consuming Phase 2's Parquet output via the ES bulk API.
4. Sanity queries: counts per category, filter by SIC code/date range.
   Verify: ES doc count matches Spark output row count; sample filtered queries return expected results.

```bash
git checkout -b phase-3-es-indexing
git add indexing/ docker-compose.yml
git commit -m "Phase 3: Elasticsearch indexing of extracted clauses"
# to run: docker compose up -d elasticsearch && uv run python -m indexing.bulk_index
```

---

## Phase 4 — Dataset Curation & Splits

**Goal**: produce the curated training set, hand-picked eval set, and leakage-safe splits.

1. Build a stratified sampling script (`curation/`) pulling from ES: 500-2,000 example training set, roughly balanced across the 10 categories.
2. Build the hand-picked evaluation set (30-50 examples) deliberately including ambiguous/edge cases — manual review, informed by the ambiguous fixtures surfaced in Phase 1.
3. Implement CIK-based train/validation/test splitting (partition by company/CIK, not by individual clause).
4. Leakage test: automated test asserting no CIK appears in more than one split.
   Verify: leakage test passes; split sizes and per-split category distribution are reported.

```bash
git checkout -b phase-4-curation
git add curation/ tests/
git commit -m "Phase 4: dataset curation, CIK-based splits, leakage test"
# to run: uv run python -m curation.build_dataset && uv run pytest tests/test_leakage.py
```

---

## Phase 5 — LoRA Fine-Tuning Pipeline + Hyperparameter Sweep

**Goal**: train and select a final LoRA adapter.

1. Build the training script (`training/`): Llama 3 8B + Unsloth + QLoRA (4-bit), LoRA on `q_proj`/`v_proj`, single default-config run end-to-end on the curated training set.
2. W&B logging: hyperparameters, loss curves, GPU memory usage, duration.
3. Early stopping on validation loss.
   Verify: a single default run completes on the RTX 3090 within VRAM budget and produces a saved adapter.
4. Run the hyperparameter sweep (rank 8/16/32 × lr 1e-4/2e-4/5e-4 × epochs 1/3/5), sequential runs on the single GPU, all logged to W&B.
5. Select the final configuration from sweep results with documented rationale.
   Verify: sweep runs are comparable in the W&B dashboard; final adapter saved and under 100MB.

```bash
git checkout -b phase-5-training
git add training/
git commit -m "Phase 5: LoRA fine-tuning pipeline and hyperparameter sweep"
# to run default: uv run python -m training.train --config configs/default.yaml
# to run sweep: uv run python -m training.sweep
```

---

## Phase 6 — Evaluation Harness

**Goal**: honest, apples-to-apples base-vs-fine-tuned comparison.

1. Build the eval harness (`evaluation/`): accuracy, per-class precision/recall/F1, confusion matrix.
   Verify: harness tested against a small fixture set with known expected metrics.
2. Run the base (non-fine-tuned) model against the full eval benchmark — record as baseline.
3. Run the fine-tuned model (Phase 5's final adapter) against the identical benchmark.
4. GPT-4o adjudication: for cases where a model's prediction disagrees with the header-derived label, call GPT-4o to determine the more defensible label; fold into final reporting.
5. Catastrophic-forgetting check: run `lm-evaluation-harness` (MMLU subset + HellaSwag) on both base and fine-tuned models, compare deltas.
   Verify: comparison report produced with metrics, confusion matrices, adjudication results, and forgetting-check deltas.

```bash
git checkout -b phase-6-evaluation
git add evaluation/ tests/
git commit -m "Phase 6: evaluation harness, GPT-4o adjudication, catastrophic-forgetting check"
# to run: uv run python -m evaluation.run_comparison
```

---

## Phase 7 — Deployment: Adapter Packaging + vLLM A/B Endpoint

**Goal**: serve both models for direct comparison.

1. Package the final LoRA adapter as a standalone artifact, separate from base model weights.
2. Stand up a single vLLM server (`--enable-lora`), base model loaded once, adapter registered alongside it.
3. Build a thin FastAPI wrapper (`serving/`): the A/B endpoint accepts a clause, fires two requests against the vLLM server (no adapter vs. adapter), returns both predicted labels + confidences side by side.
4. Contract test: fixed clause input → well-formed response containing both models' label + confidence (API-boundary test, not a model-correctness test).
   Verify: contract test passes; manual request against a real clause returns sane side-by-side output.

```bash
git checkout -b phase-7-serving
git add serving/ tests/
git commit -m "Phase 7: LoRA adapter packaging and vLLM multi-LoRA A/B endpoint"
# to run: uv run python -m serving.app
```

---

## Phase 8 — Full Docker Compose Packaging

**Goal**: one-command reproducibility across the whole pipeline.

1. Finalize `docker-compose.yml`: Spark, Elasticsearch, training script, evaluation harness, serving — all services defined; GPU passthrough (nvidia-container-toolkit) configured for the training/evaluation/serving services.
2. Document that ETL and training phases run sequentially, not concurrently (31GB RAM / ~22GB available), and verify one-command bring-up per documented sequence/profile.
   Verify: a fresh clone + the documented `docker compose` commands reproduce each stage without manual intervention beyond what's documented.

```bash
git checkout -b phase-8-docker-packaging
git add docker/ docker-compose.yml README.md
git commit -m "Phase 8: full docker-compose packaging of the pipeline"
# to run: docker compose up
```

---

## Phase 9 — Experiment Report + Demo Video

**Goal**: the artifacts that back the portfolio narrative.

1. Write `REPORT.md`: problem statement, why fine-tuning was chosen as a lifecycle/engineering showcase (not an algorithmic-optimality claim), dataset statistics and curation methodology (citing LEDGAR and CUAD), hyperparameter sweep results, head-to-head evaluation results, catastrophic-forgetting findings, and the CIK-based leakage-prevention methodology.
2. Record a demo video (<4 minutes): ETL job running, a W&B training run, base-vs-fine-tuned evaluation results, the A/B endpoint in action.
3. Polish `README.md`: setup instructions, architecture overview, links to `REPORT.md` and the video.
   Verify: report explicitly cites LEDGAR/CUAD; video is under 4 minutes and covers all four required segments.

```bash
git checkout -b phase-9-report
git add REPORT.md README.md
git commit -m "Phase 9: experiment report and demo video"
```
