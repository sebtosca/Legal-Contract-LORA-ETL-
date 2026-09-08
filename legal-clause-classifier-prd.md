# PRD: Legal Contract Clause Classification — LoRA Fine-Tuning Pipeline with Large-Scale ETL

(Project 10 from BASWE's "15 AI Engineering Projects That Actually Land Jobs" guide, expanded with a large-scale legal data ETL layer.)

## Problem Statement

The builder is a mid-level software engineer transitioning into AI engineering. Most portfolio projects in this space are shallow — "I called an LLM API and deployed a chatbot" — and don't demonstrate the full model-customization lifecycle (data curation, fine-tuning, evaluation, deployment) or the large-scale data engineering skills that distinguish a strong AI engineering candidate from someone who has only prompted an API. The builder specifically wants a project that showcases the ability to handle large-scale data ETL, since that is a skill gap most AI-engineering portfolio projects don't touch, and wants to target companies doing specialized legal AI (e.g., Noxtua) where models are fine-tuned on proprietary/domain-specific data at scale.

## Solution

Build an end-to-end pipeline that:
1. Bulk-ingests ~50,000 SEC EDGAR Exhibit-10 contract filings and extracts individual clauses using a distributed Spark ETL job.
2. Indexes extracted clauses into Elasticsearch for stratified sampling, filtering, and dataset curation.
3. Bootstrap-labels clauses by contract section headers (a methodology consistent with the published LEDGAR and CUAD legal-NLP corpora, both also built from SEC EDGAR Exhibit-10 filings), with manual spot-checks and a hand-picked edge-case evaluation set.
4. Fine-tunes Llama 3 8B with LoRA (via Unsloth + QLoRA) to classify contract clauses into 10 standard categories.
5. Evaluates the fine-tuned model against the base model on classification accuracy/F1, using GPT-4o as an adjudicator for label disagreements, plus a catastrophic-forgetting check.
6. Serves the fine-tuned model via vLLM with an A/B comparison endpoint against the base model.
7. Packages the entire pipeline (Spark, Elasticsearch, training, eval, inference) as a single reproducible docker-compose stack, tracked end-to-end in Weights & Biases.

The resulting narrative for interviews: "I built a production-style legal data pipeline *and* the full model-customization lifecycle on top of it" — an engineering/lifecycle showcase, not a claim that fine-tuning is the uniquely optimal algorithmic choice for clause classification.

## User Stories

1. As the builder, I want to bulk-download SEC EDGAR Exhibit-10 filings within SEC's fair-access rate limits, so that I can assemble a ~50,000-filing corpus without getting rate-limited or blocked.
2. As the builder, I want the download job to checkpoint progress, so that a failure partway through a multi-hour/multi-day run doesn't force a restart from zero.
3. As the builder, I want a Spark job that parses raw filing text and splits it into individual clauses, so that I can process the corpus in parallel rather than serially.
4. As the builder, I want clause boundaries detected primarily from section headers, so that extraction is robust to the inconsistent formatting across different filers' contracts.
5. As the builder, I want each extracted clause bootstrap-labeled from its section header against a fixed 10-category taxonomy, so that I get a large labeled dataset without hand-labeling every clause.
6. As the builder, I want a label-frequency check run early against a data sample, so that I discover before committing further work whether any of the 10 candidate categories are too rare to classify reliably.
7. As the builder, I want extracted, labeled clauses indexed into Elasticsearch with metadata (company/CIK, filing date, industry/SIC code, clause type), so that I can filter and stratify-sample a diverse training set rather than taking an arbitrary slice.
8. As the builder, I want to sample a curated 500-2,000 example training set and a 30-50 example hand-picked evaluation set (deliberately including ambiguous/edge cases) from the indexed corpus, so that I have both a bulk training signal and a high-quality evaluation benchmark.
9. As the builder, I want train/validation/test splits partitioned by company/CIK rather than by individual clause, so that near-duplicate boilerplate reused across a filer's contracts doesn't leak between splits and inflate apparent accuracy.
10. As the builder, I want the training pipeline to run LoRA fine-tuning on Llama 3 8B via Unsloth with QLoRA, so that I can train within my local RTX 3090's 24GB VRAM.
11. As the builder, I want every training run's hyperparameters, loss curves, GPU memory usage, and duration logged to Weights & Biases, so that every run is reproducible from its config alone.
12. As the builder, I want early stopping based on validation loss, so that I don't overfit or waste compute past the point of improvement.
13. As the builder, I want a hyperparameter sweep across LoRA rank (8/16/32), learning rate (1e-4/2e-4/5e-4), and epoch count (1/3/5), so that I can justify my final configuration with comparative data rather than a guess.
14. As the builder, I want the base (non-fine-tuned) model run against the full evaluation benchmark first, so that I have an honest baseline before any fine-tuning claims.
15. As the builder, I want the fine-tuned model evaluated on the identical benchmark and scoring criteria as the base model, so that the comparison is apples-to-apples.
16. As the builder, I want accuracy, per-class precision/recall/F1, and a confusion matrix computed for both models, so that I can show exactly where the fine-tuned model improved (and didn't).
17. As the builder, I want GPT-4o used as an adjudicator specifically when a model's predicted label disagrees with the header-derived label, so that my accuracy numbers aren't artificially depressed by imperfect bootstrap labels.
18. As the builder, I want a catastrophic-forgetting check that runs standard general-capability benchmarks on both the base and fine-tuned model, so that I can detect and document any degradation in general ability caused by fine-tuning.
19. As the builder, I want the trained LoRA adapter (not the full merged model) packaged for deployment, so that the artifact stays small (<100MB) and swappable against the base model.
20. As the builder, I want the fine-tuned model served via vLLM with the base model loaded alongside it, so that I can expose both for direct comparison.
21. As the builder, I want an A/B comparison API endpoint that accepts a clause and returns both models' predicted label and confidence side by side, so that the improvement is immediately visible without reading a report.
22. As the builder, I want the entire stack (Spark, Elasticsearch, training script, eval harness, inference server) defined in one docker-compose file, so that the whole pipeline is reproducible with a single command.
23. As the builder, I want a written experiment report documenting the problem, why fine-tuning was chosen as a lifecycle/engineering showcase (not as the uniquely optimal algorithmic choice), dataset statistics and curation methodology, hyperparameter sweep results, head-to-head evaluation results, and catastrophic-forgetting findings, so that I can back every claim in interviews with data in the repo.
24. As the builder, I want a demo video under 4 minutes showing the ETL job running, a training run in Weights & Biases, the base-vs-fine-tuned evaluation results, and the A/B endpoint in action, so that I have a persuasive artifact beyond the README.
25. As a reviewer/interviewer evaluating the portfolio, I want to see the labeling methodology cite precedented published approaches (LEDGAR, CUAD), so that I can trust the dataset wasn't naively or unreliably constructed.
26. As a reviewer/interviewer, I want to see the leakage-prevention (company/CIK split) explicitly documented, so that I can trust the reported accuracy numbers reflect real generalization.

## Implementation Decisions

- **Task**: Multi-class clause classification over a fixed 10-category taxonomy: Indemnification, Limitation of Liability, Termination, Governing Law, Confidentiality, Assignment, Force Majeure, Payment Terms, Warranty/Representations, Dispute Resolution/Arbitration. Taxonomy validated against the published CUAD (41 clause categories) and LEDGAR (large-scale multi-label legal provisions) corpora, both sourced from SEC EDGAR Exhibit-10 filings.
- **Data source**: SEC EDGAR Exhibit-10 filings specifically (material commercial contracts — NDAs, licensing, employment, supply agreements), not general 10-K/10-Q financial text. Target corpus size: ~50,000 filings.
- **Data acquisition**: Bulk download must respect SEC's fair-access rate limit (~10 requests/second per host); the download job must support checkpointing/resume so a multi-hour or multi-day run can recover from failure without restarting.
- **ETL/distributed processing**: Spark, running in local mode with multiple partitions (no multi-node cluster required), performs distributed parsing and clause extraction from raw filing text. Clause boundaries are detected primarily via section headers, which in SEC contract exhibits are frequently self-labeling (e.g., a section titled "Indemnification" is indemnification).
- **Clause store**: Elasticsearch (single-node, Dockerized) indexes extracted clauses with metadata — company/CIK, filing date, industry/SIC code, clause type — enabling stratified sampling and filtered querying when building the curated training/eval sets.
- **Labeling methodology**: Semi-automatic header-bootstrap labeling (methodology consistent with LEDGAR's published approach), followed by manual spot-checks on a sample. A label-frequency check across a data sample is run early in Phase 1 to confirm all 10 taxonomy categories appear with reasonable frequency before committing to the full pipeline.
- **Curated dataset sizing**: 500-2,000 example training set (quality-curated subset sampled from the indexed corpus, not the full 50,000), plus a separately hand-picked 30-50 example evaluation set deliberately including ambiguous/edge cases.
- **Leakage prevention**: Train/validation/test splits are partitioned by company/CIK, not by individual clause, to prevent boilerplate/template reuse across filings from leaking near-duplicate text across splits. Near-duplicate embedding-based deduplication is out of scope for v1 (see Out of Scope) but noted as a valid future extension.
- **Base model**: Llama 3 8B, chosen over Mistral 7B for its stronger baseline (making a genuine fine-tuned improvement more credible) and broader first-class tooling support (Unsloth, vLLM, HF PEFT/TRL).
- **Fine-tuning**: LoRA via Hugging Face PEFT + TRL, using Unsloth for memory-efficient training with QLoRA (4-bit quantization) as needed. LoRA targets attention layers (q_proj, v_proj baseline). Hyperparameter sweep covers rank (8/16/32), alpha (2x rank), dropout (0.05), learning rate (1e-4/2e-4/5e-4), and epoch count (1/3/5); all runs logged to Weights & Biases with early stopping on validation loss.
- **Compute target**: RTX 3090 (24GB VRAM), sufficient for the full hyperparameter sweep locally without cloud GPU rental.
- **Evaluation design**: Classification metrics (accuracy, per-class precision/recall/F1, confusion matrix) comparing base vs. fine-tuned model on the held-out evaluation set — replacing the generic "LLM-as-judge 1-5 quality score" pattern, which doesn't fit a fixed-taxonomy classification task. GPT-4o is retained but repurposed as an adjudicator: invoked only when a model's predicted label disagrees with the header-derived label, to determine which label is actually more defensible (since header labels are an imperfect proxy for ground truth). Catastrophic-forgetting check (standard general-capability benchmarks run on both base and fine-tuned model) is retained unchanged from the original project template.
- **Deployment**: Trained LoRA adapter (not the full merged model, <100MB) packaged separately from the base model. Served via vLLM, with both base and fine-tuned model loaded for a dedicated A/B comparison API endpoint returning both models' predicted label and confidence side by side.
- **Experiment tracking**: Weights & Biases (not MLflow) — no self-hosted tracking infra needed given the project already includes self-hosted Spark/Elasticsearch, and W&B dashboards are directly shareable for the portfolio without extra work.
- **Containerization**: Single docker-compose.yml covering Spark, Elasticsearch, the training script, the evaluation harness, and the vLLM inference server — one-command reproducibility across the full pipeline. Local machine has 31GB RAM (~22GB available), sufficient given ETL and training phases run sequentially rather than concurrently.
- **Timeline**: Extended from the original guide's 14-day estimate to ~19-22 days (part-time, 2-3 hrs/day) to accommodate the added large-scale ETL and Elasticsearch scope.

## Testing Decisions

Since this is a greenfield personal project (no existing codebase or issue tracker), "seams" here are the testable boundaries between pipeline stages rather than pre-existing interfaces. A good test in this pipeline verifies external behavior at a stage boundary (input filing text in, correctly labeled/split clauses out; input clause in, predicted label out) rather than internals of Spark's execution plan or the model's internal weights.

- **Highest-value seam: the extraction + labeling function** (raw filing text → structured, labeled clause records). This should be a pure function testable independently of Spark's distributed execution — write it first as a plain Python function operating on a single document, verify it against a handful of known SEC filings with manually-verified expected clause boundaries and labels, then apply it inside the Spark job via `mapPartitions` or equivalent. This keeps the actual business logic (clause segmentation, header-based labeling) unit-testable without needing a live Spark context for every test run.
- **Data leakage test**: an integration-level test asserting that no company/CIK appears in more than one of train/validation/test after splitting — this is a correctness property worth automated verification given it's the main defense against inflated eval numbers.
- **Eval harness test**: verify accuracy/precision/recall/F1 calculations against a small fixture set with known expected metrics (a classic "known distribution" test), so a bug in scoring code doesn't silently produce misleading comparison numbers.
- **Inference/A-B endpoint contract test**: given a fixed clause input, assert the endpoint returns a well-formed response containing both models' predicted label and confidence — a contract test at the API boundary, not a test of model correctness itself (model correctness is measured by the eval harness, not the API layer).
- **Prior art**: none exists in-repo since this is a new project; methodological precedent instead comes from the published LEDGAR and CUAD papers, which should be cited in the experiment report as validation of the labeling and sourcing approach.

## Out of Scope

- Near-duplicate embedding-based deduplication across the full corpus (company/CIK-level split is the required leakage guard for v1; embedding dedup is a valid future extension, not required).
- A real multi-node Spark cluster — local mode with multiple partitions is sufficient to demonstrate distributed-processing design without the operational overhead of a true cluster.
- Cloud GPU rental — the RTX 3090 covers the full training and hyperparameter sweep locally.
- Benchmarking against alternative approaches (few-shot GPT-4o prompting, classical embedding + linear classifier) to "prove" fine-tuning is the optimal choice for this task — the project's justification is framed as a lifecycle/engineering showcase, not an algorithmic-optimality claim, so this comparison isn't required.
- Full manual (non-bootstrapped) labeling of the training set from scratch.
- Any real-time or production-scale serving concerns (load balancing, autoscaling, multi-tenant rate limiting) beyond a single-machine vLLM deployment with an A/B endpoint.
- Any domain other than legal contract clauses (healthcare and code-review domains were considered and explicitly ruled out during scoping).
- Elasticsearch as a general-purpose search product feature (e.g., a search UI) — its role here is limited to backing stratified sampling/filtering for dataset curation, not building a searchable legal-document product.

## Further Notes

- **Interview narrative**: "I built a production-style legal data pipeline (large-scale distributed ETL) *and* the full model-customization lifecycle on top of it" — this is a deliberate two-part story: the ETL/data-engineering layer showcases skills a mid-level SWE already has (Docker, distributed processing, data pipelines) at unusually large scale for a portfolio project, while the fine-tuning layer demonstrates the newer AI-engineering-specific competence (LoRA, evaluation design, experiment tracking, deployment).
- **Target audience context**: the project's framing was partly inspired by Noxtua (a European "sovereign Legal AI" company whose core pitch is proprietary models trained on exclusive legal data, with products spanning legal research, document understanding, and drafting) — this project is pitched as a scoped-down version of their "document understanding" pillar, built entirely on public data.
- **Methodological precedent**: [LEDGAR](https://aclanthology.org/2020.lrec-1.155/) (a published legal-NLP benchmark built from ~60,000 SEC EDGAR Exhibit-10 contracts, semi-automatically labeled) and [CUAD](https://arxiv.org/pdf/2103.06268) (510 expert-labeled SEC EDGAR contracts, 41 clause categories) directly validate this project's data source, scale, and labeling methodology — both are citable in the experiment report as prior art, which is a stronger claim than an invented ad-hoc methodology.
- **Known risk to monitor early**: SEC EDGAR's fair-access rate limit means bulk-downloading 50,000 filings is itself a multi-hour-to-multi-day engineering task (not a quick script), and it should be tackled early in Phase 1 with checkpointing built in from the start, rather than discovered as a blocker mid-project.
- **Scope-in-progress flag**: this PRD reflects an *expanded* version of the original guide's Project 10, with the timeline extended (~19-22 days vs. the guide's original 14) specifically to accommodate the added ETL/Elasticsearch layer — this tradeoff was made deliberately in favor of a stronger data-engineering showcase over strict adherence to the guide's original timeline.
