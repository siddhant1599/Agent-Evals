# Agent-Evals

**An evaluation suite for agentic LLM systems. It scores an agent's final output and also how the agent got there, how much it spent, and how reliably it succeeds.**

Most agent benchmarks report one number: did the final answer look right? Agent-Evals breaks evaluation into separate, reproducible evals. Each one answers its own question about the agent. All of them are computed from recorded runs, with validated judges, paired statistics and full provenance.

> **Systems under test:** single agent (T0, baseline) · Self-Refine agent (T1) · Critique-Revise agent (T2) · RL-trained agent (T5). Backbones used: Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B (4-bit, one A10G GPU). Tasks: XSum, CNN/DM, SAMSum, ASQA. Each is run on 500 frozen items × 3 seeds.

---

## The eval suite

![Eight evals grouped by the question they answer. Outcome and grounding assess answer quality. Reliability and trajectory describe run behavior. Efficiency and comparative evals measure cost and benefit. Ablation and integrity test attribution and trust. Each branch lists its metrics.](docs/eval-suite.svg)

### 1. Outcome evals: is the answer faithful?
Each output is checked against the evidence it was given (the article, the dialogue or the retrieved passages) by NLI-based fact-checking judges. **MiniCheck** is the headline judge, and every comparison is made on it. **AlignScore** serves as a second, independent judge. SummaC and FactCC are reported for continuity with summarization work, and ROUGE-L and BERTScore track whether faithfulness came at the expense of content. Outputs are scored in full, never cut short.

### 2. Grounding & citation evals: are the claims attributed?
On ASQA the agent must answer from 5 retrieved passages and cite them. **Citation recall** asks whether each sentence is supported by the passages it cites. **Citation precision** asks whether each cited passage is needed. **Exact-match recall** checks whether the gold short answers appear at all. Citation markers are removed before faithfulness scoring, so citing cannot inflate the other judges.

### 3. Reliability evals: does it succeed consistently?
An item *passes* when its headline score is ≥ 0.5. Over k = 3 seeds:
- **pass@1**: the expected pass rate of a single run.
- **pass@k**: the item passes on *at least one* seed, i.e. the agent *can* do it.
- **pass^k**: the item passes on *every* seed, i.e. the agent *reliably* does it (tau-bench style).
- **Consistency gap = pass@k − pass^k**: how much single-run accuracy overstates dependability.

Reliability also covers whether runs complete. Every launched run is recorded, and failed or out-of-memory attempts count against the **run-completion rate** instead of being quietly dropped.

### 4. Trajectory evals: what did the agent do?
Every model call is logged to `trace.jsonl` with its role, round, content, token count and finish reason. From these traces:
- **LLM calls and rounds per task**: how much work the agent did.
- **Self-termination rate**: how often a loop stopped on its own stop signal rather than running out of rounds. This shows whether the agent's own judgment controlled the loop.
- **Completion-cap rate** and **blank-output rate**: degenerate trajectories (runaway rewrites, empty answers).
- **Feedback diagnostics** (`run_analysis.py`): what form the agent's self-feedback takes, and how the score changes between the first draft and the final answer.

### 5. Efficiency evals: what did it cost?
Prompt, completion and total tokens per task, and wall-clock seconds per task, are measured on a single GPU. For a trained agent, the **GPU-hours** of its training run are traced from the adapter recorded in the run manifest. Training cost is reported separately and never hidden in per-request numbers.

### 6. Comparative evals: was it worth it?
Each agent is compared with the **single-agent baseline on the same backbone, task and items**:
- **Paired per-item gain** (seeds averaged per item) with a **95% bootstrap CI** (10,000 resamples). The verdict is *better*, *worse* or *no clear difference*, depending on whether the CI excludes 0.
- **Token and latency multipliers**, and **gain per 1k extra tokens**.
- **Pareto frontier**: configurations that no other configuration beats on both faithfulness and tokens.
- **Break-even**: the number of requests after which a one-time training cost is cheaper than running an inference-time loop on every request.

### 7. Ablation evals: which part made the difference?
An ablation runs the same agent, prompts, backbone and items with one component removed. **T2 vs T2-noTool** removes only the Critic's NLI tool, so the paired difference is the tool's own contribution, separate from the critique loop.

### 8. Integrity evals: can the eval be trusted?
An eval is only as good as its judge and its data, so the suite checks itself:
- **Judge validation**: MiniCheck and AlignScore must reproduce their published numbers on human-labelled data (LLM-AggreFact) before any run is scored (`validate_metrics.py`).
- **Judge agreement**: the training-reward judge and the evaluation judge are checked against each other (r ≈ 0.7), so an agent is never graded only by the judge it was trained to please.
- **Leak checks**: every output is scored for verbatim overlap with the prompt's worked examples, so copied text cannot count as faithfulness.
- **Reward-exploitation check** *(in progress)*: RL-trained agents are re-tested on held-out RAGTruth, which is never used for training or tuning. A gain that holds on the training task but not on held-out data indicates a gamed judge.

---

## How a system is evaluated

![Evaluation workflow. Run an agent on 500 frozen items, trace each model call, judge completed outputs against their evidence, and save the run records. Aggregate matched items across agents and seeds to regenerate the scorecard, leaderboard, and figures.](docs/evaluation-workflow.svg)

---

## Results

**On XSum, the RL-trained agent scored 0.707 on MiniCheck faithfulness in one model call per summary.** Critique-Revise, the strongest inference-time loop on average, scored 0.709 in three calls. All architectures were evaluated on the same 500 articles, three backbones and three seeds.

![XSum results averaged across three backbones. Relative to the single-agent baseline, Self-Refine gained 0.018 MiniCheck at 14.6× the inference tokens, Critique-Revise gained 0.039 at 3.7×, and the RL-trained agent gained 0.037 at 1.0×.](docs/xsum-architecture-comparison.svg)

![XSum comparison. Each agentic loop beat the baseline on 2 of 3 backbones, versus 3 of 3 for GRPO. Across all three seeds, 52–55% of articles passed for loops and 57% for GRPO. Loops used 3.0–4.1 calls and 4.35–12.52 seconds per summary. GRPO used 1 call and 1.14 seconds after 33–58 GPU-hours of training per backbone. Loops completed 39 of 39 evaluation attempts and GRPO completed 9 of 16.](docs/xsum-eval-results.svg)

Loop ranges span the Self-Refine and Critique-Revise averages. A faithfulness win requires a 95% paired bootstrap interval above zero. [Full scorecard and per-backbone results](evals/SCORECARD.md)

---

## Running the evals

```bash
pip install -r requirements.txt && pip install -e .
pip install matplotlib                         # optional, for figures

python src/run_evals.py                        # score all runs in results/ -> evals/   (CPU, ~5 s)
python src/run_evals.py --pass-threshold 0.7   # stricter pass criterion
python src/validate_metrics.py ...             # integrity: reproduce judges on human labels
python src/run_analysis.py --config src/configs/analysis/asqa_diagnostics.yaml --set run.owner=<name>
                                               # trajectory / feedback diagnostics
python src/run_benchmark.py --topology all --backbone all --task tier1_xsum --set run.owner=<name>
                                               # produce new runs to evaluate (GPU)
pytest                                         # CPU test suite
```

## Where the evals live

```
evals/                         generated scorecard: SCORECARD.md, leaderboard.csv, scorecard.json, fig/
results/<run_id>/              raw eval records: manifest, per-item metrics, per-call trajectory
src/component/agent_evals/
  scorecard/                   reliability, trajectory, efficiency, comparative and ablation evals
  metrics/                     outcome, citation and leak-check judges + judge validation
  analysis/                    paired bootstrap, trajectory and feedback diagnostics
  datasets/                    frozen evaluation samples
  topologies/, agents/, rl/    the systems under test
src/docs/decisions.md          why every eval choice was made
```

