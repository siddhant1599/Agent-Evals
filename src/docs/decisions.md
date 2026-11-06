# Decision Log — Agent-Evals (formerly FactAgent-Bench)

Append-only. Newest last. See AGENTS.md §12.2 for the rules:

- An entry is **never edited or deleted.** Changing our minds means a *new* entry whose
  `Supersedes:` field names the old one.
- Every entry names the config file that holds the value. Prose in the paper cites the config
  (§11); the config traces back to here.
- **A value sitting in a config with no `D-` entry was never approved by anyone.** That is the
  point of this file.
- Open questions live in `questions_to_ask.md` until they are answered, then arrive here.

This file is the draft of the paper's methods section. Write entries as if a reviewer will read
them, because one will.

---

### D-001 · 2026-09-10 · Backbones are exactly the three named in the proposal

**Decision:**
The three backbones are `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.3`, and
`Qwen2.5-7B-Instruct`. No substitutions. `model_registry.csv` is the source of truth for the exact
revisions actually served, and code never hardcodes a model name (AGENTS.md §4.1).

**Reason:**
The proposal commits to these three by name, and a released benchmark whose backbones differ from
its own protocol is the first inconsistency a reviewer finds. The three were chosen as distinct
model families (Meta / Mistral / Alibaba), which is what makes the H3 heterogeneous-vs-homogeneous
contrast a comparison across *families* rather than across checkpoints of one lineage.

**On the feasibility gate:**
The Week 2 GRPO smoke test ran on `Qwen/Qwen3-8B`, which is not on this list. It does **not** need
re-running. Peak memory in GRPO is driven by parameter count and by the rollout group held in
memory, and every model on the list is ≤ 8B — so a gate that passed at 8B passes at 7B. The recorded
figures (13.9 GB peak reserved at a 512-token prompt cap, 22.6 GB at 1080) stand as an upper bound
for all three backbones.

When the numbers are quoted in the paper, they are cited as measured on an 8B model as an upper
bound, not as measured on the trained backbone. That sentence costs nothing and prevents the
objection.

**Source:** proposal §1 and §4 (Week 2); `RL Feasability test/grpo-smoke-test/`
**Approved:** Sameer, 2026-09-10
**Config:** `src/model_registry.csv` (to be created, Week 1–2)
**Supersedes:** —

---

### D-002 · 2026-09-10 · Evaluation sample: 500 documents per dataset, length-stratified, frozen

**Decision:**
For every Tier 1 and Tier 2 dataset, draw **500 documents** once, with a recorded seed, and freeze
them. Specifically:

- **500 per dataset**, stratified into **length terciles by whitespace word count**, equal draws
  from each tercile.
- Terciles are computed **within each dataset**, never pooled across datasets.
- The **same 500 documents** are used by all six topologies (T0–T5) and all three backbones.
- The sample is drawn **once and never resampled.** Seeds 0/1/2 vary generation only, not sampling.
- Source length is recorded per example as an analysis variable, alongside the per-backbone token
  count.

**Reason:**

*Why 500:* it sits at the top of the precedent band for XSum factuality work — Maynez et al. used
500 documents, FRANK used 250, Roit et al. used 100 for human evaluation. It is also close to the
ceiling our compute supports: the Tier 1 sweep is 6 topologies × 3 backbones × 3 datasets × 3 seeds
× 500, and T3 costs roughly ten times T0's calls per example. If Week 2's measured throughput says
500 does not fit, N is reduced by a superseding entry carrying the measured number — not by a quiet
config edit.

*Why the same documents everywhere:* it makes every topology comparison **paired**. Comparing T0 to
T3 article-by-article cancels out document difficulty, which gives materially more statistical power
per example than two independent samples would. Different samples per condition would be strictly
worse and would need a larger N for the same power.

*Why stratify by length at all:* a purely random draw follows the natural distribution, and the long
tail is rare enough that we could end up with a dozen long documents — too few to say anything
about. Stratification guarantees ~167 per bucket, which is what makes "does agentic critique help
more on long inputs?" an answerable question rather than a footnote. It also feeds Week 13's
regression directly.

*Why word count and not tokens:* Llama, Mistral and Qwen have different tokenizers, so the same
article is a different token count in each. Stratifying on tokens would produce a
backbone-specific sample, which would break the paired design and confound H2 (backbone dependence)
with sample differences. Word count is tokenizer-independent, so the sample is identical across all
three. Per-backbone token counts are still recorded separately for the cost analysis.

*Why terciles and not fixed thresholds:* the boundaries come from the measured distribution, so
there is no invented number to defend. They are reported in the paper as measured word counts.

*Why per-dataset and not pooled:* CNN/DailyMail articles are much longer than XSum's. Pooled
terciles would make "short" mean XSum and "long" mean CNN/DM, turning length into a proxy for
dataset. Per-dataset terciles keep "long XSum" and "long CNN/DM" as distinct, populated cells.

**Considered and rejected:** stratifying by abstractiveness (novel n-gram ratio), which is arguably
closer to the true hallucination driver. Rejected because the three Tier 1 datasets already span
that axis *between* datasets — XSum is highly abstractive by construction, CNN/DM more extractive,
SAMSum a different register. Length is the better *within*-dataset stratifier and doubles as the
compute variable.

**Source:** proposal §2 ("Fix a stratified evaluation sample per (task, tier) … with a documented
random seed"); Maynez et al. XSum hallucination annotations (500 docs); FRANK (250); Roit et al.
2023 §3.6 (100 for human eval)
**Approved:** Sameer, 2026-09-10
**Config:** `src/configs/data/tier1_*.yaml` (to be created)
**Supersedes:** —

---

### D-003 · 2026-09-10 · Core infrastructure parameters

**Decision:**

| Parameter | Value |
| --- | --- |
| `run.seed` | `0` per run; the reporting protocol is seeds 0, 1, 2 |
| `run.owner` | **required, no default**, validated against `known_owners` |
| `run.known_owners` | `(sameer, kartik, siddhanth)` |
| `run.output_root` | `runs` |
| `logging.console_level` | `INFO` |
| `logging.file_level` | `DEBUG` |
| `logging.flush_metrics_every` | `1` |
| `seeding.deterministic` | `false` |

**Reason:**

*Seeds 0/1/2:* AGENTS.md §3 requires at least three seeds reported as mean and standard
deviation, but names no values. 0/1/2 is obvious and repeatable, and nobody has to wonder later
where a number like 1337 came from.

*`owner` required with no default:* an unattributed run must be impossible to start. Loading
`base.yaml` on its own raises, which is deliberate. Validating against a fixed list means a typo or
an agent-invented value fails loudly rather than being silently recorded as provenance — the
weakness identified while drafting Q-003.

*`file_level` more verbose than `console_level`:* `run.log` is the forensic record for a job that
failed unattended overnight; the console is for a human watching. Different audiences, different
levels.

*`flush_metrics_every = 1`:* a buffered writer loses the tail of exactly the runs whose evidence
matters most — the ones killed by OOM. AGENTS.md §6 makes OOM a recorded outcome, which only works
if the records survive it. The cost is one flush per example, negligible next to a model call.

*`deterministic = false`:* pending Q-002. It does nothing for vLLM-served inference, which runs in
a separate process, and may raise rather than merely slow down on 4-bit QLoRA kernels.

**Not made config fields, deliberately:** the `run_id` format (AGENTS.md §3 says "nothing else", so
it is a module constant) and any `allow_dirty` flag (§3 marks a dirty run and forbids citing it —
it never refuses to run, so there is nothing to configure).

**Source:** AGENTS.md §3, §6, §12.1; questions_to_ask.md Q-002, Q-003
**Approved:** Sameer, 2026-09-10
**Config:** `src/configs/base.yaml`
**Supersedes:** —

---

### D-004 · 2026-09-10 · VRAM is recorded in GiB, and the keys say `gib`

**Decision:**
Every VRAM figure the run contract writes is in **GiB** — bytes divided by `1024**3` — and the keys
that carry it are named accordingly:

| Old key | New key | Written to |
| --- | --- | --- |
| `gpu.total_vram_gb` | `gpu.total_vram_gib` | `manifest.json` |
| `peak_vram_gb` | `peak_vram_gib` | `summary.json` |

The arithmetic does not change; only the label does. Paper prose that quotes peak VRAM writes
"GiB", and may give the vendor's decimal figure alongside it when naming the card.

**Reason:**

*Why the rename was needed:* the code already divided by `1024**3`, which is GiB, while the key said
`gb`. On this A10G that produces `total_vram_gib: 22.49` against a vendor spec sheet that says
24 GB. A reader who takes the key at its word concludes the card is smaller than it is. AGENTS.md
§11 makes peak VRAM a mandatory reported figure for every condition, so the label is not internal
bookkeeping — it lands in the paper.

*Why GiB rather than switching the arithmetic to decimal GB:* every other memory number this
project compares against is binary. `nvidia-smi` reports MiB — it is the same command this module
already shells out to for the driver version — and `torch.cuda.max_memory_reserved()` returns bytes
that the QLoRA and vLLM literature conventionally divides by `1024**3`. Reporting decimal GB would
make our peak-VRAM figures silently non-comparable with the 24 GB budget named in AGENTS.md §12.1
and with the feasibility-gate measurements, which is a worse failure than an unfamiliar suffix.

*Why now:* `runs/` does not exist yet, so no run has written either key. AGENTS.md §4.5 fixes result
keys permanently once a run has used them, and the same reasoning applies to the manifest and
summary contract in §3. This rename is free today and would cost a migration of every stored run
later.

**Not a config field, deliberately:** the unit is a property of the reproducibility contract in
AGENTS.md §3, not an experimental parameter. Making it configurable would let two runs report
`peak_vram` in different units under the same key, which is the exact ambiguity this entry closes.

**Source:** AGENTS.md §3 (manifest and summary contract), §4.5 (keys are stable once used), §11
(cost accounting is mandatory); `nvidia-smi` on the project A10G reports `23028 MiB` total
**Approved:** Sameer, 2026-09-10
**Config:** none — the values are produced by `gpu_info()` and `_peak_vram_gib()` in
`src/component/factagent_bench/run.py`
**Supersedes:** —

---

### D-005 · 2026-09-10 · Tier 1 XSum filtering: a minimum word count, and no maximum

**Decision:**

| Parameter | Value |
| --- | --- |
| `data.min_source_words` | `20` |
| `data.max_source_words` | `null` — no upper bound is applied |
| `data.sample_seed` | `123` |

Documents longer than the serving context budget are **truncated at generation time, not excluded
at load time**, and the truncated text becomes that example's `evidence`. Source word count and a
`truncated` flag are recorded per example. The `max_source_words` field exists in the config and is
left unset, so an upper bound can be introduced later by a superseding entry without a schema
change.

**Reason:**

*Why no maximum — the literature does not support one.* Both papers this project cites for its XSum
protocol were checked directly for a truncation or maximum-length policy, and neither states one:

- **Maynez et al. 2020** — the precedent D-002 cites for N = 500 — says "We randomly sampled 500
  articles from the test set to facilitate our study." The draw was purely random with **no length
  filtering of any kind**, and the paper states no input truncation or token limit, deferring model
  hyperparameters to the original model papers.
- **Roit et al. 2023** states no maximum input length, no truncation policy and no sequence-length
  limit for XSum anywhere in its experimental setup or appendix. The only length figure reported is
  *observed* output length (~17–21 tokens), which is a measurement, not an enforced cap.

D-002 already departs from Maynez once, by stratifying rather than sampling purely at random, and
argues that departure on its own merits. Adding a maximum-length filter would be a *second*
departure with nothing to cite behind it, and it would remove the longest documents — the cases the
length-stratification exists to study. An uncited exclusion of the hardest 8% of the benchmark is
exactly the methodological choice a reviewer challenges first.

*Why truncation instead.* `types.py` already fixes the rule: "When a source is truncated to fit a
context window, `evidence` must be the truncated text the model actually saw — scoring faithfulness
against text the model was never shown measures nothing." Truncation is therefore already the
project's sanctioned mechanism for the context budget, and it keeps the sample comparable to
Maynez's while filtering would not.

*Why `min_source_words = 20` — measured, and free.* Computed on the XSum test split (11,334 rows) on
2026-09-10: exactly **1 document is empty**, 4 are ≤ 10 words, and 18 are ≤ 20 words. A minimum of
20 removes 14 rows, retains 99.9% of the split, and leaves the tercile boundaries **completely
unchanged at 217 / 409 words** — verified before and after. It deletes the degenerate rows at no
cost to the distribution.

*Why `sample_seed` is a separate field from `run.seed`.* D-002 requires the evaluation sample to be
drawn once and frozen, and states that seeds 0/1/2 "vary generation only, not sampling." These are
two independent random processes and sharing a field would let them couple. If the sample seed ever
tracked `run.seed`, seeds 1 and 2 would silently evaluate *different documents*, and every paired
topology comparison in the paper would be invalid while still looking correct. `123` is arbitrary as
a value; its only requirement is not being 0, 1 or 2, so the two roles can never be confused by
eye.

**Ordering this implies:** filter → compute terciles → draw the stratified sample. Terciles are
computed on the distribution that survives filtering, or the boundaries describe data that was
excluded.

**Known risk, accepted and recorded:** 8.4% of XSum test documents exceed 800 words (measured,
2026-09-10). For those, the model sees a prefix while the reference summary was written from the
whole article, so the reference may contain facts absent from the evidence and the example will
score as unfaithful for a reason unrelated to the topology. This is why `truncated` is recorded per
example: Week 13 reports factuality with and without those rows rather than the effect going
unmeasured. Deleting the rows instead would have hidden the same problem and cost the long tercile.

**Source:** Maynez et al. 2020, *On Faithfulness and Factuality in Abstractive Summarization*
(https://aclanthology.org/2020.acl-main.173/), §4; Roit et al. 2023
(https://aclanthology.org/2023.acl-long.344.pdf), experimental setup and Appendix A — both read
directly on 2026-09-10 and checked for a length policy; D-002; `types.py` `Example` docstring;
XSum test-split distribution measured 2026-09-10
**Approved:** Sameer, 2026-09-10
**Config:** `src/configs/data/tier1_xsum.yaml` (to be created)
**Supersedes:** —

---

### D-006 · 2026-09-11 · The evaluation sample is frozen as explicit IDs, not re-derived from a seed

**Decision:**

The frozen evaluation sample is stored as an **explicit, ordered list of 500 example IDs** in
config, not as a seed that is re-drawn at load time. Two separate operations:

| Operation | What it does | When it runs |
| --- | --- | --- |
| `DatasetAdapter.draw_sample_ids()` | draws 500 IDs from the filtered, id-sorted pool using `data.sample_seed`, and returns them | **once**, by hand, at freeze time |
| `DatasetAdapter.build()` | replays a given ID list in its recorded order | every run |

`build()` **refuses to run** while `data.sample_ids` is `None`, and `draw_sample_ids()` refuses to
redraw once IDs are frozen. Neither ever silently substitutes a document. The upstream dataset
revision is pinned alongside the IDs in `data.hf_revision` — as a 40-character commit SHA, format-
validated at config load, so a mutable tag such as `main` cannot be recorded in its place.

**Reason:**

*Why a seed alone is not enough.* A seed reproduces a draw only if the input to that draw is
unchanged. `load_dataset` returns whatever the upstream repository currently holds, so if the
dataset is re-uploaded with rows in a different order, or rows are added or removed, seed `123`
silently selects a **different 500 documents**. Nothing errors, the config still reads `123`, and
the paper's paired comparisons would rest on a sample nobody could reconstruct. Recording the IDs
converts that silent substitution into a loud failure: `build()` raises with the specific missing
IDs listed.

*Why this matters more here than in most projects.* D-002 makes the identical 500 documents across
all six topologies and all three backbones the basis of the paired design — that pairing is what
buys statistical power per example. A sample that quietly drifts between one topology's run and
another's would invalidate every comparison in the paper while every individual run still looked
correct.

*Why the IDs are also a defence against our own config changes.* If `min_source_words` is later
raised, some frozen IDs fall out of the filtered pool and `build()` refuses rather than returning
a short sample. The frozen list therefore pins the *filter* as well as the draw.

*Why the pool is sorted by `id` before drawing.* Upstream row order is not part of the contract, so
sorting makes the draw depend only on the set of surviving rows and the seed — not on how the rows
happened to arrive.

*Why sampling uses a private RNG.* `seeding.make_sampling_rng()` returns a fresh `random.Random`
rather than touching the global generators that `set_seed()` controls. The sampling stream and the
generation stream are therefore incapable of coupling, which is the structural version of the
separation D-002 requires in prose.

**Redistribution:** the committed artifact is the **ID list only**, never the article text. XSum's
HuggingFace dataset card states its license as `"unknown"`, and the original release distributes a
scraper rather than the BBC article text. 500 IDs plus a pinned revision reproduce the sample
exactly without redistributing anything (3.9 KB of IDs against 1.10 MB of article text — the choice
is about provenance, not size). Any local materialization of the rows is a git-ignored cache.

**Considered and rejected:** storing the 500 documents as a CSV or parquet file in the repository.
It would make the benchmark runnable offline and immune to upstream removal, which is genuinely
attractive, but it redistributes news text under an unstated license in a repository intended for
public release. The pinned revision already guarantees the exact bytes.

**Source:** D-002 (frozen, paired evaluation sample); AGENTS.md §3 (a run must be reproducible from
its recorded config alone); XSum dataset card license field, checked 2026-09-11; sizes measured
2026-09-11
**Approved:** Sameer, 2026-09-11
**Config:** `src/configs/data/tier1_xsum.yaml`, with the ID list in
`src/configs/data/tier1_xsum_sample.yaml`
**Supersedes:** —

---

### D-007 · 2026-09-12 · Sampling temperature 0.7, in one decoding config shared by T0–T5

**Decision:**
Generation uses **temperature 0.7**, set once in a single decoding config that every topology
reads — T0 through T5. T0 is therefore **sampled, not greedy**, and is reported at seeds 0/1/2 like
every other condition. This resolves Q-001 as its Option B. Approved "for now": it changes only by
a superseding entry.

**Reason:**

*Why sampled rather than greedy.* Two commitments already made rule greedy T0 out:

1. AGENTS.md §3 requires every reported result to be ≥ 3 seeds, mean ± std. Under greedy decoding
   the same prompt and model produce byte-identical output on every seed, so T0 would cost three
   runs and report std = 0 — the only condition in the benchmark with no error bar, turning every
   "T1 beats T0 by X" into a comparison against a point estimate.
2. Homogeneous T3 cannot run greedy: three debaters sharing one backbone and prompt would emit
   identical text, leaving nothing to debate. If T0 were greedy while T3 sampled, the conditions
   would differ in decoding strategy as well as topology, and H1 — whether factuality rises with
   topology sophistication — would be confounded. Decoding is held constant instead.

*Why 0.7.* It is the only decoding temperature stated by any topology paper this project cites.
Madaan et al. 2023 (Self-Refine), §3.1: "We use greedy decoding with a temperature of 0.7 for all
setups." That sentence is self-contradictory — greedy decoding and temperature 0.7 are mutually
exclusive — so the operative value is cited as **temperature 0.7** and their wording is not
repeated. Du et al. 2023 (debate) states no temperature at all. Both papers use frontier models
only, so 0.7 is a deliberate choice anchored to the one available precedent, and is described as
such — never as "the value from the literature".

**Not decided here:** `top_p` and the maximum completion length. Self-Refine states neither (paper
re-checked 2026-09-12 for both), so they are proposed separately with their own sources.

**Source:** questions_to_ask.md Q-001; Madaan et al. 2023, *Self-Refine*, §3.1 (arXiv 2303.17651),
read 2026-09-10 and re-checked 2026-09-12; Du et al. 2023 (arXiv 2305.14325); AGENTS.md §3
**Approved:** Sameer, 2026-09-12
**Config:** `src/configs/base.yaml`, `sampling.temperature` (to be created with `serving/`)
**Supersedes:** —

---

### D-008 · 2026-09-12 · Serve through vLLM 0.27.1's built-in bitsandbytes path (NF4 + double quant)

**Decision:**
All backbones are served by **vLLM 0.27.1** with **bitsandbytes 0.50.2**, quantized in flight to
4-bit **NF4 with double quantization** when the BF16 checkpoint loads. This is the Q-006 choice of
one quantizer for serving and QLoRA training, implemented on the vLLM release that still ships
bitsandbytes in-tree. **Provisional:** Q-006 still requires Dr. Jafari's confirmation.

**Reason:**

*vLLM no longer ships bitsandbytes.* Commit `073c510c91` (PR #43529, "[Migration] Migrate
bitsandbytes support to OOT plugin", merged 2026-08-09) moved it to `vllm-bnb-plugin`. Checked on
2026-09-12: the vLLM 0.29.0 wheel registers 30 quantization methods and `bitsandbytes` is not among
them; `vllm/model_executor/layers/quantization/bitsandbytes.py` exists at tag `v0.27.1` and not at
`v0.28.0`. So 0.27.1 is the last release with the path Q-006 assumed.

*It really is NF4 with double quantization.* The in-flight loader calls
`quantize_4bit(..., compress_statistics=True, quant_type="nf4")` — `compress_statistics` is
bitsandbytes' name for double quantization (v0.27.1 `bitsandbytes_loader.py`, lines 435–436). The
`fp4` / no-double-quant defaults visible in its `BitsAndBytesConfig` class apply only when reading
an already-quantized checkpoint that omits those keys; in the plugin, whose loader has the identical
call, the surrounding code was read in full and confirms the split between the pre-quantized and the
in-flight (`_unquantized_generator`) paths. NF4 with double quantization is the format Dettmers et
al. 2023 show recovering 16-bit LoRA quality (their Table 2), and the one QLoRA training uses.

*Why 0.27.1 and not vLLM 0.29.0 with the plugin.* The plugin is at v0.0.3 (released 2026-08-24),
declares `vllm` with no version bound, builds its releases on Python 3.12, and its latest commit
exists only to follow a method renamed on vLLM main. vLLM 0.29.0 was released 2026-09-09 — after
the plugin's last release — so that pairing is untested. A static check found all 24 `vllm` names
the plugin imports still resolve in the 0.29.0 wheel; that is necessary, not sufficient. For a
baseline other work depends on, the configuration vLLM itself released and tested was preferred.
Every release from 0.27.1 to 0.29.0 pins the same torch (D-009), so the choice costs nothing there.

**Not yet verified at runtime:** loading a backbone through this path on the A10G. Minimum compute
capability in the source is 7.0 (v0.27.1 `bitsandbytes.py`, `get_min_capability`); the A10G is 8.6.

**Source:** Q-006; vLLM PR #43529 and commit `073c510c91`; vLLM 0.29.0 wheel and `v0.27.1` /
`v0.28.0` tags, inspected 2026-09-12; `vllm-bnb-plugin` repository and PyPI metadata, 2026-09-12;
Dettmers et al. 2023 (arXiv 2305.14314), Table 2
**Approved:** Sameer, 2026-09-12 (provisional pending Dr. Jafari)
**Config:** `requirements.txt`; serving config under `src/configs/` (to be created with `serving/`)
**Supersedes:** —

---

### D-009 · 2026-09-12 · torch 2.14.0 → 2.13.0, one environment for serving and training

**Decision:**
The project venv moves to **torch 2.13.0** and **numpy 2.4.6**, and vLLM is installed into the same
environment rather than a separate one. `requirements.txt` is re-pinned accordingly.

**Reason:**
Every vLLM release from 0.27.1 through 0.29.0 declares `torch==2.13.0` exactly (PyPI metadata,
checked 2026-09-12); no release supports 2.14. Installing vLLM 0.27.1 therefore changed, measured
from a before/after `pip freeze`: torch 2.14.0 → 2.13.0, numpy 2.5.3 → 2.4.6, nvidia-cudnn-cu13
9.24.0.43 → 9.20.0.48, nvidia-nccl-cu13 2.30.7 → 2.29.7, triton 3.8.0 → 3.7.1, setuptools
84.0.0 → 80.10.2, and added 141 packages. torch 2.13.0 is still the CUDA 13.0 build
(`2.13.0+cu130`), which the instance driver supports.

One environment was chosen over a separate server venv because it keeps a single torch and
bitsandbytes build under both serving and QLoRA training — the same reasoning Q-006 applies to the
quantizer. The trade-off: vLLM's exact pin now dictates the torch the training stack runs on, and
any future vLLM upgrade moves torch with it.

**Verified after the change:** 183 tests pass, ruff and black clean, `gpu_info()` reports the A10G
available, and the manifest records torch 2.13.0, vLLM 0.27.1, bitsandbytes 0.50.2. No benchmark
run existed before the change, so no evidence was produced under the old versions.

**Source:** vLLM 0.27.1 / 0.28.0 / 0.29.0 PyPI metadata; pip freeze before and after, 2026-09-12
**Approved:** Sameer, 2026-09-12
**Config:** `requirements.txt`
**Supersedes:** —

---

### D-010 · 2026-09-12 · Qwen2.5-7B-Instruct is the backbone for the first T0 pilot

**Decision:**
The first T0 run — the pilot for the 2026-09-15 advisor meeting — uses
**`Qwen/Qwen2.5-7B-Instruct`** pinned at revision `a09a35458c702b33eeacc393d103063234e8bc28`.
This selects a backbone for one pilot only: D-001 is unchanged, all three backbones remain in scope,
and the deep-track backbone is still chosen from the Week 2 sanity checks.

**Reason:**
Of D-001's three backbones, the HuggingFace API reports `Llama-3.1-8B-Instruct` as gated
(`"manual"`) and the other two as open. Agents never handle HF tokens (AGENTS.md §12.7), so a
Llama pilot waits on Sameer authenticating. Between the two open models, Qwen2.5-7B-Instruct is the
smaller download — 14.19 GiB of safetensors against 27.0 GiB in the Mistral-7B-Instruct-v0.3
repository — and ships a `chat_template` in `tokenizer_config.json`, which vLLM's chat endpoint
requires. Downloaded 2026-09-12 to the local HF cache at the pinned revision.

The pilot is a single-backbone, single-seed capacity run and is labelled as such (AGENTS.md §3); it
is not a reported result.

**Source:** HuggingFace model API (`gated`, `sha`, safetensors sizes), 2026-09-12; D-001
**Approved:** Sameer, 2026-09-12
**Config:** `src/model_registry.csv` (to be created with `serving/`)
**Supersedes:** —

---

### D-011 · 2026-09-12 · Sampling uses each backbone's own shipped defaults

**Decision:**
Every sampling setting — temperature, top-p, top-k, min-p, repetition penalty — comes from **each
backbone's own `generation_config.json`** at its pinned revision. The serving client sends none of
them; vLLM runs with its default `--generation-config auto`, which applies the model's file to any
parameter a request leaves unset. Only the per-request `seed` and `max_tokens` (D-012) are sent.
This supersedes D-007's single shared decoding config at temperature 0.7.

**Reason:**
The project studies multi-agent topologies on these models **as they ship**, not how decoding
settings change their outputs. Holding every backbone to one hand-picked configuration would make
each model something other than what a practitioner downloading it actually runs, and would make
decoding a project-chosen variable rather than a property of the model.

*What D-007 was protecting still holds.* D-007's first concern was that topology comparisons must
not be confounded by decoding. Under this decision every topology on a given backbone uses that
backbone's defaults, so T0-versus-T1–T5 comparisons — H1 and H5 — still hold decoding constant.
D-007's second concern was that sampling is required for seed variance and for homogeneous debate;
both backbones whose defaults could be checked sample rather than decode greedily (below).

*What changes, and must be stated in the paper.* Decoding now differs **across** backbones by the
vendor's choice, so backbone comparisons (H2) and heterogeneous ensembles (H3) compare models
including their shipped decoding behaviour. That is the intended reading under this decision, not a
confound to correct.

**Effective defaults, read from the pinned files (2026-09-12):**

| Backbone | Shipped sampling settings | What vLLM applies |
| --- | --- | --- |
| Qwen2.5-7B-Instruct @ `a09a354` | `temperature` 0.7, `top_p` 0.8, `top_k` 20, `repetition_penalty` 1.05, `do_sample` true | those values |
| Mistral-7B-Instruct-v0.3 @ `c170c70` | none — the file holds only token IDs | vLLM's neutral defaults: `temperature` 1.0, `top_p` 1.0, `top_k` 0 (off), `min_p` 0.0, `repetition_penalty` 1.0 |
| Llama-3.1-8B-Instruct @ `0e9e39f` | **unverified** — the repository is gated | to be read once Sameer authenticates |

Mistral's row is vLLM's fallback rather than a vendor choice, and is described that way. The
effective values for every backbone are recorded in `model_registry.csv` so every run's decoding is
reconstructible from its recorded revision.

**Source:** vLLM 0.27.1 source — `ModelConfig.generation_config` default `"auto"` and
`get_diff_sampling_param` (`config/model.py`), `ChatCompletionRequest.to_sampling_params` and
`_DEFAULT_SAMPLING_PARAMS` (`entrypoints/openai/chat_completion/protocol.py`); the pinned
`generation_config.json` files; D-007
**Approved:** Sameer, 2026-09-12
**Config:** `src/model_registry.csv` (to be created with `serving/`); no sampling fields in
`src/configs/`, by design
**Supersedes:** D-007

---

### D-012 · 2026-09-12 · Maximum completion length: 256 tokens

**Decision:**
Every generation request sets `max_tokens` to **256**, for every topology and backbone. Every
response's `finish_reason` is recorded, and the share of outputs stopped by the cap
(`finish_reason == "length"`) is reported for every run.

**Reason:**
*A cap is required.* vLLM's chat endpoint has no default: with `max_tokens` unset it generates until
the model stops or the context fills (`get_max_tokens`, `entrypoints/serve/utils/api_utils.py`) —
about 2,400 tokens for a median prompt under D-013's context length. An unbounded rambling output
inflates token cost, which is a reported result (H4).

*The only published value is the XSum paper's.* Narayan et al. 2018, "Model Parameters and
Optimization": "the length of the summary limited to 90 tokens" — lowercased *word* tokens, for the
models they trained. No other paper this project cites states a completion cap: Roit et al. 2023,
Maynez et al. 2020, Madaan et al. 2023 and Du et al. 2023 were each checked.

*Why 256 rather than their 90.* A trained summarizer learns to stop short; an instruct model can
overrun, and a tight cap would cut a sentence mid-way and score the fragment as a poor summary.
Measured on the frozen 500 with Qwen's tokenizer, reference summaries run median 26, p99 42, maximum
57 tokens, at 1.26 tokens per word. Narayan's 90 words is therefore about 113 Qwen tokens, and 256
tokens is about 200 words: the cap is **looser** than the source's, never stricter, and 4.5× the
longest reference.

*Why the exact number matters little.* A cap changes no output it does not reach: generation stops
at the model's end-of-sequence token, and sampling a token does not depend on the cap. The reported
`"length"` share is what makes the value defensible — if it is non-zero, the cap is binding and is
raised by a superseding entry.

**Source:** Narayan et al. 2018 (arXiv 1808.08745), "Model Parameters and Optimization"; Roit et
al. 2023; Maynez et al. 2020; Madaan et al. 2023; Du et al. 2023; vLLM 0.27.1 source; token counts
measured on the frozen XSum sample 2026-09-12
**Approved:** Sameer, 2026-09-12
**Config:** `src/configs/base.yaml`, `sampling.max_tokens` (to be created with `serving/`)
**Supersedes:** —

---

### D-013 · 2026-09-12 · Serving context length for Qwen2.5-7B-Instruct: 2,809 tokens

**Decision:**
vLLM serves Qwen2.5-7B-Instruct with `max_model_len` **2,809**. GPU memory utilization is **not
set**, so vLLM 0.27.1's default of 0.92 applies (`CacheConfig`, `config/cache.py`).

**Reason:**
The context must hold the longest prompt plus the completion cap, or that example fails. Measured
on all 500 frozen XSum documents with Qwen's tokenizer and its own chat template applied (default
system message included, per D-011's use of shipped templates): prompt tokens median 424, p99 1,724,
maximum **2,553**, under the "in one sentence" wording. 2,553 + 256 (D-012) = 2,809, so **no frozen
document is truncated**. The model supports 32,768 positions (`max_position_embeddings`), so this is
well inside its range. A larger value would reserve KV-cache capacity no request uses.

The figure is specific to Qwen's tokenizer and template. Each other backbone gets its own measured
value before it is served.

**Source:** token counts measured 2026-09-12; D-012; Qwen2.5-7B-Instruct `config.json` at `a09a354`
**Approved:** Sameer, 2026-09-12
**Config:** `src/model_registry.csv` (to be created with `serving/`)
**Supersedes:** —

---

### D-014 · 2026-09-12 · Serving client, backbone check command, timeouts and secrets

**Decision:**

| Item | Value |
| --- | --- |
| `serving.request_timeout_s` | `120` |
| `serving.startup_timeout_s` | `600` |
| HTTP client | Python standard library (`urllib.request`); no new dependency |
| Backbone check | `python3 src/check_backbone.py --backbone <name>`, names from `model_registry.csv`: `qwen2.5-7b-instruct`, `mistral-7b-instruct-v0.3`, `llama-3.1-8b-instruct` |
| Secrets | `.env` at the repository root, git-ignored (`.gitignore:151`), file mode 600, holding `HF_TOKEN` and `FACTAGENT_VLLM_URL` |

**Reason:**

*Request timeout, 120 s.* The smoke request on 2026-09-12 (one XSum article: 283 prompt tokens, 56
completion tokens) took 1.31 s cold and 0.76 s warm. A timed-out call is recorded as a failure and
is never retried silently, so the value only decides when a stuck call is abandoned. Set too low,
slow but valid calls under concurrent load would be recorded as failures. No paper gives a value;
this is a deliberate default with that trade-off.

*Startup timeout, 600 s.* The server became ready after 150 s, measured on the second start — after
a crashed first attempt had already created vLLM's compile-cache directory — so a start with an
empty cache, or a larger backbone, may be slower. 600 s is four times the measurement. Set too low,
it aborts a healthy startup; set too high, it only delays reporting a server that has hung.

*Standard library, not a new dependency.* One JSON POST with a timeout is a few lines of
`urllib.request`, and AGENTS.md §9 rules out a library for that. Tests substitute the transport, so
the suite still needs no network.

*Secrets never appear on a command line.* Process arguments are visible to every user of the
machine through `ps`, so the token reaches vLLM only through the environment the server inherits.
The `.env` loader reports variable names and never values, and error messages name the line rather
than its contents. `.env` holds secrets and the endpoint URL only — never an experimental parameter.

*The check prompt is a connectivity probe, not a benchmark prompt.* `check_backbone.py` sends one
short fixed message to confirm the server loads the backbone and returns text, token counts and a
finish reason. Its output is never scored, so its wording is not part of the T0 prompt decision.

**Source:** smoke test logs 2026-09-12 (a check, not a run: no `runs/` directory written); AGENTS.md
§2, §9, §12.7
**Approved:** Sameer, 2026-09-12
**Config:** `src/configs/base.yaml` (`serving.*`); `src/model_registry.csv`; `.env`
**Supersedes:** —

---

### D-015 · 2026-09-12 · Llama-3.1-8B-Instruct access verified; its shipped defaults recorded

**Decision:**
Completes the row D-011 left unverified. Gated access to `meta-llama/Llama-3.1-8B-Instruct` works
with Sameer's token: `config.json`, `generation_config.json` and `tokenizer_config.json` downloaded
at the pinned revision `0e9e39f249a16976918f6564b8830bc894c89659`. The token was read from `.env`
into the environment of the checking process only and was never displayed. The weights were not
downloaded.

| Setting | Value | Origin |
| --- | --- | --- |
| `temperature` | 0.6 | model `generation_config.json` |
| `top_p` | 0.9 | model `generation_config.json` |
| `do_sample` | true | model `generation_config.json` |
| `top_k`, `min_p`, `repetition_penalty` | 0, 0.0, 1.0 | absent from the file, so vLLM's neutral defaults apply |

The model samples by default, so D-011's requirement that every backbone samples rather than decodes
greedily holds for all three. `max_position_embeddings` is 131,072. Its chat template is present and
contains the text "Cutting Knowledge Date" — backbone-specific system text used unchanged under
D-011; the exact rendered prompt is examined when its context length is measured.

**Not yet done:** Llama's `max_model_len` (D-013's rule — longest prompt with this backbone's own
tokenizer and template, plus 256), which waits on the approved T0 prompt wording.

**Source:** HuggingFace downloads at the pinned revision, 2026-09-12; D-011; D-013
**Approved:** Sameer, 2026-09-12
**Config:** `src/model_registry.csv`
**Supersedes:** —

---

### D-016 · 2026-09-12 · XSum prompt: "Summarize the following article in one sentence:"

**Decision:**
For XSum, the generation step sends each article as **one user message**: the line
`Summarize the following article in one sentence:`, a blank line, then the article text. No system
message and no examples: T0 is the zero-shot baseline, and each backbone's own chat template wraps
the message unchanged (D-011). The wording is identical for every topology's generation step and
every backbone on XSum. It is a versioned template owned by the Generator role module, and its
version is recorded in each run's manifest.

The wording is XSum's, not a universal prompt: CNN/DailyMail, SAMSum and the Tier 2 datasets each
get an instruction matching their own reference format when their loaders arrive.

**Reason:**
XSum is defined as single-sentence summarization. Narayan et al. 2018, abstract: "The idea is to
create a short, one-sentence news summary answering the question 'What is the article about?'";
§2: "Our extreme summarization dataset (which we call XSum) consists of BBC articles and
accompanying single sentence summaries", with an average of 1.00 sentences per summary (Table 1).
Without the instruction an instruct model writes a multi-sentence summary, and output length then
confounds every measurement against the one-sentence references: ROUGE-L and BERTScore against
roughly 21-word references, the number of claims that factuality metrics check, completion token
cost (H4), and how often D-012's cap binds. The phrase costs 3 tokens of prompt (measured with Qwen's
tokenizer on the frozen sample).

The prompt is kept deliberately minimal — no persona, no formatting rules, no examples — because T0
measures the backbone, not prompt engineering. The proposal describes XSum as single-sentence but
specifies no prompt, so the wording is this entry's, anchored to the dataset paper.

**Source:** Narayan et al. 2018 (arXiv 1808.08745), abstract and §2 with Table 1; proposal §2;
prompt-token cost measured 2026-09-12; D-011; D-012
**Approved:** Sameer, 2026-09-12
**Config:** Generator prompt template in `agents/` (to be created), versioned in the run manifest
**Supersedes:** —

---

### D-017 · 2026-09-12 · Serving context lengths: Mistral-7B-Instruct-v0.3 3,189, Llama-3.1-8B-Instruct 2,738

**Decision:**
`model_registry.csv` records `max_model_len` **3,189** for Mistral-7B-Instruct-v0.3 and **2,738**
for Llama-3.1-8B-Instruct, by D-013's rule: the longest prompt among the frozen 500 XSum documents
under D-016's wording, counted with that backbone's own tokenizer and chat template, plus D-012's
256-token cap.

**Measured 2026-09-12** (prompt tokens, chat template applied):

| Backbone | median | p99 | max | context (max + 256) | model's position limit |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 424 | 1,724 | 2,553 | 2,809 — D-013, re-confirmed | 32,768 |
| Mistral-7B-Instruct-v0.3 | 453 | 1,947 | 2,933 | **3,189** | 32,768 |
| Llama-3.1-8B-Instruct | 426 | 1,725 | 2,482 | **2,738** | 131,072 |

No frozen document is truncated for any backbone.

**Reason:**
*Mistral is counted two ways.* Its repository ships both a HuggingFace tokenizer (`tokenizer.json`)
and Mistral's own format (`params.json`, `tokenizer.model.v3`), and vLLM's default
`tokenizer_mode="auto"` can resolve to the latter (`vllm/tokenizers/registry.py`,
`resolve_tokenizer_args`). Counting with the HuggingFace chat template and with `mistral_common`
1.11.7's `encode_chat_completion` gave identical distributions — median 453, p99 1,947, maximum
2,933 — so the value holds whichever tokenizer vLLM selects.

*Llama's count includes its template's own system text.* The template renders a system block
reading "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024" ahead of the user message;
it is part of every prompt under D-011 and is counted.

**Not yet verified at runtime:** neither backbone has been served. Their weights are not downloaded
(about 14.96 GiB for Llama; Mistral's repository holds 27.0 GiB of safetensors, including a
consolidated copy of the same weights).

**Source:** token counts measured 2026-09-12 with each backbone's tokenizer at its pinned revision;
`config.json` at those revisions; D-012, D-013, D-016
**Approved:** Sameer, 2026-09-12 — values measured under the D-013 rule he approved
**Config:** `src/model_registry.csv`
**Supersedes:** —

---

### D-018 · 2026-09-12 · SummaC-ZS and SummaC-Conv: official configuration, ported code, reproduction target

**Decision:**
Both SummaC variants are computed, in the configuration the official repository recommends, by a
**port of the official code** into `metrics/`. Every setting below is inherited from that code or its
README — none was chosen by this project.

| Setting | Value | Origin |
| --- | --- | --- |
| NLI model | `tals/albert-xlarge-vitaminc-mnli` @ `3082ba54344bd9ddada2be1c5e9b4131721d2a5d` ("vitc") | README; `model_map`; SummaC paper's default "MNLI+VitaminC" |
| Label indices | entailment 0, contradiction 1, neutral 2 | `model_map["vitc"]` |
| Precision on GPU | half precision (`model.half()`) | `SummaCImager.load_nli` |
| Granularity | sentence, for article and summary | README; paper recommends (sentence, sentence) |
| Sentence splitter | `nltk.tokenize.sent_tokenize`, sentences of ≤ 10 characters dropped | `split_sentences` |
| Article limits | first 100 sentences; pair input truncated to 500 tokens, article side only | `max_doc_sents`, `max_input_length`, `truncation_strategy="only_first"` |
| SummaC-ZS | per summary sentence: max entailment − max contradiction over article sentences; mean over summary sentences | code defaults `op1="max"`, `op2="mean"`, `use_ent=True`, `use_con=True` |
| SummaC-Conv | percentile bin edges (23 bins), entailment channel only, first 10 summary sentences, mean aggregation, score = softmax class 1 | README `bins="percentile"`, `nli_labels="e"`, `agg="mean"` |
| Conv weights | `summac_conv_vitc_sent_perc_e.bin` from commit `525cb081993ad1c872d73a9d5d46ea3c3f73c24f` — the only commit touching the file; 1,811 bytes; git blob `8233d99b8fdd147640eadada85325d72f0627991`; sha256 `cd880581f59bd47b8968c7f48d754cd617dfe9b256d8d76fdcbc0fa4a3b3c6fe` | GitHub API, verified 2026-09-12 |

**Changes made by the port, and no others:** the pair-encoding call (transformers 5 removed
`batch_encode_plus`); logging in place of `print`; pinned model revision and checkpoint with a hash
check, replacing a `wget` of the moving `master` branch into the working directory; no hardcoded
cache folder; balanced accuracy in numpy rather than a scikit-learn dependency. The source is
Apache-2.0 and is credited in the ported module.

**Reproduction target (validation of the port):** SummaC benchmark, XSumFaith test set, balanced
accuracy — **SummaC-ZS 58.4, SummaC-Conv 66.4** (Laban et al. 2022, Table 2). The data is rebuilt
exactly as the official loader does: annotation file pinned to commit
`4abdbb2c807cb93c1df1188a1992d20a55cdcc34` (sha256
`fa7fb66a36cc0f32ede4135985d0d65591dc2a8d21103a0bacd0583d77d4c8ea`), grouped by (article, system)
with a majority label, alternately assigned to validation and test. This gives 2,500 pairs — 1,250
validation and 1,250 test, matching the paper's 1,250 — with 130 consistent and 1,120 inconsistent
in test. All 500 articles are present in the pinned `EdinburghNLP/xsum` test split, which replaces
the official loader's `load_dataset("xsum")` (a script loader datasets 5 no longer runs). The
threshold is selected on the validation half and applied to the test half, following the paper:
"The threshold is selected using the validation set, allowing for a different threshold for each
dataset in the benchmark." The achieved values are reported against the targets.

**Reason:**

*Why both variants.* Neither dominates across the evidence. On the SummaC benchmark SummaC-Conv
leads (balanced accuracy 74.4 vs 72.1 overall; 66.4 vs 58.4 on XSumFaith; Laban et al. 2022,
Table 2). On LLM-generated text the order reverses: SummaC-ZS 67.9 vs SummaC-Conv 62.1 average
balanced accuracy on LLM-AggreFact (Tang et al. 2024, Table 2).

*Why the official configuration.* It is what the closest published comparator uses. "Self-Consistent
Decoding for More Factual Open Responses" (arXiv 2403.00696) runs SummaC "in the zero-shot (ZS) and
convolutional (Conv) variants at sentence granularity, using the 'vitc' model" on XSum summaries from
Llama-2-13B-chat and Mistral-7B-Instruct-v0.2. It is also how AggreFact and AlignScore run their
SummaC baselines. Any other configuration would make those numbers non-comparable.

*Why a port rather than the package.* The package was tried first. `summac` 0.0.4 declares
`huggingface-hub<=0.17.0`, while transformers 5.17 and vLLM need hub 1.31, so it cannot be installed
with its dependencies. Installed with `--no-deps`, the official code fails on our stack with
`AlbertTokenizer has no attribute batch_encode_plus`. The published code therefore does not run
unmodified, and the minimal port above is the closest faithful implementation.

*Why reproduce against the paper.* The stack differs from the one the published numbers came from,
so the port is trusted only once it reproduces them on the same data and procedure. The official
`evaluate()` selects the threshold on the same half it reports, which is optimistic; the paper's
validation-selected threshold is used instead.

*Why SummaC-ZS subtracts contradiction.* The paper's prose describes the entailment maximum and mean;
the code's defaults also subtract the contradiction maximum. The code is what produced the published
numbers, so the port follows the code.

**Known caveats:**
- SummaC-ZS is weak on modern XSum summaries: 56.4 ± 1.2 balanced accuracy on AggreFact-XSum
  (fine-tuned state-of-the-art summarizers, single threshold), against a 50.0 trivial baseline;
  SummaC-Conv 65.0 ± 2.2 (Tang et al. 2023, Table 4). This is why stronger checks are added
  alongside it rather than relied upon alone.
- The NLI checkpoint's model card declares no license (it cites Schuster et al. 2021). This does not
  affect use in research but must be resolved before the benchmark's public release.

**Source:** Laban et al. 2022, *SummaC* (arXiv 2111.09525) — Table 2, Table 4, Appendix Table A1,
§4 threshold selection; SummaC repository (`tingofurro/summac`, Apache-2.0) README and `summac` 0.0.4
`model_summac.py`, `benchmark.py`, `utils_scorer.py`, read 2026-09-12; Tang et al. 2023, *AggreFact*
(arXiv 2205.12854), Tables 3–4; Tang et al. 2024, *MiniCheck* (arXiv 2404.10774), Table 2;
arXiv 2403.00696, §4 and Table 2; smoke tests of the package on the project stack, 2026-09-12
**Approved:** Sameer, 2026-09-12 (port plan; reproduction target 58.4 / 66.4)
**Config:** metric configuration under `src/configs/` (to be created with `metrics/`)
**Supersedes:** —

---

### D-019 · 2026-09-12 · BERTScore: its English default, roberta-large layer 17

**Decision:**
BERTScore is computed with `bert-score` 0.3.13 at its default English configuration: model
`roberta-large`, representations from **layer 17**, no IDF weighting and no baseline rescaling. The
precision, recall and F1 of every output are all recorded.

**Reason:**
These are the values the library itself selects for English (`bert_score/utils.py`:
`lang2model["en"] = "roberta-large"`; `model2layers["roberta-large"] = 17`), so they are what any
paper that runs BERTScore with `lang="en"` and no further arguments reports. Staying on the default
keeps our numbers comparable with that literature and involves no project-chosen value. All three
components are stored so that no later choice of headline number requires a re-run.

*Recorded oddity:* pip installed `bert-score` 0.3.13, but the imported module reports `__version__`
0.3.12. The pip-installed version is the one pinned.

**Source:** `bert_score/utils.py` in the `Tiiiger/bert_score` repository, read 2026-09-12; installation
2026-09-12
**Approved:** Sameer, 2026-09-12
**Config:** metric configuration under `src/configs/` (to be created with `metrics/`); `requirements.txt`
**Supersedes:** —

---

### D-020 · 2026-09-12 · Model requests are sent one at a time

**Decision:**
During a run, the serving client sends generation requests **sequentially** — one request, wait for
its response, then the next. No concurrent requests.

**Reason:**
*Latency is a reported result.* Per-example wall-clock latency feeds H4 and H5. Under concurrent
requests vLLM batches them, so one example's latency would depend on which other examples happened
to be in flight, not on the example and the topology.

*Reproducibility is only verified for sequential requests.* The 2026-09-12 smoke test showed that two
identical sequential requests with the same seed return byte-identical text (Q-002). Whether a fixed
seed still reproduces under concurrent batching is unverified.

*Trade-off.* A sequential sweep takes longer in wall-clock time than a concurrent one. A single
request on one XSum article took 0.76 s once warm (smoke test), so the cost is modest at this scale,
but it has not been measured over a full run.

**Source:** Q-002; smoke test 2026-09-12; D-003
**Approved:** Sameer, 2026-09-12
**Config:** serving client behaviour in `serving/`; no config field (a concurrency setting would invite
changing it)
**Supersedes:** —

---

### D-021 · 2026-09-12 · Metric dependencies: what was installed, and how

**Decision:**

| Package | Pinned as | Note |
| --- | --- | --- |
| `rouge-score` | 0.1.2 | ROUGE-L |
| `bert-score` | 0.3.13 | BERTScore (D-019); the module reports `__version__` 0.3.12 |
| `nltk` | 3.10.3 | sentence splitting for SummaC; data `punkt_tab` downloaded into `.venv/nltk_data` |
| `spacy` | 3.8.16 | sentence splitting for AlignScore |
| `en_core_web_sm` | 3.8.0, wheel pinned by URL and sha256 `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85` | spaCy's compatibility table maps spaCy 3.8 to this release |
| `minicheck` | git commit `b58b9fa69acbd1015ec970fa65dd752413a053d2` (package version 0.1.0) | installed from the authors' repository, never by name |
| `summac` | 0.0.4, `--no-deps` | **temporary**: installed only to test the official code (D-018); replaced by the port and removed once the port is validated; not in `requirements.txt` |

Transitive additions: absl-py 2.5.0, blis 1.3.3, catalogue 2.0.10, cloudpathlib 0.25.0, confection
1.3.3, contourpy 1.4.0, cycler 0.12.1, cymem 2.0.13, defusedxml 0.7.1, fonttools 4.65.0, joblib
1.6.0, kiwisolver 1.5.1, matplotlib 3.11.2, murmurhash 1.0.15, preshed 3.0.13, pyparsing 3.3.2,
smart-open 8.0.1, spacy-legacy 3.0.12, spacy-loggers 1.0.5, srsly 2.5.3, thinc 8.3.13, wasabi 1.1.3,
weasel 1.0.0, wrapt 2.4.1.

**The pinned stack is unchanged**, verified by before/after `pip freeze` and import: torch 2.13.0,
transformers 5.17.0, numpy 2.4.6. Every metric package was dry-run against the stack before
installation.

**Reason:**

*MiniCheck must never be installed by name.* The PyPI package called `minicheck` (0.4.0) is an
unrelated project — its description reads "An explicit-state model checker in ~2900 lines". The
MiniCheck authors' README installs from GitHub
(`pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"`); the commit is pinned
instead of `main` so the code cannot change under the benchmark.

*Why AlignScore brings spaCy.* AlignScore splits text with spaCy (README: "we use spaCy for
sentenization"). Substituting another splitter would change its scores, so the authors' splitter is
kept and pinned.

*Why SummaC is not a dependency.* See D-018: its declared `huggingface-hub<=0.17.0` conflicts with the
stack, and its code does not run on transformers 5 unmodified.

**Recorded observations:**
- Importing one of the installed packages downloaded nltk's older `punkt` data into
  `.venv/nltk_data` on first use, without being asked. Packages that download at import time are a
  reproducibility risk and are watched for.
- nltk warns that `.venv` is group-writable and treats its data directory as untrusted. Loading
  worked, but the permissions should be tightened.

**Source:** PyPI metadata, repository requirement files, dry-run resolutions and `pip freeze`
before/after, all 2026-09-12; AlignScore README; MiniCheck README
**Approved:** Sameer, 2026-09-12
**Config:** `requirements.txt`
**Supersedes:** —

---

### D-022 · 2026-09-12 · Seven metrics for XSum outputs, and what prior work reports for each

**Decision:**
Every XSum output is scored on seven metrics:

| # | Metric | Measures | Origin |
| --- | --- | --- | --- |
| 1 | SummaC-ZS | factual consistency with the article | proposal |
| 2 | SummaC-Conv | factual consistency with the article | proposal |
| 3 | FactCC | factual consistency with the article | proposal |
| 4 | AlignScore | factual consistency with the article | **added** |
| 5 | MiniCheck | factual consistency with the article | **added** |
| 6 | ROUGE-L | overlap with the reference summary | proposal |
| 7 | BERTScore | semantic similarity to the reference summary | proposal |

**Not included:** ALCE citation metrics apply to Tier 2 question answering, not summarization.
SelfCheckGPT appears in the proposal's Week 1 install list but not in its Week 3 metric pipeline, and
it scores consistency across several extra generations per example, multiplying generation cost; it
is not added unless separately decided. The checkpoint and settings of FactCC, AlignScore, MiniCheck
and ROUGE-L are decided in their own entries.

**Reason:**

*The proposal's pipeline.* The proposal's Week 3 metric pipeline is FactCC, SummaC, BERTScore and
ROUGE-L, validated against AggreFact's human labels.

*Why AlignScore and MiniCheck are added — a departure from the proposal.* Neither appears in the
proposal. They were added on Sameer's decision because the published evidence below shows SummaC
weak on modern XSum summaries, while AlignScore and MiniCheck rank above it on the benchmarks built
for that question, including one built from LLM-generated text.

**Evidence: how well each metric agrees with human labels.** Read directly from each paper's tables
on 2026-09-12. Numbers across rows use different data, thresholds and metrics, so they are comparable
within a row, not between rows.

| Source (table) | Setting | SummaC-ZS | SummaC-Conv | FactCC | AlignScore | MiniCheck |
| --- | --- | --- | --- | --- | --- | --- |
| Laban et al. 2022 (T2) | SummaC benchmark, balanced accuracy, validation threshold — overall | 72.1 | 74.4 | 62.8 (FactCC-CLS) | — | — |
| Laban et al. 2022 (T2) | same — XSumFaith | 58.4 | 66.4 | 57.6 (FactCC-CLS) | — | — |
| Tang et al. 2023 (T4) | AggreFact-XSum, recent fine-tuned summarizers, single threshold, balanced accuracy | 56.4 ± 1.2 | 65.0 ± 2.2 | — | — | — |
| Tang et al. 2023 (T4) | AggreFact-CNN, same setting | 64.0 ± 3.8 | 61.0 ± 3.9 | — | — | — |
| Tang et al. 2023 (T3) | AggreFact-XSum, recent summarizers, threshold per dataset | 56.1 | 67.0 | — | — | — |
| Zha et al. 2023 (T2) | SummaC benchmark, AUC-ROC — average | 79.0 | 81.0 | 68.8 | base 87.4 / large 88.6 | — |
| Zha et al. 2023 (T2) | same — XSumFaith | 58.0 | 70.3 | 55.1 | base 79.4 / large 75.8 | — |
| Tang et al. 2024 (T2) | LLM-AggreFact, balanced accuracy, no threshold tuning — average of 10 datasets | 67.9 | 62.1 | — | 70.4 (RoBERTa-L, 355M per T3) | FT5 74.7 / RoBERTa 72.7 / DeBERTa 72.6 |
| Tang et al. 2024 (T2) | same — AggreFact-XSum | 61.5 | 54.5 | — | 71.4 | FT5 74.3 / RoBERTa 70.8 / DeBERTa 71.0 |

For reference, the trivial baseline in balanced accuracy is 50.0, and GPT-4 on LLM-AggreFact scores
75.3 average and 76.5 on AggreFact-XSum (Tang et al. 2024, T2).

**Evidence: scores prior work reports for LLM-written XSum summaries.** The closest published system
comparison is arXiv 2403.00696, Table 2 (XSum, greedy decoding, 175 documents from the FRANK test
subset, SummaC with "vitc" at sentence granularity):

| Model | SummaC-ZS | SummaC-Conv |
| --- | --- | --- |
| Llama-2-13B-chat | .356 | .364 |
| Mistral-7B-Instruct-v0.2 | .413 | .402 |

These use different documents, model versions and decoding from this project, so they are context
for magnitude, not a paired comparison; the valid comparison is between this project's topologies on
its identical frozen 500. Roit et al. 2023 does not use SummaC at all: it reports the share of
summaries entailed by a T5-XXL NLI model trained on ANLI, and Q².

*Overlap with human annotations.* 23 of the frozen 500 XSum articles are among the 500 articles
annotated in XSumFaith (Maynez et al. 2020); too few to serve as a separate validation set.

**Source:** proposal §4 (Weeks 1 and 3); Laban et al. 2022 (arXiv 2111.09525); Tang et al. 2023
(arXiv 2205.12854); Zha et al. 2023 (arXiv 2305.16739); Tang et al. 2024 (arXiv 2404.10774); arXiv
2403.00696; Roit et al. 2023 (arXiv 2306.00186) — all read 2026-09-12; overlap computed 2026-09-12
**Approved:** Sameer, 2026-09-12 (AlignScore and MiniCheck added at his decision)
**Config:** metric configuration under `src/configs/` (to be created with `metrics/`)
**Supersedes:** —

---

### D-023 · 2026-09-12 · AlignScore: the large checkpoint in its default `nli_sp` mode, ported

**Decision:**
AlignScore is computed with **AlignScore-large** in its default **`nli_sp`** mode, by a port of the
official inference code (MIT) at commit `a0936d5afee642a46b22f6c02a163478447aa493`. Every setting is
inherited from that code:

| Setting | Value | Origin |
| --- | --- | --- |
| Checkpoint | `AlignScore-large.ckpt` from `yzha/AlignScore` @ `8509e78d25bb914939fc585c626500c9b2944249` (MIT) | README "Checkpoints" |
| Backbone | RoBERTa-large (355M); config and tokenizer from `FacebookAI/roberta-large` @ `722cf37b1afa9454edce342e7895e588b6ff1d59` | `BERTAlignModel`; README |
| Article chunks | article split with `nltk.sent_tokenize`; `n_chunk = words // 350 + 1`; sentences grouped `max(len(sentences) // n_chunk, 1)` per chunk | `inference_per_example` |
| Summary units | summary split with `nltk.sent_tokenize` | same |
| Encoding | (chunk, sentence) pairs, padded and truncated to the tokenizer's maximum (512), article side only | `batch_tokenize` |
| Score | pooled RoBERTa output → 3-way head → softmax probability of class 0; max over chunks for each summary sentence, mean over summary sentences | `model.forward`; `inference`; `inference_per_example` |

**Changes made by the port, and no others:** the PyTorch Lightning checkpoint is loaded into plain
modules without Lightning, with the backbone and 3-way-head weights checked to be present (the
official loader uses `strict=False`, which would silently keep random weights); spaCy is not loaded;
logging replaces progress bars and `print`.

**Reproduction target:** AUC-ROC **75.8** on the SummaC benchmark XSumFaith test split (Zha et al.
2023, Table 2). AlignScore's own evaluation scores `SummaCBenchmark(cut='test')` (`evaluate.py`), so
the same 1,250 test pairs D-018 rebuilds apply, and AUC needs no threshold.

**Reason:**
*Why large.* It is the variant with the higher average in its own paper — AUC-ROC 88.6 against 87.4
for base on the SummaC benchmark (Zha et al. 2023, Table 2) — and the one MiniCheck reports as its
AlignScore comparison ("RoBERTa-L, 355M"; Tang et al. 2024, Table 3), so our AlignScore numbers sit on
the same model as the published comparisons. **Caveat:** on XSumFaith alone the order reverses — base
79.4, large 75.8.

*Why `nli_sp`.* The README: "`nli_sp` is the default setting of AlignScore."

*Why a port.* The package declares `torch>=1.12.1,<2`, `transformers>=4.20.1,<5`, `numpy<2`,
`datasets<3` and `pytorch_lightning<2` (`pyproject.toml`), none of which can coexist with this stack.

**Correction to D-021.** D-021 gave "AlignScore splits text with spaCy" as the reason spaCy was
installed, following the README's "we use spaCy for sentenization". The `nli_sp` code path does not
use spaCy: `Inferencer.__init__` loads it, but `inference_per_example` splits both texts with
`nltk.sent_tokenize`; spaCy appears only in other code paths. See D-027.

**Source:** Zha et al. 2023 (arXiv 2305.16739), Table 2; Tang et al. 2024 (arXiv 2404.10774), Table 3;
AlignScore repository (`yuh-zha/AlignScore`, MIT) at the pinned commit — `README.md`, `pyproject.toml`,
`src/alignscore/inference.py`, `src/alignscore/model.py`, `evaluate.py` — read 2026-09-12
**Approved:** Sameer, 2026-09-12
**Config:** metric configuration under `src/configs/` (to be created with `metrics/`)
**Supersedes:** —

---

### D-024 · 2026-09-12 · MiniCheck: Flan-T5-Large, threshold 0.5, per-sentence scores averaged

**Decision:**
MiniCheck is computed with **MiniCheck-Flan-T5-Large** (`lytang/MiniCheck-Flan-T5-Large` @
`96eafd01cee2d16cf81aaa2fb226b14f422a37b3`, MIT, 2.92 GiB), through the authors' package installed at
commit `b58b9fa69acbd1015ec970fa65dd752413a053d2` with `model_name="flan-t5-large"` passed explicitly.

| Setting | Value | Origin |
| --- | --- | --- |
| Article chunks | whole sentences packed into chunks of up to 500 whitespace words | package default for `flan-t5-large` (`minicheck.py`, `inference.py`) |
| Chunk aggregation | maximum "supported" probability over chunks | `inference.py`, `max_support_prob` |
| Threshold | 0.5 | Tang et al. 2024: "Unless otherwise specified, we set t = 0.5"; package `prob > 0.5` |
| Output units | each output split into sentences with `nltk.sent_tokenize` and scored as its own claim | README |
| Output score | **mean** of the per-sentence scores | this decision — see Reason |

The package hardcodes checkpoint names without a revision, so the pinned snapshot is downloaded in
advance and loaded from it.

**Reason:**
*Why Flan-T5-Large.* It is the strongest MiniCheck model in the paper — 74.7 average and 74.3 on
AggreFact-XSum balanced accuracy on LLM-AggreFact, above RoBERTa (72.7 / 70.8) and DeBERTa
(72.6 / 71.0) (Tang et al. 2024, Table 2) — and the README calls it "the best fact-checking model with
size < 1B" that "reaches GPT-4 performance". The stronger Bespoke-MiniCheck-7B is licensed CC BY-NC
4.0 (non-commercial), and it is also **the package's default** `model_name`; passing the model
explicitly prevents it from being loaded by accident.

*Why split outputs into sentences.* The README: "MiniCheck(document, sentence) -> [0, 1] is a
sentence-level fact-checking model. In order to fact-check a multi-sentence claim, the claim should
first be broken up into sentences to achieve optimal performance."

*Why the mean.* The authors do not say how to combine sentence scores. The mean matches how SummaC-ZS
(D-018) and AlignScore (D-023) combine summary sentences, so all three sentence-level metrics
aggregate the same way. The trade-off: a mean lets well-supported sentences offset an unsupported one,
whereas a minimum would fail the whole summary on a single unsupported sentence. Under D-016's
one-sentence prompt most outputs are expected to be a single sentence, where the two coincide; the
share of multi-sentence outputs is reported.

**Reproduction target:** balanced accuracy **74.3** on the AggreFact-XSum portion of the LLM-AggreFact
test set at threshold 0.5 (Tang et al. 2024, Table 2), from `lytang/LLM-AggreFact` @
`981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4` (CC BY-ND 4.0; gated — terms accepted by Sameer
2026-09-12). The dataset's claims are already sentences.

**Source:** Tang et al. 2024 (arXiv 2404.10774), §3 and Tables 2–3; MiniCheck repository
(`Liyan06/MiniCheck`, Apache-2.0) README and package source at the pinned commit; Bespoke-MiniCheck-7B
model card; all read 2026-09-12
**Approved:** Sameer, 2026-09-12
**Config:** metric configuration under `src/configs/` (to be created with `metrics/`); `requirements.txt`
**Supersedes:** —

---

### D-025 · 2026-09-12 · FactCC: the HuggingFace conversion of the original checkpoint, P(CORRECT)

**Decision:**
FactCC is computed with **`manueldeprada/FactCC`** @ `c7b3148015d4ddc263f6e2acb2689e90ac061669`
(BSD-3-Clause-Clear), a `BertForSequenceClassification` whose configuration names
`factCC/factcc-checkpoint` as its origin and labels `0: CORRECT`, `1: INCORRECT`. The score is the
softmax **probability of CORRECT** for the pair (article, whole summary), encoded to 512 tokens with
padding to the maximum and truncation on the article side only — the usage the model card gives.

**Reason:**
*Why this checkpoint.* FactCC is in the proposal's metric pipeline. Salesforce's original checkpoint
(`storage.googleapis.com/sfr-factcc-data-research/factcc-checkpoint.tar.gz`, linked from the archived
`salesforce/factCC` repository) returns HTTP 403 as of 2026-09-12, so the conversion's weights cannot
be compared to it directly. It is therefore validated by behaviour instead. It loads completely on the
project stack (201 weight tensors, transformers 5.17 / torch 2.13) and separates a constructed
consistent from a contradicting claim (P(CORRECT) 0.9986 vs 0.0005; smoke check, not a result).

*Why P(CORRECT).* It is the consistency-oriented score, and it matches how AlignScore's evaluation
turns FactCC output into a consistency score (`1 - output` in `FactCCScorer`).

**Reproduction target:** AUC-ROC **55.1** on the SummaC benchmark XSumFaith test split (Zha et al.
2023, Table 2), produced by AlignScore's `FactCCScorer`, which runs the original FactCC scripts.

*Why not SummaC's 57.6.* The SummaC paper's FactCC row is "FactCC-CLS", described as "a Roberta-base
model finetuned on the synthetic training portion of the FactCC dataset" — a different model from the
original BERT checkpoint, so its number is not a target for this one.

**Known caveat:** FactCC is weak on XSum. An AUC-ROC of 55.1 is close to the 50.0 of an uninformative
scorer; it is kept because the proposal names it and so prior work can be compared.

**Source:** Kryscinski et al. 2020 (arXiv 1910.12840), via the model card; `manueldeprada/FactCC` model
card and `config.json`; `salesforce/factCC` README; Laban et al. 2022, §4 and Table 2; Zha et al. 2023,
Table 2 and `baselines.py`; all read 2026-09-12; load and smoke check 2026-09-12
**Approved:** Sameer, 2026-09-12
**Config:** metric configuration under `src/configs/` (to be created with `metrics/`)
**Supersedes:** —

---

### D-026 · 2026-09-12 · ROUGE-L: F1, no stemming

**Decision:**
ROUGE-L is computed with `rouge-score` 0.1.2 as `rougeL` with **`use_stemmer=False`** and the default
tokenizer. F1 is the reported value; precision and recall are recorded alongside it.

**Reason:**
No stemming is the library's default (`RougeScorer.__init__(rouge_types, use_stemmer=False, ...)`) and
the default of the HuggingFace ROUGE wrapper, both in current `evaluate` and in `datasets` 2.3.2 — the
version arXiv 2403.00696 reports using for its ROUGE scores. The `rouge-score` README shows
`use_stemmer=True` only as an example. `rougeL` rather than `rougeLsum`: `rougeLsum` splits on
newlines, and XSum's references are single sentences. The proposal names ROUGE-L; the comparator paper
reports ROUGE-1, so its ROUGE numbers are not directly comparable in any case.

**Source:** `rouge_score/rouge_scorer.py` (installed 0.1.2); `huggingface/evaluate`
`metrics/rouge/rouge.py`; `huggingface/datasets` 2.3.2 `metrics/rouge/rouge.py`; proposal §4; all read
2026-09-12
**Approved:** Sameer, 2026-09-12
**Config:** metric configuration under `src/configs/` (to be created with `metrics/`)
**Supersedes:** —

---

### D-027 · 2026-09-12 · spaCy removed: no chosen metric uses it

**Decision:**
`spacy` 3.8.16 and `en_core_web_sm` 3.8.0 are uninstalled and removed from `requirements.txt`, together
with the transitive packages installed only for them.

**Reason:**
They were installed for AlignScore (D-021). D-023 establishes that AlignScore's `nli_sp` mode splits
text with nltk, not spaCy, and no other chosen metric uses spaCy. An unused dependency is unpinned
risk with no benefit.

**Source:** D-021; D-023
**Approved:** Sameer, 2026-09-12
**Config:** `requirements.txt`
**Supersedes:** D-021 — only its `spacy` and `en_core_web_sm` rows and its "Why AlignScore brings
spaCy" reason; the rest of D-021 stands

---

### D-028 · 2026-09-12 · Metric ports: SummaC truncation correction, pinned artifacts, inherited batch sizes

**Decision:**

*1. SummaC encodes sentence pairs with `longest_first` truncation, not article-side-only.* This
corrects D-018's "Article limits" row, which read the official call's
`truncation_strategy="only_first"` at face value.

*2. The metric ports use these verified artifacts:*

| Artifact | Pin | Verified |
| --- | --- | --- |
| AlignScore-large checkpoint | `yzha/AlignScore` @ `8509e78d`, `AlignScore-large.ckpt` | 4,895,673,790 bytes; sha256 `ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d`; loads with `torch.load(weights_only=True)`; 405 tensors (base_model 392, mlm_head 7, bin_layer 2, tri_layer 2, reg_layer 2); `tri_layer.weight` (3, 1024); `hyper_parameters.model = "roberta-large"` |
| MiniCheck-Flan-T5-Large | `lytang/MiniCheck-Flan-T5-Large` @ `96eafd01` | snapshot downloaded |
| SummaC NLI model | `tals/albert-xlarge-vitaminc-mnli` @ `3082ba54` | snapshot downloaded |
| RoBERTa-large (BERTScore; AlignScore backbone) | `FacebookAI/roberta-large` @ `722cf37b` | snapshot downloaded |
| LLM-AggreFact test | `lytang/LLM-AggreFact` @ `981dfd0b`, `data/test-00000-of-00001.parquet` | 17,298,205 bytes; sha256 `ba6976c3da990a9d7b11670efaf4e3c93c1bd0530061ec742afd6b2d5bce698c`; AggreFact-XSum 558 claims (285 supported, 273 unsupported) |
| LLM-AggreFact dev | same revision, `data/dev-00000-of-00001.parquet` | 35,453,136 bytes; sha256 `ca9a5bda9561bbc6f7327ef81ab41727fa10e96f96e9dc74ef32ba3af3c1d8a8`; AggreFact-XSum 777 claims |

*3. Batch sizes are the official defaults where one exists:* SummaC 128 (`SummaCImager.build_images`
and `SummaCZS.score`), AlignScore 32 (`Inferencer.__init__`), MiniCheck 16 (`MiniCheck.__init__`).

*4. MiniCheck's model is loaded by a subclass of the package's `Inferencer`* that reads the pinned
snapshot; the package's chunking, prompt construction and scoring methods are used unchanged.

**Reason:**

*1.* In transformers 4.x, `_get_padding_truncation_strategies` (read at tag v4.35.2) applies the
legacy `truncation_strategy` argument only when `truncation is None`. The official SummaC call passes
`truncation=True`, which resolves to `TruncationStrategy.LONGEST_FIRST`; `"only_first"` is popped
and ignored. The published numbers were therefore produced with longest-first truncation, and a
faithful port must use it. In practice summary sentences are short, so the longer side truncated is
almost always the article.

*2.* Pins and digests make every metric score reconstructible, and the AlignScore checkpoint loads
without executing pickled code.

*3.* Batch size is left at the value the authors' code uses so the ports stay as close to the
published implementations as possible. One known deviation: official SummaC-Conv builds its sentence
matrix one pair at a time in batches of 20, whereas the port builds one matrix per example that both
SummaC variants share. In half precision, batch composition can move scores in late decimal places;
the XSumFaith reproduction (D-018) measures whether it matters.

*4.* The package loads its checkpoint by name with no revision and with `device_map="auto"`, which
requires `accelerate` — not installed and not otherwise needed. Overriding only the loader keeps the
authors' scoring code while honouring the pin.

**Source:** `transformers` v4.35.2 `tokenization_utils_base.py`; `summac` 0.0.4 `model_summac.py`;
AlignScore `inference.py` at `a0936d5a`; `minicheck` at `b58b9fa6` (`minicheck.py`, `inference.py`);
download log and checkpoint inspection, 2026-09-12
**Approved:** Sameer, 2026-09-12 (ports and reproduction plan)
**Config:** `src/configs/base.yaml`, `metrics.*`
**Supersedes:** D-018 — only its "Article limits" row, for the truncation strategy

## D-029 — Tokenizer pair encoding and nltk data under transformers 5 (findings for the metric ports)

**Decision:**

*1. Ports must request `token_type_ids` explicitly (`return_token_type_ids=True`) for every
sentence-pair encoding.* Under transformers 5.17.0 the ALBERT tokenizer (SummaC's
`tals/albert-xlarge-vitaminc-mnli` @ `3082ba54`) lists `model_input_names` as `input_ids`,
`attention_mask` and returns no `token_type_ids` by default. With the flag set it returns the correct
segment ids (`[0, 0, 0, 0, 0, 1, 1]` for a toy pair). The BERT tokenizer (FactCC @ `c7b31480`)
still returns them by default; the flag is set anyway so the behaviour never depends on the default.

*2. nltk data is installed once, never at import time during a run.* `minicheck/__init__.py`
(commit `b58b9fa6`) calls `nltk.download('punkt')` when `tokenizers/punkt` is missing. That was the
unprompted punkt download. `punkt` and `punkt_tab` are now both present in `.venv/nltk_data`, so the
import no longer touches the network. `rouge-score` and `bert-score` contain no nltk downloads.

**Reason:**

*1.* In transformers v4.35.2 the ALBERT and BERT tokenizers do not override `model_input_names`, so
they inherit the base default `["input_ids", "token_type_ids", "attention_mask"]`
(`tokenization_utils_base.py` line 1558). The published SummaC numbers were therefore produced with
segment ids. Without them ALBERT treats article and summary as one segment, so the port would silently
diverge from the paper.

*2.* A metric import that downloads data breaks the offline, reproducible run requirement and hides
a dependency that `requirements.txt` cannot pin.

**Source:** local checks on transformers 5.17.0, 2026-09-12; transformers v4.35.2
`models/albert/tokenization_albert{,_fast}.py`, `models/bert/tokenization_bert{,_fast}.py` and
`tokenization_utils_base.py`; `minicheck` `__init__.py` at `b58b9fa6`
**Approved:** pending (findings)
**Config:** none

---

### D-030 · 2026-09-13 · Model-based metric ports: batch size, blank outputs, precision and validation runner

**Decision:**

*1. FactCC batch size 16.* Inputs are padded to a fixed 512 tokens, so batch size changes speed only.
It matches MiniCheck's official default.

*2. A blank generated output is never scored.* Every metric rejects an output that is empty or
whitespace only. The caller records it as a generation failure, counted like `finish_reason`. The
frozen 500 XSum articles and references were checked: none is blank.

*3. D-029 approved, with one clarification.* The rule to request `token_type_ids` explicitly applies
where transformers 4 returned them by default: ALBERT (SummaC) and BERT (FactCC). RoBERTa (AlignScore)
and T5 (MiniCheck) tokenizers never returned them, so the ports do not request them there, as the
original code did not.

*4. Weights load in float32 explicitly.* transformers 5 loads weights in their stored dtype by
default, where transformers 4 used float32. Every checkpoint used is stored in float32 (header and
state-dict check), so the result is the same, but the dtype is passed so it does not depend on the
default. SummaC then halves its NLI model on CUDA, as the official code does.

*5. SummaC edge case kept as official.* When no article or no summary sentence survives the
10-character filter, the official imager returns an all-zero image. The port does the same:
SummaC-ZS is 0.0, and SummaC-Conv scores one zero-valued histogram entry.

*6. Validation runner.* `src/validate_metrics.py` (logic in `metrics/reproduce.py`) loads one
metric at a time and computes each published number. It writes the summary and per-item scores to
`~/.cache/factagent_bench/metrics/validation/`. It is a check of the ports, not a run, so it writes
nothing to `runs/`. A gap to a target is reported, not treated as a failure.

**Reason:** 1 and 2 were approved by Sameer on 2026-09-13. 3–5 keep the ports as close to the
published code as the current stack allows.

**Known deviations still present:** transformers 5 defaults to SDPA attention where the authors'
transformers 4 versions used eager attention. The two are equal up to floating-point error. SummaC
batches pairs across examples as `build_images` does (D-028). The XSumFaith and LLM-AggreFact
reproductions measure whether any of this matters.

**Source:** checkpoint headers and state dicts inspected 2026-09-13; `summac` 0.0.4
`model_summac.py`; AlignScore `inference.py` at `a0936d5a`; `minicheck` `inference.py` at `b58b9fa6`
**Approved:** Sameer, 2026-09-13 (1, 2 and D-029)
**Config:** `src/configs/base.yaml`, `metrics.factcc.batch_size`
**Supersedes:** —

---

### D-031 · 2026-09-13 · Run outputs go to a committed `results/` folder, per-example values in CSV

**Decision:**
- The run folder is renamed from `runs/` to **`results/`** at the repository root. It still holds one
  folder per run, `results/<run_id>/`, with the same run-id format.
- `results/` is **committed to git**, so the evidence behind every number is on GitHub with the code.
  `runs/` and `*.jsonl` are removed from `.gitignore`.
- Per-example (inference) or per-step (training) values are written to **`metrics.csv`** instead of
  `metrics.jsonl`. The other four files are unchanged: `config.resolved.yaml`, `manifest.json`,
  `run.log` and `summary.json`.
- The CSV writer fixes the columns from the first record and rejects a later record with different
  keys. It rejects non-scalar values instead of turning them into strings. It refuses to open a
  file that already exists.

**Reason:** Sameer's decision, 2026-09-13: the team wants all results in one visible, versioned
folder. The outputs are small text, not model weights. A CSV per run opens directly in a spreadsheet
or pandas. The writer's checks keep the file loadable as one table.

**Consequences to watch:**
- *Size.* GitHub rejects files over 100 MB and warns above 50 MB. T0 rows hold only an ID, a
  one-sentence output, token counts, latency and the metric scores. The first run's actual file size
  is recorded when it exists.
- *Multi-agent traces.* The CSV holds one flat row per example. Full per-message traces for later
  topologies do not fit that shape and will need their own file, decided when those topologies are
  built.
- *Provenance in public.* `manifest.json` records `owner` and the machine's hostname. Once committed,
  both are visible to anyone with repository access, which affects the open anonymous-review question.

**Source:** Sameer, 2026-09-13
**Approved:** Sameer, 2026-09-13
**Config:** `src/configs/base.yaml`, `run.output_root`; `.gitignore`
**Supersedes:** D-030 item 6 wording "writes nothing to `runs/`" (now `results/`); the `runs/` and
`metrics.jsonl` conventions in AGENTS.md §1 and §3, updated to match

---

### D-032 · 2026-09-13 · All five metric ports validated; SummaC's published numbers use a test-tuned threshold

**Decision:** SummaC-ZS, SummaC-Conv, AlignScore, FactCC and MiniCheck are accepted as faithful ports
and may score benchmark outputs. D-018's assumption about the reproduction protocol is corrected
below; D-018 itself is left unchanged.

**Results** (`src/validate_metrics.py`, 2026-09-13; records in
`~/.cache/factagent_bench/metrics/validation/`):

| Metric | Data | Statistic | Achieved | Published |
| --- | --- | --- | --- | --- |
| SummaC-ZS | XSumFaith test (1,250) | balanced accuracy, threshold tuned on validation | 54.2 | 58.4 (Laban T2) |
| SummaC-Conv | same | same | 61.4 | 66.4 (Laban T2) |
| SummaC-ZS | same | balanced accuracy, threshold tuned on test | 57.6 | 58.4 (Laban T2) |
| SummaC-Conv | same | same | 66.5 | 66.4 (Laban T2) |
| SummaC-ZS | same | ROC AUC | 56.9 | 58.0 (Zha T2) |
| SummaC-Conv | same | ROC AUC | 70.4 | 70.3 (Zha T2) |
| AlignScore-large | same | ROC AUC | 78.1 | 75.8 (Zha T2) |
| FactCC | same | ROC AUC | 57.4 | 55.1 (Zha T2) |
| MiniCheck-FT5 | LLM-AggreFact AggreFact-XSum test (558) | balanced accuracy, threshold 0.5 | 74.3 | 74.3 (Tang 2024 T2) |

**Reason:**
*SummaC.* D-018 assumed the paper's 58.4 / 66.4 used a threshold chosen on the validation half,
following the paper's prose. With the validation-chosen threshold the port falls 4–5 points short. The
same scores agree with both published numbers when the threshold is chosen on the test half, as the
official `evaluate()` does: Conv 66.5 against 66.4 balanced accuracy, and 70.4 against 70.3 AUC.
ZS is within about one point on both. AUC needs no threshold, and it matches independently, so the
scores themselves are reproduced. The shortfall comes from the threshold protocol, and the published
table was most likely produced by the official code's test-tuned threshold.

*Others.* AlignScore and FactCC exceed their targets by 2.3 AUC points each, and MiniCheck matches
exactly. A higher reproduction is not a failure of faithfulness. Plausible sources are the numeric
stack (transformers 5, SDPA attention; D-030) and differences between our rebuilt XSumFaith split and
the one AlignScore's evaluation loaded. Neither was isolated.

**Consequences:** metric thresholds are never tuned on the benchmark's own outputs. T0 and later
topologies are compared on the raw scores. The temporary `summac` package (D-021) can now be
uninstalled.

**Source:** validation records above; `summac` 0.0.4 `benchmark.py` `evaluate()`; Laban et al. 2022
Table 2; Zha et al. 2023 Table 2; Tang et al. 2024 Table 2
**Approved:** Sameer, 2026-09-13
**Config:** none
**Supersedes:** D-018 — only its assumption that the published SummaC numbers use a validation-tuned
threshold

---

### D-033 · 2026-09-13 · T0 pipeline: Generator, T0 topology and `run_benchmark.py`

**Decision:**

| Item | Value |
| --- | --- |
| Generator prompt | template `xsum_one_sentence@v1`: D-016's instruction, a blank line, then `Example.source`; recorded in `manifest.json` under `prompt_templates` |
| Request seed | `run.seed`, sent with every request of the run (D-003) |
| T0 | registered `t0_single_agent`; one Generator call per example; `n_rounds` 1, `converged` true; wall-clock measured around the call |
| Entrypoint | `python3 src/run_benchmark.py --topology <name\|all> --backbone <name\|all> --task <name\|all> --set run.owner=<name> --set run.seed=<N>`; logic in `benchmarks/runner.py` |
| Run name | `<topology>_<task>_<backbone>`, so run ids read `YYYYMMDD_HHMMSS_t0_single_agent_tier1_xsum_qwen2.5-7b-instruct_seed0` |
| Phases | generate all outputs with vLLM running; stop vLLM; load, score and free each metric in turn; then write `metrics.csv` |
| Metric suite | `metrics.suite` = `summac, factcc, alignscore, minicheck, rouge_l, bertscore` (D-022; `summac` emits both SummaC variants) |
| `metrics.csv` columns | `run_id, output, prompt_tokens, completion_tokens, total_tokens, wall_clock_s, n_rounds, converged, n_messages, example_id, source_words, finish_reason, hit_length_cap, blank_output`, then every metric key |
| Blank output | row kept with `blank_output` true and empty metric cells; counted in `summary.json` (D-030) |
| `summary.json` additions | `n_examples, n_blank_outputs, length_cap_rate, prompt_tokens, completion_tokens, total_tokens, generation_wall_clock_s, mean_example_wall_clock_s, metric_means` |
| `manifest.json` additions | `cell`, the backbone's registry row, `prompt_templates`, `metric_suite`, `metric_keys`, `n_examples` |
| vLLM server log | the system temp directory, `factagent_vllm_<run_id>.log`, keeping the run folder at its five files |
| `Message.finish_reason` | new optional field, so D-012's per-response finish reason survives into the trace |

**Reason:** each item implements an earlier decision: D-003 seeds, D-012 cap reporting, D-016
prompt, D-020 sequential requests, D-022 metrics, D-030 blank outputs, D-031 results folder.
Separate generation and scoring phases are forced by GPU memory: the NF4 backbone reserves about
18 GiB (weights plus KV cache), and the scorers need several GiB more.

**Known limitations:**
- Outputs live in memory until scoring ends. A crash during scoring loses that run's generations,
  and the cell is re-run. The failure is still recorded in `summary.json`.
- A run from an uncommitted working tree is marked `dirty` and may not be cited (AGENTS §3). Commit
  before the seeds whose numbers will be reported.
- `run_matrix.csv` does not exist yet, so run ids are not yet written back to a matrix row.
- `peak_vram_gib` in `summary.json` covers the scoring process only. vLLM runs in its own process.

**Source:** D-003, D-012, D-016, D-020, D-022, D-030, D-031; AGENTS.md §3, §4.3, §4.4
**Approved:** Sameer, 2026-09-13 (go-ahead to build and run the Qwen seed-0 check)
**Config:** `src/configs/base.yaml` (`metrics.suite`); `src/configs/data/tier1_xsum.yaml`
**Supersedes:** —

---

### D-034 · 2026-09-13 · vLLM is stopped as a whole process group

**Decision:** `VLLMServer` starts vLLM in its own session (`start_new_session=True`). Stopping it
sends SIGTERM to the whole process group, escalates to SIGKILL after the grace period, and SIGKILLs
anything still in the group once the API server has exited, logging a warning if something was left.

**Reason:** Interrupting the first full T0 run during vLLM startup (2026-09-13, run
`20260913_170212_t0_single_agent_tier1_xsum_qwen2.5-7b-instruct_seed0`, status `failed`, 0 rows)
stopped the API server process, but its `VLLM::EngineCore` child survived and held 22,302 MiB of GPU
memory until it was killed by hand. Scoring could not have loaded after it, and every later run would
have failed to start. A CPU test now starts a real parent that forks a child which outlives SIGTERM,
and asserts `stop()` kills both.

**Source:** `nvidia-smi --query-compute-apps` after the interrupt, 2026-09-13; `src/tests/test_serving.py`
**Approved:** — (bug fix)
**Config:** none
**Supersedes:** —

---

### D-035 · 2026-09-13 · Tier 1 CNN/DailyMail: source, filtering, sample seed and duplicate removal

**Decision:**

| Parameter | Value |
| --- | --- |
| `data.name` | `tier1_cnndm` |
| `data.hf_path` | `abisee/cnn_dailymail` |
| `data.hf_revision` | `96df5e686bee6baa90b8bee7c28b81fa3fa6223d` |
| `data.hf_config` (new field, default `null`) | `3.0.0`, the non-anonymized version |
| `data.split` | `test` |
| `data.n_samples` / `n_strata` / `stratify_by` | `500` / `3` / `whitespace_words` (D-002) |
| `data.sample_seed` | `123` |
| `data.min_source_words` | `20` |
| `data.max_source_words` | `null` |
| `data.dedupe_sources` (new field, default `false`) | `true`: rows with identical article text keep only the lowest `id` |
| Trainable splits | none; evaluation-only (AGENTS.md §4.2) |

Order: remove duplicates → filter by length → compute terciles → draw. Both steps log before → after
counts.

**Reason:**

*Why 3.0.0.* The dataset card: "Version 3.0.0 provided a non-anonymized version of the data, whereas
both the previous versions were preprocessed to replace named entities with unique identifier
labels." Entity labels such as `@entity5` would reach the backbones and the NLI-based metrics as
literal text. See et al. 2017 §4 also "operate directly on the original text (or non-anonymized
version of the data)".

*Why these filtering values — measured on the test split at this revision, 2026-09-13.* 11,490 rows,
11,490 unique IDs, no empty articles or highlights. Article whitespace words: min 55, median 613,
mean 683.5, p99 1,689, max 1,954; 30.7% exceed 800 words, none exceed 2,000. A minimum of 20 removes
**0 rows** and the tercile boundaries are **484 / 772** words with or without it. It is kept for the
same reason as XSum's (D-005): it guards against degenerate rows at no cost, and one threshold for
Tier 1 is easier to defend than two. No maximum, for D-005's reasons; D-036 shows no truncation is
needed.

*Why `sample_seed` 123.* D-005's only requirement is that it is not 0, 1 or 2. Reusing it keeps one
sampling seed across the project; the samples are independent anyway because the pools differ.

*Why remove duplicates.* Two pairs of rows carry identical article text and identical highlights
under different IDs: `5c0c00dcd3a745b98a380431609355658b9e4163` / `ac5f7eb411c744562e9ff10c537288b3a125610c`
(169 words) and `26252d601e52ff5ffabde202bc4883ee24b4ba49` / `5e7b53548bb3a7998b9ed67d23bb3507ecef8ee0`
(1,678 words). If both copies were drawn, one document would count twice in every paired
comparison. Keeping the lowest ID follows D-006's rule that the draw depends on the set of rows, not
on upstream row order. Expected effect: 11,490 → 11,488; boundaries are recomputed at freeze. The
field defaults to `false`, so XSum's configuration and frozen sample are unchanged.

**Redistribution:** the card states `apache-2.0`. Whether that covers the CNN and Daily Mail article
text was not verified. As with XSum (D-006), only IDs are committed.

**Source:** `abisee/cnn_dailymail` dataset card and file listing at the pinned revision, checked
2026-09-13; See et al. 2017 (arXiv 1704.04368) §4, read via ar5iv HTML 2026-09-13; split statistics
measured 2026-09-13 with `DatasetAdapter`'s own filter and partition code; D-002, D-005, D-006
**Approved:** Sameer, 2026-09-13
**Config:** `src/configs/data/tier1_cnndm.yaml` (to be created), with the ID list in
`src/configs/data/tier1_cnndm_sample.yaml`
**Supersedes:** —

---

### D-036 · 2026-09-13 · Serving context lengths cover XSum and CNN/DailyMail: Qwen 3,399, Mistral 3,770, Llama 3,395

**Decision:**
Each backbone keeps **one** `max_model_len` in `model_registry.csv`, now set to cover every dataset
added so far: the longest prompt, counted with that backbone's tokenizer and chat template, plus
D-012's 256-token cap (the D-013 rule). For CNN/DailyMail the maximum is taken over **all 11,490
test rows**, not only the frozen 500, so any sample fits without re-measuring.

| Backbone | CNN/DM prompt median | p99 | max | new `max_model_len` | old (XSum) | position limit |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 817 | 2,210 | 3,143 | **3,399** | 2,809 (D-013) | 32,768 |
| Mistral-7B-Instruct-v0.3 | 871 | 2,415 | 3,514 | **3,770** | 3,189 (D-017) | 32,768 |
| Llama-3.1-8B-Instruct | 812 | 2,180 | 3,139 | **3,395** | 2,738 (D-017) | 131,072 |

Under the old values 8 / 3 / 6 CNN/DM test rows would not fit. Mistral's HuggingFace template and
`mistral_common` 1.11.7 again gave identical counts.

**Timing:** the registry is edited only after the Mistral and Llama XSum seeds 0–2 finish, so every
XSum seed of a backbone runs under the same serving settings. Before editing, the counts are re-run
with D-037's instruction (they were measured with D-016's XSum instruction as a stand-in, one word
different), and the registry takes the re-measured values.

**Reason:** a single length per backbone keeps the registry schema and serving code unchanged. The
alternatives were rejected: per-dataset lengths need a registry and launcher change for a
difference of about 600 tokens; truncation would affect only 17 row-backbone pairs but reintroduce
D-005's risk of references citing text the model never saw.

**Not yet verified:** that vLLM starts at these lengths on the A10G with NF4 weights. To be checked
with `src/check_backbone.py` before the first CNN/DM smoke test.

**Source:** prompt token counts measured 2026-09-13 on the pinned CNN/DM test split with each
backbone's tokenizer at its registry revision; D-012, D-013, D-017
**Approved:** Sameer, 2026-09-13 (option A: one length per backbone covering all datasets)
**Config:** `src/model_registry.csv`
**Supersedes:** D-013 and D-017 (their values; the measurement rule stands), effective when the
registry is edited

---

### D-037 · 2026-09-13 · CNN/DailyMail prompt: "Summarize the following article in three sentences:"

**Decision:**
For CNN/DailyMail the generation step sends one user message: the line `Summarize the following
article in three sentences:`, a blank line, then the article. No system message, no examples. The
template is `cnndm_three_sentences@v1`, registered as `TEMPLATES["tier1_cnndm"]` in the Generator
module and recorded in each manifest. The layout matches D-016. `max_tokens` stays 256 (D-012).

**Reason:**
*Precedent for three sentences.* Goyal et al. 2022 §2 use "Article: {{ article }} Summarize the above
article in N sentences." and Appendix A sets "N = 3 for CNN and DailyMail"; they report that
summaries "followed the given sentence length constraint 98% of the time". Zhang et al. 2023 §3.1:
"For CNN/DM, we solicit LLM summaries with the following prompt template “ Article: [article].
Summarize the article in three sentences. Summary: ” For XSUM, we modify the prompt template to
summarize in one sentence to match the style of the reference summaries." The one-sentence XSum
wording of D-016 therefore has the same precedent pattern.

*Reference format.* See et al. 2017 §4: "multi-sentence summaries (3.75 sentences or 56 tokens on
average)". Measured on the test split 2026-09-13: highlight lines median 4, mean 3.9; nltk sentences
median 4, mean 3.8; words median 51.

*Why three rather than four.* Both cited setups use three; four has no precedent we have read.
Disclosed for completeness: Goyal et al. Appendix A prompt the T0 model with "Summarize the article
below in 3 to 4 sentences?"; that was their prompt for a different model, not their standard one.

*Instruction placement.* Both papers put the article first; this project keeps the instruction
first, as in D-016, so the layout does not vary between datasets.

**Known trade-off:** outputs will run about one sentence shorter than the median reference, which may
lower ROUGE-L recall. It applies equally to every topology and backbone, so comparisons are
unaffected. The smoke test reports the length-cap rate and output sentence counts.

**Source:** Goyal, Li & Durmus 2022 (arXiv 2209.12356) §2 and Appendix A; Zhang et al. 2023 (arXiv
2301.13848) §3.1; See et al. 2017 §4 — all read via ar5iv HTML 2026-09-13; highlight statistics
measured 2026-09-13; D-012, D-016
**Approved:** Sameer, 2026-09-13
**Config:** Generator prompt template in `src/component/factagent_bench/agents/generator.py`,
versioned in the run manifest
**Supersedes:** —

---

### D-038 · 2026-09-13 · The Hub loader lives in `DatasetAdapter`; CNN/DailyMail highlights are kept verbatim

**Decision:**
1. `DatasetAdapter._load_raw` is concrete: `load_dataset(hf_path, name=hf_config, revision=hf_revision,
   split=split, token=False)`. `XSumAdapter` loses its copy; dataset subclasses implement only
   `_map_row`. Tests still override `_load_raw` with offline fixtures.
2. CNN/DailyMail loads through the Hub config `3.0.0` (D-035's `hf_config`), not a `data_files`
   path to the test parquet.
3. `reference` is the `highlights` string unchanged, one highlight per line.

**Reason:**
1. XSum and CNN/DailyMail load identically apart from the config name, and AGENTS.md §7.1 treats
   duplicated logic between two implementations as a bug. For XSum the call is unchanged:
   `name=None` is `load_dataset`'s default.
2. `datasets` 5.0.1 prepares every split of a config before selecting one
   (`load.py` calls `download_and_prepare` without a split; the parquet builder downloads all of
   the config's `data_files`), so this costs a one-time 837,094,602-byte download (the file
   listing at the pinned revision) instead of the 30,000,471-byte test file. Accepted to keep one
   loading path for every dataset and no file layout in config.
3. Changing the reference text would change what ROUGE-L and BERTScore compare against; outputs
   are likewise recorded unchanged (D-033).

**Source:** AGENTS.md §7.1; `datasets` 5.0.1 source (`load.py`, `packaged_modules/parquet/parquet.py`),
read 2026-09-13; Hub file listing for `abisee/cnn_dailymail` at `96df5e68`
**Approved:** Sameer, 2026-09-13
**Config:** `src/configs/data/tier1_cnndm.yaml`
**Supersedes:** —

---

### D-039 · 2026-09-13 · Serving context length 32,768 for every backbone

**Decision:**
`model_registry.csv` sets `max_model_len` to **32,768** for Qwen2.5-7B-Instruct,
Mistral-7B-Instruct-v0.3 and Llama-3.1-8B-Instruct. The value is no longer measured per dataset.
Applied on the CNN/DailyMail branch now: Kartik and Siddhanth run their XSum seeds on their own
machines from `main`, so the change reaches them only when merged.

**Reason:**
*What the setting does in vLLM 0.27.1 (source read 2026-09-13).* `max_model_len` is the largest
request (prompt plus completion) the server accepts. The KV-cache pool is sized from available
memory alone — `num_blocks = int(available_memory // page_size // num_layers)` in
`v1/core/kv_cache_utils.py` (`get_num_blocks`) — and does not depend on `max_model_len`. At startup,
`check_enough_kv_cache_memory` raises if that pool cannot hold one request of `max_model_len`
tokens. So a larger value does not change memory reserved or, with requests sent one at a time
(D-020), the generated text; it only has to fit in the pool.

*Correction.* D-013's reason "A larger value would reserve KV-cache capacity no request uses", which
D-017 and D-036 relied on, is wrong for this vLLM version for the reason above.

*Why 32,768.* It is the model's own limit (`max_position_embeddings`) for Qwen2.5-7B-Instruct and
Mistral-7B-Instruct-v0.3, and within Llama-3.1-8B-Instruct's 131,072. KV cache for one request at
this length, computed from each `config.json` (2 × layers × KV heads × head dim × 2 bytes per token):
Qwen 1.75 GiB, Mistral 4.00 GiB, Llama 4.00 GiB. Llama's own default of 131,072 would need
16.00 GiB for one request, which is unlikely to fit beside the weights on a 24 GiB A10G (D-033
records about 18 GiB reserved in total). One value for all three backbones covers every dataset
measured so far (longest CNN/DM prompt plus 256: 3,770 tokens, D-036) and leaves room for the longer
multi-round prompts of T1–T4 without re-measuring. Every backbone still reads every input uncut, with
the same 256-token completion cap (D-012).

**Not yet verified:** that vLLM starts at 32,768 on the A10G for each backbone
(`src/check_backbone.py`). The Qwen XSum results (D-033 runs) were produced at 2,809; by the source
above the outputs should be identical at 32,768, which has not been tested.

**Source:** vLLM 0.27.1 `v1/core/kv_cache_utils.py` (`get_num_blocks`, `check_enough_kv_cache_memory`)
and `config/model.py` (`_get_and_verify_max_len`), read 2026-09-13; each backbone's `config.json` at
its registry revision; D-012, D-020, D-033, D-036
**Approved:** Sameer, 2026-09-13
**Config:** `src/model_registry.csv`
**Supersedes:** D-013, D-017, D-036 (context-length values; D-013's memory reason is corrected above)

---

### D-040 · 2026-09-13 · 32,768-token serving verified on the A10G for all three backbones

**Finding:** `src/check_backbone.py` served each backbone at `max_model_len` 32,768 (NF4 via
bitsandbytes, vLLM's default `gpu_memory_utilization`) and got a reply with `finish_reason=stop`.
Figures from each vLLM server log, 2026-09-13:

| Backbone | Available KV cache memory | GPU KV cache size | Max concurrency at 32,768 | Check wall time |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 12.35 GiB | 231,200 tokens | 7.06x | 197 s |
| Mistral-7B-Instruct-v0.3 | 15.34 GiB | 125,680 tokens | 3.84x | 177 s |
| Llama-3.1-8B-Instruct | 12.37 GiB | 101,328 tokens | 3.09x | 198 s |

Llama's pool of 101,328 tokens is below its own 131,072-token default, so vLLM's startup check
would have refused the unset default on this GPU, as D-039 expected. GPU memory returned to 0 MiB
after each check.

**Source:** `/tmp/factagent_vllm_<backbone>.log` and the check output, 2026-09-13; D-039
**Approved:** — (verification of D-039)
**Config:** `src/model_registry.csv`
**Supersedes:** —

---

### D-041 · 2026-09-27 · Quantization confirmed by Dr. Jafari; D-008 is final

**Decision:**
Dr. Jafari confirmed the quantization proposed in Q-006: bitsandbytes 4-bit NF4 with double
quantization for serving (vLLM 0.27.1, D-008) and for T5 QLoRA/GRPO training. D-008 is no longer
provisional. No setting changes.

**Reason:** D-008 was approved provisionally, pending this confirmation.

**Source:** Dr. Jafari, confirmation relayed by Sameer on 2026-09-27; Q-006; D-008
**Approved:** Dr. Jafari; recorded by Sameer, 2026-09-27
**Config:** `requirements.txt`; `src/model_registry.csv` (`quantization`)
**Supersedes:** D-008 — only its provisional status

---


### D-045 · 2026-10-05 · XSum primary factuality metrics: AlignScore and MiniCheck

**Decision:**
On XSum, AlignScore (large checkpoint, D-023) and MiniCheck (Flan-T5-Large, D-024) are the primary
factuality metrics for every topology comparison and for the confirmatory hypothesis tests.
SummaC-ZS, SummaC-Conv and FactCC are still computed for every run and are reported as secondary.
ROUGE-L and BERTScore remain NLG-quality metrics and are not read as factuality. This entry covers
XSum only; primary metrics for CNN/DailyMail and SAMSum are not decided here.

**Reason:**
The two primary metrics separate faithful from unfaithful XSum summaries best, in both our
validation (D-032) and the published evaluations:

| Metric | Data | Statistic | Ours (D-032) | Published |
| --- | --- | --- | --- | --- |
| AlignScore-large | XSumFaith test | ROC AUC | 78.1 | 75.8 (Zha T2) |
| MiniCheck-FT5 | AggreFact-XSum test (LLM-AggreFact) | balanced accuracy | 74.3 | 74.3 (Tang 2024 T2) |
| SummaC-Conv | XSumFaith test | ROC AUC | 70.4 | 70.3 (Zha T2) |
| SummaC-ZS | XSumFaith test | ROC AUC | 56.9 | 58.0 (Zha T2) |
| FactCC | XSumFaith test | ROC AUC | 57.4 | 55.1 (Zha T2) |

SummaC-ZS and FactCC are within 8 AUC points of chance (50) on XSum. SummaC-Conv discriminates,
but less well than either primary metric. Two primary metrics also keep the confirmatory set small
for the multiple-comparisons correction (proposal, Week 13).

**Timing:** Recorded on 2026-10-05. The first T1 runs on XSum (Mistral-7B-Instruct-v0.3, seeds 0-2:
`20261004_184105_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed0`,
`20261004_191342_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed1`,
`20261004_194653_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed2`) were committed on
2026-10-04, before this entry. No T0-vs-T1 comparison had been computed when this decision was
made, and the decision uses only the validation evidence above, all of which predates those runs.

**Consequences:**
- Tables, figures and slides for XSum lead with AlignScore and MiniCheck; the secondary metrics
  appear alongside them, labelled as secondary.
- Confirmatory hypothesis tests on XSum use the two primary metrics; tests on the secondary
  metrics are exploratory.
- If T5 stays in scope, its XSum reward is built on a primary metric, not on SummaC-ZS or FactCC,
  since optimizing a near-chance metric invites reward hacking (Instructor Review #3).
- Metric keys and the per-run metric suite are unchanged (AGENTS.md §4.5).

**Source:** D-032 validation table; Zha et al. 2023 Table 2; Tang et al. 2024 Table 2; Instructor
Review #3 (issue #33): "Pre-register AlignScore and MiniCheck as primary for XSum"; proposal,
Week 13 (confirmatory vs. exploratory tests, multiple-comparisons correction)
**Approved:** Sameer, 2026-10-05
**Config:** none (analysis and reporting rule; no run parameter changes)
**Supersedes:** -

---

### D-046 · 2026-10-05 · T5 is trained with GRPO only

**Decision:**
T5 is trained with GRPO (Shao et al. 2024) on QLoRA adapters over the 4-bit NF4 base (D-008), as
the proposal specifies. No second training method is planned: offline DPO on metric-ranked samples
and a SCRPO-style critique-and-refine preference run were considered and not adopted. If GRPO does
not fit the A10G at a defensible group size (Q-004), the proposal's own fallback applies (proposal
§7 item 9), and a new entry records it.

**Reason:** GRPO is the method the proposal and H5 are written around, and it needs no value
network, which matters on 24 GB. One method keeps T5 to a single, citable recipe. The SCRPO-style
alternative and its literature are recorded in the literature review
(`reports/Latex_report/FactAgent_Bench_Report.tex`, Related Work, "Training for factual
consistency") so the choice can be revisited with a new entry.

**Source:** proposal §1 objective 4 and §4 Phase 3.5; Shao et al. 2024 (Q-004); literature review
(`reports/Latex_report/FactAgent_Bench_Report.tex`), Related Work: "GRPO"
**Approved:** Sameer, 2026-10-05
**Config:** `src/configs/rl/grpo_xsum.yaml` (to be created with the `rl/` module)
**Supersedes:** —

---

### D-047 · 2026-10-05 · T5 scope: standalone only, three backbones, XSum then ASQA; H5b and H5c move to future work

**Decision:**

| Item | Value |
| --- | --- |
| What T5 is compared with | T0, T1, T2, T3 and T4 results on the same backbone, the same frozen evaluation sample, the same metrics and seeds |
| Re-running T5 inside T1–T4 | **Not done.** H5b (train and orchestrate) and H5c (its generality) move to future work. |
| Backbones trained | All three: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct |
| Training datasets | XSum train split, then ASQA train split |
| Fine-tunes | 6 (3 backbones × 2 datasets) |
| Deep/shallow tracks | Removed. Every backbone gets the same treatment. |
| Order | All three XSum runs first. ASQA runs start only after the ASQA loader, ALCE metrics, ALCE reward and T0–T4 ASQA results exist. |
| Still recorded | Token cost and latency per example (AGENTS §5) and one-time training GPU-hours, so H5a's break-even request volume is computable |
| Tier discipline | Unchanged: only the XSum and ASQA train splits are trainable; CNN/DM, SAMSum and ELI5 are evaluation-only (transfer checks); Tier 3 is never trained on |

**Reason:** Instructor Reviews #1–#3 asked for a written T5 scope decision. The proposal's re-entry
runs (Weeks 8–10) are the largest block of T5 work, and the schedule has about nine weeks to the
2026-12-08 submission. Training all three backbones instead of one deep-track backbone gives H5a on
three model families, which the closest prior work (SCRPO, Qwen only) lacks. Dropping H5b removes
the least-explored contribution, and the paper says so as a limitation and future work.

**Consequences:**
- `DatasetAdapter.assert_trainable(split, track)` checks deep/shallow tracks; the track concept
  must be removed or redefined (shared-core change, separate PR with review).
- Six `run_matrix.csv` rows with owners and GPU-hour estimates are needed before the first run
  (AGENTS §12.4).
- T5 on ASQA has no comparison until T0–T4 run on ASQA.

**Source:** proposal §1 objective 4, §4 Phase 3.5, §5, §7 items 9–11; Instructor Review #3
(2026-10-04); literature review (`reports/Latex_report/FactAgent_Bench_Report.tex`), Related
Work: "Training for factual consistency" and "The gap"
**Approved:** Sameer, 2026-10-05 (to be communicated to Dr. Jafari)
**Config:** `src/configs/rl/` (to be created); `src/run_matrix.csv` (to be created)
**Supersedes:** — (replaces the proposal's deep/shallow two-track design and Weeks 8–10 re-entry
plan, neither of which had a D-entry)

---

### D-048 · 2026-10-05 · T5 XSum reward is AlignScore; MiniCheck is the held-out judge

**Decision:**
The GRPO reward for T5 on XSum is the registered `alignscore` metric (AlignScore-large, `nli_sp`,
D-023) scored against the source article, used through `metrics/` and never reimplemented
(AGENTS §4.6). Its output range is [0, 1]. MiniCheck (D-024) is the held-out judge: T5 claims rest
on MiniCheck, together with human evaluation and Tier 3. T5's AlignScore numbers are still
reported, labelled as the training reward. This narrows D-045 for T5 only: AlignScore is not one of
T5's primary metrics, and D-045 applies unchanged to T0–T4 on XSum. SummaC-ZS and FactCC are not
used as the reward. The ASQA reward is a separate decision (open question on the ASQA reward
design).

**Reason:** D-032 shows SummaC-ZS (ROC AUC 56.9) and FactCC (ROC AUC 57.4) close to chance on
XSumFaith, so a reward built on them would push the policy almost at random with respect to
faithfulness. Instructor Review #3 names this a reward-hacking risk. AlignScore is the strongest
validated port (ROC AUC 78.1), and Gao et al. 2023 find stronger reward models overoptimize more
slowly. Training and judging with different checkers avoids the circularity of Roit et al. 2023,
who used one NLI model for both (their §3.5). MiniCheck is a separately trained model with a
different architecture (Flan-T5-Large vs RoBERTa-large), and it is the judge used by SCRPO, the
closest prior work, so T5 results are comparable with theirs. AlignScore-large (RoBERTa-large,
355M, D-023) is the lighter of the two to keep on the GPU beside the policy; the memory is still to
be measured (Q-004).

**Known limitation:** AlignScore and MiniCheck are both "is this supported by the source" checkers,
so both may favour short or copied text. Length and extractiveness are monitored during training
(open question on training monitors), and human evaluation remains the final check.

**Source:** D-023, D-024, D-032, D-045; Instructor Review #3 (2026-10-04); Roit et al. 2023 §3.5;
Gao et al. 2023; Hu et al. 2025 (SCRPO); literature review
(`reports/Latex_report/FactAgent_Bench_Report.tex`), Related Work: "Training for factual
consistency" and the AlignScore and MiniCheck paragraphs
**Approved:** Sameer, 2026-10-05
**Config:** `src/configs/rl/grpo_xsum.yaml` (to be created)
**Supersedes:** — (replaces the proposal's Week 4 wording "SummaC/FactCC entailment"; no earlier
D-entry)

---

### D-049 · 2026-10-05 · XSum reward has no ROUGE-L term; a length floor is a contingency only

**Decision:**
The T5 XSum reward is the AlignScore term alone (D-048) plus GRPO's KL penalty. The proposal's
ROUGE-L-against-reference term is dropped. A length floor is **not** part of the initial reward. It
is switched on only if the training monitors show outputs getting shorter (the trigger and the
floor value are open, under the training-monitor question). If a second reward term is ever added,
each term is normalized separately before combining (GDPO), never summed and then
group-normalized. This answers and closes Q-005.

**Reason:** In Maynez et al. 2020 (Table 2), 73.1% of XSum gold summaries contain extrinsic
hallucinations. A ROUGE-L reward against those references pays the policy for reproducing
unsupported content, working against the entailment term. ROUGE is also a weak training objective
compared with learned preferences (Stiennon et al. 2020). Roit et al. 2023 used no length or ROUGE
term. Their small policy did collapse to short summaries (§5), which is why the length floor is
kept as a contingency rather than dropped. GDPO (Liu et al. 2026) shows that summing terms before
GRPO's group normalization loses reward resolution, which is the mechanism behind AGENTS §4.6's
mixed-scales warning.

**Known deviations from Roit et al. (for the paper's limitations section):**
- Their reward model is an 11B T5-XXL NLI. Ours is AlignScore-large (355M); a model that size
  cannot share 24 GB with the policy and its rollouts.
- Their optimizer is actor-critic with GAE; ours is GRPO (D-046).
- Their KL anchor is a task-supervised model; ours is the instruction-tuned base.

**Source:** Q-005; Maynez et al. 2020 Table 2 (full text); Stiennon et al. 2020; Roit et al. 2023
§2.1, §5 (full text); Liu et al. 2026 (GDPO); literature review
(`reports/Latex_report/FactAgent_Bench_Report.tex`), Related Work: "RL from entailment feedback"
**Approved:** Sameer, 2026-10-05
**Config:** `src/configs/rl/grpo_xsum.yaml` (to be created)
**Supersedes:** — (closes Q-005)

---

### D-042 · 2026-09-30 · Tier 1 SAMSum: source, filtering and frozen sample

**Decision:**

| Parameter | Value |
| --- | --- |
| `data.name` | `tier1_samsum` |
| `data.hf_path` | `knkarthick/samsum` |
| `data.hf_revision` | `6b929ff10edec703164e3ddb2e94aae058c9ab5f` |
| `data.hf_config` | `null` |
| `data.split` | `test` |
| `data.n_samples` / `n_strata` / `stratify_by` | `500` / `3` / `whitespace_words` (D-002) |
| `data.sample_seed` | `123` (D-035) |
| `data.min_source_words` | `20` (D-005, D-035) |
| `data.max_source_words` | `null` |
| `data.dedupe_sources` | `true` |
| Trainable splits | none; evaluation-only (proposal §2; AGENTS.md §4.2) |

The sample was drawn on 2026-09-30 and is frozen in `tier1_samsum_sample.yaml`. A second draw
from the same config reproduces it exactly.

**Reason:**

*Why this repository.* The proposal's source, `Samsung/samsum`, returns HTTP 401 and is absent
from Hub search (checked 2026-09-30). The authors' release is still distributed as an arXiv
ancillary file (`arxiv.org/src/1911.12237v2/anc/corpus.7z`, 2,944,100 bytes, sha256
`a97674c66726f66b98a08ca5e8868fb8af9d4843f2b05c4f839bc5cfe91e8899`). `knkarthick/samsum` at the
pinned revision was compared with it row by row. Test split: the same 819 IDs, all 819 summaries
identical, and all 819 dialogues identical except line endings (724 original dialogues use `\r\n`;
the copy uses `\n`). Validation: 818 rows, identical on the same terms. The Hub copy loads through
the existing `DatasetAdapter._load_raw` (D-038); the arXiv archive would need a second loading
path and a 7z dependency.

*Why these filtering values — measured on the test split, 2026-09-30.* 819 dialogues, none empty,
no duplicate dialogue text. Dialogue whitespace words: min 9, median 74, max 516. The Tier 1
minimum of 20 words is kept for consistency with XSum and CNN/DailyMail; unlike those datasets it
removes non-degenerate rows here: 819 → 774 (45 short dialogues, 5.5%). Length boundaries on the
filtered pool: 54 / 109 words; stratum sizes 262 / 255 / 257. `dedupe_sources` has no effect on
this split and is set as for CNN/DailyMail. `sample_seed` 123 follows D-035.

*IDs.* Test IDs are strings of digits, some with a suffix such as `-1`; 12 of the 500 sampled IDs
carry one. They load as strings and are stored quoted (D-006).

*Serving context.* Longest sampled prompt under D-043's wording, with each backbone's tokenizer and
chat template: Qwen 738, Mistral 801, Llama 744 tokens. With the 256-token cap (D-012) every prompt
fits D-039's 32,768.

**Known risks:**
- The factuality metrics were validated on news (D-032). Dialogue input — one `Speaker: message`
  turn per line, with placeholders such as `<file_gif>` — is untested for them, and nltk sentence
  splitting on unpunctuated chat lines is unverified. A sample of scored outputs is checked by hand
  after the first run.
- Redistribution: the release is licensed CC BY-NC-ND 4.0 (`licence.txt` in the arXiv archive). As
  with XSum (D-006), only IDs are committed. Whether model outputs committed under `results/` count
  as derivatives under the ND term is not resolved; to be confirmed with Dr. Jafari before public
  release.

**Source:** Gliwa et al. 2019 (ACL Anthology D19-5409); arXiv 1911.12237v2 ancillary `corpus.7z`
(`README.txt`, `licence.txt`, `test.json`, `val.json`); `knkarthick/samsum` at the pinned
revision; split statistics and the row-by-row comparison measured 2026-09-30; prompt token counts
measured 2026-09-30 at each backbone's registry revision; D-002, D-005, D-006, D-012, D-035, D-038,
D-039
**Approved:** Sameer, 2026-09-30 (500 samples; minimum 20 words)
**Config:** `src/configs/data/tier1_samsum.yaml`, with the ID list in
`src/configs/data/tier1_samsum_sample.yaml`
**Supersedes:** —

---

### D-043 · 2026-09-30 · SAMSum prompt: "Generate a summary of the following dialogue snippet:"

**Decision:**
For SAMSum the generation step sends one user message: the line `Generate a summary of the
following dialogue snippet:`, a blank line, then the dialogue. No system message, no examples, no
length instruction. The template is `samsum_dialogue_snippet@v1`, registered as
`TEMPLATES["tier1_samsum"]` in the Generator module and recorded in each manifest. `max_tokens`
stays 256 (D-012).

**Reason:**
*Precedent.* Ramprasad, Ferracane & Lipton 2024 §2.2 generate SAMSum and DialogSum summaries
zero-shot with GPT-4 and Alpaca-13B: "we use the default settings and prompt zero-shot using the
following template to generate summaries: Generate a summary of the following dialogue snippet:
{{Dialogue}}". Their study measures the faithfulness of LLM dialogue summaries, which is this
project's question on this dataset. Their use of default settings matches D-011, and they chose
zero-shot "to better understand the model's inherent capabilities", which is T0's role.

*Layout.* Their template places the dialogue after the instruction on the same line. This project
keeps D-016's layout, instruction then a blank line, as D-037 did for CNN/DailyMail, so the layout
does not vary between datasets. The wording is theirs, unchanged.

**Considered and rejected:** Laskar et al. 2023, Appendix L, Table 27: "Write a very short and
concise summary of the following dialogue in not more than 20 words: [DIALOGUE]". It controls
length, as D-016 and D-037 do for the news datasets, but it comes from a broad ChatGPT benchmark
rather than a faithfulness study, and ChatGPT exceeded its word limit on 56.4% of SAMSum outputs
(their Table 18).

**Known trade-off:** unlike D-016 and D-037, the prompt sets no length. Outputs may run longer than
the references (filtered test pool: median 19 words, mean 20.6), which can lower ROUGE-L and
BERTScore precision and raises completion token cost (H4). It applies equally to every topology and
backbone, so comparisons within SAMSum are unaffected; output length and the length-cap rate are
reported.

**Source:** Ramprasad, Ferracane & Lipton 2024, ACL (2024.acl-long.677; arXiv 2406.03487) §2.2;
Laskar et al. 2023, Findings of ACL (2023.findings-acl.29; arXiv 2305.18486) Appendix L, Tables 18
and 27 — both read 2026-09-30; reference lengths measured 2026-09-30; D-011, D-012, D-016, D-037
**Approved:** Sameer, 2026-09-30 (option A)
**Config:** Generator prompt template in `src/component/factagent_bench/agents/generator.py`,
versioned in the run manifest
**Supersedes:** —

---

### D-050 · 2026-10-06 · A blank completion receives GRPO reward 0.0

**Decision:**
During T5's GRPO training, a completion that is empty or whitespace-only receives a reward of
**0.0** and is never passed to the reward's metric. Every other completion is scored by the
metric as usual (D-048).

**Reason:** `Metric.score()` refuses blank predictions: "A blank generated output is never
scored" (`metrics/base.py`). The reward still needs one value per completion. AGENTS.md §4.6
requires one float per completion and no `None`, which rules out the alternative TRL 1.14.1
offers: a reward of `None`, which leaves the completion out of its group's mean and gives it zero
advantage (`trl/trainer/grpo_trainer.py`, `_calculate_rewards` and the unscorable mask). 0.0 is
the bottom of AlignScore's [0, 1] output range (D-023), so GRPO never prefers a blank to a real
summary, and the policy is pushed away from blanks. No paper found sets this value; it is a
deliberate default (AGENTS.md §12.1, reason 4).

**Trade-off:** a blank ties with a completely unsupported summary rather than ranking below it,
and the constructed value enters its group's mean and standard deviation. The blank rate is to be
logged at every training step (training-monitor question) so its effect is visible.

**Source:** `src/component/factagent_bench/metrics/base.py` (module docstring and
`_check_inputs`); TRL 1.14.1 `trl/trainer/grpo_trainer.py`, read 2026-10-06; AGENTS.md §4.6,
§12.1; D-023; D-048
**Approved:** Sameer, 2026-10-06
**Config:** `reward.blank_reward` (`src/component/factagent_bench/config/reward.py`, dataclass
default); `src/configs/rl/grpo_xsum.yaml` once created
**Supersedes:** —

---

### D-051 · 2026-10-06 · T1 Self-Refine follows Madaan et al. (2023): up to 4 rounds, few-shot feedback and refine, no system prompt

**Decision:**

1. **Up to 4 rounds.** One round is one feedback step plus one refine step. This adopts Self-Refine's
   general limit ("up to a maximum of 4 iterations", Madaan et al. 2023, §3.1). The paper never
   evaluated summarization, so this is its general cap carried over, not a summarization-specific
   finding. It replaces the proposal's "up to 3 rounds".
2. **Few-shot feedback and refine prompts**, as in the paper (§3.1, App. S). Every demonstration
   comes from the XSum **train** split, never the evaluation sample (which is drawn from the test
   split, D-002, D-006).
3. **No system prompt for T1.** Every T1 call is a single user message, as T0's is (D-016).

| Item | Value |
| --- | --- |
| Draft (round 0) | the Generator's T0 call, unchanged: `xsum_one_sentence@v1` (D-016). The first draft is the T0 baseline, matching the paper's comparison against the same model without refinement |
| Feedback | role `feedback`, template `xsum_self_feedback@v1`: "List each phrase in the summary that the article does not support, quoting it exactly, one per line. If the article supports the whole summary, reply with only NONE." Sees the article and the latest summary |
| Refine | role `refiner`, template `xsum_self_check_revise@v2`: "Rewrite the summary using the feedback. Reply with only the rewritten one-sentence summary." Sees the article and every earlier summary with its feedback, oldest first (paper §2). v1's "check … then give a corrected summary" wording is dropped |
| Loop | draft → feedback → stop if NONE → otherwise refine → repeat; stops after the 4th refine with no further feedback call. At most 9 model calls per example |
| Stop signal | the whole feedback reply is `NONE`, ignoring case, surrounding whitespace, a trailing full stop, and quote or emphasis marks; anything else is feedback |
| Output | the latest summary: the draft or the last rewrite (never the feedback) |
| `converged` | true only when the loop stopped on NONE; false when it reached 4 rounds |
| `n_rounds` | rounds actually run, counted as feedback calls made (1 to 4); a round that ended on NONE counts |
| Token cost | summed over every call |
| Demonstrations | 3, provisional (see Open): XSum train IDs `32549602`, `38537239`, `36133214`. The feedback prompt sends all three; the refine prompt sends the two with a rewrite. For `32549602` and `36133214` the summary under review is XSum's own reference summary, which contains unsupported phrases; the feedback quotes them and the rewrite uses only the article. For `38537239` the summary is written by us to be fully supported, and the feedback is NONE |
| Roles and backbone | the three roles share one serving client: this is self-refinement. The topology decides which role speaks; no role chooses its prompt from `round_index` |
| Datasets | XSum only. CNN/DailyMail has no T1 templates until its demonstrations are chosen from its own train split; building T1 for it fails with an `AgentError` |

**Reason:** T1 previously ran a fixed second call with a zero-shot "check and correct" prompt and
no feedback step, which is not Self-Refine as published. A reviewer comparing T1 to Madaan et al.
would flag the missing feedback step, the missing history, the fixed round count and the zero-shot
prompts. Keeping round 0 identical to T0 makes the draft the T0 baseline, so any T1 gain is
attributable to the refinement loop.

**Open:**
- How many demonstrations to use. The paper used 2–6 depending on the task. Three is provisional;
  changing the set is a new template version.
- The demonstration summaries, feedback and rewrites need a second reader before the first cited
  run.
- Numbering, to confirm with Kartik before merge. As of 2026-10-06 the remote branches already
  use D-042–D-043 (`datasets-tier1-samsum`), D-045 (`primary-metrics-xsum`), D-046–D-049
  (`t5-rl-decisions`) and D-050 (`rl-reward`), so T1 takes D-051 and D-052. T2's code
  (`T2Script`) cites D-043, D-044 and D-047, which clash with those entries and with nothing here;
  T2 needs numbers from D-053 on.

**Source:** Madaan et al. (2023), *Self-Refine: Iterative Refinement with Self-Feedback*, §2, §3.1,
App. S; D-002, D-006, D-016
**Approved:** — (pending team review)
**Config:** `src/configs/base.yaml` (`topology.max_rounds: 4`); prompts in
`src/component/factagent_bench/agents/self_refine.py`
**Supersedes:** the proposal's "up to 3 rounds" for T1. Earlier T1 code cited D-042 for its prompt,
but D-042 is the SAMSum entry; no logged entry is superseded.

---

### D-052 · 2026-10-06 · The nine T1 runs made before D-051 are superseded, kept as a possible one-step ablation

**Decision:**
The nine T1 XSum runs below used the design D-051 replaces, and are **superseded**. They are
**not deleted**, and no T1 result in the paper is cited from them. We may keep them as a
**one-step ablation**: a single zero-shot "check and correct" call after the T0 draft, with no
feedback step and no loop.

The old design, as these runs recorded it in `manifest.json`:

| Item | Old design (these runs) | D-051 |
| --- | --- | --- |
| Roles | `generator` only | `generator`, `feedback`, `refiner` |
| Templates | `xsum_one_sentence@v1`, `xsum_self_check_revise@v1` | `xsum_one_sentence@v1`, `xsum_self_feedback@v1`, `xsum_self_check_revise@v2` |
| Calls per example | 2: draft, then one zero-shot check-and-correct | 2 to 9: draft, then feedback and refine until NONE or 4 rounds |
| Prompts | zero-shot | few-shot (XSum train split) |
| Saved per example | final output only | every message, in `traces.jsonl` |

| Run ID | Backbone | Seed | Code commit | Committed on |
| --- | --- | --- | --- | --- |
| `20261004_184105_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed0` | Mistral-7B-Instruct-v0.3 | 0 | `c7c4df7` | `feature/T1-topology-libraries` (`d1709f4`) |
| `20261004_191342_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed1` | Mistral-7B-Instruct-v0.3 | 1 | `c7c4df7` | `feature/T1-topology-libraries` (`d1709f4`) |
| `20261004_194653_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed2` | Mistral-7B-Instruct-v0.3 | 2 | `c7c4df7` | `feature/T1-topology-libraries` (`d1709f4`) |
| `20261004_223847_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B-Instruct | 0 | `d1709f4` | `feature/T1-topology-benchmarks` (`4b2b383`) |
| `20261004_230131_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B-Instruct | 1 | `d1709f4` | `feature/T1-topology-benchmarks` (`4b2b383`) |
| `20261004_232343_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B-Instruct | 2 | `d1709f4` | `feature/T1-topology-benchmarks` (`4b2b383`) |
| `20261004_235811_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed0` | Qwen2.5-7B-Instruct | 0 | `d1709f4` | `feature/T1-topology-benchmarks` (`4b2b383`) |
| `20261005_002103_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed1` | Qwen2.5-7B-Instruct | 1 | `d1709f4` | `feature/T1-topology-benchmarks` (`4b2b383`) |
| `20261005_004221_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed2` | Qwen2.5-7B-Instruct | 2 | `d1709f4` | `feature/T1-topology-benchmarks` (`4b2b383`) |

All nine finished with `status=success` on the full 500-example XSum sample. `d1709f4` only adds
result files, so all nine ran the same T1 code.

**Telling old runs from new:** rerunning T1 does not overwrite these. Every run gets a new
`results/<run_id>/` folder, and the run writers refuse to replace an existing file. But a rerun's
folder name differs from these only in its time stamp. Select T1 runs by the run IDs above or by
`manifest.json` → `prompt_templates`, never by name pattern or "latest run": a D-051 run lists
the `feedback` and `refiner` roles and `xsum_self_check_revise@v2`.

**Reason:** D-051 changed T1 from one fixed check-and-correct call to Self-Refine as published, so
these runs no longer measure T1. They still measure a real and cheaper variant, and rerunning it
later would cost the same GPU time again, so they are kept rather than deleted.

**Open:** whether the one-step variant is reported as an ablation. If it is, it needs a name
distinct from T1 in every table, and the D-051 runs must use the same 500 examples and seeds.

**Source:** D-051; the runs' `manifest.json` and `summary.json`
**Approved:** — (pending team review)
**Config:** none changed; the old design is recorded in each run's `config.resolved.yaml` and
`manifest.json`
**Supersedes:** the nine runs above as T1 results. It corrects D-051's note that "no logged entry
is superseded": no *decision* is superseded, but these runs are.

---

### D-053 · 2026-10-09 · T5 training code: TRL's GRPOTrainer, transformers rollouts, adapters outside results/

**Decision:**

| Item | Decision |
| --- | --- |
| Trainer | TRL 1.14.1 `GRPOTrainer` on a LoRA adapter over the 4-bit base (D-046), in `rl/trainer.py`; entry point `src/run_training.py` |
| Rollout generation | TRL's built-in transformers `generate`, not vLLM inside training |
| KL weight | `training.beta` must be > 0 |
| Prompt | the dataset's T0 Generator template (D-016), sent as one user message, as T0 sends it |
| Completion cap | `sampling.max_tokens` (D-012), the same cap as inference |
| LoRA targets | all linear layers |
| 4-bit base | NF4 with double quantization, loaded in `serving/` (D-008, D-041) |
| Adapters | written to `training.adapter_root/<run_id>/`, default `~/.cache/factagent_bench/adapters`, outside the repository; the path is recorded in `summary.json` |
| Unapproved values | group size, batch size, learning rate, KL weight, loss type, LoRA rank and alpha, rollout sampling, step count and the other GRPO settings are required config fields with no default |
| Step log | one `metrics.csv` row per optimizer step from TRL's log, including reward and KL |
| Training gate | `assert_trainable(split)`; the deep/shallow tracks are removed (D-047) |

**Reason:**
- *KL weight.* TRL 1.14.1 defaults `beta` to 0.0, and then neither builds a reference nor logs KL
  (`grpo_trainer.py`: `ref_model = None` and the `kl` metric only when `beta != 0`). AGENTS.md
  §4.6 requires KL every step, and D-049 relies on the KL penalty.
- *Loss type.* TRL's default `loss_type` is `dapo`, not the original GRPO loss of Shao et al.
  2024, so it is left to an explicit decision rather than inherited.
- *Rollouts.* The transformers path needs no vLLM engine reserving part of the 24 GB beside the
  policy and the reward model. Whether it is fast enough is not yet measured (Q-004).
- *Prompt and cap.* T5 is T0 plus the adapter, so H5a compares like with like.
- *LoRA targets.* LoRA on query and value only underperforms (Dettmers et al. 2023 §4; AGENTS.md
  §12.1).
- *Adapters.* `results/<run_id>/` holds exactly the five run files (`run.py`), and AGENTS.md §10.1
  warns against committing adapter weights.
- *Required fields.* The same mechanism as `run.owner`: the code can merge before the values are
  approved, and a run cannot start until each is set in `src/configs/rl/` with a decision-log
  entry behind it.
- *Seed.* TRL's trainer calls transformers' `set_seed(args.seed)` when constructed; `args.seed` is
  the run's seed, so it matches what `seeding.py` set.

**Source:** TRL 1.14.1 `trl/trainer/grpo_config.py` and `trl/trainer/grpo_trainer.py`, read
2026-10-09; AGENTS.md §4.6, §10.1, §12.1; `src/component/factagent_bench/run.py`; D-008, D-012,
D-016, D-041, D-046, D-047, D-049
**Approved:** Sameer, 2026-10-09
**Config:** `src/component/factagent_bench/config/training.py`; `src/configs/rl/grpo_xsum.yaml`
once its values are approved
**Supersedes:** —

---

### D-054 · 2026-10-09 · T5 GRPO settings: Dr. GRPO loss, KL 0.04, LR 1e-5, LoRA r16/α32, unmodified sampling

**Decision:** These values are the defaults in `TrainingConfig`. Group size, batch size, gradient
accumulation, step count and checkpoint interval stay undecided (Q-004, Q-011).

| Setting | Value | Source |
| --- | --- | --- |
| `beta` (KL weight) | 0.04 | Shao et al. 2024 §4.2 |
| `loss_type` / `scale_rewards` | `dr_grpo` / `none` | Liu et al. 2025 (Dr. GRPO) |
| `epsilon` (ratio clipping) | 0.2 | Schulman et al. 2017 Table 1 |
| `num_iterations` | 1 | Shao et al. 2024 §4.2 |
| `learning_rate` | 1e-5 | Shao et al. 2024 §4.2 (1e-6, full fine-tuning) × ~10 for LoRA (Schulman / Thinking Machines 2025) |
| `lr_scheduler_type` / `warmup_steps` | constant / 0 | Dettmers et al. 2023 App. B.2 (constant); no warmup is stated |
| `optim` | `paged_adamw_32bit` | Dettmers et al. 2023 §3, App. B.2 |
| `max_grad_norm` | 0.3 | Dettmers et al. 2023 App. B.2 |
| `gradient_checkpointing` | true | Dettmers et al. 2023 §3 |
| `mask_truncated_completions` | false | this decision |
| `temperature` / `top_p` / `top_k` / `repetition_penalty` | 1.0 / 1.0 / 0 / 1.0 | Liu et al. 2025 (temperature 1.0); TRL 1.14.1 |
| `lora_rank` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.0 | Hu et al. 2025 (SCRPO) §4.2; TRL 1.14.1 |
| `compute_dtype` | bfloat16 | Dettmers et al. 2023 App. B.2 |

**Reason:**
- *Loss.* GRPO as published averages each completion's loss over its own length, so a short
  completion with a positive advantage gets a larger per-token update and a long one with a
  negative advantage a smaller one (Liu et al. 2025). Shortening is the main reward-hacking risk
  for an entailment reward (Roit et al. 2023 §5), so the loss is divided by a constant instead
  (`dr_grpo`). Advantages are not divided by the group's standard deviation (`none`), which would
  enlarge small score differences in groups whose completions score alike. This is a departure
  from GRPO as published in Shao et al. 2024 (D-046); Liu et al.'s evidence is on math reasoning
  only, which the paper states.
- *KL weight.* GRPO's own value. Liu et al. 2025 use 0, but T5 needs the KL penalty (D-049) and KL
  logged (D-053).
- *Learning rate.* Shao et al.'s 1e-6 is for full fine-tuning. Schulman / Thinking Machines 2025
  find the best LoRA rate about 10 times the full fine-tuning rate, in RL as in supervised
  learning (about 15 times for runs under ~100 steps). The value is derived, not taken from a
  paper directly.
- *LoRA.* SCRPO is the closest setup: Qwen2.5-7B-Instruct, LoRA, faithful summarization. Alpha 32
  is also the value Schulman / Thinking Machines 2025 use. No source fits dropout for RL: TRL
  notes that dropout makes the same input give different log-probabilities, which adds noise to
  GRPO's probability ratio, so it is 0. Dettmers et al.'s 0.05 and 0.1 are for supervised
  fine-tuning and disagree within that paper (App. A.1 and App. B.2).
- *Sampling.* TRL 1.14.1 applies temperature both when sampling and when computing
  log-probabilities, but top-p, top-k and repetition penalty only when sampling. Any other value
  of those three would make the samples come from a different distribution than the one the
  loss assumes. Temperature 1.0 samples the policy unchanged. Rollouts therefore differ from
  T5's evaluation sampling, which uses each backbone's defaults (D-011).
- *Optimizer and clipping.* Paged AdamW absorbs memory spikes under gradient checkpointing on the
  24 GB card; QLoRA's gradient-clipping norm comes with it.
- *Truncated completions.* A one-sentence summary that runs to the cap is a failure; leaving it in
  the loss lets the reward score it.

**Source:** Shao et al. 2024 (arXiv 2402.03300) §4.1–4.2; Liu et al. 2025 (arXiv 2503.20783) and
its Table 6; Schulman et al. 2017 (arXiv 1707.06347) §6.1, Table 1; Schulman / Thinking Machines
2025, "LoRA Without Regret" (thinkingmachines.ai/blog/lora); Dettmers et al. 2023 (arXiv
2305.14314) §3, App. A.1 and B.2; Hu et al. 2025 (arXiv 2512.05387) §4.2; TRL 1.14.1
`grpo_config.py` and `grpo_trainer.py`; all read 2026-10-09. Roit et al. 2023 §5; D-011, D-046,
D-049, D-053
**Approved:** Sameer, 2026-10-09
**Config:** `src/component/factagent_bench/config/training.py` (dataclass defaults);
`src/configs/rl/grpo_xsum.yaml` once the undecided values exist
**Supersedes:** — (refines D-046: GRPO with the Dr. GRPO loss aggregation)

---

### D-055 · 2026-10-09 · T5 trains on the XSum train split without test articles; the Q-004 memory measurement grid

**Decision:**

*Training data (answers Q-012 except the length cap and the draw seed):*

| Item | Decision |
| --- | --- |
| Source | XSum train split, at the revision evaluation uses (D-006) |
| Filters | the 20-word floor (D-005); duplicate articles dropped, lowest id kept (the D-035 rule); every article identical to a test-split article dropped (`data.exclude_sources_from_split: test`) |
| Long articles | excluded above a cap in words, never truncated. The cap is chosen from the measurement below, in a new entry |
| Size | a frozen list of 10,000 articles, drawn at random through the existing freezing tool with a recorded seed (seed decided with the cap); the same list for every backbone; each article used at most once |
| Monitor set | a separate set of training-split articles for the held-out MiniCheck monitor, sized with Q-009 |
| Prompt | T0's prompt (D-016, D-053), wrapped by each backbone's own chat template; no reference summary is shown |

*Memory measurement (Q-004):*

| Item | Decision |
| --- | --- |
| Backbones | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, in that order |
| Prompt caps | 1,024 / 1,536 / 2,048 tokens, chat template included |
| Articles | per cap, the 10 longest eligible training articles whose prompt fits the cap under all three tokenizers |
| Group sizes | 2, 4, 8, ascending; a cell that runs out of memory skips the larger G at that backbone and cap |
| Batch | per-device batch = G (one prompt per forward pass); gradient accumulation 1 |
| Steps | 10 per cell; step 1 is left out of the seconds-per-step average |
| GRPO settings | D-054 |
| Execution | one process per cell; adapters deleted; every run directory kept |

**Reason:**
- *Test articles.* On the pinned train split, 6 articles are identical to a test article, 2 of them
  in the frozen 500-article evaluation sample. Training on them would let T5 learn articles it is
  later scored on. With all filters, the eligible pool is 204,045 → 203,846 (duplicates) →
  203,840 (test overlap) → 203,679 (20-word floor).
- *Length.* On the 203,793 train articles of at least 20 words, T0's prompt with the chat
  template measures, in tokens, median / p99 / maximum: Qwen 420 / 1,777 / 35,595; Mistral
  445 / 1,982 / 39,280; Llama 418 / 1,759 / 35,193. The proposal's 768–896-token range would
  leave out 13–22% of articles, all from the long end, while the evaluation sample spans every
  length (D-002). Excluding rather than truncating keeps the loader's no-truncation rule and the
  reward's evidence intact; a cap in words gives all three backbones the same articles.
- *Size.* 10,000 is SCRPO's training-set size (Hu et al. 2025 §4.2). Roit et al. 2023 report 20,000
  RL steps at batch size 32 (App. A, Table 5) but not how many articles they used.
- *Measurement.* The longest articles that fit are the worst case for memory at a cap. The group
  sizes span 2-GRPO (Wu et al. 2025; math, code and vision reasoning only), the Week 2 gate's 4,
  and Dr. GRPO's 8 (Liu et al. 2025, Table 6); DeepSeekMath's 64 cannot fit 24 GB (Q-004). A
  process per cell keeps one cell's memory out of the next. Ten steps is a deliberate default:
  every batch here is a longest batch, so peak memory is reached early; the trade-off is a
  seconds-per-step average over nine steps.

**Source:** XSum train and test splits at `7d4d486c2f8ef850b1a11aead99b894ff3dd7da9`, measured
2026-10-09 (filter counts from the measurement's `select` log); Hu et al. 2025 (arXiv 2512.05387)
§4.2; Roit et al. 2023 (arXiv 2306.00186) App. A; Wu et al. 2025 (arXiv 2510.00977); Liu et al.
2025 (arXiv 2503.20783) Table 6; Q-004; Q-012; D-002, D-005, D-006, D-016, D-035, D-053, D-054
**Approved:** Sameer, 2026-10-09
**Config:** `src/configs/rl/grpo_xsum_memory.yaml` and its per-cap article lists;
`data.exclude_sources_from_split` (`src/component/factagent_bench/config/data.py`)
**Supersedes:** —

---

### D-056 · 2026-10-09 · Q-004 timing cells on median-length training articles

**Decision:** In addition to the worst-case grid (D-055), time one cell per backbone on the 10
training articles whose prompt is closest to the median prompt of the training pool. The pool is
the eligible training articles whose prompt, chat template included, is within 1,536 tokens under
all three tokenizers; the cells run at G = 8, per-device batch 8, 10 steps, with the D-054
settings. They are capacity runs, exempt from the three-seed rule.

**Reason:** The grid used the longest articles that fit each cap, so its seconds per step are the
worst case. The 1,536-token G = 8 cells took 24.4 s (Qwen), 26.6 s (Llama) and 38.1 s
(Mistral) per article (`20261009_190413_grpo_xsum_memory_1536_qwen2.5-7b-instruct_seed0`,
`20261009_173514_grpo_xsum_memory_1536_llama-3.1-8b-instruct_seed0`,
`20261009_181902_grpo_xsum_memory_1536_mistral-7b-instruct-v0.3_seed0`), so one pass over a
10,000-article pool is bounded by about 68, 74 and 106 GPU-hours. The median training prompt is
far shorter, so a typical step is expected to be faster; the timing cells measure by how much, so
that the step count (Q-011) is planned from typical rather than worst-case cost. The 1,536-token
pool and G = 8 are the values proposed from the grid and await approval; if either changes, the
timing is re-run.

**Source:** the 27 grid cells of D-055 (run IDs in `src/run_matrix.csv`, commit `1e57915`); Q-004;
Q-011; D-054; D-055
**Approved:** Sameer, 2026-10-09
**Config:** `src/configs/rl/grpo_xsum_memory.yaml` (`measurement.typical_within_cap`,
`measurement.typical_group_size`); `src/configs/rl/grpo_xsum_memory_typical.yaml`
**Supersedes:** —

---

### D-057 · 2026-10-09 · T5 group size 8; training prompts capped at 1,536 tokens by a token filter; pool seed 123

**Decision:**

| Setting | Value |
| --- | --- |
| `training.num_generations` (G) | 8 |
| `training.per_device_train_batch_size` | 8: one article's whole group per forward pass |
| Length cap | an article is eligible for training only if its T0 prompt, chat template included, is at most 1,536 tokens under all three backbones' tokenizers. Applied once, when the 10,000-article pool is frozen. This replaces D-055's cap in words |
| Pool draw seed | 123 |

**Reason:**
- *Memory.* No cell of the Q-004 grid ran out of memory (D-055). At 1,536 tokens and G = 8, on
  the longest articles that fit, peak reserved memory was 15.5 GiB (Llama,
  `20261009_173514_grpo_xsum_memory_1536_llama-3.1-8b-instruct_seed0`), 16.0 GiB (Mistral,
  `20261009_181902_grpo_xsum_memory_1536_mistral-7b-instruct-v0.3_seed0`) and 17.79 GiB (Qwen,
  `20261009_190413_grpo_xsum_memory_1536_qwen2.5-7b-instruct_seed0`), against the A10G's
  22.49 GiB. That leaves at least 4.7 GiB for the CUDA context, which the figure excludes, and for
  summaries longer than the 50–130 tokens measured. At 2,048 tokens, G = 8 peaked at 21.2–21.3 GiB,
  too close to the limit; 1,024 tokens would keep only 87.7% of articles.
- *Group size.* 8 is the largest group measured, and the group size Liu et al. 2025 use (Table 6).
  A larger group gives a steadier group mean (Q-004).
- *Token filter, not word cap.* Tokens per word vary between articles: one 552-word article needs
  2,117 prompt tokens. Over the 203,679 eligible training articles, a word cap that guarantees
  1,536 tokens (548 words) keeps 79.3% of them; filtering on tokens keeps 97.0%. Filtering on the
  longest of the three prompts gives every backbone the same articles, which was D-055's reason
  for counting words.
- *Seed.* 123 is the seed the project's frozen samples already use (D-035).

**Source:** the Q-004 grid (D-055) and timing cells (D-056), run IDs in `src/run_matrix.csv`;
prompt lengths of the eligible training pool under the three tokenizers, measured 2026-10-09;
Liu et al. 2025 (arXiv 2503.20783) Table 6; Q-004; D-035; D-055; D-056
**Approved:** Sameer, 2026-10-09
**Config:** `src/configs/rl/grpo_xsum.yaml` and the frozen training pool, both to be written
**Supersedes:** D-055 — only its "Long articles" row (a cap in words)

---

### D-058 · 2026-10-09 · T5 XSum training length: 10,000 articles, 16 per step, 625 steps; checkpoints scored with MiniCheck

**Decision:**

| Setting | Value |
| --- | --- |
| Articles per fine-tune | 10,000: the whole frozen pool (D-057), each article used once |
| Articles per optimizer step | 16: `per_device_train_batch_size` 8 (one article's group, D-057) × `gradient_accumulation_steps` 16, i.e. 128 summaries per update |
| `training.max_steps` | 625 |
| `training.save_steps` | 125: a checkpoint every 2,000 articles, five per fine-tune |
| Checkpoint evaluation | every checkpoint is scored with MiniCheck on a held-out set of training-split articles, disjoint from the pool; the set's size is decided once its evaluation cost is measured (Q-009) |

**Reason:**
- *Size.* 10,000 is the training-set size of SCRPO, the closest prior work (Hu et al. 2025 §4.2).
  One pass over the frozen pool keeps D-055's rule that no article repeats.
- *Articles per step.* Shao et al. 2024 §4.2 train on 1,024 outputs per batch with 64 outputs per
  question, i.e. 16 questions per step; at G = 8 that is 128 summaries per update. Gradient
  accumulation adds no memory, so the measured per-step memory of D-057 still applies.
- *Cost.* At the per-article cost of D-056 (12.0, 19.7 and 11.0 s for Llama, Mistral and Qwen,
  interpolated to the pool's mean prompt of 517 tokens), one pass costs about 33, 55 and 31
  GPU-hours. Sameer approved up to three to four days per fine-tune.
- *Learning rate.* At 625 steps the approved 1e-5 (D-054) stands: the larger LoRA factor of
  Schulman / Thinking Machines 2025 applies to runs under about 100 steps.
- *Checkpoint evaluation.* Scoring each checkpoint on held-out articles shows whether 10,000
  articles were enough: a plateau before 10,000 supports the size, a rising curve is reported as
  a limitation. The same checkpoints give a training-size analysis without extra training.

**Source:** Hu et al. 2025 (arXiv 2512.05387) §4.2; Shao et al. 2024 (arXiv 2402.03300) §4.2;
Schulman / Thinking Machines 2025, "LoRA Without Regret"; D-054; D-055; D-056; D-057; Q-009; Q-011
**Approved:** Sameer, 2026-10-09
**Config:** `src/configs/rl/grpo_xsum.yaml` (to be written)
**Supersedes:** —

---

### D-059 · 2026-10-09 · T5 training pool frozen; runs resume from checkpoints; forced-length worst-case cells; Sameer runs all six fine-tunes

**Decision:**

| Item | Decision |
| --- | --- |
| Training pool | frozen once, by `python -m factagent_bench.rl.pool`, into `src/configs/rl/grpo_xsum_pool.yaml`: the eligible XSum training articles (D-055), those whose T0 prompt is within 1,536 tokens under all three tokenizers, then 10,000 drawn at random with seed 123 through the project's isolated sampler (D-057, D-058). Training replays the IDs; it never re-filters or redraws |
| Training config | `src/configs/rl/grpo_xsum.yaml`, one config for the three backbones |
| Resume | `run_training.py --resume results/<run_id>` continues an interrupted run from its latest checkpoint, with that run's own resolved config. The continuation writes checkpoints into the same adapter directory and is recorded as a new run directory whose manifest names the first run (`resumed_from`) and the checkpoint used; no earlier run directory is changed. A run's GPU-hours are the sum over its continuations |
| Forced-length cells | before the first fine-tune, the 10 longest 1,536-token articles (D-055) run once per backbone at G = 8 with every completion forced to the 256-token cap (`training.force_full_completions`, measurement only). An out-of-memory cell is recorded as a result |
| Owner | Sameer runs all six T5 fine-tunes, one after another on his instance |

**Reason:**
- *Pool.* The filter needs three tokenizers, so it runs once at freeze time and the frozen list is
  the evidence, as for the evaluation samples (D-006). The freeze measured 203,679 eligible
  articles, 197,582 of them within 1,536 tokens.
- *Resume.* A fine-tune runs for one to two and a half days (D-058). An interruption without
  resume restarts it from zero; resuming within the same run directory would rewrite evidence,
  which AGENTS.md §12.6 forbids. Lowering G after an out-of-memory error cannot be resumed
  either: the run would mix two group sizes.
- *Forced-length cells.* The grid's summaries were 50–130 tokens (D-055); training can lengthen
  them up to the 256-token cap, and memory grows with summary length. These cells measure the
  most memory any training step can need, before a day-long run depends on it.
- *Owner.* Sameer owns the RL track (AGENTS.md §10) and chose to run it on his instance alone.

**Source:** the pool freeze of 2026-10-09 (`src/configs/rl/grpo_xsum_pool.yaml` header); D-006,
D-055, D-056, D-057, D-058; AGENTS.md §10, §12.6
**Approved:** Sameer, 2026-10-09
**Config:** `src/configs/rl/grpo_xsum.yaml`; `src/configs/rl/grpo_xsum_pool.yaml`;
`src/configs/rl/grpo_xsum_memory.yaml` (`measurement.forced_cap`);
`training.force_full_completions` (`src/component/factagent_bench/config/training.py`)
**Supersedes:** —

---

### D-060 · 2026-10-09 · T5 is the final checkpoint; each fine-tune is trained once; no automatic stop

**Decision:**

| Item | Decision |
| --- | --- |
| Which checkpoint is T5 | the final checkpoint, step 625. The MiniCheck scores of the five checkpoints (D-058) are reported as analysis and never used to choose |
| Training runs per fine-tune | one, at seed 0. T5 is evaluated on the frozen evaluation sample at inference seeds 0, 1 and 2, like every topology |
| Stopping | every run trains all 625 steps; there is no automatic stop. Reward, KL and summary length, logged every step, are watched by hand. If summaries shorten, the run is stopped and D-049's length floor is switched on through a new entry |

**Reason:**
- *Final checkpoint.* Choosing among checkpoints after seeing their held-out scores would select on
  held-out data. Fixing the final checkpoint before training removes that choice.
- *One training run.* AGENTS.md §3 asks for three seeds per reported result. Three training seeds
  for six fine-tunes would be 18 runs of one to two and a half days each (D-058), which the
  schedule cannot hold. Inference seeds still give three runs per reported T5 result; the
  variance between training runs is not measured, and the paper states it as a limitation.
- *No automatic stop.* No cited source gives a threshold for halting on KL, reward or length
  (Q-009; proposal §7 item 8 names the pattern only). A threshold chosen now could not be argued;
  stopping by hand, with each intervention logged, keeps every decision on record.

**Source:** AGENTS.md §3; proposal §7 item 8; D-049; D-058; Q-009; Q-011
**Approved:** Sameer, 2026-10-09
**Config:** `src/configs/rl/grpo_xsum.yaml` (`training.max_steps`, `run.seed`)
**Supersedes:** —

---

### D-061 · 2026-10-09 · Deterministic torch kernels stay off for T5 training (answers Q-002)

**Decision:** `seeding.deterministic` stays false, its default in `src/configs/base.yaml`, for every
T5 run: `torch.use_deterministic_algorithms` is not enabled.

**Reason:** Operations without a deterministic implementation raise once the switch is on, and
`seeding.py` records that 4-bit QLoRA training may reach one; the switch also needs
`CUBLAS_WORKSPACE_CONFIG` set before the process starts, and slows some operations by an amount
not measured here. It governs PyTorch's own operations; whether custom kernels such as
bitsandbytes' 4-bit matmuls honour it is not established. Each fine-tune is trained once (D-060),
so a bit-exact rerun would not change what the paper claims: a run is reproducible from its
config, seed and data up to GPU floating-point differences, and the variance between training
seeds remains the limitation D-060 states.

**Source:** `src/component/factagent_bench/seeding.py` (`set_seed`) and the `SeedingConfig`
docstring; Q-002; D-060
**Approved:** Sameer, 2026-10-09
**Config:** `src/configs/base.yaml` (`seeding.deterministic`)
**Supersedes:** —

---

### D-062 · 2026-10-09 · Forced-length cells did not run out of memory; G = 8 and the 1,536-token cap stand

**Decision:** D-057's group size and prompt cap stand. The forced-length cells of D-059 ran on all
three backbones without running out of memory.

| Backbone | Forced-length run | Peak reserved memory | Same articles, unforced (D-057) |
| --- | --- | --- | --- |
| Llama | `20261009_210316_grpo_xsum_memory_1536_forced_llama-3.1-8b-instruct_seed0` | 17.99 GiB | 15.5 GiB |
| Mistral | `20261009_211351_grpo_xsum_memory_1536_forced_mistral-7b-instruct-v0.3_seed0` | 13.51 GiB | 16.0 GiB |
| Qwen | `20261009_212439_grpo_xsum_memory_1536_forced_qwen2.5-7b-instruct_seed0` | 15.01 GiB | 17.79 GiB |

**Reason:**
- *Worst case.* Each cell trained 10 steps on the 10 longest 1,536-token articles at G = 8, and all
  80 summaries per cell were exactly 256 tokens, the completion cap. No training step at this cap
  and group size holds longer sequences.
- *Headroom.* The highest peak of either kind, 17.99 GiB, leaves 4.07 GiB of the 22.06 GiB PyTorch
  can use on the A10G. D-057 compared against 22.49 GiB, the total `nvidia-smi` reports; the
  driver reserves 439 MiB of it.
- *Lower forced peaks.* The figure is the memory PyTorch's allocator holds, its cache included.
  Summaries of one fixed length can reuse the same blocks, while summaries of varying length
  leave cached blocks of many sizes; this is the likely reason Mistral and Qwen peaked lower when
  forced, and it was not measured. Training summaries vary in length, so planning uses the higher
  figure per backbone.
- *If memory still runs out.* The run is resumed from its latest checkpoint (D-059).

**Source:** the three run directories above; `src/run_matrix.csv` rows `q004_*_1536_g8_forced`;
D-057; D-059
**Approved:** Sameer, 2026-10-09
**Config:** none changed; `src/configs/rl/grpo_xsum.yaml` keeps G = 8 and the 1,536-token pool
**Supersedes:** —

---

### D-063 · 2026-10-09 · T5 generates one article's group at a time

**Decision:** During T5 training, completions are generated one forward pass at a time:
`generation_batch_size` equals `per_device_train_batch_size`, so each generation call holds one
article's group of G = 8. TRL's default, which generates all 16 articles of an optimizer step at
once (128 completions), is not used. `grpo_args` derives the setting, and every manifest records it
under `grpo_args`.

**Reason:**
- *The failure.* The first Qwen fine-tune, `20261009_213948_grpo_xsum_qwen2.5-7b-instruct_seed0`,
  ran out of memory before step 1. TRL 1.14.1 sets `generation_batch_size` to
  `per_device_train_batch_size × gradient_accumulation_steps`, here 128, unless told otherwise. The
  Q-004 cells ran with gradient accumulation 1, so they generated 8 completions at a time; their
  peaks (D-057, D-062) hold for that setting only.
- *Same training.* With `num_iterations` 1 the policy does not change within an optimizer step, so
  every completion of the step comes from the same model either way, and TRL applies no importance
  correction in either case. The Dr. GRPO loss of each forward pass is divided by a constant and by
  the accumulation steps, not by the generation batch, so a step's gradient is the same sum.
  Advantages are computed within each article's group, which one call holds whole. Only the order
  in which random samples are drawn changes.
- *Time.* The timing cells (D-056) also generated 8 completions at a time, so D-058's estimates
  stand.

**Source:** run `20261009_213948_grpo_xsum_qwen2.5-7b-instruct_seed0`; TRL 1.14.1 `GRPOConfig`
(`generation_batch_size`, `steps_per_generation`) and `GRPOTrainer` (`_prepare_inputs`, the
`dr_grpo` loss); D-056; D-057; D-058; D-062
**Approved:** Sameer, 2026-10-09
**Config:** `src/component/factagent_bench/rl/trainer.py` (`grpo_args`);
`training.per_device_train_batch_size` in `src/configs/rl/grpo_xsum.yaml`
**Supersedes:** —

---

### D-068 · 2026-10-15 · T5 XSum fine-tunes complete; no evaluation article in the training pool; actual peak memory; driver change

**Decision:** The three runs below are T5 for XSum as D-060 defines it: each run's `checkpoint-625`,
whose weights are identical to its `final` adapter. No article of the frozen evaluation sample is in
the training pool. No setting changes.

| Backbone | Run | Steps | GPU-hours | Peak reserved memory |
| --- | --- | --- | --- | --- |
| Qwen | `20261009_214506_grpo_xsum_qwen2.5-7b-instruct_seed0` | 625 / 625 | 33.0 | 21.28 GiB |
| Llama | `20261011_064822_grpo_xsum_llama-3.1-8b-instruct_seed0` | 625 / 625 | 36.1 | 18.64 GiB |
| Mistral | `20261012_185905_grpo_xsum_mistral-7b-instruct-v0.3_seed0` | 625 / 625 | 58.0 | 21.43 GiB |

All three started from commit `25c9d40` with a clean tree, and none was resumed. The adapters are
outside git (D-053) and are backed up off the instance.

**Reason:**
- *Overlap check.* The evaluation sample (500 test-split IDs, `tier1_xsum_sample.yaml`) and the
  training pool (10,000 train-split IDs, `grpo_xsum_pool.yaml`) were compared at revision `7d4d486`:
  0 shared IDs, and none of the 500 IDs is in the train split; 0 identical article texts after
  collapsing whitespace and lower-casing (also 0 pool articles identical to any test article, as
  D-055 requires); 0 identical reference summaries; no pool article shares half or more of its
  13-word sequences with the evaluation sample. 24 pool articles share 10–32%: different articles
  about the same events, and BBC boilerplate such as its note on the term "migrant", each with a
  different reference summary.
- *Peak memory.* Qwen and Mistral reserved up to 21.28 and 21.43 GiB of the 22.06 GiB PyTorch can
  use, against 17.99 GiB in the forced-length cells (D-062). The figure is the allocator's reserved
  memory, cache included: over 625 steps of varying summary length the cache grew beyond what ten
  fixed-length steps showed, as D-062 expected. No run ran out of memory; the margin was about
  0.6 GiB.
- *Driver.* After training, an automatic kernel update needed a matching NVIDIA module, and
  installing it upgraded the driver from 595.91.07, which the three training manifests record, to
  595.99.02, which later runs record. PyTorch 2.13.0 (CUDA 13.0) and the GPU are unchanged.

**Source:** the three run directories above; the overlap check of 2026-10-15, method as stated;
D-053, D-055, D-060, D-062
**Approved:** Sameer, 2026-10-15
**Config:** none changed
**Supersedes:** —

---

### D-100 · 2026-10-15 · Multi-round topology support: a `topology` config group, `trace.jsonl` as a sixth run file, and `TopologyResult.finish_reason`

**Decision:**

1. **A `topology` config group.** `TopologyConfig` gains `max_rounds`: the maximum number of
   feedback-to-refine iterations of a multi-round topology, None for a single-pass one (an integer
   >= 1 otherwise; booleans are rejected). A topology with parameters keeps them in
   `src/configs/topology/<name>.yaml`. For each cell, the benchmark runner merges that file after
   the task config and before `--set` overrides (`load_config(..., extra_paths=...)`), so
   `config.resolved.yaml` records the group and the run can be repeated from it alone. A topology
   without such a file binds only its name and records `topology: null`, as T0 does. The runner
   refuses a config whose `topology.name` is not the cell's topology. T1's `max_rounds: 4` moves
   from `base.yaml`, where D-051 put it, to `src/configs/topology/t1_self_refine.yaml`; the value
   is unchanged.
2. **`trace.jsonl`, a sixth file in every run directory**, for every topology: one JSON line per
   message of every result, oldest first, with the keys `run_id`, `example_id`, `message_index`,
   `role`, `round_index`, `content`, `prompt_tokens`, `completion_tokens`, `total_tokens`,
   `latency_s` and `finish_reason`. It is created when the run starts and flushed on
   `logging.flush_metrics_every`, so it exists, possibly empty, on every exit path. It replaces
   the one-line-per-example `traces.jsonl` that D-051 introduced; the three runs that wrote
   `traces.jsonl` keep it. `metrics.csv` keeps the same columns in the same order.
3. **`TopologyResult.finish_reason`**, a required field: why the model stopped while writing the
   output, taken from the output's own message. A multi-round topology can end on a message that
   is not its output, such as a final feedback check, so `finish_reason`, `hit_length_cap` and
   `length_cap_rate` are now read from this field instead of from the last message in the trace.
   Metric keys and `metrics.csv` columns are unchanged.
4. **Result assembly in the `Topology` base class** (`Topology._assemble`): wall-clock time, the
   token cost summed over the whole trace, the output and its finish reason are computed once for
   every topology (AGENTS.md §7.1). T0's results are unchanged.

**Reason:** T1 is the first topology with a round budget; later multi-round topologies bind theirs
the same way (AGENTS.md §4.4). A per-topology file keeps T1's parameters out of every other run's
config while still recording them in its own run directory. Per-round analysis, and checks such as
whether a refinement repeats its input, need every draft and every feedback, which `metrics.csv`
does not hold: D-031 noted that "full per-message traces for later topologies do not fit that shape
and will need their own file". One line per message keeps each line flat. Making `finish_reason`
required means a topology that does not set it fails when the result is built, instead of
reporting a length-cap rate taken from the wrong message.

**Source:** review of PR #34, 2026-10-05 (part A); D-031; D-051; AGENTS.md §3, §4.4, §7.1
**Approved:** Sameer, 2026-10-05 (review of PR #34); recorded 2026-10-15
**Config:** `src/configs/topology/t1_self_refine.yaml`; `TopologyConfig`
(`src/component/factagent_bench/config/topology.py`); `logging.flush_metrics_every`
(`src/configs/base.yaml`)
**Supersedes:** D-051's location for `topology.max_rounds` (`base.yaml`) and its `traces.jsonl`
format. Amends D-031 and AGENTS.md §3: a run directory holds six files, not five.

---

### D-101 · 2026-10-15 · T1 Self-Refine follows Algorithm 1: a separate `self_feedback` role, a final check, the last summary as output

**Decision:** T1 implements Self-Refine's Algorithm 1 (Madaan et al. 2023, §2) with two roles on one
backbone:

1. **Roles.** `generator` writes the draft with the unchanged T0 prompt `xsum_one_sentence@v1`
   (D-016), and every rewrite. `self_feedback` writes the feedback. `self_feedback` is a prompt
   with no tools; T2's Critic is the counterpart that has an NLI tool. The Generator picks its
   prompt from the history it is given (empty: the T0 prompt; ending in self-feedback: the refine
   prompt), never from the round number.
2. **Feedback input:** the article and the current summary only (Eq. 2,
   `fb_t = M(p_fb ‖ x ‖ y_t)`).
3. **Refine input:** the article and every earlier summary with its feedback, oldest first
   (Eq. 4, `y_{t+1} = M(p_refine ‖ x ‖ y_0 ‖ fb_0 ‖ … ‖ y_t ‖ fb_t)`).
4. **Loop:** draft, feedback, then either stop or rewrite and give feedback again. Feedback also
   runs on the last summary the round budget allows, because Algorithm 1 checks the stop
   condition after every feedback call. A run makes at most `2 × max_rounds + 2` model calls (10
   at the budget of D-102).
5. **Stop rule:** the feedback text, stripped of surrounding whitespace, with one trailing full
   stop removed and stripped again, equals `NONE` in any letter case. Anything else is feedback,
   including a blank reply, `NONE` followed by more text, and `**NONE**`.
   `SelfFeedback.is_stop` is the only place the stop word is interpreted. No model output is
   post-processed anywhere.
6. **Output:** the last summary, never feedback (Algorithm 1, line 10; §2: "we use the last
   refinement y_t as the output").
7. **Result fields:** `n_rounds` is the number of summaries written (1 to `max_rounds + 1`).
   `converged` is true when some feedback was the stop word and false when the budget ran out
   without one. A summary and the feedback on it share one `round_index`, starting at 0 for the
   draft. `finish_reason` is the output summary's own (D-100). `token_cost` sums every call.

**Reason:** a reviewer comparing T1 with Madaan et al. should find the same steps: feedback
separate from refinement, feedback on the current output only, refinement over the whole
history, and a stop indicator taken from the feedback (§2, "Iterating Self-Refine"). The final
check keeps `converged` meaningful for the last summary: without it, an example that used the
whole budget could not count as converged even when its last summary passed. Keeping round 0
identical to T0 makes the draft the T0 baseline, as the paper compares Self-Refine with the same
base model without refinement (§3.1). Earlier T1 code cited "D-042" for its prompt, an entry that
was never logged for T1 (D-042 is the SAMSum entry).

**Source:** Madaan et al. (2023), *Self-Refine: Iterative Refinement with Self-Feedback*, arXiv
2303.17651v2, §2 (Algorithm 1, Eq. 2, Eq. 4), §3.1; review of PR #34, 2026-10-05; D-016; D-100
**Approved:** team decision, 2026-10-05 (review of PR #34); recorded by Sameer, 2026-10-15
**Config:** `src/configs/topology/t1_self_refine.yaml`; roles in
`src/component/factagent_bench/agents/self_feedback.py` and `agents/generator.py`; loop in
`topologies/t1_self_refine.py`
**Supersedes:** D-051's design: the separate `refiner` role and the `feedback` role name, the
loop that ended after the last rewrite without a feedback check (at most 9 calls), `n_rounds`
counted as feedback calls, and the stop rule that also ignored quote and emphasis marks

---

### D-102 · 2026-10-15 · T1 round budget: at most 4 feedback-to-refine iterations

**Decision:** `topology.max_rounds: 4`. T1 makes at most 4 rewrites, so at most 5 summaries, and
the fifth still gets its feedback check (D-101).

**Reason:** Madaan et al. (2023), §3.1: "The FEEDBACK-REFINE iterations continue until the
desired output quality or task-specific criterion is reached, up to a maximum of 4 iterations."
The paper does not evaluate summarization; its seven tasks are listed in §3. This adopts the
paper's general limit, not a summarization-specific finding, and the limit is not uniform across
its own tasks: its Dialogue Response Generation setup used at most 3 iterations (App. M.2).

**Source:** Madaan et al. (2023), §3, §3.1, App. M.2; proposal Week 4 (`docs/proposal.md`)
**Approved:** team decision, 2026-10-05 (review of PR #34); recorded by Sameer, 2026-10-15
**Config:** `src/configs/topology/t1_self_refine.yaml` (`topology.max_rounds`)
**Supersedes:** the proposal's "up to 3 rounds" for T1 (Week 4) and AGENTS.md §4.4's cap of 3
rounds, for T1. D-051 item 1 set the same value; it is restated here for the D-101 loop.

---

### D-103 · 2026-10-15 · T1 worked examples: three XSum train articles of the evaluation median length

**Decision:**

- **Count and source.** 3 worked examples, shared by the feedback and refine prompts
  (`agents/self_refine_examples.py`). The articles come from the XSum `train` split at
  `data.hf_revision`, never from the frozen evaluation sample, which is drawn from the test split.
- **Selection rule.** The median whitespace word count of the 500 articles in the frozen
  evaluation sample (`tier1_xsum_sample.yaml`; the definition of `source_words`) is 300,
  computed 2026-10-15. The candidates are the 375 train articles of exactly 300 words; all pass
  `min_source_words` (20), and none has the same text as any test-split article. Sorted by ID and
  shuffled with `make_sampling_rng(123)` (`rng.sample(pool, len(pool))`; 123 is the evaluation
  sample's `data.sample_seed`), the first three that are each a single news story are taken.
  None was skipped: **24880761** (filming a rocket launch at Baikonur), **34640927** (a shell
  ship sculpture in Liverpool) and **28486128** (a replica Sutton Hoo brooch).
- **Content.** Each example holds the article verbatim; a one-sentence draft with two factual
  errors, each of one of these kinds: a wrong number, a wrong name or place, or a detail the
  article does not give; feedback with one line per error, starting with `- `, quoting the wrong
  words and saying what the article states instead or that it does not say it; and a revised
  one-sentence summary that the article fully supports. The drafts, feedback and revisions were
  written for this project, not taken from XSum or the paper, and checked sentence by sentence
  against each article by Sameer.
- **Order.** Feedback prompt: example 1 as draft and feedback, example 2 as revised summary and
  `NONE`, example 3 as draft and feedback. Refine prompt: all three as draft, feedback and revised
  summary.

**Reason:** Madaan et al. implement FEEDBACK and REFINE as few-shot prompts (§3.1), with feedback
examples as input-output-feedback triples and refine examples as input-output-feedback-refined
quadruples (§2; App. S). The paper's count varies by task: three for FEEDBACK and REFINE in
Acronym Generation ("3 such examples are used for REFINE and FEEDBACK", App. S), two in Math
Reasoning and six in Constrained Generation (App. S). For Dialogue Response Generation it gives
both three (App. M.2) and six (App. M.1). Three follows Acronym Generation, the count the paper
states unambiguously for both prompts. The length rule makes the examples typical of the
evaluation articles in length, and the seeded shuffle makes the choice reproducible. The `NONE`
case keeps the model from learning that every summary has an error; in the middle it is not the
most recent example.

**Trade-off:** the three articles add 900 words to every feedback and refine prompt (D-051's
articles added 360 words to its feedback prompt and 222 to its refine prompt), which raises T1's
prompt-token cost. That cost is part of T1's measured result (H4).

**Source:** Madaan et al. (2023), §2, §3.1, App. M.2, App. S; review of PR #34, 2026-10-05;
`src/configs/data/tier1_xsum.yaml` (`hf_revision`, `sample_seed`); `tier1_xsum_sample.yaml`
**Approved:** Sameer, 2026-10-15 (selection rule and examples); count and order: team decision,
2026-10-05
**Config:** `src/component/factagent_bench/agents/self_refine_examples.py`. Changing any example
requires a version bump of both templates that use them.
**Supersedes:** D-051's demonstrations (train IDs 32549602, 38537239 and 36133214: 92, 138 and 130
words, chosen without a recorded rule, two of them using XSum reference summaries as the flawed
summary) and its two-example refine prompt

---

### D-104 · 2026-10-15 · T1 prompts `xsum_self_feedback@v2` and `xsum_self_refine@v1`; feedback covers factual consistency only

**Decision:**

- `xsum_self_feedback@v2` instruction: "Check whether every part of the summary is supported by
  the article. For each part that the article does not support, write one line starting with
  "- " that quotes the unsupported words from the summary and says what the article states
  instead, or that the article does not mention it. If every part of the summary is supported by
  the article, reply with only the word NONE."
- `xsum_self_refine@v1` instruction: "Rewrite the latest summary so that it fixes every problem
  listed in its feedback. Use only information stated in the article. Keep it to one sentence.
  Reply with only the rewritten summary." "One sentence" restates D-016's task format; it is not
  length feedback.
- Layout: the instruction, then each worked example, then the item under work, as blocks joined
  by a blank line, with `###` on its own line between examples. The feedback prompt shows
  `Article:`, `Summary:` and `Feedback:` and ends with `Feedback:`. The refine prompt numbers the
  history (`Summary 1:`, `Feedback 1:`, …) and ends with the next `Summary <k+1>:` label.
- Feedback scope: content the article does not support. Length and style are out of scope.

**Reason:** the project measures factual consistency against the source (AGENTS.md §4.5), so the
feedback asks about nothing else. Madaan et al. ask for feedback that is "actionable" (it contains
a concrete action) and "specific" (it identifies concrete phrases to change) (§2). Quoting the
unsupported words makes the feedback specific, and saying what the article states instead makes
it actionable. In their ablation, generic feedback scored below Self-Refine feedback on all three
tasks tested (Table 2).

**Source:** Madaan et al. (2023), §2, Table 2; review of PR #34, 2026-10-05; D-016; AGENTS.md §4.5
**Approved:** team decision, 2026-10-05 (review of PR #34); recorded by Sameer, 2026-10-15
**Config:** `src/component/factagent_bench/agents/self_feedback.py` (`XSUM_SELF_FEEDBACK`);
`src/component/factagent_bench/agents/generator.py` (`XSUM_SELF_REFINE`)
**Supersedes:** D-051's prompts: `xsum_self_feedback@v1`, which listed unsupported phrases without
saying what the article states, and `xsum_self_check_revise@v2`. The feedback template keeps its
name and moves to `v2` because D-051's runs recorded `xsum_self_feedback@v1` for the old wording;
one label names one wording.

---

### D-105 · 2026-10-15 · No system prompt for T1; T1 covers XSum only for now

**Decision:** every T1 call is a single user message with no system message, as T0's is (D-016).
T1's feedback and refine templates exist for `tier1_xsum` only; building T1 for another dataset
fails at construction with an `AgentError`. CNN/DailyMail and SAMSum each get their own worked
examples from their own train split later, as new decisions.

**Reason:** with no system prompt, the feedback-refine loop is the only difference between T0
and T1, so a T1 gain is attributable to it. Worked examples from one dataset would show the
model the wrong source genre and summary length on another (CNN/DailyMail asks for three
sentences, D-037; SAMSum summarizes dialogue, D-043). When T2 merges with its
`AgentState.use_system_prompt` field (default true), every T1 call must set it to false.

**Source:** review of PR #34, 2026-10-05; D-016; D-037; D-043
**Approved:** team decision, 2026-10-05 (review of PR #34); recorded by Sameer, 2026-10-15
**Config:** `FEEDBACK_TEMPLATES` (`agents/self_feedback.py`) and `REFINE_TEMPLATES`
(`agents/generator.py`)
**Supersedes:** — (D-051 item 3 and its dataset scope said the same; restated for the D-101 design)

---

### D-106 · 2026-10-15 · The twelve earlier T1 runs are superseded

**Decision:** the twelve T1 XSum runs below used designs that D-101 replaces. They stay in
`results/` unchanged and are not reported as Self-Refine.

| Run ID | Design | Notes |
| --- | --- | --- |
| `20261004_184105_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed0` | one-step check and correct | D-052 |
| `20261004_191342_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed1` | one-step check and correct | D-052 |
| `20261004_194653_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed2` | one-step check and correct | D-052 |
| `20261004_223847_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed0` | one-step check and correct | D-052 |
| `20261004_230131_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed1` | one-step check and correct | D-052 |
| `20261004_232343_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed2` | one-step check and correct | D-052 |
| `20261004_235811_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed0` | one-step check and correct | D-052 |
| `20261005_002103_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed1` | one-step check and correct | D-052 |
| `20261005_004221_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed2` | one-step check and correct | D-052 |
| `20261006_131646_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed0` | D-051 loop | dirty |
| `20261006_151034_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed1` | D-051 loop | dirty |
| `20261006_170610_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed2` | D-051 loop | dirty |

The first nine may still be reported as a one-step ablation, named distinctly from T1 (D-052).
The last three record `git.dirty: true` at commit `4b2b383` in `manifest.json`: the D-051 code
was not yet committed when they ran (it was committed in `1be891f`), so under AGENTS.md §3 they
may not be cited at all.

**Reason:** none of the twelve ran the D-101 design. New T1 runs are told apart from these by
`manifest.json` → `prompt_templates`, which lists the `self_feedback` role,
`xsum_self_feedback@v2` and `xsum_self_refine@v1`, never by run-ID pattern or "latest run".

**Source:** each run's `manifest.json` and `summary.json`; D-051; D-052; AGENTS.md §3
**Approved:** Sameer, 2026-10-15
**Config:** none changed
**Supersedes:** extends D-052 to the three D-051 runs

---

### D-107 · 2026-10-15 · T1 Self-Refine: what follows Madaan et al. (2023) and what is the project's own choice

**Decision:** T1 (D-101 to D-106) separates the parts taken from Madaan et al. (2023, arXiv
2303.17651v2) from the project's own choices, so the paper's methods section can state each
departure. Every citation below was checked against the paper on 2026-10-15.

**Taken from the paper:**

| Element | T1 | Paper |
| --- | --- | --- |
| One model for every step | the draft, feedback and rewrites come from the same backbone | §2, Figure 1 |
| Loop order | feedback, then the stop check, then a rewrite; the stop check follows every feedback call | Algorithm 1, lines 2–8 |
| Feedback input | the article and the current summary only | §2, Eq. 2 |
| Refine input | the article and every earlier summary with its feedback | §2, Eq. 4 |
| Output | the last summary | §2 ("we use the last refinement y_t as the output"); Algorithm 1, line 10 |
| Stop signal from the feedback | the feedback prompt asks for `NONE` when nothing is unsupported | §2 ("the model can be prompted to generate a stopping indicator") |
| Round budget | at most 4 feedback-to-refine iterations | §3.1 ("up to a maximum of 4 iterations") |
| Few-shot prompts for both steps | feedback and refine both carry worked examples | §3.1 |
| Shape of the worked examples | (article, summary, feedback) for feedback; (article, draft, feedback, revised) for refine | §2; App. S |
| Kind of feedback | quotes the unsupported words (specific) and says what the article states (actionable) | §2; Table 2 (generic feedback scored lower on all three tasks tested) |
| Number of worked examples | 3 | App. S, Acronym Generation ("3 such examples are used for REFINE and FEEDBACK") |
| Worked examples written by the researchers | drafts, feedback and revisions written for this project | App. S (the authors write the feedback and improved outputs) |
| Baseline | T0, the same model without refinement | §3.1 |
| Prompt layout | examples separated by `###`, labelled fields | Figures 17 and 18 |

**The project's own choices:**

| Element | T1 | Paper, for comparison | Entry |
| --- | --- | --- | --- |
| Task and data | one-sentence summarization of XSum | not among the paper's seven tasks (§3) | D-102 |
| Backbones | 7–8B open instruction-tuned models | GPT-3.5, ChatGPT and GPT-4 (§3.1); a weaker model, Vicuna-13B, "was not able to consistently generate the feedback in the required format" (§4) | proposal |
| Draft prompt | zero-shot T0 instruction, so the draft is the T0 baseline | few-shot generation prompts for most tasks (App. S); §2 allows "a task-specific few-shot prompt (or instruction)" | D-101 |
| Written instruction | each feedback and refine prompt opens with an instruction before the examples | the prompt figures show examples only (Figures 16–18) | D-104 |
| Feedback scope | factual support only, no scores | several scored aspects in some tasks (App. M.1: 10; App. Q: 5) | D-104 |
| Output selection | always the last summary | in scored multi-aspect tasks, the highest-scoring output (App. M.2; §4) | D-101 |
| Stop rule | the exact word `NONE`, with one trailing full stop and case ignored | the indicator is left to each task (§2) | D-101 |
| Check at the budget | feedback also runs on the last summary | Algorithm 1's order; the paper does not state the budget case | D-101 |
| Budget across tasks | 4 for XSum | 4 in general (§3.1), but at most 3 for Dialogue Response Generation (App. M.2) | D-102 |
| `NONE` example | the middle feedback example is a correct summary with `NONE` | closest precedent: Figure 17's feedback examples include a high-scoring acronym (SCUBA, 23/25) beside a weaker one; a score, not a "nothing to fix" reply | D-103 |
| Example count source | 3, following Acronym Generation | the paper gives both 3 (App. M.2) and 6 (App. M.1) for Dialogue feedback | D-103 |
| Choosing the examples | XSum train articles of the evaluation median length (300 words), seeded shuffle | not described | D-103 |
| Errors in the drafts | two per draft: a wrong number, a wrong name or place, or a detail the article does not give | not described | D-103 |
| System message | none | not stated | D-105 |
| Decoding | each backbone's defaults (D-011), a 256-token cap per call (D-012), the same seed on every call (D-003) | "greedy decoding with a temperature of 0.7" (§3.1) | D-003, D-011, D-012 |
| Evaluation | factuality metrics on the final summary only; every intermediate summary saved in `trace.jsonl` | task metrics and human or GPT-4 preference (§3.2); per-iteration scores (Figure 4) | D-100 |

**Not checked:** the authors' released code, and the prompt figures for the paper's other six
tasks (Figures 19–35).

**Reason:** a reviewer comparing T1 with Self-Refine needs to see which parts reproduce the paper
and which are this project's adaptation to summarization with smaller open models. Recording the
split once, with citations, keeps the methods section and the code consistent.

**Source:** Madaan et al. (2023), arXiv 2303.17651v2: §2, Algorithm 1, Eq. 2, Eq. 4, Figure 1,
§3, §3.1, §3.2, §4, Table 2, Figure 4, App. M.1, App. M.2, App. Q, App. S, Figures 16–18; read
2026-10-15
**Approved:** Sameer, 2026-10-15
**Config:** none; the entries in the last column name their configs
**Supersedes:** —

---

### D-108 · 2026-10-16 · T1 XSum results: nine runs; three failed Llama starts recorded and their folders deleted

**Decision:** the nine runs below are T1's XSum results under D-101 (3 backbones × seeds 0–2).
Each finished with status `success`, 500 rows and `git.dirty: false`.

| Run ID | Backbone | Seed | Commit |
| --- | --- | --- | --- |
| `20261016_012133_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed0` | Qwen2.5-7B-Instruct | 0 | `55620f1` |
| `20261016_030408_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed1` | Qwen2.5-7B-Instruct | 1 | `55620f1` |
| `20261016_044540_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed2` | Qwen2.5-7B-Instruct | 2 | `55620f1` |
| `20261016_014509_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed0` | Mistral-7B-Instruct-v0.3 | 0 | `55620f1` |
| `20261016_032823_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed1` | Mistral-7B-Instruct-v0.3 | 1 | `55620f1` |
| `20261016_050951_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed2` | Mistral-7B-Instruct-v0.3 | 2 | `55620f1` |
| `20261016_162704_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B-Instruct | 0 | `1b2a819` |
| `20261016_184526_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B-Instruct | 1 | `1b2a819` |
| `20261016_205956_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B-Instruct | 2 | `1b2a819` |

`1b2a819` adds only the Qwen and Mistral result folders to `55620f1`, so all nine runs used the
same code.

Three earlier Llama runs failed before generating anything. Their folders were deleted, not
committed, and they are recorded here only: `20261016_030109_…_seed0`, `20261016_044241_…_seed1` and `20261016_062206_…_seed2`
(`…` = `t1_self_refine_tier1_xsum_llama-3.1-8b-instruct`). Each records status `failed` with
"vLLM exited with code 1 before becoming ready", about 2.9 minutes after start.

**Reason:** the three failures each began immediately after a Mistral run in the same
back-to-back chain. The same Llama runs succeeded when relaunched one at a time, each waiting
until the GPU reported less than 500 MiB in use. A team member reports the same failure and fix
during T0. The likely cause is GPU memory not yet released by the previous run's vLLM server;
this is not verified, because vLLM's own logs (in `/tmp`) were lost when the instance restarted on
2026-10-16. Their run IDs and error are recorded here so the failures stay on record without their folders.

**Source:** each run's `summary.json`, `manifest.json` and `run.log`
**Approved:** Sameer, 2026-10-16
**Config:** none changed
**Supersedes:** the three failed Llama runs above, as T1 results

---

### D-109 · 2026-10-19 · T1 few-shot prompts leak their worked examples on Llama and Mistral; the D-108 runs are not reported as Self-Refine, and T1 is re-run on all three backbones with a revised prompt

**Finding:** in the D-108 runs, a large share of Llama and Mistral T1 outputs summarize one of the
three worked examples (D-103) instead of the article under work. An output counts as leaked when
it contains a term specific to a worked example (Baikonur, Soyuz, Olympic torch, Bold Street,
FACT centre, shell, Ledbury, brooch, Sutton) that does not occur in the article being summarized.
An exact copy is an output identical to a worked example's draft or revised summary.

| Backbone | Leaked of 500, seed 0 / 1 / 2 | Exact copies | MiniCheck mean of leaked outputs | Run MiniCheck mean (T0, 3-seed mean) |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | 201 / 198 / 184 | 58 / 41 / 34 | .011 / .018 / .016 | .425 / .422 / .447 (.679) |
| Mistral-7B-Instruct-v0.3 | 253 / 245 / 235 | 34 / 40 / 36 | .100 / .093 / .093 | .384 / .367 / .397 (.683) |
| Qwen2.5-7B-Instruct | 1 / 0 / 1 | 0 / 0 / 0 | — | .694 / .687 / .663 (.647) |

In each run's `trace.jsonl`, the first leaked message is most often Llama's first refine call and
Mistral's first feedback call. The prompts are not truncated: refine prompts are about 1,850 tokens
against a 32,768-token context (`model_registry.csv`).

The superseded D-051 design leaked in the same way. Its three Mistral runs
(`20261006_131646`, `20261006_151034`, `20261006_170610`; dirty and not citable, D-106) contain a
D-051 demonstration term (dinghy, Great Yarmouth, Gorleston, Bridge of Dee, Dumfries, Carney,
Salford, Red Devils) absent from the article in 247 / 229 / 213 of 500 outputs. The failure
therefore comes from the prompt layout both designs share (full worked-example articles in one
user message, labelled like the item under work), not from the D-103 choice of examples alone.

Llama's self-feedback rarely returns the stop word: `converged` is true in 5 / 4 / 5 of 500
examples per seed, against 338 / 366 / 349 for Mistral and 476 / 474 / 478 for Qwen.

**Decision:**

1. The nine D-108 runs are not reported as Self-Refine results. They stay in `results/` unchanged.
   The Llama and Mistral runs are reported as a prompt-contamination failure, with the counts above.
   The Qwen runs did not leak, but are superseded so that the whole T1 column uses one prompt
   version.
2. T1 is re-run on all three backbones with a revised prompt, following Instructor Review #5
   (2026-10-19, Critique 1):
   - worked examples sent as earlier chat turns (user and assistant messages), not one text block;
   - the item under work labelled explicitly as the article to summarize;
   - an automatic leak check that flags any output sharing an 8-word sequence with a worked
     example, run on every T1 run and covered by a test.
   The exact wording, message layout and leak-check threshold are set in a later entry, after a
   pilot on a small subset has been read by hand.
3. The convergence rate is reported per backbone with every T1 result.

**Reason:** a T1 score computed on summaries of a different article measures prompt-following, not
self-refinement. Madaan et al. report that a weaker model, Vicuna-13B, "was not able to
consistently generate the feedback in the required format" (§4); these runs show a related failure
at 7–8B scale.

**Source:** the D-108 runs' `metrics.csv` and `trace.jsonl`; the D-051 runs' `metrics.csv`;
`agents/self_refine_examples.py` (D-103); Instructor Review #5 (2026-10-19), Critique 1;
Madaan et al. (2023), §4
**Approved:** Sameer, 2026-10-19
**Config:** none changed yet; the revised prompt gets its own entry and new template versions
**Supersedes:** the nine D-108 runs, as T1 results

---

### D-110 · 2026-10-19 · T1 prompts send the worked examples as chat turns and label the item under work; a worked-example overlap metric is scored on every run

**Decision:**

1. **Worked examples as chat turns.** Each worked example (D-103) is one earlier user message
   answered by one assistant message, sent before the final user message that holds the item under
   work. Feedback (`xsum_self_feedback@v3`): the user message is the instruction, the article and a
   summary; the assistant message is that summary's feedback, in D-103's order (draft, revised with
   `NONE`, draft). Refine (`xsum_self_refine@v2`): the user message is the instruction, the article,
   `Summary 1:` (the draft) and `Feedback 1:`; the assistant message is the revised summary.
2. **Instruction repeated** at the top of every user message, so each example has the same form as
   the real task. The D-104 wording is unchanged.
3. **The item under work is labelled.** After the instruction, the final user message adds one
   line: "Now check the summary of the article below." (feedback) or "Now rewrite the latest
   summary of the article below." (refine).
4. **No open label at the end.** The final user message ends with the summary (feedback) or the
   last feedback (refine). It no longer ends with an open `Feedback:` or `Summary <k+1>:` label,
   because in a chat layout the assistant reply is the answer. The worked examples no longer use
   the `###` separator.
5. **Not adopted:** repeating the article's first sentence after the feedback (Review #5). It
   changes what the model reads beyond the layout; it is considered only if the pilot still leaks.
6. **Serving:** `ChatClient.complete` takes optional earlier `turns`, as (user, assistant) pairs
   sent between the system message and the final user message. With no turns, the request is
   unchanged, so T0 and T2 calls are byte-identical to before.
7. **Worked-example overlap metric** (`worked_example_overlap`, in every run's metric suite):
   1.0 when an output shares a sequence of `metrics.worked_example_overlap.ngram` = 8 words
   (Review #5) with any worked example's article, draft, feedback or revised summary, and that
   sequence does not occur in the output's own article; otherwise 0.0. Words are lowercased runs
   of letters and digits. The run mean is the leak rate.

| Parameter | Value | Reason | Source |
| --- | --- | --- | --- |
| Example layout | one user turn + one assistant turn per example | separates examples from the item under work | Review #5, Critique 1 |
| Instruction placement | every user turn | each example has the same form as the real task | deliberate default |
| Target labels | the two sentences in item 3 | marks the item under work explicitly | Review #5 ("label the target explicitly"); wording ours |
| `metrics.worked_example_overlap.ngram` | 8 | the length Review #5 asked for | Review #5, Critique 1 |

**Checks:** on the D-108 runs, the metric flags 191 / 189 / 174 Llama outputs, 198 / 200 / 179
Mistral and 1 / 0 / 1 Qwen (seeds 0 / 1 / 2), and 0 of 500 in each of three T0 and three T2 Qwen
runs. It counts fewer than D-109's term match on Mistral, whose leaked outputs often paraphrase an
example rather than copy eight words in a row, so it is a lower bound on leakage. The new prompts
render with all three backbones' chat templates at about 1,800–2,150 prompt tokens per call.

**Source:** Instructor Review #5 (2026-10-19), Critique 1; D-103; D-104; D-109
**Approved:** Sameer, 2026-10-19
**Config:** `agents/self_feedback.py` (`XSUM_SELF_FEEDBACK`), `agents/generator.py`
(`XSUM_SELF_REFINE`), `serving/base.py`, `src/configs/base.yaml` (`metrics.suite`,
`metrics.worked_example_overlap.ngram`)
**Supersedes:** D-104's layout (one user message with `###`-separated examples and an open final
label) and D-105's "every T1 call is a single user message"; D-105's "no system prompt" stands

---

### D-111 · 2026-10-19 · T5 inference: T0's single call served with each backbone's GRPO adapter (`t5_rl_agent`)

**Decision:**

1. **Topology.** `t5_rl_agent` sends exactly T0's call: one user message with
   `xsum_one_sentence@v1` (D-016), no role preface, no system message. GRPO trained on that exact
   prompt as one user message (`rl/trainer.py` on `grpo-training-pool`), so each adapter sees at
   inference the prompt it was trained on. It is registered under its own name so that every T5
   result is labelled T5 (AGENTS.md §4.4).
2. **Adapters.** Each backbone is served with the final checkpoint (`checkpoint-625`, D-060) of
   its XSum fine-tune (D-068), named in `serving.lora_adapters` in
   `src/configs/topology/t5_rl_agent.yaml`:

   | Backbone | Fine-tune run ID |
   | --- | --- |
   | Llama-3.1-8B-Instruct | `20261011_064822_grpo_xsum_llama-3.1-8b-instruct_seed0` |
   | Mistral-7B-Instruct-v0.3 | `20261012_185905_grpo_xsum_mistral-7b-instruct-v0.3_seed0` |
   | Qwen2.5-7B-Instruct | `20261009_214506_grpo_xsum_qwen2.5-7b-instruct_seed0` |

   `checkpoint-625` and `final` hold byte-identical `adapter_model.safetensors` for all three
   (checked 2026-10-17). Every T5 run records the adapter directory and the SHA-256 of its weights
   in `manifest.json` → `lora_adapter`.
3. **Serving.** vLLM starts with `--enable-lora --max-lora-rank 16 --lora-modules
   <backbone>-lora=<adapter directory>` on the same 4-bit base it serves for T0 to T2, and every
   request addresses `<backbone>-lora`. Without an adapter entry the serve command and requests are
   unchanged. `topology.requires_adapter: true` makes the runner refuse a T5 cell whose backbone has
   no adapter, or whose adapter directory has no weights, before any model call, so T5 can never
   run as the base model and be reported as T5.
4. **Sampling and seeds** as T0: each backbone's own defaults (D-011), a 256-token cap per call
   (D-012), seeds 0, 1 and 2. The seeds vary inference only; each adapter comes from one training
   run (D-060).

| Parameter | Value | Reason | Source |
| --- | --- | --- | --- |
| Adapter checkpoint | `checkpoint-625` | T5 is the final checkpoint | D-060 |
| Prompt | `xsum_one_sentence@v1`, no preface | the prompt the adapters were trained on | `rl/trainer.py` |
| Sampling | backbone defaults, `max_tokens` 256 | identical to T0, so T5 − T0 isolates the training | D-011, D-012 |
| Seeds | 0, 1, 2 | ≥ 3 seeds for reported results | AGENTS.md §3 |
| `serving.max_lora_rank` | 16 | the adapters' rank | D-054 |

**Scope:** these are the three 7–8B fine-tunes. The instructor's 10-06 recommendation of a small
language model for T5 (Reviews #4 and #5) is put to the instructor on the Review #5 issue, with
the case for keeping the same three backbones; the instructor's ruling is pending.

**Not yet verified:** vLLM serving a LoRA adapter on these bitsandbytes 4-bit backbones has not
been run on the instance; the first T5 run is that check.

**Source:** D-011, D-012, D-016, D-054, D-060, D-068; `rl/trainer.py` (`grpo-training-pool`);
each adapter's `adapter_config.json`; Instructor Reviews #4 and #5
**Approved:** Sameer, 2026-10-19
**Config:** `src/configs/topology/t5_rl_agent.yaml` (`topology.requires_adapter`,
`serving.max_lora_rank`, `serving.lora_adapters`); `config/serving.py`; `config/topology.py`
**Supersedes:** —

---

### D-112 · 2026-10-20 · T5 serving reserves 0.85 of GPU memory; seven T5 runs that ran out of memory at startup are re-run

**Finding:** of the nine T5 XSum runs started on 2026-10-20 from `9bf9a08` (D-111), two succeeded
and seven failed before generating anything, each with "vLLM exited with code 1 before becoming
ready". The server logs show a CUDA out-of-memory error during CUDA-graph capture with the LoRA
adapter loaded ("this process has 21.95 GiB memory in use", 22.06 GiB total). In the successful
Llama seed-0 run, vLLM reserved 13.28 GiB of KV cache under its default memory share (0.92, 20.3
GiB) and graph capture then took 2.01 GiB more; in the failed Llama seed-1 run the KV-cache
reservation came out at 13.94 GiB, and capture no longer fitted.

| Run ID | Status |
| --- | --- |
| `20261020_044314_t5_rl_agent_tier1_xsum_llama-3.1-8b-instruct_seed0` | success |
| `20261020_045933_t5_rl_agent_tier1_xsum_llama-3.1-8b-instruct_seed1` | failed (out of memory at startup) |
| `20261020_050124_t5_rl_agent_tier1_xsum_llama-3.1-8b-instruct_seed2` | failed (out of memory at startup) |
| `20261020_050303_t5_rl_agent_tier1_xsum_mistral-7b-instruct-v0.3_seed0` | failed (out of memory at startup) |
| `20261020_050703_t5_rl_agent_tier1_xsum_mistral-7b-instruct-v0.3_seed1` | failed (out of memory at startup) |
| `20261020_050837_t5_rl_agent_tier1_xsum_mistral-7b-instruct-v0.3_seed2` | failed (out of memory at startup) |
| `20261020_051011_t5_rl_agent_tier1_xsum_qwen2.5-7b-instruct_seed0` | success |
| `20261020_052602_t5_rl_agent_tier1_xsum_qwen2.5-7b-instruct_seed1` | failed (out of memory at startup) |
| `20261020_052917_t5_rl_agent_tier1_xsum_qwen2.5-7b-instruct_seed2` | failed (out of memory at startup) |

**Decision:**
1. New field `serving.gpu_memory_utilization` (default None: the flag is not passed, so T0 to T2
   serve exactly as before). `src/configs/topology/t5_rl_agent.yaml` sets 0.85, an 18.75 GiB budget
   that leaves about 3.3 GiB for graph capture.
2. The seven failed runs are re-run with this setting from a clean commit. The two successful runs
   stand. The setting changes only how much memory vLLM reserves for its KV cache; requests are sent
   one at a time, so the generated text does not depend on it. The two settings within the T5
   column are recorded here and in each run's `config.resolved.yaml`.
3. The seven failed run folders stay in `results/` unchanged, recorded above.

| Parameter | Value | Reason | Source |
| --- | --- | --- | --- |
| `serving.gpu_memory_utilization` (T5 only) | 0.85 | leaves room for LoRA graph capture (2.01 GiB measured) above the KV cache | the runs' vLLM logs |

**Source:** each run's `summary.json` and its vLLM server log (`/tmp/factagent_vllm_<run_id>.log`)
**Approved:** Sameer, 2026-10-20
**Config:** `src/configs/topology/t5_rl_agent.yaml` (`serving.gpu_memory_utilization`);
`config/serving.py`; `serving/launch.py`
**Supersedes:** the seven failed runs above, as T5 results

---

### D-114 · 2026-10-20 · T5 XSum results: nine runs; the seven D-112 failures stay on record

**Decision:** the nine runs below are T5's XSum results under D-111 (3 backbones × seeds 0–2,
each backbone served with its GRPO adapter `checkpoint-625`). Each finished with status `success`,
500 rows and `git.dirty: false`. Each run's `manifest.json` records the adapter path and sha256.

| Run ID | Backbone | Seed | Commit | `gpu_memory_utilization` |
| --- | --- | --- | --- | --- |
| `20261020_044314_t5_rl_agent_tier1_xsum_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B-Instruct | 0 | `9bf9a08` | default (0.92) |
| `20261020_112244_t5_rl_agent_tier1_xsum_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B-Instruct | 1 | `53d1314` | 0.85 |
| `20261020_113822_t5_rl_agent_tier1_xsum_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B-Instruct | 2 | `53d1314` | 0.85 |
| `20261020_115056_t5_rl_agent_tier1_xsum_mistral-7b-instruct-v0.3_seed0` | Mistral-7B-Instruct-v0.3 | 0 | `53d1314` | 0.85 |
| `20261020_121424_t5_rl_agent_tier1_xsum_mistral-7b-instruct-v0.3_seed1` | Mistral-7B-Instruct-v0.3 | 1 | `53d1314` | 0.85 |
| `20261020_123542_t5_rl_agent_tier1_xsum_mistral-7b-instruct-v0.3_seed2` | Mistral-7B-Instruct-v0.3 | 2 | `53d1314` | 0.85 |
| `20261020_051011_t5_rl_agent_tier1_xsum_qwen2.5-7b-instruct_seed0` | Qwen2.5-7B-Instruct | 0 | `9bf9a08` | default (0.92) |
| `20261020_125719_t5_rl_agent_tier1_xsum_qwen2.5-7b-instruct_seed1` | Qwen2.5-7B-Instruct | 1 | `53d1314` | 0.85 |
| `20261020_131141_t5_rl_agent_tier1_xsum_qwen2.5-7b-instruct_seed2` | Qwen2.5-7B-Instruct | 2 | `53d1314` | 0.85 |

`9bf9a08` and `53d1314` differ only by D-112's memory setting, which changes how much memory vLLM
reserves and not the generated text (D-112, decision 2). The seven failed runs listed in D-112 are
committed with the successful ones and are not T5 results.

**Reason:** D-112 called for re-running the seven runs that ran out of memory at startup; with those
re-runs every backbone has three seeds.
### D-113 · 2026-10-20 · T1 XSum results after the D-110 prompt fix: nine runs, no worked-example overlap

**Decision:** the nine runs below are T1's XSum results under D-110 (3 backbones × seeds 0–2).
Each finished with status `success`, 500 rows and `git.dirty: false`, all from the same commit.

| Run ID | Backbone | Seed | Commit |
| --- | --- | --- | --- |
| `20261019_222000_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B-Instruct | 0 | `6c1892a` |
| `20261019_232219_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B-Instruct | 1 | `6c1892a` |
| `20261020_001623_t1_self_refine_tier1_xsum_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B-Instruct | 2 | `6c1892a` |
| `20261020_011129_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed0` | Mistral-7B-Instruct-v0.3 | 0 | `6c1892a` |
| `20261020_020741_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed1` | Mistral-7B-Instruct-v0.3 | 1 | `6c1892a` |
| `20261020_030140_t1_self_refine_tier1_xsum_mistral-7b-instruct-v0.3_seed2` | Mistral-7B-Instruct-v0.3 | 2 | `6c1892a` |
| `20261020_035418_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed0` | Qwen2.5-7B-Instruct | 0 | `6c1892a` |
| `20261020_041142_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed1` | Qwen2.5-7B-Instruct | 1 | `6c1892a` |
| `20261020_042733_t1_self_refine_tier1_xsum_qwen2.5-7b-instruct_seed2` | Qwen2.5-7B-Instruct | 2 | `6c1892a` |

`worked_example_overlap` (D-110, 8-gram) is 0.0 in every run's `summary.json`, so no final summary
copies a worked example. No run failed.

**Reason:** D-109 withdrew the D-108 runs as Self-Refine results and called for a re-run on all three
backbones with the revised prompt; these are those runs.

**Source:** each run's `summary.json` and `manifest.json`
**Approved:** Sameer, 2026-10-20
**Config:** none changed
**Supersedes:** the D-108 runs, as T1 results (D-109)

---

### D-115 · 2026-10-22 · Headline metric and aggregation: MiniCheck, paired per-article gain over T0, 95% bootstrap CI

**Decision:**
Every topology is compared with T0 on the same backbone and dataset, and the reported number is
computed as follows.

1. **Headline metric: MiniCheck** (D-024). AlignScore (D-023) is reported next to it as the second
   primary metric for T0–T4 (D-045); for T5 it is labelled as the training reward (D-048).
   SummaC-ZS, SummaC-Conv and FactCC stay secondary (D-045). ROUGE-L and BERTScore are NLG-quality
   checks and are not read as factuality.
2. **Unit: the per-article paired difference.** For article *i* of the frozen 500-article sample
   (D-002), *d_i* = (mean of the topology's scores on *i* over seeds 0–2) − (mean of T0's scores on
   *i* over seeds 0–2), on the same backbone. Seeds are averaged within each article before the
   difference is taken, so seed indices are not matched across topologies.
3. **Point estimate:** the mean of *d_i* over the 500 articles.
4. **Uncertainty:** a 95% percentile bootstrap confidence interval. The 500 values of *d_i* are
   resampled with replacement, the mean is recomputed for each resample, and the 2.5th and 97.5th
   percentiles of those means are the interval. A gain is reported as present only when the interval
   excludes 0.
5. **Cost:** every gain is reported with the mean total tokens per article (all calls summed,
   D-100) for the topology and for T0, and their ratio. Wall-clock time is reported alongside
   (AGENTS.md §5).
6. **Main comparisons are each topology against T0.** Pairwise topology-against-topology
   comparisons go in the paper's appendix.
7. **Across datasets, gains are compared, not raw scores.** Each dataset's gain is taken over that
   dataset's own T0.

| Parameter | Value | Reason | Source |
| --- | --- | --- | --- |
| `analysis.bootstrap.n_resamples` | 10,000 | deliberate default (§12.1, reason 4): more resamples only reduce the Monte Carlo noise in the interval ends; the trade-off is computation time only | approved by Sameer, 2026-10-19 |
| `analysis.bootstrap.seed` | 123 | the project's existing seed for the frozen evaluation sample, reused so no new arbitrary value is introduced | D-002; `src/configs/data/tier1_xsum_sample.yaml` |
| `analysis.bootstrap.ci_level` | 0.95 | deliberate default (§12.1, reason 4): the conventional level; a wider level would make fewer gains count as present | approved by Sameer, 2026-10-16 |

**Reason:**
- **One headline metric, held out from training.** AlignScore is T5's training reward (D-048), so it
  cannot judge T5 without circularity. MiniCheck is the held-out judge (D-048) and is validated on
  XSum at 74.3 balanced accuracy, matching the published value (D-032, D-045). Using it for every
  topology puts T0–T5 on one scale and keeps the confirmatory set small for the
  multiple-comparisons correction (proposal, Week 13; D-045).
- **Paired per-article differences.** Comparing the two topologies on the same article removes the
  variation in difficulty between articles from the comparison. Generation is sampled at
  temperature 0.7 (D-007), so a single article's score moves between seeds; averaging the three
  seeds per article and then the 500 articles keeps one seed's draw from deciding a result.
- **Bootstrap over articles.** The proposal (Week 13) and AGENTS.md §3 require bootstrapped
  confidence intervals on every factuality delta. Articles are the resampling unit because the 500
  articles are the sample drawn from the dataset (D-002); the seeds are not a sample of anything
  beyond the run.
- Instructor Review #4 (2026-10-11) asked for this unit to be declared: the per-example paired
  difference against T0, averaged over seeds, with a bootstrap CI over the 500 items.

**Scope:** XSum. Primary metrics for CNN/DailyMail and SAMSum are not decided (D-045); before this
rule is applied to SAMSum, MiniCheck's validation coverage for dialogue is to be checked.

**Not decided here:**
- the multiple-comparisons correction (the proposal, Week 13, names Holm-Bonferroni or
  Benjamini-Hochberg FDR);
- an optional thresholded rate ("share of summaries judged faithful"); if adopted, its threshold is
  fixed on human-labelled data, never on our outputs.

**Source:** proposal, Week 13; AGENTS.md §3 and §5; D-002, D-007, D-024, D-032, D-045, D-048, D-100;
Instructor Review #4 (2026-10-11)
**Approved:** Sameer, 2026-10-16 (metric and method) and 2026-10-19 (bootstrap values)
**Config:** `src/configs/analysis/aggregation.yaml` (to be created with the analysis code)
**Supersedes:** — (narrows D-045 by naming MiniCheck as the single headline metric)

---

### D-116 · 2026-10-23 · Tier 2 data: ALCE's ASQA file at a pinned revision; ASQA before ELI5

**Decision:**
1. **Source.** Tier 2 ASQA is `asqa_eval_gtr_top100.json` from the ALCE release
   (`princeton-nlp/ALCE-data` on the HuggingFace Hub), revision
   `334fa2e7dd32040c3fef931a123c4be1a81e91a0`, file `ALCE-data.tar`, SHA-256
   `eda837bf659a91b3648dc6e7ab6b17197664d93593857e8fdf3800b6aa6a98f0` (the value the Hub publishes for
   that file). The loader refuses a file whose checksum differs. The file lives in the local Hub
   cache and is never committed.
2. **Split and IDs.** The file holds ASQA's development split: 948 questions. Each item is
   identified by its `sample_id`. Checked against the official ASQA release
   (`gs://gresearch/ASQA/data/ASQA.json`): all 948 `sample_id`s are the official dev IDs and all 948
   question texts are identical.
3. **Passages.** Each item carries ALCE's top-100 GTR retrieval results from the 2018-12-20
   Wikipedia snapshot (ALCE §2, §4.1). We run no retriever: every topology, backbone and seed reads
   the same passages from the pinned file.
4. **Order.** ASQA first. ELI5 follows, also from ALCE (`eli5_eval_bm25_top100.json`), by its own
   entry.

**Reason:**
- The proposal (§2) names ALCE (Gao et al. 2023) as the Tier 2 evaluation framework and rules out
  live retrieval. ALCE ships retrieval results with the data, which satisfies both.
- Self-RAG (Asai et al. 2024, §4.3), Huang et al. 2024 (App. B.2) and Trust-Align (Song et al. 2025,
  §5) evaluate on these same ALCE files, so our inputs match theirs.
- ALCE evaluates on the development set (§2). ASQA's training split stays available for T5 (D-047)
  and does not overlap the evaluation questions.
- ASQA before ELI5: ASQA's correctness metric is string matching and needs no model (D-120), and
  T5's ASQA track (D-047) needs it. ELI5 is the weak-grounding contrast the proposal describes:
  ALCE's top-5 passages contain 56.8% of ASQA's gold short answers but cover 9.6% of ELI5's gold
  claims (ALCE Tables 12 and 14). The proposal's ELI5 link (`eli5_category`) has no passages, so
  ELI5 needs ALCE's version.

**Source:** proposal §2 (Tier 2 table, "Dataset / Pipeline Preparation"); Gao et al. 2023 (ALCE,
EMNLP), §2, §4.1, Tables 12 and 14; Asai et al. 2024 (Self-RAG, ICLR), §4.3; Huang et al. 2024
(arXiv 2402.04315), App. B.2; Song et al. 2025 (Trust-Align, ICLR), §5; Stelmakh et al. 2022
(ASQA); D-047; checksum and ID checks run 2026-10-23
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/tier2_asqa.yaml` (to be created)
**Supersedes:** — (replaces the proposal's ELI5 source `eli5_category`; no earlier D-entry)

---

### D-117 · 2026-10-23 · ASQA input: the top 5 of ALCE's retrieved passages

**Decision:**
The model reads the first 5 of ALCE's top-100 passages for each question, in ALCE's order. Each
passage is formatted as in ALCE's prompt file: `Document [i](Title: <title>): <text>`, numbered from
1. `Example.evidence` holds exactly these 5 passages, so every factuality check runs against the
text the model saw. Measured on the 948 dev questions: every passage text is 100 words; with titles,
the 5 passages total median 516 and maximum 557 words per question.

**Reason:**
- ALCE's VANILLA setting uses 5 passages for open models with a 4K context ("at most k = 5 for
  models with 4K context window", §4.1, footnote 9), and its Llama-2 results use 5 (Table 4).
- Self-RAG uses "the author-provided top 5 documents by GTR-XXL across all baselines for a fair
  comparison" (§4.3). Huang et al. (App. B.2) and Trust-Align (§5) also use the top 5.
- Using the same number keeps our results comparable with these four papers.

**Known consequence:** the top 5 GTR passages contain 56.8% of ASQA's gold short answers; the top 20
contain 70.3% (ALCE Table 12). Exact-match recall (D-120) therefore has a ceiling below 100% for an
answer grounded only in the passages. Our 32,768-token serving context (D-039) could hold more
passages; that is not adopted, to keep comparability.

**Source:** Gao et al. 2023, §4.1 footnote 9, Table 4, Table 12; Asai et al. 2024, §4.3; Huang et
al. 2024, App. B.2; Song et al. 2025, §5; passage lengths measured 2026-10-23
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/tier2_asqa.yaml` (to be created)
**Supersedes:** —

---

### D-118 · 2026-10-23 · ASQA prompt: ALCE's instruction and two of its four worked examples, drawn per seed, sent as chat turns

**Decision:**
1. **Wording.** ALCE's `prompts/asqa_default.json` (ALCE repository, commit `246c476`, MIT
   licence): its instruction verbatim and its passage format (D-117). The prompt file's four worked
   examples are copied verbatim into the Generator module with that commit recorded.
2. **Two examples per run, drawn per seed.** Each run draws 2 of the 4 examples, without
   replacement, from the run's seed, and uses the same 2 for every question in that run, as ALCE's
   `run.py` does. The two drawn examples are recorded in the run's manifest.
3. **Layout (D-110).** Each example is one user turn (instruction, question, its 5 passages)
   answered by one assistant turn (its cited answer). The final user turn holds the question under
   work in the same form. ALCE's open `Answer:` label is dropped because the assistant turn is the
   answer (D-110 item 4). No system message (D-016).
4. **Leak check.** The worked-example overlap metric (D-110) is scored on every ASQA run against the
   drawn examples.
5. **Every topology uses this prompt for its generation step,** as D-016 does on XSum. T0 on ASQA
   is therefore few-shot, while T0 on XSum, CNN/DailyMail and SAMSum is zero-shot.

**Reason:**
- *Why worked examples at all.* ALCE Table 16 (App. G.3) reports ChatGPT on ASQA with 0, 1 and 2
  examples: correctness stays level (41.9, 39.8, 40.4) while citation recall rises from 69.3 with
  none to 74.6 with one; "using at least one demonstration ensures high citation recall". For the
  original LLaMA, "more demonstrations lead to better performance". Huang et al. 2024 (§2.1) found
  that "even with in-context learning, a base LLaMA-2-7B can only generate citations for a very
  small portion of sentences" (citation recall 14.12 on ASQA, Table 1). Without examples, low
  citation scores would measure the output format rather than faithfulness.
- *Why two.* Two is the standard setting: ALCE (§4.2), Huang et al. 2024 (§3.4) and Trust-Align
  (§5), whose models include LLaMA-3-8B and Qwen-2.5-7B, two of our backbone families. Self-RAG
  evaluates zero-shot (§4.1); we follow the majority setting.
- *Why ALCE's examples.* They were written and published by ALCE's authors, so no example was chosen
  by us or tuned on our outputs. All four are ASQA training questions, none in the dev split
  (checked against the official release), so no evaluation question appears as an example.
- *Why re-drawn per seed.* ALCE "run[s] each model three times with different seeds and each time
  we sample two demonstrations from a pool of four" (App. C). Following it keeps our protocol the
  same as the benchmark we compare with, and our three seeds (AGENTS.md §3) map onto ALCE's three
  runs. Consequence: seed-to-seed spread on ASQA includes which examples were shown, as in ALCE, and
  not sampling alone.
- *Why chat turns and the leak check.* D-109 found that 7–8B backbones copy worked examples into
  their output; D-110's layout and overlap metric are the project's fix, applied here before the
  first run.

**Source:** Gao et al. 2023, §4.2, App. C, App. G.3 (Table 16); ALCE repository commit `246c476`
(`prompts/asqa_default.json`, `run.py`); Huang et al. 2024, §2.1, §3.4, Table 1; Song et al. 2025,
§5; Asai et al. 2024, §4.1; D-016; D-109; D-110
**Approved:** Sameer, 2026-10-23
**Config:** Generator module (ASQA template and worked examples, to be created);
`src/configs/data/tier2_asqa.yaml` (to be created)
**Supersedes:** —

---

### D-119 · 2026-10-23 · ASQA completion cap: 300 tokens

**Decision:**
Generation requests on ASQA set `max_tokens` to **300**. Tier 1 keeps 256 (D-012). As D-012
requires, each run reports the share of outputs stopped by the cap, and ASQA runs also report the
mean output length in tokens next to exact-match recall (D-120).

**Reason:**
- *Published value.* ALCE generates with `max_new_tokens` 300 (the `run.py` default, used by its
  configurations). Self-RAG describes it as "the original 300 tokens in the ALCE paper"
  (App. B.2).
- *Measured on ASQA.* The 1,896 human long answers in the dev split (two per question) run median
  90 / 98 / 85 tokens and 99th percentile 217 / 239 / 205 tokens under the Qwen / Mistral / Llama
  tokenizers. 3 / 5 / 2 of them (at most 0.26%) exceed 300 tokens. ALCE reports an average of 65
  words (App. B); we measure a median of 62.
- *Why not higher.* On ASQA, length is not neutral: exact-match recall rises with output length.
  Self-RAG found Llama2-chat's longer outputs "resulting in inflated str-em scores" and cut its
  baselines' cap to 100 tokens (App. B.2). Huang et al. 2024 used 200 "to control the conciseness of
  answers" (App. C). A higher cap only gives verbose outputs more room to raise exact-match recall.
  300 keeps ALCE's setting, and reporting length makes any length effect visible.

**Source:** ALCE repository commit `246c476` (`run.py`, `--max_new_tokens`); Gao et al. 2023,
App. B; Asai et al. 2024, App. B.2; Huang et al. 2024, App. C; ASQA dev answer lengths measured
2026-10-23 with each backbone's tokenizer; D-012
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/tier2_asqa.yaml` (to be created; overrides `sampling.max_tokens`)
**Supersedes:** — (D-012 stands for Tier 1)

---

### D-120 · 2026-10-23 · ASQA metrics: MiniCheck headline, exact-match recall for correctness, MiniCheck-judged citations as secondary

**Decision:**
1. **Headline: MiniCheck** (D-024). Each output sentence, with citation markers removed, is checked
   against the 5 passages the model saw (D-117). D-115's aggregation applies unchanged: paired
   per-question gain over T0, 95% bootstrap CI.
2. **Correctness: exact-match recall** (ALCE's `str_em`): for each question, the share of its
   sub-questions for which at least one gold short answer appears in the output. Output and answers
   are normalized as in SQuAD (lowercase; punctuation, articles and extra whitespace removed), and
   citation markers are removed first (ALCE `eval.py`, `utils.py`). Reported next to every headline
   number, with mean output length (D-119).
3. **Second primary: AlignScore** (D-023), against the same 5 passages, as on XSum (D-045).
4. **Secondary: citation recall and citation precision,** computed as ALCE defines them (§3.3; at
   most 3 citations per sentence), with MiniCheck as the entailment judge in place of ALCE's TRUE
   model.

**Reason:**
- *Two separate questions.* Exact-match recall asks whether the answer contains the right facts.
  MiniCheck asks whether every sentence is supported by the passages given. ALCE separates the same
  two dimensions, correctness and citation quality (§3). A model can answer correctly from
  parametric knowledge without support in the passages, so the two can disagree, and reporting both
  shows it.
- *MiniCheck is validated on this setting.* LLM-AggreFact (Tang et al. 2024, Table 2) includes
  three "retrieve-then-generate" datasets, model answers checked against retrieved documents.
  Balanced accuracy, MiniCheck-FT5 vs T5-NLI-Mixed (TRUE, the T5-11B NLI model ALCE uses):
  LFQA 85.2 vs 61.8, ClaimVerify 74.6 vs 59.5, ExpertQA 59.0 vs 55.6.
- *One headline judge across tiers.* The same metric as XSum (D-115) keeps the confirmatory set
  small and the tiers on one scale.
- *Why MiniCheck judges citations.* ALCE's judge is TRUE, a T5-11B model (§3.2, App. C), which
  scores below MiniCheck-FT5 on all three datasets above. Huang et al. 2024 also replaced TRUE with a
  smaller NLI model, during training, for efficiency (App. C). Consequence: our citation scores are
  not directly comparable with ALCE's published citation numbers, which is why they are secondary.
- *Exact-match recall is the standard ASQA correctness metric* in ALCE (§3.2), Self-RAG (§4.1) and
  Huang et al. 2024 (§3.2). ASQA's own QA-model metric (Disambig-F1; Stelmakh et al. 2022) is not
  used: ALCE found it "not as stable" (App. C).

**Known limitations:**
- ExpertQA is close to chance for every checker in Tang et al. Table 2, GPT-4 included (59.2).
- The top 5 passages contain 56.8% of gold short answers (D-117), which caps exact-match recall for
  fully grounded answers.
- Exact-match recall rises with output length (D-119).

**Not decided here:** MAUVE; whether the other XSum metrics (FactCC, SummaC, ROUGE-L, BERTScore) are
run on ASQA; ALCE's truncation of outputs at the first newline before evaluation.

**Source:** Gao et al. 2023, §3, §3.2, §3.3, App. C; ALCE repository commit `246c476` (`eval.py`,
`utils.py`); Tang et al. 2024 (MiniCheck, arXiv 2404.10774), Table 2; Asai et al. 2024, §4.1;
Huang et al. 2024, §3.2, App. C; Stelmakh et al. 2022; D-023; D-024; D-045; D-115
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/tier2_asqa.yaml` (metric suite, to be created)
**Supersedes:** — (extends D-115's scope from XSum to ASQA)

---

### D-121 · 2026-10-23 · ASQA decoding: the project's own default sampling settings, not ALCE's

**Decision:**
ASQA generation uses the same sampling settings as every other run in the project: each
backbone's own shipped defaults (D-011), recorded in `model_registry.csv`. At the pinned revisions
these are temperature 0.7 and top-p 0.8 for Qwen2.5-7B-Instruct, 1.0 and 1.0 for
Mistral-7B-Instruct-v0.3, and 0.6 and 0.9 for Llama-3.1-8B-Instruct. Requests send only the seed and
`max_tokens` (D-119). ALCE's decoding settings are not adopted.

**Reason:**
- One decoding rule for every dataset and topology keeps decoding constant in every topology
  comparison on a backbone (D-011).
- ALCE has no single setting to adopt. Its decoding differs by model "based on preliminary
  experiments": temperature 0.5 for ChatGPT and GPT-4, nucleus sampling with top-p 0.95 for
  open-source models (App. C), with temperature 1.0 in its Llama-2 configuration. Taking it would
  make ASQA the only dataset whose decoding is chosen by the project rather than by each model's
  vendor.

**Consequence:** absolute ASQA scores are not directly comparable with ALCE's published open-model
results. Paired gains over T0 (D-115, D-120) compare runs with identical decoding and are
unaffected.

**Source:** D-011; `src/model_registry.csv`; Gao et al. 2023, App. C; ALCE repository commit
`246c476` (`configs/asqa_llama2_shot2_ndoc5_gtr_default.yaml`)
**Approved:** Sameer, 2026-10-23
**Config:** `src/model_registry.csv`; no sampling fields in `src/configs/`, by design (D-011)
**Supersedes:** —

---

### D-122 · 2026-10-23 · ASQA evaluation sample: 500 dev questions in three strata by number of sub-questions, frozen as IDs

**Decision:**
1. **Pool:** the 948 dev questions of D-116.
2. **Strata by number of sub-questions** (the length of each item's `qa_pairs`). The dev split has
   2 to 6 sub-questions per question, none with 1. Three groups:

   | Group | Sub-questions | Questions available | Drawn |
   | --- | --- | --- | --- |
   | Low ambiguity | 2 | 379 | 167 |
   | Medium | 3–4 | 373 (170 + 203) | 167 |
   | High | 5–6 | 196 (72 + 124) | 166 |

   Equal draws follow D-002; the freeze tool gives the remainder to the first groups.
3. **Seed 123,** the project's sample seed (D-002, `tier1_xsum_sample.yaml`).
4. **Frozen as an explicit, ordered list of `sample_id`s** in `tier2_asqa_sample.yaml`, drawn once
   and never re-drawn (D-006).

**Reason:**
- *Why not length.* D-002 stratifies by length so that "does critique help more on long inputs?"
  can be answered. On ASQA, length barely varies: questions run median 8 and maximum 22 words, and
  every passage is 100 words (D-117).
- *Why sub-questions.* Ambiguity is what defines ASQA: each sub-question is one interpretation of
  the ambiguous question (Stelmakh et al. 2022), and the number of sub-questions sets how many facts
  a complete answer must cover. It is the denominator of exact-match recall (D-120). Stratifying on
  it makes "does critique help more on more ambiguous questions?" answerable, as length does for
  Tier 1.
- *Why three groups.* Equal draws need at least 167 questions in each group. The 5-sub-question
  level has 72, so one group per level is impossible. Of the three-group splits into consecutive
  levels that leave at least 167 in every group, this one has the most even pools (196 to 379,
  against 196 to 549 for {2–3}/{4}/{5–6} and 170 to 399 for {2}/{3}/{4–6}). Three groups also match
  D-002's terciles.

**Consequence:** high-ambiguity questions are over-represented relative to the dev split (33% of
the sample against 21% of the pool), as long articles are on Tier 1 (D-002). Sample means are
stratified, not weighted to the dev split's distribution.

**Source:** D-002; D-006; D-116; Stelmakh et al. 2022; sub-question counts measured on the ALCE file
2026-10-23; `datasets/base.py` (`draw_sample_ids`)
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/tier2_asqa.yaml` and `src/configs/data/tier2_asqa_sample.yaml` (to be
created)
**Supersedes:** D-002's length stratification, for ASQA only

---

### D-123 · 2026-10-23 · ASQA secondary metrics: the full XSum metric suite is run and reported

**Decision:**
Every ASQA run computes and reports the full metric suite used on XSum (D-022), as secondary
metrics next to the primary ones of D-120:
- **SummaC-ZS, SummaC-Conv and FactCC** score the output, with citation markers removed, against
  the 5 passages the model saw (D-117).
- **ROUGE-L** (D-026) and **BERTScore** (D-019) score the output against ASQA's human long answers.
  Each dev question has two; the output is scored against both and the higher score is kept, as
  ALCE does for ROUGE (`eval.py`, `compute_rouge`).

**Reason:** every dataset reports the same metric columns, so no metric is shown on one tier and
hidden on another. The ports and their validation (D-032) are unchanged.

**Known limitations, stated with the numbers:**
- SummaC-ZS, SummaC-Conv and FactCC were validated on XSum (D-032), not on question answering.
- Our ROUGE-L is F1 without stemming (D-026); ALCE uses ROUGE-Lsum with stemming. ASQA ROUGE values
  are therefore not directly comparable with published ASQA ROUGE.

**Not used: MAUVE.** ALCE reports MAUVE for fluency but "mainly employ[s] it as a sanity check",
noting that it "is sensitive to output length and text style" (§3.1). Huang et al. 2024 leave it
out of their main analysis "since it is sensitive to output length and unstable" (App. B.3). It
would also add a new dependency (AGENTS.md §9) for a sanity check, while ROUGE-L and BERTScore
already compare outputs with the human answers.

ALCE's truncation of outputs at the first newline is decided in D-125.

**Source:** D-019; D-022; D-026; D-032; D-117; D-120; ALCE repository commit `246c476`
(`eval.py`, `compute_rouge`); Gao et al. 2023, §3.1; Huang et al. 2024, App. B.3
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/tier2_asqa.yaml` (metric suite, to be created)
**Supersedes:** — (settles D-120's open items on the other XSum metrics and MAUVE)

---

### D-124 · 2026-10-23 · Before the first ASQA run, the MiniCheck and AlignScore ports are validated on LLM-AggreFact's long-form QA data

**Decision:**
Before any ASQA run is made, the MiniCheck-FT5 port (D-024) and the AlignScore-large port (D-023)
are scored on two test subsets of LLM-AggreFact, with the same tool and pinned file as D-032
(`src/validate_metrics.py`; `lytang/LLM-AggreFact` at revision `981dfd0`, SHA-256 in
`src/configs/base.yaml`). The statistic is balanced accuracy at threshold 0.5, the setting of Tang
et al.'s Table 2 ("the midpoint of the output score range ... 0.5 for most fact-checkers", §4.1).

| Subset | Test rows (pinned file) | Published MiniCheck-FT5 | Published AlignScore |
| --- | --- | --- | --- |
| LFQA | 1,911 | 85.2 | 84.5 |
| ClaimVerify | 1,088 | 74.6 | 69.6 |

Published values are from Tang et al. 2024, Table 2. The results are recorded in their own entry
before ASQA runs begin.

**Reason:** D-120 makes MiniCheck the ASQA headline and AlignScore the second primary metric,
because Tang et al. evaluate both on retrieve-then-generate data. D-032 checked our ports only on
XSum data (MiniCheck on AggreFact-XSum, 74.3, an exact match; AlignScore on XSumFaith by ROC AUC).
This check shows that our ports, not only the published models, hold on answers checked against
retrieved documents, the setting ASQA puts them in.

**Note:** the pinned file is the dataset release used for D-032, which reproduced Tang et al.'s
AggreFact-XSum value exactly. A difference on LFQA or ClaimVerify is checked first for a change in
the subset between releases.

**Source:** D-023; D-024; D-032; D-120; Tang et al. 2024, §4.1 and Table 2; row counts read from
the pinned file 2026-10-23
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/base.yaml` (`metrics.validation`, the subset name)
**Supersedes:** —

---

### D-125 · 2026-10-23 · ASQA outputs are scored in full, not cut at the first line break; the share with a line break is reported

**Decision:**
Every ASQA metric (D-120, D-123) scores the whole output. ALCE's evaluation step that keeps only
the text before the first line break (`eval.py`, `main`: `output.strip().split("\n")[0]`) is not
applied. Each ASQA run reports, next to the length-cap share (D-012), the share of outputs that
contain a line break.

**Reason:**
- ALCE gives its reason for the cut: "non-instruction-tuned models may generate more content after
  new lines that are irrelevant" (App. C). Our three backbones are instruction-tuned chat models
  whose answers can run over more than one paragraph.
- Cutting would discard answer text a reader sees and lower exact-match recall and MiniCheck for a
  formatting reason rather than a factual one. Tier 1 outputs are not cut either.
- None of ALCE's four worked examples (D-118) contains a line break, so mostly single-paragraph
  answers are expected; this is an expectation, not yet measured.

**Consequence:** for outputs with a line break, our scores differ from what ALCE's evaluation would
give. The reported share shows how many outputs that affects.

**Source:** Gao et al. 2023, App. C; ALCE repository commit `246c476` (`eval.py`,
`prompts/asqa_default.json`); D-012; D-118; D-120; D-123
**Approved:** Sameer, 2026-10-23
**Config:** none (scoring rule); the share is written to each run's `summary.json`
**Supersedes:** —

---

### D-126 · 2026-10-23 · Shared contract: `Example.gold` for annotations one reference cannot hold; citation markers removed before scoring; two new run statistics

**Decision:**
1. **`Example.gold`.** `Example` and the common dataset schema gain one optional field, `gold`: a
   JSON object as a string, or None. It holds gold annotations that a single `reference` string
   cannot: for ASQA, the short answers of every sub-question (D-120) and both human answers
   (D-123). Every Tier 1 dataset leaves it None.
2. **Metrics receive it.** `Metric.score()` and `Metric.compute()` take an optional `gold`,
   aligned with the predictions. A metric that cannot score without it declares
   `NEEDS_GOLD = True` and refuses any example that lacks it. Every existing metric ignores it, so
   its scores are unchanged.
3. **Citation markers.** A new data setting, `data.strip_citations` (default False). When True,
   the runner removes markers with ALCE's `remove_citations`, ported unchanged, before every
   metric except those that declare `READS_CITATIONS = True` (the citation metrics of D-120). An
   output with no text left after removal is recorded as a blank output and not scored (D-030).
4. **Two run statistics,** written to every run's `summary.json`: `newline_rate`, the share of
   outputs whose text, stripped of surrounding whitespace, contains a line break (D-125); and
   `mean_output_tokens`, the mean completion tokens of the call whose text is the output, taken
   from the trace (D-119).

**Reason:**
- *A new field rather than the alternatives.* Packing the annotations into `reference` as JSON
  would make the existing ROUGE-L and BERTScore metrics score against JSON text without any error.
  Letting ASQA's metrics re-read the ALCE file is not possible, because metrics never see example
  IDs and so cannot match rows.
- *Optional, defaulting to None,* so every Tier 1 dataset, metric and earlier run is unchanged.
  Tier 3 datasets with structured labels (RAGTruth's spans) are expected to use the same field.
- *Markers removed before scoring,* as ALCE does before exact match and ROUGE (`eval.py`, `main`).
  A marker such as `[1]` is not part of the answer's claims, and a checker would otherwise read it
  as text.
- *Output tokens read from the trace,* not from a new `TopologyResult` field, so the topology
  contract is unchanged. The output is one message's text verbatim (`Topology._assemble`), so the
  latest message with that text is the call that wrote it.

**Interfaces changed, every caller updated:**
- `Example`: new optional field `gold`, after `question`; positional construction is unchanged.
- `Metric.score` and `Metric.compute`: new optional argument `gold`. `Metric._score_examples`:
  new parameter `gold`, added to all seven registered metrics and to the test stubs.
- `DataConfig`: new field `strip_citations`.
- `summary.json`: new keys `newline_rate` and `mean_output_tokens`.

**Source:** D-030; D-119; D-120; D-123; D-125; ALCE repository commit `246c476` (`utils.py`,
`remove_citations`; `eval.py`, `main`)
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/*.yaml` (`data.strip_citations`, default False)
**Supersedes:** — (extends the common schema of AGENTS.md §4.2 and the metric interface of §4.5)

---

### D-127 · 2026-10-23 · ASQA implementation: reference text, frozen sample, prompt delivery and prompt lengths

**Decision:**
1. **Reference.** `Example.reference` is ALCE's `answer` field, which is the second of the two
   human answers for all 948 dev questions (checked). `Example.gold` holds `short_answers`, one
   list of accepted answers per sub-question, and `long_answers`, both human answers in ASQA's
   order (D-123 scores against both).
2. **Frozen sample** (D-122). Pool: all 948 dev questions; no filter removes any. Drawn
   2026-10-23: 167 / 167 / 166 per stratum; by number of sub-questions, 2: 167, 3: 79, 4: 88,
   5: 60, 6: 106. The IDs are in `tier2_asqa_sample.yaml`.
3. **Input** (D-117). `source` and `evidence` are the 5 passages, one per line in ALCE's format:
   median 527 and maximum 567 whitespace words per question in the sample, including the
   `Document [i](Title: …)` labels.
4. **Prompt** (D-118). Template `asqa_alce_vanilla@v1`. The instruction and the four worked
   examples are copied from ALCE's prompt file and checked byte for byte against it; a test pins
   their SHA-256. Every user message is the instruction, `Question: <question>` and the passages,
   separated by blank lines: ALCE's example layout without its `Answer:` label. Each run draws its
   2 examples with the project's sampling stream (`make_sampling_rng`) from the run seed, and its
   manifest lists them by ASQA train ID in the order sent. Seed 0 draws the *Planet of the Apes*
   and US-independence examples; seed 1, US independence and the field-goal record; seed 2, the
   rainiest place and *Planet of the Apes*.
5. **Prompt lengths,** measured with each backbone's chat template on the 500 sample questions
   for seeds 0–2: median 2,655 to 3,244 tokens, maximum 3,508 (Mistral-7B-Instruct-v0.3, seed 2).
   With the 300-token cap (D-119) the longest request is 3,808 tokens, within the 32,768-token
   serving context (D-039).

**Reason:** these record how D-116 to D-122 were implemented and checked, so the paper can state
the input, the prompt and the sample exactly. Using ALCE's `answer` field as the reference keeps
the reference text the same as ALCE's.

**Deviation from ALCE's delivery, stated for the paper:** ALCE sends the whole few-shot prompt as
a single text. For ChatGPT it adds the system message "You are a helpful assistant that answers
the following questions with proper citations."; for open models it applies no chat template
(`run.py`). We send the examples as chat turns (D-110) with no system message (D-016), as D-118
records.

**Source:** D-016; D-039; D-110; D-116 to D-123; ALCE repository commit `246c476`
(`prompts/asqa_default.json`, `run.py`); checks and measurements run 2026-10-23
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/tier2_asqa.yaml`, `src/configs/data/tier2_asqa_sample.yaml`;
`agents/generator.py` (`ASQA_ALCE_VANILLA`), `agents/asqa_examples.py`
**Supersedes:** —

---

### D-128 · 2026-10-23 · ASQA metrics: keys, two-answer scoring, and the leak check against all four ALCE examples

**Decision:**
1. **Keys,** fixed from the first run: `str_em` (metric `str_em`); `citation_recall` and
   `citation_precision` (metric `citation`); the ASQA leak check is metric
   `asqa_worked_example_overlap` and emits the existing key `worked_example_overlap` (D-110).
2. **`str_em`** is ALCE's `compute_str_em`, `exact_presence` and `normalize_answer`, unchanged,
   reported in [0, 1] rather than as a percentage. It reads `short_answers` from `Example.gold`
   and scores the output with citation markers removed (D-126) and not cut at a line break
   (D-125).
3. **`citation`** is ALCE's `compute_autoais` algorithm (§3.3):
   - sentences split by NLTK;
   - at most 3 citations read per sentence (`metrics.citation.max_citations_per_sentence`);
   - cited passages given to the judge as `Title: <title>`, the text on the next line, joined by
     newlines.
   The judge is the MiniCheck port's `claim_support` above the 0.5 threshold (D-024, D-120).
   Passages are recovered from the evidence, which must render back to the same text. One
   deliberate difference: a citation `[0]` counts as out of range, where ALCE's code would read it
   as the last passage.
4. **ROUGE-L and BERTScore** score the output against both human answers. They keep the
   precision, recall and F1 of the answer with the higher F1; on a tie, the later answer's, as
   ALCE's `compute_rouge` does. BERTScore's own multi-reference mode is not used, because it takes
   the maximum of each of the three separately. With one reference (Tier 1), both metrics run
   exactly as before.
5. **Leak check against all four ALCE examples.** It checks each example's question, each of its
   passages (title and text) and its answer without markers, as separate texts, with D-110's
   8-word rule and the question's own passages excluded.

**Reason for item 5:** D-118 said the check runs against the drawn examples, but metrics do not
receive the run seed. A match with an example the run did not show can only be a coincidence, so
checking all four never misses a leak that a check on the drawn pair would catch.

**Checks** (2026-10-23, CPU):
- The tests reproduce ALCE's worked examples: Figure 2 gives `str_em` 1/3, and Figure 3 gives
  citation recall 2/3 and precision 4/6 (with a stub judge).
- Scoring the first human answer of each of the 500 sample questions as if it were an output:
  mean `str_em` 0.972, and ROUGE-L F1 1.0 for every question (it matches itself).
- The real MiniCheck judge on ALCE's four worked answers against their own passages gives
  citation recall 0.50, 1.00, 0.00 and 1.00 (rainiest place, US independence, field goal,
  *Planet of the Apes*). The field-goal answer is a single sentence citing [1][2] that MiniCheck
  scores 0.482, just under the threshold. This shows how much one borderline judgment moves the
  recall of a short answer. Whether the judge agrees with people is what D-124 tests.

**Source:** Gao et al. 2023, §3.2, §3.3, Figures 2 and 3; ALCE repository commit `246c476`
(`eval.py`: `compute_str_em`, `exact_presence`, `compute_autoais`, `compute_rouge`; `utils.py`:
`normalize_answer`); `bert-score` (`BERTScorer.score`); D-024; D-110; D-118; D-120; D-123; D-125;
D-126
**Approved:** Sameer, 2026-10-23 (key names; the leak check against all four examples)
**Config:** `src/configs/base.yaml` (`metrics.citation`); `src/configs/data/tier2_asqa.yaml`
(`metrics.suite`)
**Supersedes:** D-118 item 4 ("scored on every ASQA run against the drawn examples"), in which
examples are checked

---

### D-129 · 2026-10-23 · ASQA judges validated: MiniCheck and AlignScore reproduce Tang et al. on LFQA and ClaimVerify

**Decision:** the MiniCheck-FT5 and AlignScore-large ports are accepted as ASQA's judges (D-120).
Run as D-124 specifies (`src/validate_metrics.py --metric minicheck_qa --metric alignscore_qa`,
2026-10-23):

| Port | Subset (test) | n | Balanced accuracy, threshold 0.5 | Published (Tang et al. 2024, Table 2) | Gap |
| --- | --- | --- | --- | --- | --- |
| MiniCheck-FT5 | LFQA | 1,911 | 85.2 | 85.2 | 0.0 |
| MiniCheck-FT5 | ClaimVerify | 1,088 | 74.6 | 74.6 | 0.0 |
| AlignScore-large | LFQA | 1,911 | 84.8 | 84.5 | +0.3 |
| AlignScore-large | ClaimVerify | 1,088 | 69.3 | 69.6 | −0.3 |

MiniCheck matches both published values exactly; AlignScore is within 0.3 points on both.

**Note:** during the MiniCheck pass the PyTorch allocator logged three failed allocations of
3–4 GB on long LFQA documents and retried; the run completed and the values match the published
ones. The ASQA inputs are shorter (at most 567 words of passages, D-127).

**Source:** validation records `minicheck_qa-20261023T133227Z.json` and
`alignscore_qa-20261023T133707Z.json` in `~/.cache/factagent_bench/metrics/validation/`; Tang et
al. 2024, Table 2; D-124
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/base.yaml` (`metrics.validation`)
**Supersedes:** —

---

### D-130 · 2026-10-26 · ASQA T0 results: nine runs; three Llama starts that ran out of memory stay on record

**Decision:** the nine runs below are T0's ASQA results under D-116 to D-128 (3 backbones × seeds
0–2, D-122's 500 questions, `asqa_alce_vanilla@v1`). Each finished with status `success`, 500 rows
and `git.dirty: false`, all from commit `cbb57f7`.

| Run ID | Backbone | Seed | Commit |
| --- | --- | --- | --- |
| `20261026_132201_t0_single_agent_tier2_asqa_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B-Instruct | 0 | `cbb57f7` |
| `20261026_134546_t0_single_agent_tier2_asqa_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B-Instruct | 1 | `cbb57f7` |
| `20261026_140655_t0_single_agent_tier2_asqa_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B-Instruct | 2 | `cbb57f7` |
| `20261023_150629_t0_single_agent_tier2_asqa_mistral-7b-instruct-v0.3_seed0` | Mistral-7B-Instruct-v0.3 | 0 | `cbb57f7` |
| `20261023_160249_t0_single_agent_tier2_asqa_mistral-7b-instruct-v0.3_seed1` | Mistral-7B-Instruct-v0.3 | 1 | `cbb57f7` |
| `20261023_165920_t0_single_agent_tier2_asqa_mistral-7b-instruct-v0.3_seed2` | Mistral-7B-Instruct-v0.3 | 2 | `cbb57f7` |
| `20261023_144555_t0_single_agent_tier2_asqa_qwen2.5-7b-instruct_seed0` | Qwen2.5-7B-Instruct | 0 | `cbb57f7` |
| `20261023_153956_t0_single_agent_tier2_asqa_qwen2.5-7b-instruct_seed1` | Qwen2.5-7B-Instruct | 1 | `cbb57f7` |
| `20261023_163758_t0_single_agent_tier2_asqa_qwen2.5-7b-instruct_seed2` | Qwen2.5-7B-Instruct | 2 | `cbb57f7` |

The first three Llama runs below failed at vLLM startup (`ServingError`: vLLM exited with code 1
before becoming ready) and wrote no rows. They are committed with the successful runs and are not
T0 results.

| Run ID | Backbone | Seed | Commit | Status |
| --- | --- | --- | --- | --- |
| `20261023_153655_t0_single_agent_tier2_asqa_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B-Instruct | 0 | `cbb57f7` | failed |
| `20261023_163448_t0_single_agent_tier2_asqa_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B-Instruct | 1 | `cbb57f7` | failed |
| `20261023_172833_t0_single_agent_tier2_asqa_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B-Instruct | 2 | `cbb57f7` | failed |

**Cause and fix:** `--backbone all` runs the three backbones in one process, and that process kept
about 326 MiB of GPU memory after the first two backbones' metrics; Llama ran third and its
128k-token vocabulary needs the most memory at startup, so vLLM found too little free memory. The
server logs (`/tmp/factagent_vllm_<run_id>.log`) are no longer on disk, so the run folders record
the error type and message only. Llama was re-run on 2026-10-26 with one process per run; the
three re-runs are the Llama rows in the first table. Qwen and Mistral were not re-run. Their
numbers do not depend on the fix, because their startup had free memory. Open item: the runner
should free GPU memory between backbones (or run each backbone in its own process).

**Checks on the nine runs** (from each run's `summary.json`): no blank outputs, `worked_example_overlap`
0.0 (D-128's leak check), and a length-cap rate of at most 0.006 (D-119).

Mean over the three seeds, computed from the runs above. These are diagnostics only; the headline
comparison is D-115's paired gain over T0.

| Backbone | MiniCheck | AlignScore | `str_em` | Citation recall | Citation precision | Mean output tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | 0.768 | 0.786 | 0.388 | 0.578 | 0.602 | 68 |
| Mistral-7B-Instruct-v0.3 | 0.731 | 0.759 | 0.389 | 0.519 | 0.497 | 104 |
| Qwen2.5-7B-Instruct | 0.822 | 0.817 | 0.358 | 0.676 | 0.617 | 57 |

**Reason:** D-124 required the judges to be validated first (D-129); with that done and the pilot
clean, T0 was run on the full 500 questions with every backbone and seed.

**Source:** each run's `summary.json` and `manifest.json`; D-119, D-122, D-128, D-129
**Approved:** Sameer, 2026-10-26
**Config:** none changed
**Supersedes:** the three failed Llama runs above, as T0 results

---

### D-131 · 2026-10-26 · T5 transfer results: eighteen runs on CNN/DM and SAMSum, each backbone served with its XSum adapter

**Decision:** the eighteen runs below are T5's transfer results under D-111 (3 backbones × seeds
0–2 on each of CNN/DM and SAMSum, `data/tier1_cnndm.yaml` and `data/tier1_samsum.yaml`, 500
articles each). Each backbone is served with its XSum GRPO adapter `checkpoint-625` (D-114); no
model was trained for this. Each run finished with status `success`, 500 rows and
`git.dirty: false`, all from commit `cbb57f7`. Each run's `manifest.json` records the adapter path
and sha256.

CNN/DM:

| Run ID | Backbone | Seed | Commit |
| --- | --- | --- | --- |
| `20261023_184340_t5_rl_agent_tier1_cnndm_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B-Instruct | 0 | `cbb57f7` |
| `20261023_202957_t5_rl_agent_tier1_cnndm_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B-Instruct | 1 | `cbb57f7` |
| `20261023_221601_t5_rl_agent_tier1_cnndm_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B-Instruct | 2 | `cbb57f7` |
| `20261023_180301_t5_rl_agent_tier1_cnndm_mistral-7b-instruct-v0.3_seed0` | Mistral-7B-Instruct-v0.3 | 0 | `cbb57f7` |
| `20261023_194907_t5_rl_agent_tier1_cnndm_mistral-7b-instruct-v0.3_seed1` | Mistral-7B-Instruct-v0.3 | 1 | `cbb57f7` |
| `20261023_213544_t5_rl_agent_tier1_cnndm_mistral-7b-instruct-v0.3_seed2` | Mistral-7B-Instruct-v0.3 | 2 | `cbb57f7` |
| `20261023_173133_t5_rl_agent_tier1_cnndm_qwen2.5-7b-instruct_seed0` | Qwen2.5-7B-Instruct | 0 | `cbb57f7` |
| `20261023_191814_t5_rl_agent_tier1_cnndm_qwen2.5-7b-instruct_seed1` | Qwen2.5-7B-Instruct | 1 | `cbb57f7` |
| `20261023_210442_t5_rl_agent_tier1_cnndm_qwen2.5-7b-instruct_seed2` | Qwen2.5-7B-Instruct | 2 | `cbb57f7` |

SAMSum:

| Run ID | Backbone | Seed | Commit |
| --- | --- | --- | --- |
| `20261023_233312_t5_rl_agent_tier1_samsum_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B-Instruct | 0 | `cbb57f7` |
| `20261024_003444_t5_rl_agent_tier1_samsum_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B-Instruct | 1 | `cbb57f7` |
| `20261024_013556_t5_rl_agent_tier1_samsum_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B-Instruct | 2 | `cbb57f7` |
| `20261023_231052_t5_rl_agent_tier1_samsum_mistral-7b-instruct-v0.3_seed0` | Mistral-7B-Instruct-v0.3 | 0 | `cbb57f7` |
| `20261024_001222_t5_rl_agent_tier1_samsum_mistral-7b-instruct-v0.3_seed1` | Mistral-7B-Instruct-v0.3 | 1 | `cbb57f7` |
| `20261024_011344_t5_rl_agent_tier1_samsum_mistral-7b-instruct-v0.3_seed2` | Mistral-7B-Instruct-v0.3 | 2 | `cbb57f7` |
| `20261023_225111_t5_rl_agent_tier1_samsum_qwen2.5-7b-instruct_seed0` | Qwen2.5-7B-Instruct | 0 | `cbb57f7` |
| `20261023_235217_t5_rl_agent_tier1_samsum_qwen2.5-7b-instruct_seed1` | Qwen2.5-7B-Instruct | 1 | `cbb57f7` |
| `20261024_005342_t5_rl_agent_tier1_samsum_qwen2.5-7b-instruct_seed2` | Qwen2.5-7B-Instruct | 2 | `cbb57f7` |

**Checks on the eighteen runs** (from each run's `summary.json`): no blank outputs, and a
length-cap rate of at most 0.018. No run failed.

**What these are for:** D-047 keeps CNN/DM and SAMSum evaluation-only, as transfer checks of an
adapter trained on XSum. This entry records the T5 side. The comparison with T0 on the same
articles is D-115's paired gain, in the aggregation step, and is not made here.

**Reason:** D-047 sets T5's scope as standalone on three backbones, with CNN/DM and SAMSum as
transfer checks; the XSum adapters (D-114) are complete.

**Source:** each run's `summary.json` and `manifest.json`; D-047, D-111, D-114, D-115
**Approved:** Sameer, 2026-10-26
**Config:** none changed
**Supersedes:** —

---

### D-132 · 2026-10-23 · ASQA training passages: Huang et al. 2024's GTR top-5 release

**Decision:**
1. **Source.** The passages for ASQA's training questions are the top-5 GTR passages released by
   Huang et al. 2024 (repository `HCY123902/atg-w-fg-rw`, Google Drive data folder, Apache-2.0):
   `asqa_train.json` (3,853 questions) and `asqa_dev.json` (500), SHA-256
   `62580fd30c64c907fe927c49f808b25c727a670cf64d674342bcbc0bb050e056` and
   `4ea8ae12e06a765b43bf4f9733bb63f884a22eb2c88145712b138066361c28bd`. The loader refuses a file
   whose checksum differs. A private copy is kept off the instance (location to be named).
2. **Fields used.** Only `question`, `qa_pairs` and `docs` (title, text). Each question's gold
   annotations come from the official ASQA release. Their ChatGPT `output` and `prompt` fields are
   not used.
3. **Exclusions.** ALCE's four worked examples (D-118) are ASQA training questions and are in these
   files; they are removed from training, leaving 4,349.

**Reason:**
- *Same passages as evaluation.* Huang et al. retrieve "the top-5 passages from the 2018-12-20
  Wikipedia snapshot" with GTR on ASQA (App. B.2), the corpus and retriever of ALCE's file (D-116,
  D-117). Their `asqa_test.json` (ALCE's dev split; SHA-256
  `4952ba628cf809b0030c6d70d279be9fbb95cdb9fe58da880f0a3fde9f01ff5d`) has the same top-5 passage
  IDs, in the same order, as our pinned ALCE file for all 948 questions. The policy therefore
  trains on input produced exactly as its evaluation input is.
- *Coverage and no overlap.* The two files together hold all 4,353 official ASQA training
  questions: every question text and every `qa_pairs` entry matches the official `ASQA.json`. None
  is a dev question, so none is in the 500-question evaluation sample (D-122).
- *Alternatives not adopted.* Re-running GTR needs ALCE's ~31 GB prebuilt index (ALCE README) and
  a separate instance, and would reproduce these passages. BM25 or ASQA's own `knowledge` passages
  would give training input that differs from evaluation input.

**Source:** Huang et al. 2024 (arXiv 2402.04315), App. B.1, B.2; their data release (files and
checksums above, downloaded 2026-10-23); official ASQA release `ASQA.json` (SHA-256
`8de0c824824372243b73b10314727dac1a02d9ecc729c34c752c30f8a3c1e58a`); ALCE README; D-116, D-117,
D-118, D-122; checks run 2026-10-23
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/data/tier2_asqa_train.yaml` (to be created)
**Supersedes:** —

---

### D-133 · 2026-10-23 · T5 is trained on ASQA as planned; its reward follows Huang et al. 2024

**Decision:**
1. **ASQA training stays.** D-047's three ASQA fine-tunes (one per backbone) go ahead, so H5a is
   answered on two task types: summarization (XSum) and grounded long-form QA (ASQA). The
   proposal's fallback of scoping H5a to Tier 1 only (proposal §7 item 9) is not taken.
2. **The ASQA reward follows Huang et al. 2024.** It has their three terms: correctness
   (exact-match recall of the gold short answers), citation recall and citation precision, as
   ALCE defines them (Huang et al. §2.2, App. A). The entailment checker, the term weights, how the
   terms are combined under GRPO, and the deviations from their setup are decided in later entries.

**Reason:**
- *What T5 is for.* T5 is one of the six topologies compared in H5a: training once against
  orchestrating on every request (proposal, Week 12). The contribution is the comparison, not a
  new training method. For the comparison to be fair, T5 is trained with an established recipe
  for the task. A reward designed by us would leave open whether a weak T5 result reflects
  training or the reward.
- *Why Huang et al.* It is the closest published RL setup for this task: ASQA with ALCE's GTR
  top-5 passages (App. B.2), the passages T5 trains on (D-132), and a 7B open model. Their
  ablation on ASQA (Table 2) shows that both parts are needed: trained on correctness alone,
  citation recall falls from 77.65 to 49.51 and answers lengthen from 51.34 to 56.34 tokens;
  trained on citation quality alone, correctness recall falls from 40.24 to 39.79.
- *What this project adds beyond Huang et al.:* their models are LLaMA-2-7B (main experiments)
  and T5-large (App. D); T5 here is trained on three backbone families (D-047). Their baselines are
  in-context prompting and their own training variants (Table 1); T5 here is compared with T0–T4
  on the same backbone, the same 500 questions and the same metrics (D-047, D-122). Every T5 and
  T0–T4 run records token cost and latency, and T5 its training GPU-hours, which H5a's break-even
  needs (AGENTS.md §5). Their RL stage scores citations with AttrScore-Flan-T5-XL and evaluation
  with TRUE (App. C); here the headline judge, MiniCheck, is kept out of the reward (D-048, D-120).
- *Precedent in this project.* T1 follows Self-Refine's published setup rather than a new method
  (D-051, D-107), for the same reason.

**Source:** proposal §7 item 9 and Week 12 (H5a); Huang et al. 2024 (arXiv 2402.04315), §2.2,
App. A, App. B.2, App. C, App. D, Tables 1 and 2; D-047, D-048, D-051, D-107, D-120, D-122, D-132
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/rl/grpo_asqa.yaml` (to be created)
**Supersedes:** — (confirms D-047's ASQA fine-tunes; answers Q-010 question 1)

---

### D-134 · 2026-10-23 · ASQA reward terms are combined with GDPO; D-054's advantage scaling is overridden for ASQA only

**Decision:**
1. The three ASQA reward terms (D-133) are combined as GDPO does (Liu et al. 2026, §3, Eq. 5–6):
   each term is normalized within its question's group of completions (minus the group mean,
   divided by the group standard deviation), the normalized terms are summed with their weights,
   and the sum is normalized once more. In TRL 1.14.1 this is
   `multi_objective_aggregation: normalize_then_sum`.
2. **For ASQA only,** this replaces D-054's `scale_rewards: none` (no division by the group
   standard deviation). TRL's `normalize_then_sum` divides by the standard deviation whatever
   `scale_rewards` says, so the ASQA config states the aggregation explicitly and every manifest
   records it under `grpo_args`. Every other D-054 setting, including the `dr_grpo` loss, is
   unchanged. XSum training (D-054, D-068) is unchanged.
3. The term weights are a separate decision.

**Reason:**
- *The problem is specific to GRPO.* Huang et al. 2024 add their three terms with fixed weights
  and train with PPO, which has a value model and no group normalization (§2.2, Eq. 1–2); their
  paper does not address combining terms within a group. Under GRPO, summing the terms first and
  then normalizing maps different combinations of term scores to the same advantage, so the
  policy cannot tell an answer that satisfies one term from one that satisfies two (Liu et al.
  2026, §2, Fig. 2).
- *D-054's setting was tested for several rewards and failed.* Liu et al. compare GRPO, GRPO
  without standard-deviation normalization (D-054's setting) and GDPO. Removing the standard
  deviation "only slightly increases the number of distinct advantage groups" (Fig. 3). On tool
  calling with Qwen2.5-1.5B-Instruct (five runs), GRPO without it learned the correctness reward
  but not the format reward: 0% correct format, against 80.66% for GDPO (Table 2, §4.1.1). GDPO
  outperformed both on tool calling, math and coding.
- *D-054's reason does not carry over.* D-054 removed the division because it enlarges small
  differences in groups whose completions score alike (Liu et al. 2025). That argument was made for
  a single reward. With several terms on different scales, per-term division is what gives each
  term an equal say.
- *Already the project's rule.* D-049 states that a second reward term is normalized separately
  (GDPO), never summed and then normalized.

**Comparability with XSum:** T5 is compared only within a dataset, with T0–T4 on the same backbone
and frozen sample (D-047, D-115); across datasets, gains over each dataset's own T0 are compared,
not raw scores (D-115 item 7). The XSum and ASQA rewards already differ in their terms (D-048,
D-133), so T5's recipe is task-specific; the paper states that a difference between T5's XSum and
ASQA gains reflects both the task and its reward.

**Not tested by Liu et al., stated as limitations:**
- Their rewards are mostly binary; ASQA's terms are fractions in [0, 1].
- Their tasks are tool calling, math and coding with 1.5B–7B models, not grounded QA.
- Enlargement of small differences in a group whose completions score alike, D-054's concern, is
  not examined. Each term is logged every step (AGENTS.md §4.6) so this stays visible.
- GDPO's final normalization is over the batch. Generation here holds one question's group at a
  time (D-063), so that batch is one group; this follows from TRL's implementation and is not a
  setting of Liu et al.

**Source:** Liu et al. 2026 (GDPO, arXiv 2601.05242), §2, §3, §4.1.1, Figs. 2–3, Table 2; Huang
et al. 2024 (arXiv 2402.04315), §2.2; Liu et al. 2025 (arXiv 2503.20783); TRL 1.14.1
`grpo_config.py` (`multi_objective_aggregation`) and `grpo_trainer.py` (`normalize_then_sum`
branch), read 2026-10-23; D-049, D-054, D-063, D-115, D-133
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/rl/grpo_asqa.yaml` (to be created);
`src/component/factagent_bench/config/training.py` (an aggregation field, to be added)
**Supersedes:** D-054 — only `scale_rewards: none`, and only for ASQA training

---

### D-135 · 2026-10-23 · ASQA reward: AlignScore judges the citation terms; the three terms are weighted equally

**Decision:**

| Term (D-133) | Computed by | Range |
| --- | --- | --- |
| Correctness | the registered `str_em` metric (D-128): share of sub-questions with a gold short answer in the answer | [0, 1] |
| Citation recall | ALCE's algorithm as in the registered `citation` metric (D-128), with **AlignScore-large** (D-023) as the entailment judge at threshold 0.5 | [0, 1] |
| Citation precision | as citation recall | [0, 1] |

1. **Judge.** The `citation` metric gains a judge setting (MiniCheck, its current judge, or
   AlignScore); the metric is extended, not copied (AGENTS.md §0.4). Evaluation keeps MiniCheck
   (D-120, D-128); the reward uses AlignScore. Both apply the same algorithm.
2. **Weights.** Equal: 1.0 for each term (TRL `reward_weights`), applied to the normalized terms
   (D-134).
3. **One score per answer.** Each term is the answer's ALCE score, as GRPO needs one reward per
   completion. A blank answer scores 0.0 on every term (D-050). An answer without citations scores
   0 on both citation terms, as ALCE defines them.

**Reason:**
- *Held-out judge.* MiniCheck is ASQA's headline judge and judges the reported citation metrics
  (D-120, D-128). Training on it would make the evaluation circular, which is D-048's rule for
  XSum.
- *Why AlignScore.* It is validated on long-form answers checked against retrieved documents:
  balanced accuracy 84.8 on LFQA and 69.3 on ClaimVerify, within 0.3 points of Tang et al. 2024
  (D-129), at the 0.5 threshold of that validation. It already ran beside the policy as the XSum
  reward (D-048, D-068), and it is a different model from MiniCheck (RoBERTa-large against
  Flan-T5-Large, D-048).
- *Departure from Huang et al.* During RL they replace TRUE with AttrScore-Flan-T5-XL "for
  efficiency" (App. C). AttrScore is not ported or validated in this project, and AlignScore is.
- *Equal weights.* Huang et al. set each weight to 0.2 and report that this "serves our goal of
  optimizing citation quality, while ensuring correctness" (§3.3). Equal weights keep their
  balance; the scale does not matter because GDPO normalizes the sum (D-134).

**Deviations from Huang et al., for the paper:**
- They sum ±w per gold answer, per sentence and per citation (App. A). Their correctness term is
  an affine function of exact-match recall within a question, so after normalization within the
  group it gives the same advantages. Their citation terms are sums over sentences and citations,
  which grow with the number of sentences; ours are ALCE's means, which do not reward length.
- Their rewards are assigned to tokens and trained with PPO; ours are one score per answer under
  GRPO (D-046, D-134).

**Known risks, monitored:**
- Liu et al. 2026 report that weights do not always give the intended balance when one objective
  is much easier than the others (§3.2). Each term is logged every step (AGENTS.md §4.6).
- Exact-match recall rises with answer length (D-119; Huang et al. Table 2). Answer length is
  logged every step, and D-049's length contingency applies.
- T5's ASQA AlignScore numbers are reported labelled as related to the training reward, as on XSum
  (D-048). The headline remains MiniCheck (D-120).

**Source:** Huang et al. 2024 (arXiv 2402.04315), §2.2, §3.3, App. A, App. C, Table 2; Liu et al.
2026 (arXiv 2601.05242), §3.2; `metrics/citation.py`, `metrics/str_em.py`; D-023, D-046, D-048,
D-049, D-050, D-068, D-119, D-120, D-128, D-129, D-133, D-134
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/rl/grpo_asqa.yaml` (to be created; reward terms and weights);
`src/configs/base.yaml` (`metrics.citation`, a judge field, to be added)
**Supersedes:** — (answers Q-010 questions 2 and 3)

---

### D-136 · 2026-10-23 · ASQA GRPO memory test: two prompt settings on the longest training prompts, plus forced-length, timing, reward-signal and smoke checks

**Decision:** before any ASQA fine-tune, measure which training prompt fits the A10G. Both settings
are measured; the training prompt is then chosen from the results in a new entry.

| Setting | Training prompt | Group size G |
| --- | --- | --- |
| 1 | ALCE instruction with 2 worked examples as chat turns, as evaluation sends it (D-118) | 4; 2 only if 4 runs out of memory |
| 2 | the same instruction with no worked examples | 8 |

| Parameter | Value | Source |
| --- | --- | --- |
| Questions per setting | the 10 training questions with the longest prompt, longest over the three tokenizers | D-055 |
| Worked examples in setting 1 | seed 2's pair, which gives the longest prompts for every backbone | worst case, as D-055 |
| Steps per cell | 10; step 1 left out of the seconds-per-step average | D-055 |
| Completion cap | 300 tokens | D-119 |
| Backbones, in order | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct | D-055 |
| Reward in every cell | the ASQA reward: three terms, AlignScore-judged citations above 0.5, equal weights, GDPO | D-133, D-134, D-135; D-129 |
| GRPO settings | D-054, except advantage scaling (D-134) | D-054, D-134 |

**Further checks, each run by Sameer after the GPU is free:**

| Check | What | Why |
| --- | --- | --- |
| Forced-length cells | the setting chosen, its longest prompts, every completion forced to 300 tokens | the most memory a step can need (D-059); real XSum training peaked above its forced cells (21.43 against 17.99 GiB, D-062, D-068) |
| Timing cells | the setting chosen, the 10 training questions closest to the median prompt | seconds per step with the citation judge loaded, for the step count and GPU-hours (Q-011), as D-056 |
| Reward-signal check | the three reward terms scored on the outputs of the nine T0 ASQA runs; no new generation | agreement of AlignScore- and MiniCheck-judged citation recall; how often answers to one question tie on a term, which leaves GRPO no signal for it (D-134's untested concern) |
| Smoke run | two optimizer steps of real ASQA training on one backbone | catches failures of the reward code on real outputs before a multi-day run |
| Zero-shot citation pilot | 30 questions × 3 backbones, no worked examples, inference only | only if setting 1 does not fit: GRPO can learn citations only if some zero-shot answers cite |

**Training data used by the cells:** ASQA's training questions with Huang et al.'s passages
(D-132), downloaded from the original sources and checked against their SHA-256, with the four
ALCE worked examples removed.

**Reason:** the evaluation prompt with 2 worked examples measures up to 3,689 tokens on the training
questions (Mistral; zero-shot maximum 1,415), measured 2026-10-23 with each chat template. GRPO
memory was measured only up to 2,048-token prompts, where G = 8 peaked at 21.2–21.3 GiB (D-057),
and XSum training at 1,536 tokens peaked at 21.43 GiB of 22.06 (D-068). Setting 1 keeps T5's
prompt identical to T0's, as on XSum (D-111), and avoids teaching citations from zero-shot answers;
setting 2 is the fallback, with the larger group of D-057.

**Source:** D-054, D-055, D-056, D-057, D-059, D-062, D-068, D-111, D-118, D-119, D-129, D-132–D-135;
Q-011; prompt lengths measured 2026-10-23
**Approved:** Sameer, 2026-10-23
**Config:** `src/configs/rl/grpo_asqa_memory.yaml` and its per-setting files (to be created);
`src/run_matrix.csv`
**Supersedes:** —

---

### D-137 · 2026-10-24 · An empty claim scores 0.0 under the AlignScore citation judge; the ASQA memory test is re-run in full

**Finding:** the first ASQA memory sweep (D-136), from commit `cda8343`, stopped after three cells:

| Run | Status |
| --- | --- |
| `20261024_024429_grpo_asqa_memory_fewshot_4096_llama-3.1-8b-instruct_seed0` | success, 10 steps, peak 21.23 GiB |
| `20261024_025118_grpo_asqa_memory_fewshot_4096_mistral-7b-instruct-v0.3_seed0` | failed at step 10 (9 steps logged), peak 18.56 GiB |
| `20261024_030045_grpo_asqa_memory_zeroshot_4096_llama-3.1-8b-instruct_seed0` | failed at step 2 (1 step logged), peak 13.75 GiB |

Both failures are `AlignScore needs a summary with at least one sentence`, raised by the citation reward, not running out of memory. A completion that places citation markers on their own after a sentence (`Paris is the capital [1]. [2]`) is split by NLTK into a sentence `[2]`, whose claim is empty once the markers are removed (ALCE's `compute_autoais` removes them the same way). MiniCheck, the evaluation judge, accepts an empty claim; AlignScore's port raises. The tests used stub judges and did not reach this path.

**Decision:**
1. Under the AlignScore judge, a claim with no text scores 0.0, so the sentence counts as unsupported and its citations as imprecise. It is applied in `AlignScore.claim_support`, the judge path only; AlignScore as a metric and every MiniCheck-judged score are unchanged.
2. All six cells of D-136 are re-run from a commit that holds the fix. The three runs above stay in `results/` and are not used.

**Reason:** an empty claim states nothing the passages could support; 0.0 is the bottom of the score range, so GRPO never rewards stray markers. Dropping empty sentences from the count instead would change how ALCE's algorithm counts sentences, and with it the reported, MiniCheck-judged citation scores (D-128).

**Source:** the three run directories above (`summary.json`, error and traceback); `~/asqa_memtest.log`; ALCE `eval.py` (`compute_autoais`), commit `246c476`; D-128, D-135, D-136
**Approved:** Sameer, 2026-10-24
**Config:** none; `src/component/factagent_bench/metrics/alignscore.py` (`claim_support`)
**Supersedes:** — (the three runs above, as D-136 measurements)

---

### D-138 · 2026-10-24 · ASQA training prompt: the evaluation prompt with 2 worked examples, at G = 4

**Decision:** T5 trains on ASQA with setting 1 of D-136: ALCE's instruction with 2 worked examples
as chat turns, the prompt T0 is evaluated with (D-118), and G = 4. Setting 2 (no worked examples,
G = 8) is not used. The G = 2 fallback cells were not needed.

**Measurements** (peak reserved memory of the 22.06 GiB PyTorch can use; 10 steps per cell on
the 10 longest training prompts, up to 3,689 tokens; runs from commits `0939d03` and `31125f5`,
none dirty; run IDs in `src/run_matrix.csv`, rows `asqa_mem_*`):

| Backbone | Setting 1 | Setting 1, every completion forced to 300 tokens | Setting 2 | Setting 1 s/step | Setting 2 s/step |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | 21.23 GiB | 18.16 GiB | 14.34 GiB | 35.3 | 43.0 |
| Mistral-7B-Instruct-v0.3 | 19.79 GiB | 19.75 GiB | 17.80 GiB | 42.3 | 42.6 |
| Qwen2.5-7B-Instruct | 21.66 GiB | 18.70 GiB | 20.55 GiB | 30.2 | 29.8 |

Every cell completed 10 steps. The allocator freed its cache and retried an allocation in the
setting-1 Llama and Qwen cells and in the setting-2 Qwen cell (`~/asqa_memtest.log`); no forced
cell did.

**Reason:**
- *Same prompt as T0.* With setting 2, T5 would either be evaluated on a prompt it was not trained
  on, which D-111 rules out on XSum, or be evaluated zero-shot while T0 is few-shot, so H5a could
  not credit a difference to RL alone. Setting 1 keeps the comparison to one change, as D-136
  preferred.
- *It fits.* No setting-1 cell ran out of memory, including the forced cells, which hold the
  longest sequences a training step can have. As on XSum (D-062), the forced cells peaked lower
  than the unforced ones; the figure includes the allocator's cache, which completions of varying
  length enlarge.
- *Remaining risk.* XSum's fine-tunes peaked 0.65 to 5.43 GiB above the higher of their memory and
  forced cells (D-057, D-062, D-068) and still completed. If an ASQA fine-tune runs out of memory,
  it resumes from its latest checkpoint (D-059).
- *Signal at G = 4* (from the setting-1 cells' per-term standard deviations): all four answers to
  a question tied on exact-match recall in 6, 8 and 7 of 10 steps (Llama, Mistral, Qwen), on
  citation recall in 2, 0 and 4, on citation precision in 2, 0 and 4, and on all three terms in
  0, 0 and 2. GDPO normalizes each term within its group (D-134), so a tied term adds nothing to
  that group's advantages and the other terms still do.

**Source:** the run directories in `src/run_matrix.csv` rows `asqa_mem_*`; `~/asqa_memtest.log`,
`~/asqa_forced.log`; D-057, D-059, D-062, D-068, D-111, D-118, D-134, D-136, D-137
**Approved:** Sameer, 2026-10-24
**Config:** `src/configs/rl/grpo_asqa_memory_fewshot.yaml` (worked-example seed 2, G = 4);
the training config follows from it; `src/run_matrix.csv`
**Supersedes:** —

---

### D-139 · 2026-10-24 · ASQA fine-tunes: one pass over 4,349 questions, 16 per step, per-question worked examples, three instances in parallel

**Decision:**

| Setting | Value |
| --- | --- |
| Training pool | all 4,349 training questions (D-132), frozen by `python -m factagent_bench.rl.pool` into `src/configs/rl/grpo_asqa_pool.yaml` in an order drawn with seed 123 (D-057). Every prompt fits the 4,096-token bound under all three tokenizers |
| Questions per optimizer step | 16: `per_device_train_batch_size` 4 (one question's group, D-138) × `gradient_accumulation_steps` 16, i.e. 64 answers per update |
| `training.max_steps` | 271: one pass over 4,336 questions, each used once; the last 13 do not fill a step |
| `training.save_steps` | 50: checkpoints at steps 50, 100, 150, 200 and 250, plus the final adapter at 271 |
| Learning rate | 1e-5 (D-054) |
| Worked examples | `training.worked_examples_draw: per_question`: each question draws its own 2 of ALCE's 4 examples, from a seed derived from the run seed and the question's ID (`rl/trainer.py`, `question_seed`). All 12 ordered pairs occur, 341 to 377 questions each; the manifest records the counts |
| Reward-signal check | runs before the fine-tunes, on the six T0 ASQA runs that completed (Qwen and Mistral, seeds 0 to 2; D-136): `python -m factagent_bench.rl.reward_check`. The three Llama T0 ASQA runs of 2026-10-23 failed before generating (vLLM did not start; the server log was in `/tmp` and is gone) |
| Smoke run | 2 optimizer steps of `grpo_asqa.yaml` on Qwen, before the fine-tunes (D-136) |
| Where | each backbone on its own `g5.2xlarge` (one A10G), at the same time, from the same pushed commit: Qwen on Sameer's instance, Llama and Mistral on two new instances set up as `src/docs/t5_asqa_parallel_setup.md` describes. Sameer runs all three |

**Timing cells** (D-136; the 10 training questions closest to the 3,242-token median prompt, G = 4,
10 steps, commit `7d92e55`, none dirty; rows `asqa_mem_*_typical` in `src/run_matrix.csv`):

| Backbone | Seconds per question | Peak reserved memory | Mean answer | Seconds per question, forced to 300 tokens (D-138) |
| --- | --- | --- | --- | --- |
| Llama | 32.5 | 18.32 GiB | 77 tokens | 43.7 |
| Mistral | 40.7 | 20.26 GiB | 110 tokens | 47.3 |
| Qwen | 29.3 | 21.70 GiB | 71 tokens | 41.1 |

The Qwen cell's allocator freed its cache and retried once (step 5, `~/asqa_timing.log`).

**Cost:** 4,336 questions at the typical rate take about 39 (Llama), 49 (Mistral) and 35 (Qwen)
GPU-hours; if answers lengthen to the 300-token cap during training, up to 53, 57 and 50. In
parallel the three finish in about 1.5 to 2.4 days of wall clock, against 5 to 7 back to back.

**Reason:**
- *One pass, each question once.* D-055 and D-058 on XSum: no question repeats, so no reward is
  earned by memorizing an answer. ASQA has 4,349 training questions, fewer than XSum's 10,000, so
  the whole split is used.
- *16 questions per step.* As D-058, from Shao et al. 2024 §4.2 (16 questions per step). At G = 4
  that is 64 answers per update. Gradient accumulation adds no memory (D-055).
- *Learning rate.* 271 steps is above the roughly 100 steps under which D-058 would raise it.
- *Per-question worked examples.* T0 and T5 are evaluated with seeds 0, 1 and 2, each drawing its
  own pair (D-118). One pair for the whole of training would let T5 see the other seeds' pairs only
  at evaluation. The longest prompt under per-question draws is 3,634 tokens, within the 3,689 the
  memory and forced cells used (D-138), measured 2026-10-24 with all three tokenizers.
- *Three instances.* Back to back, a failure in the last backbone would surface only after four to
  five days. In parallel, a failure on any backbone surfaces within hours. Identical instance
  type, pinned packages and commit keep the runs comparable; each manifest records its GPU,
  driver and host.

**Source:** the timing run directories (`src/run_matrix.csv`); D-054, D-055, D-057, D-058, D-059,
D-118, D-132, D-136, D-138; Shao et al. 2024 (arXiv 2402.03300) §4.2
**Approved:** Sameer, 2026-10-24
**Config:** `src/configs/rl/grpo_asqa.yaml`; `src/configs/rl/grpo_asqa_pool.yaml`;
`src/configs/rl/asqa_reward_check.yaml`; `src/run_matrix.csv`
**Supersedes:** D-059's "one after another on his instance", for the ASQA fine-tunes only

---

### D-140 · 2026-10-24 · Reward-signal check: AlignScore and MiniCheck agree at r ≈ 0.7; the fine-tunes proceed and the gap is reported

**Finding** (run `20261024_190842_asqa_reward_check_seed0`, commit `a66655d`, not dirty): the 3,000
answers of the six completed T0 ASQA runs (Qwen and Mistral, seeds 0 to 2; D-139) rescored with the
reward's citation judge, AlignScore above 0.5 (D-135), against the MiniCheck-judged scores the T0
runs recorded (D-120).

| Over all 3,000 answers | Citation recall | Citation precision |
| --- | --- | --- |
| Pearson correlation | 0.73 | 0.67 |
| Spearman correlation | 0.74 | 0.66 |
| Answers scored exactly the same | 71.9% | 62.2% |
| Mean, MiniCheck | 0.597 | 0.557 |
| Mean, AlignScore | 0.652 | 0.607 |

Per backbone, Pearson is 0.74 and 0.67 (Mistral; recall, precision) and 0.71 and 0.65 (Qwen), and
AlignScore's mean is higher on both terms for both backbones, by 0.04 to 0.07.

Ties across a question's three seeds, each seed with its own pair of worked examples:

| Backbone | Exact-match recall | Citation recall (AlignScore) | Citation precision (AlignScore) |
| --- | --- | --- | --- |
| Mistral | 62.6% | 8.8% | 7.2% |
| Qwen | 82.6% | 51.0% | 39.0% |

**Decision:**
1. The fine-tunes of D-139 proceed with AlignScore as the citation judge (D-135 stands).
2. T5's ASQA results report both the training reward (AlignScore-judged) and the evaluation scores
   (MiniCheck-judged) per checkpoint, so that any gain that appears under AlignScore but not under
   MiniCheck is visible. That gap is reported as a limitation, not hidden.

**Reason:**
- *Agreement.* The judges rank answers alike (Spearman 0.66 to 0.74), so raising AlignScore-judged
  citation scores is expected to raise MiniCheck-judged ones. Judging training with MiniCheck
  instead would remove the held-out evaluation judge (D-135).
- *Leniency.* AlignScore accepts more citations than MiniCheck on the same answers. A policy can
  learn citations that only the lenient judge accepts; reporting both judges per checkpoint is the
  check for it.
- *Ties.* Exact-match recall ties for most questions, so the citation terms carry most of the
  signal. GDPO normalizes each term within its group (D-134), so a tied term drops out of that
  group without cancelling the others. Qwen's answers tie more often, which may slow its learning;
  its training curve is read with that in mind. These ties are across T0 seeds with different
  worked examples, an approximation of a GRPO group, not a measurement of one.

**Source:** run `20261024_190842_asqa_reward_check_seed0` (`summary.json`); the six T0 runs named
in its manifest; D-120, D-134, D-135, D-136, D-139
**Approved:** Sameer, 2026-10-24
**Config:** none changed; `src/configs/rl/asqa_reward_check.yaml`
**Supersedes:** —

---

### D-141 · 2026-10-26 · T5 on ASQA, Qwen2.5-7B-Instruct: fine-tune finished; on the 500 dev questions it beats T0 on faithfulness and citations and ties it on correctness

**Decision:** the Qwen ASQA fine-tune of D-139 is complete, and its `final` adapter (byte-identical to
`checkpoint-271`, checked 2026-10-26) is the T5 adapter for Qwen on ASQA. It was evaluated as
`t5_rl_agent` (D-111) on the 500 frozen dev questions (D-122), seeds 0, 1 and 2, with T0's prompt,
sampling and metric suite (D-118, D-119, D-120). The adapter was named on the command line
(`--set serving.lora_adapters=qwen2.5-7b-instruct=<adapter>/final`); the shipped
`topology/t5_rl_agent.yaml` still lists only the XSum adapters.

**Fine-tune** (run `20261024_213959_grpo_asqa_qwen2.5-7b-instruct_seed0`, commit `599eec6`, not dirty):
271 of 271 steps, 35.1 GPU-hours, peak reserved memory 21.68 GiB of 22.06, no out-of-memory,
resume or NaN. Training reward, mean of the first 20 steps against the last 20:

| Term | Steps 1–20 | Steps 252–271 |
| --- | --- | --- |
| Citation recall | 0.68 | 0.80 |
| Citation precision | 0.61 | 0.75 |
| Exact-match recall | 0.39 | 0.32 |
| KL from the base model | 0.001 | 0.063 |

These are training questions in a rotating order, not a held-out measure.

**Evaluation** (runs `20261026_160800`, `20261026_163028`, `20261026_165035`, `…t5_rl_agent_tier2_asqa_qwen2.5-7b-instruct_seed0/1/2`,
commit `599eec6`, not dirty; 500 answers each, none blank, none at the 300-token cap) against T0 on the same
questions and seeds (runs `20261023_144555`, `20261023_153956`, `20261023_163758`,
`…t0_single_agent_tier2_asqa_qwen2.5-7b-instruct_seed0/1/2`). Mean over the three seeds, seed values in brackets:

| Metric | T0 | T5 | Difference |
| --- | --- | --- | --- |
| MiniCheck (headline) | 0.822 (0.820, 0.820, 0.826) | 0.865 (0.869, 0.863, 0.863) | +0.043 |
| AlignScore | 0.817 | 0.857 | +0.040 |
| Citation recall, MiniCheck-judged | 0.676 (0.679, 0.657, 0.691) | 0.786 (0.792, 0.780, 0.785) | +0.110 |
| Citation precision, MiniCheck-judged | 0.617 (0.609, 0.607, 0.635) | 0.736 (0.743, 0.732, 0.734) | +0.119 |
| Exact-match recall | 0.358 (0.350, 0.372, 0.352) | 0.358 (0.356, 0.363, 0.356) | 0.000 |
| ROUGE-L F1 | 0.317 | 0.317 | +0.001 |
| BERTScore F1 | 0.894 | 0.895 | +0.001 |
| Mean answer length, tokens | 57.3 | 53.9 | −3.4 |

The worked-example overlap check (D-118) scored 0.0 on all three T5 runs.

**Reading:**
- The headline gain is under the evaluation judge, which training never used (D-135); the citation
  scores reported here are MiniCheck-judged (D-120), and they rose by about 0.11. D-140's concern, a
  gain that appears only under the lenient training judge, is not seen on this backbone.
- Correctness, as exact-match recall, is unchanged.
- Each difference exceeds the seed-to-seed range of either system, which is at most 0.034, for
  MiniCheck, AlignScore, citation recall and citation precision. The paired per-question gain over T0
  with its 95% bootstrap interval (D-115) is not yet computed.
- One backbone and one fine-tune: this does not yet say anything about H5a across backbones.

**Not used:** three first attempts to evaluate (`20261026_160404`, `20261026_160409`,
`20261026_160414`, `…t5_rl_agent_tier2_asqa_qwen2.5-7b-instruct_seed0/1/2`) stopped before generation because the worktree has no
`.env` and `FACTAGENT_VLLM_URL` was unset; they stay in `results/` as failed, and the three runs above
set it in the command.

**Source:** the run directories above (`summary.json`, `manifest.json`, `metrics.csv`); `src/run_matrix.csv`
rows `t5_asqa_qwen2.5-7b-instruct`, `t5_eval_asqa_*`; D-111, D-118, D-119, D-120, D-122, D-135, D-139, D-140
**Approved:** Sameer, 2026-10-26
**Config:** none changed; `src/configs/topology/t5_rl_agent.yaml` (adapter named on the command line)
**Supersedes:** —

---

### D-142 · 2026-10-29 · T5 on ASQA, Llama-3.1-8B and Mistral-7B-v0.3: fine-tunes finished; on all three backbones T5 beats T0 on faithfulness and citations, with mixed correctness

**Decision:** the Llama and Mistral ASQA fine-tunes of D-139 are complete, and each one's `final`
adapter (byte-identical to its `checkpoint-271`, checked 2026-10-29) is the T5 adapter for that
backbone on ASQA. Both were evaluated exactly as Qwen was in D-141: `t5_rl_agent` (D-111) on the 500
frozen dev questions (D-122), seeds 0, 1 and 2, with T0's prompt, sampling and metric suite (D-118,
D-119, D-120), the adapter named on the command line
(`--set serving.lora_adapters=<backbone>=<adapter>/final`). This entry also adds the paired
per-question gain over T0 with its 95% bootstrap interval (D-115) for all three backbones, which
D-141 left open.

**Fine-tunes** (commit `599eec6`, not dirty; 271 of 271 steps; no out-of-memory, resume or NaN):

| | Llama-3.1-8B | Mistral-7B-v0.3 |
| --- | --- | --- |
| Run | `20261025_190010_grpo_asqa_llama-3.1-8b-instruct_seed0` | `20261025_202728_grpo_asqa_mistral-7b-instruct-v0.3_seed0` |
| GPU-hours (wall clock) | 39.6 | 49.7 |
| Peak reserved memory | 19.87 GiB | 21.67 GiB |

Training reward, mean of the first 20 steps against the last 20 (Qwen from D-141 for comparison):

| Term | Qwen | Llama | Mistral |
| --- | --- | --- | --- |
| Citation recall | 0.68 → 0.80 | 0.56 → 0.85 | 0.63 → 0.86 |
| Citation precision | 0.61 → 0.75 | 0.56 → 0.85 | 0.63 → 0.85 |
| Exact-match recall | 0.39 → 0.32 | 0.42 → 0.36 | 0.43 → 0.39 |
| KL from the base model | 0.001 → 0.063 | 0.001 → 0.088 | 0.032 → 0.092 |
| Mean completion length, tokens | 65 → 61 | 82 → 65 | 101 → 118 |

These are training questions in a rotating order, not a held-out measure. A first Llama attempt
(`20261025_184748_grpo_asqa_llama-3.1-8b-instruct_seed0`) failed after 113 s, before step 1: Triton's
gcc build of its CUDA helper exited with status 1. It stays in `results/` as failed and is not used.

**Evaluation** (commit `e772c53`, not dirty; 500 answers each, none blank; the worked-example overlap
check, D-118, scored 0.0 on all six runs):

| Backbone | T5 runs (seeds 0, 1, 2) | Paired T0 runs (D-130) |
| --- | --- | --- |
| Llama-3.1-8B | `20261027_205944`, `20261027_212417`, `20261027_214458` | `20261026_132201`, `20261026_134546`, `20261026_140655` |
| Mistral-7B-v0.3 | `20261029_131518`, `20261029_135144`, `20261029_142439` | `20261023_150629`, `20261023_160249`, `20261023_165920` |

Full IDs are `<timestamp>_t5_rl_agent_tier2_asqa_<backbone>_seed<n>` and
`<timestamp>_t0_single_agent_tier2_asqa_<backbone>_seed<n>`. Answers at the 300-token cap: Llama 0 of
1,500; Mistral 1 of 1,500 (T0: 4 of 1,500).

**Results (D-115: seeds averaged per question, paired difference against T0, 95% percentile
bootstrap over the 500 questions, 10,000 resamples, seed 123).** Qwen's means are D-141's; its
intervals are new here.

| Backbone | Metric | T0 | T5 | Paired gain | 95% CI |
| --- | --- | --- | --- | --- | --- |
| Qwen | MiniCheck (headline) | 0.822 | 0.865 | +0.0426 | +0.0314, +0.0540 |
| Qwen | AlignScore | 0.817 | 0.857 | +0.0402 | +0.0308, +0.0499 |
| Qwen | Citation recall | 0.676 | 0.786 | +0.1101 | +0.0883, +0.1321 |
| Qwen | Citation precision | 0.617 | 0.736 | +0.1192 | +0.0971, +0.1413 |
| Qwen | `str_em` | 0.358 | 0.358 | +0.0004 | −0.0090, +0.0098 |
| Mistral | MiniCheck (headline) | 0.731 | 0.887 | +0.1553 | +0.1402, +0.1705 |
| Mistral | AlignScore | 0.759 | 0.889 | +0.1301 | +0.1176, +0.1429 |
| Mistral | Citation recall | 0.519 | 0.806 | +0.2865 | +0.2656, +0.3075 |
| Mistral | Citation precision | 0.497 | 0.801 | +0.3043 | +0.2841, +0.3244 |
| Mistral | `str_em` | 0.389 | 0.407 | +0.0181 | +0.0050, +0.0313 |
| Llama | MiniCheck (headline) | 0.768 | 0.908 | +0.1405 | +0.1239, +0.1575 |
| Llama | AlignScore | 0.786 | 0.909 | +0.1225 | +0.1082, +0.1372 |
| Llama | Citation recall | 0.578 | 0.818 | +0.2406 | +0.2138, +0.2669 |
| Llama | Citation precision | 0.602 | 0.827 | +0.2247 | +0.1991, +0.2495 |
| Llama | `str_em` | 0.388 | 0.370 | −0.0183 | −0.0299, −0.0069 |

Citation scores are MiniCheck-judged (D-120). Seed values, MiniCheck: Mistral T0 0.750, 0.722, 0.723
and T5 0.887, 0.895, 0.878; Llama T0 0.761, 0.775, 0.768 and T5 0.915, 0.912, 0.899.

Secondary metrics and cost, mean over the three seeds:

| Backbone | ROUGE-L F1, T0 → T5 | BERTScore F1, T0 → T5 | Mean answer length, tokens | Tokens per question |
| --- | --- | --- | --- | --- |
| Qwen | 0.317 → 0.317 | 0.894 → 0.895 | 57.3 → 53.9 | 2,905 → 2,901 |
| Mistral | 0.298 → 0.316 | 0.885 → 0.887 | 103.7 → 109.2 | 3,160 → 3,166 |
| Llama | 0.320 → 0.330 | 0.891 → 0.894 | 67.6 → 57.9 | 2,814 → 2,805 |

**Reading:**
- On all three backbones T5 is above T0 on MiniCheck, the held-out evaluation judge (D-135), with
  intervals that exclude 0; the gain is more than three times larger on Mistral and Llama (+0.155,
  +0.141) than on Qwen (+0.043), the backbone with the highest T0 score.
- Citation recall and precision rise by 0.11–0.30 on every backbone, under MiniCheck as judge.
- AlignScore judged the citation terms of the training reward (D-135), and exact-match recall is
  itself a reward term (D-133); neither is independent of training in the way MiniCheck is.
- Correctness, as exact-match recall, is mixed: unchanged on Qwen, +0.018 on Mistral, −0.018 on Llama.
  Mistral's answers got longer (training completions 101 → 118 tokens; evaluation answers 103.7 →
  109.2) and Llama's shorter (67.6 → 57.9), and exact-match recall rises with length (D-119), so these
  two changes may be partly length effects; this was not tested.
- T5 costs the same tokens per question as T0: one call, no added rounds.
- This is one fine-tune per backbone (seed 0); the three evaluation seeds measure sampling variation,
  not training variation.

**Source:** the run directories above (`summary.json`, `manifest.json`, `metrics.csv`); D-141's Qwen
runs; `src/run_matrix.csv` rows `t5_asqa_*`, `t5_eval_asqa_*`; D-111, D-115, D-118, D-119, D-120,
D-122, D-130, D-133, D-135, D-139, D-141
**Approved:** Sameer, 2026-10-29
**Config:** none changed; `src/configs/topology/t5_rl_agent.yaml` (adapter named on the command line)
**Supersedes:** —

---

### D-150 · 2026-10-26 · T1 Self-Refine on ASQA: feedback checks each sentence against the passages it cites; prompts `asqa_self_feedback@v1` and `asqa_self_refine@v1`; three worked examples checked by MiniCheck

*Numbering:* D-132 to D-141 are used by the T5 work on branch `t5-asqa`, and D-142 to D-149 are
left for it. The T1 ASQA entries start at D-150.

**Decision:**

1. **Feedback scope.** The feedback checks one thing: whether every sentence of the answer is
   supported by the search results it cites. A sentence that cites none counts as unsupported.
   Completeness is out of scope, meaning parts of an ambiguous question that the answer leaves
   out, and so are length and style.
2. **`asqa_self_feedback@v1` instruction:** "Check whether every sentence of the answer is
   supported by the search results it cites. For each sentence that cites no search result, or
   whose cited search results do not support it, write one line starting with "- " that quotes the
   unsupported words and says which search result supports them, or that none does. If every
   sentence is supported by the search results it cites, reply with only the word NONE."
3. **`asqa_self_refine@v1` instruction:** "Rewrite the latest answer so that it fixes every problem
   listed in its feedback. Use only information stated in the search results. Write an accurate,
   engaging, and concise answer in an unbiased and journalistic tone. Always cite for any factual
   claim. When citing several search results, use [1][2][3]. Cite at least one document and at most
   three documents in each sentence. If multiple documents support the sentence, only cite a
   minimum sufficient subset of the documents. Reply with only the rewritten answer." The first two
   sentences and the last follow `xsum_self_refine` (D-104). The middle sentences restate T0's ALCE
   instruction (D-118), as D-104 restates D-016's "one sentence": its four citation sentences
   ("Always cite …" to the end) word for word, and its first two sentences shortened to one
   ("Write an accurate, engaging, and concise answer in an unbiased and journalistic tone"), whose
   "using only the provided search results" the second sentence already states.
4. **Layout:** as D-104 and D-110. Worked examples are sent as chat turns, and the item under work
   is labelled "Now check the answer to the question below." (feedback) and "Now rewrite the latest
   answer to the question below." (refine). `Question:` and the five passages in ALCE's format
   (D-117) take the place of `Article:`, and `Answer` takes the place of `Summary`.
5. **Draft prompt unchanged.** T1's draft is generated with `asqa_alce_vanilla@v1` (D-118), the
   same template object T0 uses, with the same two worked examples drawn from the run seed (D-101).
6. **Worked examples.** There are three, taken from ALCE's four ASQA examples (D-118: train split,
   none in the evaluation sample), in the order D-103 fixes. The feedback prompt shows the first and
   third as draft with feedback, and the second as its correct answer with `NONE`. The refine prompt
   shows all three as draft, feedback, then revision. Each draft carries two errors, each from this
   list: a wrong citation number, a missing citation, or a fact that no passage states. The
   questions, passages and the ALCE answers used as revisions are ALCE's text, unchanged. The
   drafts, the feedback and the field-goal revision were written for this project.

   | Order | ALCE question (train `sample_id`) | Draft errors | Revision |
   | --- | --- | --- | --- |
   | 1 | "Who played galen in planet of the apes?" (`-8254578338731748300`) | wrong citation ([3] for [2]); missing citation (sentence 2) | ALCE's answer |
   | 2 | "When did the us break away from england?" (`-3207884824372475669`) | fact no passage states (1777 for 1776); missing citation (sentence 2) | ALCE's answer; the feedback prompt shows it with `NONE` |
   | 3 | "Who set the record for longest field goal?" (`3219470493874132834`) | fact no passage states (65 for 64 yards); wrong citation ([4] for [2]) | written for this project |

   - Example 1 draft: "In the 1968 film Planet of the Apes, Galen was played by Wright King [3].
     In the 1974 television series, Galen was played by Roddy McDowall." Feedback: "- "Galen was
     played by Wright King [3]": search result [3] does not mention Galen or Wright King; search
     result [2] supports it." and "- "In the 1974 television series, Galen was played by Roddy
     McDowall.": this sentence cites no search result; search result [1] supports it."
   - Example 2 draft: "The United States declared independence from Great Britain on July 2, 1777
     [2]. The Treaty of Paris was later signed on September 3, 1783, formally separating the United
     States from the British Empire." Feedback: "- "July 2, 1777": no search result says this;
     search result [2] states that the United States declared independence on July 2, 1776." and
     "- "The Treaty of Paris was later signed on September 3, 1783, formally separating the United
     States from the British Empire.": this sentence cites no search result; search result [3]
     supports it."
   - Example 3 draft: "Matt Prater set the NFL record for the longest field goal, 65 yards, on
     December 8, 2013 [1]. The longest field goal in recorded football history was 69 yards, kicked
     by collegiate kicker Ove Johansson in a 1976 game for Abilene Christian University against East
     Texas State University [4]." Feedback: "- "65 yards": no search result says this; search
     result [1] states that Matt Prater's record kick was 64 yards." and "- "69 yards, kicked by
     collegiate kicker Ove Johansson in a 1976 game for Abilene Christian University against East
     Texas State University [4]": search result [4] does not mention Ove Johansson; search result [2]
     supports it." Revision: "Matt Prater set the NFL record for the longest field goal, 64 yards, on
     December 8, 2013 [1]. The longest field goal in recorded football history was 69 yards, kicked
     by collegiate kicker Ove Johansson in a 1976 game for Abilene Christian University against East
     Texas State University [2]."
7. **Check of the examples.** Every draft and revision was scored with the project's citation
   metric (D-120: ALCE's citation recall, MiniCheck judge, threshold 0.5) on CPU at commit
   `67a0ace`. All four ALCE answers were also scored, to choose among them.

   | Text | Citation recall | MiniCheck per cited sentence |
   | --- | --- | --- |
   | Example 1 draft | 0.00 | 0.025; no citation |
   | Example 1 revision (ALCE) | 1.00 | 0.972, 0.976 |
   | Example 2 draft | 0.00 | 0.050; no citation |
   | Example 2 revision (ALCE), the `NONE` case | 1.00 | 0.909, 0.982 |
   | Example 3 draft | 0.00 | 0.060, 0.016 |
   | Example 3 revision (written here) | 1.00 | 0.972, 0.871 |
   | ALCE's field-goal answer, not used | 0.00 | 0.482 (one sentence citing [1][2]) |
   | ALCE's rainiest-place answer, not used | 0.50 | 0.196; 0.897 |

   Every revision and the `NONE` answer pass the judge, and every injected error fails it. ALCE's
   rainiest-place and field-goal answers do not pass, so neither is shown as a correct answer; the
   rainiest-place question is left out.

**How T1 compares with T0 on ASQA:**

- *Same draft.* The draft uses T0's template object and seeded example draw, so T1's first answer
  is produced exactly as T0's is. It need not match T0's text token for token, because sampling
  under vLLM is not guaranteed to repeat across runs. How often the two match is measured before the
  full runs.
- *Same rules for the rewrite.* The refine prompt restates T0's instruction, so a rewritten answer
  is held to the same task and citation rules as T0's answer. A difference from T0 then comes from
  the feedback-and-rewrite loop, not from different instructions.
- *Same decoding, backbones, seeds, 500 questions and metrics* as D-130 (D-011, D-119, D-120,
  D-122).
- *What differs and is reported:* T1 makes up to 10 model calls per question (D-101, D-102), so its
  token cost, latency and mean output length are reported next to T0's. Feedback limited to support
  may lead the model to delete claims, which would lower `str_em`. So `str_em` is also reported for
  each question's draft and final answer, taken from `trace.jsonl`.

**Taken from Madaan et al. (2023) and the project's own choice.** Everything D-107 lists as taken
from the paper applies unchanged: the loop and its order, the feedback and refine inputs, the last
answer as output, the `NONE` stop signal, at most 4 rounds, three few-shot examples, and specific,
actionable feedback. Self-Refine was not evaluated on question answering. Its seven tasks are
sentiment reversal, dialogue response, code optimization, code readability, math reasoning, acronym
generation and constrained generation (§3). So there is no published Self-Refine prompt for ASQA to
reuse. The project's own choices are:

| Element | T1 on ASQA | Basis |
| --- | --- | --- |
| Task and data | ASQA with ALCE's passages | not among the paper's seven tasks; D-116 |
| Feedback scope | support by the cited passages only | D-104's single aspect, with ALCE's definition of support for a cited sentence (Gao et al. 2023, §3.3) |
| Instruction wording | D-104's sentences with ALCE's citation rules | D-104, D-118 |
| Worked examples | ALCE's ASQA examples; drafts, feedback and one revision written here | D-103's method; D-118's pool |
| Example check | MiniCheck citation recall: revisions 1.00, drafts fail | this entry |

**Reason:**

- *Scope:* one factual-support aspect keeps T1 the same intervention on XSum and ASQA (D-104).
  "Supported by the passages it cites" is how T0's own instruction defines a good answer ("Always
  cite for any factual claim"), and it is ALCE's definition of citation recall (§3.3).
- *Scope, what was rejected:* adding completeness would push answers to grow longer. `str_em` rises
  with length (D-119, D-120), so a gain could not be told apart from length. It would also add a
  second aspect for 7–8B models, which Self-Refine reports struggle to keep feedback in the required
  format (§4, Vicuna-13B). Checking support against all five passages, ignoring citations, was also
  rejected: it misses the uncited and miscited sentences that T0's instruction forbids.
- *Wording:* every sentence comes from, or shortens, a prompt the project has already approved
  (D-104, D-118), so the only new choices are where the paper and T0 give none.
- *Examples:* scoring them with the evaluation's own judge means the examples never teach the model
  that a revision the metric rejects is correct.
- *Caveat:* the headline MiniCheck score removes citation markers (D-120). A rewrite that only fixes
  a citation therefore moves citation recall and precision, which are secondary, but not the
  headline.

**Not decided here:** the leak check for the new examples (it gets its own metric key, so T0's
`asqa_worked_example_overlap` keeps the definition D-128 gave it), the pilot, and the T2 ASQA
critic, whose scope should match this one.

**Source:** Madaan et al. (2023), arXiv 2303.17651v2, §2, §3, §3.1, §4; Gao et al. 2023, §3.3,
App. C; ALCE repository commit `246c476` (`prompts/asqa_default.json`); D-101, D-103, D-104, D-107,
D-110, D-117, D-118, D-119, D-120, D-122, D-128, D-130; MiniCheck check of the examples, 2026-10-26
**Approved:** Sameer, 2026-10-26
**Config:** `src/component/factagent_bench/agents/self_feedback.py` (`ASQA_SELF_FEEDBACK`),
`src/component/factagent_bench/agents/generator.py` (`ASQA_SELF_REFINE`),
`src/component/factagent_bench/agents/self_refine_examples.py` (`ASQA_SELF_REFINE_EXAMPLES`)
**Supersedes:** —

---

### D-151 · 2026-10-26 · ASQA T1 leak check: `asqa_self_refine_example_overlap`, key `self_refine_example_overlap`, against the text written for T1's ASQA examples

**Decision:** a metric registered as `asqa_self_refine_example_overlap`, emitting
`self_refine_example_overlap`, is added to the ASQA suite after `asqa_worked_example_overlap`. It
is D-110's leak check (1.0 when an output shares a sequence of `metrics.worked_example_overlap.ngram`
words with a worked example and that sequence is not in the output's own passages; otherwise 0.0;
the run mean is the leak rate). The texts checked are those written for D-150's examples: each
draft, each feedback, and the field-goal revision, with citation markers removed. The ALCE
questions, passages and answers stay with `asqa_worked_example_overlap` (D-128). The word-sequence
length is the existing 8 (D-110), unchanged.

**Reason:**
- *Why a check is needed.* D-109 found Llama and Mistral copying T1's XSum worked examples into a
  large share of outputs, which is why D-110 scores a leak check on every run. T1's ASQA prompts
  show text that T0's do not (the drafts, feedback and one revision), and D-128's check does not
  cover it.
- *Why a separate key.* T0's ASQA leak rate (0.0 in every D-130 run) was computed against ALCE's
  four examples. Adding T1's texts to that check would change what `worked_example_overlap` means on
  ASQA between T0 and T1, and AGENTS.md §4.5 keeps a key's meaning fixed once a run has used it.
- *Overlap between the two checks.* The drafts were written from ALCE's answers, so some phrases
  occur in both (for example, "Planet of the Apes, Galen was played by"). An output copying such a
  phrase is flagged by both checks. This is expected and is covered by a test.
- *On T0 runs.* T0's prompts never show these texts, so on T0 the metric measures chance overlap,
  a baseline for T1's value. The nine D-130 runs predate the metric and are not re-scored.

**Source:** D-109, D-110, D-128, D-130, D-150; AGENTS.md §4.5
**Approved:** Sameer, 2026-10-26
**Config:** `src/configs/data/tier2_asqa.yaml` (`metrics.suite`);
`src/component/factagent_bench/metrics/worked_example_overlap.py` (`ASQASelfRefineExampleOverlap`);
`src/configs/base.yaml` (`metrics.worked_example_overlap.ngram`)
**Supersedes:** —

---

### D-152 · 2026-10-26 · ASQA T1 pilot: the pipeline works on all three backbones; the nine runs go ahead with D-150's prompts and D-101's stop rule unchanged

**Decision:** the nine T1 ASQA runs (3 backbones × seeds 0–2, D-122's 500 questions) are run with
`asqa_self_feedback@v1` and `asqa_self_refine@v1` as D-150 sets them, and with D-101's stop rule
unchanged: the loop stops only when a feedback reply, stripped and with one trailing full stop
removed, is the word `NONE`.

**Pilot.** One run per backbone at seed 0 on 30 of the 500 questions, spread evenly over the frozen
sample so all three strata appear, with the full ASQA metric suite, at commit `a3b1dac`
(`git.dirty: false`). The pilot is a check of the pipeline. It writes to the git-ignored
`instance/pilot/` and is not a reported result. Run IDs: `20261026_174840_…_qwen2.5-7b-instruct_seed0`,
`20261026_175212_…_mistral-7b-instruct-v0.3_seed0`, `20261026_180154_…_llama-3.1-8b-instruct_seed0`
(`…` = `t1_self_refine_tier2_asqa`). The T0 runs compared are the seed-0 runs of D-130.

| | Qwen2.5-7B | Mistral-7B-v0.3 | Llama-3.1-8B |
| --- | --- | --- | --- |
| Leaks, `worked_example_overlap` / `self_refine_example_overlap` | 0 / 0 of 30 | 0 / 0 of 30 | 0 / 0 of 30 |
| T1 draft identical to T0's answer | 30 / 30 | 19 / 30 | 30 / 30 |
| Questions stopped with `NONE` | 24 / 30 | 19 / 30 | 0 / 30 |
| Answers written per question (count of questions) | 1: 24, 5: 6 | 1: 8, 2: 6, 3: 4, 5: 12 | 5: 30 |
| Non-stop feedback replies that contain `NONE` or say every sentence is supported | 1 of 30 | 12 of 73 | 0 of 150 |
| Calls cut at the 300-token cap (feedback, generator) | 0, 0 | 11, 3 | 0, 0 |
| Largest prompt, tokens (generator, feedback) | 4,800, 4,018 | 6,484, 4,423 | 5,112, 3,845 |
| `str_em`, T1 draft → T1 final | 0.331 → 0.331 | 0.406 → 0.364 | 0.337 → 0.314 |
| Generation time per question | 3.06 s | 10.33 s | 13.79 s |

The "contain `NONE` or say every sentence is supported" count is a pattern match on the reply text
(the word `NONE` anywhere, or "all"/"every" followed by "supported"/"correct" in a reply without a
bullet); it is a diagnostic, not a metric. Examples from Mistral: "All sentences are supported by the
given search results. NONE", "NONE (every sentence is supported by the cited search results)", and
bullets ending in a line "- NONE".

**Findings:**
- The pipeline works: no leaks, no blank outputs, every prompt within the 32,768-token context.
- For Qwen and Llama the T1 draft is T0's answer token for token, so their T1-vs-T0 comparison
  differs only by the feedback-and-rewrite loop. For Mistral, 11 of 30 drafts differ from T0's
  answer. The cause is not verified; Mistral samples at temperature 1.0 with no top-k (D-011,
  `model_registry.csv`), which makes small numerical differences between runs more likely to change
  the text. Mistral's loop effect is therefore also reported as T1 final against T1 draft from the
  same run, as D-150 planned.
- Llama never returns the stop word on these 30 questions, as on XSum (D-109: 5 / 4 / 5 of 500).
- Mistral often signals "no problems" in a form the stop rule does not accept, so the loop continues
  and rewrites answers the feedback did not fault.

**Reason:**
- *Prompts unchanged.* No mechanical failure was found. Changing the wording because of the scores
  on evaluation questions would tune the prompt on the test set. D-109's prompt change differs: it
  fixed a mechanical failure (copied worked examples), not a score.
- *Stop rule unchanged.* It is D-101's rule, the rule the XSum T1 results use. Loosening it for
  ASQA would make T1 a different procedure on the two tiers. Mistral's non-conforming stop replies
  are reported as a finding, with the share of feedback replies affected in the full runs.

**Time estimate** (written to `run_matrix.csv`): per run, the pilot's generation time per question
× 500 plus the scoring and start-up time of the same backbone's seed-0 T0 run (D-130): Qwen 0.63 h,
Mistral 1.73 h, Llama 2.14 h; 13.5 GPU-hours for the nine runs. Rough: Mistral's and Llama's time
per question depends on the number of rounds.

**Source:** the three pilot runs' `summary.json`, `manifest.json`, `metrics.csv` and `trace.jsonl`;
`instance/pilot/asqa_t1_pilot.py` (report); D-101, D-109, D-122, D-130, D-150, D-151
**Approved:** Sameer, 2026-10-26
**Config:** none changed; `src/run_matrix.csv` (GPU-hour estimates)
**Supersedes:** —

---

### D-153 · 2026-10-27 · ASQA T1 results: nine runs; no gain on Qwen, a loss on Mistral, a gain on Llama; two Mistral outputs copy example text

**Decision:** the nine runs below are T1's ASQA results (3 backbones × seeds 0–2), under D-150's
prompts and D-101's stop rule, unchanged from the D-152 pilot. All nine have status `success`, 500
rows, no blank outputs, and `git.dirty: false` from commit `6224d4f` on branch `tier2-asqa-t1`.
Each is compared with the T0 run of the same backbone and seed.

| Run ID | Backbone | Seed | Paired T0 run | Wall-clock |
| --- | --- | --- | --- | --- |
| `20261026_182356_t1_self_refine_tier2_asqa_qwen2.5-7b-instruct_seed0` | Qwen2.5-7B | 0 | `20261023_144555` | 0.64 h |
| `20261026_190239_t1_self_refine_tier2_asqa_qwen2.5-7b-instruct_seed1` | Qwen2.5-7B | 1 | `20261023_153956` | 0.65 h |
| `20261026_194132_t1_self_refine_tier2_asqa_qwen2.5-7b-instruct_seed2` | Qwen2.5-7B | 2 | `20261023_163758` | 0.61 h |
| `20261026_201800_t1_self_refine_tier2_asqa_mistral-7b-instruct-v0.3_seed0` | Mistral-7B-v0.3 | 0 | `20261023_150629` | 1.81 h |
| `20261026_220654_t1_self_refine_tier2_asqa_mistral-7b-instruct-v0.3_seed1` | Mistral-7B-v0.3 | 1 | `20261023_160249` | 1.99 h |
| `20261027_000618_t1_self_refine_tier2_asqa_mistral-7b-instruct-v0.3_seed2` | Mistral-7B-v0.3 | 2 | `20261023_165920` | 1.83 h |
| `20261027_015610_t1_self_refine_tier2_asqa_llama-3.1-8b-instruct_seed0` | Llama-3.1-8B | 0 | `20261026_132201` | 2.33 h |
| `20261027_041612_t1_self_refine_tier2_asqa_llama-3.1-8b-instruct_seed1` | Llama-3.1-8B | 1 | `20261026_134546` | 2.42 h |
| `20261027_064127_t1_self_refine_tier2_asqa_llama-3.1-8b-instruct_seed2` | Llama-3.1-8B | 2 | `20261026_140655` | 2.26 h |

Total 14.5 h against the 13.5 h estimated in D-152.

**Results (D-115: seeds averaged per question, paired difference against T0, 95% percentile
bootstrap over the 500 questions, 10,000 resamples, seed 123):**

| Backbone | Metric | T0 | T1 | Paired gain | 95% CI |
| --- | --- | --- | --- | --- | --- |
| Qwen | MiniCheck | 0.822 | 0.826 | +0.0041 | −0.0037, +0.0119 |
| Qwen | AlignScore | 0.817 | 0.818 | +0.0011 | −0.0056, +0.0078 |
| Qwen | `str_em` | 0.358 | 0.352 | −0.0054 | −0.0094, −0.0016 |
| Qwen | citation recall | 0.676 | 0.682 | +0.0060 | −0.0025, +0.0148 |
| Qwen | citation precision | 0.617 | 0.622 | +0.0049 | −0.0043, +0.0142 |
| Mistral | MiniCheck | 0.731 | 0.700 | −0.0313 | −0.0420, −0.0212 |
| Mistral | AlignScore | 0.759 | 0.721 | −0.0373 | −0.0471, −0.0278 |
| Mistral | `str_em` | 0.389 | 0.373 | −0.0160 | −0.0245, −0.0074 |
| Mistral | citation recall | 0.519 | 0.449 | −0.0702 | −0.0867, −0.0545 |
| Mistral | citation precision | 0.497 | 0.447 | −0.0498 | −0.0665, −0.0336 |
| Llama | MiniCheck | 0.768 | 0.803 | +0.0352 | +0.0209, +0.0493 |
| Llama | AlignScore | 0.786 | 0.818 | +0.0319 | +0.0195, +0.0445 |
| Llama | `str_em` | 0.388 | 0.376 | −0.0124 | −0.0214, −0.0036 |
| Llama | citation recall | 0.578 | 0.604 | +0.0268 | +0.0040, +0.0493 |
| Llama | citation precision | 0.602 | 0.614 | +0.0124 | −0.0110, +0.0360 |

Cost, mean over the three seeds:

| Backbone | Tokens per question, T0 → T1 | Ratio | Generation time per run, T0 → T1 |
| --- | --- | --- | --- |
| Qwen | 2,905 → 12,928 | 4.5× | 521 s → 1,636 s |
| Mistral | 3,160 → 29,043 | 9.2× | 791 s → 5,722 s |
| Llama | 2,814 → 40,416 | 14.4× | 610 s → 7,746 s |

**The loop, by backbone (per seed, or pooled over the three seeds):**

| | Qwen | Mistral | Llama |
| --- | --- | --- | --- |
| Questions ending with `NONE` (500 per seed) | 418 / 423 / 428 | 340 / 299 / 292 | 3 / 0 / 8 |
| Questions that stop after the first answer | 388 / 386 / 389 | 140 / 132 / 148 | 1 / 0 / 3 |
| Questions that reach the 4-round cap (5 answers written) | 82 / 78 / 73 | 179 / 230 / 233 | 498 / 500 / 494 |
| Feedback replies, total | 2,564 | 4,849 | 7,472 |
| … `NONE` | 1,269 | 931 | 11 |
| … well-formed bullets | 1,203 | 1,716 | 7,447 |
| … `NONE` followed by more text | — | 383 | — |
| … other prose | 82 | 1,527 | 9 |
| … cut at the 300-token cap | 10 | 292 | 5 |
| Calls cut at the cap: feedback / first draft / refine | 22 / 0 / 0 | 595 / 5 / 200 | 67 / 0 / 6 |
| Final answers cut at the cap (T0) | 0.0% (0.0%) | 5.4–6.0% (0.0–0.6%) | 0.0–0.4% (0.0%) |

`str_em` of the first draft against the final answer, pooled over 1,500 answers per backbone:

| Backbone | Draft | Final | Final lower on | Final higher on |
| --- | --- | --- | --- | --- |
| Qwen | 0.358 | 0.352 | 33 | 12 |
| Mistral | 0.386 | 0.373 | 98 | 45 |
| Llama | 0.389 | 0.376 | 138 | 85 |

MiniCheck is computed on final answers only; the drafts are not scored by it, so there is no T1-draft
MiniCheck to compare with.

**Leak checks (D-110, D-128, D-151):** 0 flagged answers in eight of the nine runs. In
`20261026_201800_…mistral…_seed0` one final answer is flagged by `worked_example_overlap` (8-word
overlap with an ALCE example) and two by `self_refine_example_overlap`:
- question `-3176814404414677007` (field goal): the answer repeats the sentences written for the
  field-goal worked example (MiniCheck 0.04);
- question `-2934971361413993618`: the final output contains the checker's feedback text and a
  line "Answer 3:", i.e. the model reproduced the loop's turn format (MiniCheck 0.61).

That is 1 and 2 of the 1,500 Mistral answers, in one run. These answers stay in the means above;
nothing was filtered.

**Findings:**
- **Qwen:** no detectable gain on MiniCheck, AlignScore or citation scores; `str_em` falls by 0.005.
  The checker answers `NONE` on most questions (about 77% of questions stop after the first answer),
  so the loop rarely changes the draft.
- **Mistral:** T1 is below T0 on every metric, with intervals that exclude 0. 45% of its feedback
  replies are not `NONE` or `- "…"` bullets (pilot: 12 of 73). 5–6% of final answers hit the
  300-token cap, against at most 0.6% for T0. The strict stop rule (D-152) leaves many questions
  running to the cap; each rewrite lowers citation recall.
- **Llama:** T1 is above T0 on MiniCheck, AlignScore and citation recall. Llama writes `NONE` on 11 of
  1,500 questions, so the gain comes from four forced rounds, not from the checker clearing
  answers; the checker's well-formed bullets (99.7% of replies) mean it always reports a fault. The
  cost is 14.4× the tokens.
- **`str_em`** is lower after the loop on all three backbones (final below draft on more answers than
  above it), which fits rewrites that remove claims the cited passages do not support.
- **Against XSum (D-113):** Qwen +0.02 on XSum and none on ASQA; Mistral flat on XSum and −0.031 on
  ASQA; Llama +0.04 on XSum and +0.035 on ASQA. Token ratio against T0 on XSum is 5.9× / 14.7× /
  23.3× (Qwen / Mistral / Llama), against 4.5× / 9.2× / 14.4× on ASQA. The XSum T0 runs used for that
  comparison are the 13–14 September runs, not re-checked against the current T0 setup.
- **Not established:** why Mistral's replies deviate more on ASQA than on XSum; the cause was not
  traced answer by answer. Llama's gain is not attributed to error-finding.

**Reason:** these runs follow D-152's decision to run the nine without changing the prompts or the
stop rule; the results are reported as they came out, including the Mistral loss and the two
flagged answers.

**Source:** the nine runs' `summary.json`, `manifest.json`, `metrics.csv` and `trace.jsonl`; the
paired T0 runs listed above; D-101, D-110, D-113, D-115, D-128, D-150, D-151, D-152
**Approved:** Sameer, 2026-10-29
**Config:** none changed; `src/run_matrix.csv` (run IDs and statuses for the nine rows)
**Supersedes:** —

---

### D-155 · 2026-10-30 · Tier 3 is evaluation only; RAGTruth is the generation test; HaluEval is used only to check the judges

**Decision:**
1. **No training on Tier 3.** Tier 3 is evaluated, never trained or tuned on, under any condition.
   The existing T5 adapters (D-114, D-141, D-142) are evaluated on it; no new fine-tune is run.
2. **RAGTruth is the generation test.** Each topology generates new outputs for RAGTruth's source
   items (QA, Data2txt and Summary), and the outputs are scored against each item's source (D-158 to
   D-162).
3. **HaluEval checks the judges only.** It is used in its own recognition protocol, to measure how
   well MiniCheck and AlignScore tell its right outputs from its hallucinated ones (D-163). No
   topology generates on HaluEval.

This departs from the proposal's wording, which lists both datasets as the stress test for generated
text (proposal §2, Tier 3).

**Reason:**
- *Evaluation only.* Proposal §2: "Tier 3 — Held-Out Hallucination Stress Test (evaluation only —
  never tuned or trained on)"; "Tier 3 is never trained on, under any condition — it is the primary
  instrument for detecting reward overfitting." D-047 restates it. It is enforced in code:
  `DatasetAdapter.assert_trainable` raises "Tier 3 is never trainable" before any model loads
  (`datasets/base.py`), tested in `test_datasets.py` and `test_trainer.py`.
- *HaluEval is a recognition benchmark.* Li et al. 2023 build it to test whether a model can
  *recognize* hallucinations: "we randomly select the hallucinated or normal output … Table 5
  presents the accuracy of evaluated LLMs to classify whether the sample output contains hallucinated
  information" (§3.2.1). The hallucinated outputs were written by ChatGPT from instructions (§2.1).
  The paper defines no train/test split. Generating new outputs on HaluEval is a protocol we found no
  established precedent for.
- *HaluEval's errors are synthetic.* Tang et al. 2024 left HaluEval out of LLM-AggreFact "since they
  are synthetic, with errors in summaries generated via instruction prompts", errors they call
  unnatural for detecting real generation errors (App. C.3). As a check of our judges, its recognition
  protocol is still a published, reproducible test, so it is kept for that role.
- *RAGTruth fits generation.* It has an official test split, releases the prompt used for every
  source item, and labels the responses of six models at span level (Niu et al. 2024, §3, §5.2,
  App. B). RAG-HAT (Song et al. 2024, §5.2, Table 3) regenerates on the RAGTruth test set (450
  examples) and reports the hallucination rate of the new outputs; FaithJudge (Tamber et al. 2025, §7)
  scores new models on "up to 150 sources" per RAGTruth task.
- *HaluEval 2.0 was considered and not used.* Li et al. 2024 judge answers "according to the world
  knowledge" with no source document (§4.1), so a source-grounded judge such as MiniCheck cannot
  score them.

**Source:** proposal §2 and Week 11; Li et al. 2023 (EMNLP), §2.1, §3.2.1, Table 5; Tang et al. 2024
(MiniCheck), App. C.3; Niu et al. 2024 (RAGTruth, arXiv v2), §3.2, §5.2, App. B; Song et al. 2024
(RAG-HAT, EMNLP Industry), §5.2, Table 3; Tamber et al. 2025 (arXiv 2505.04847v2), §7; Li et al.
2024 (HaluEval 2.0), §4.1; D-047
**Approved:** Sameer, 2026-10-30
**Config:** none yet; Tier 3 data configs are created on the data branch (D-166)
**Supersedes:** — (narrows proposal §2's use of HaluEval)

Numbering note: D-154 is on branch `asqa-analysis`, not yet merged into `main`; Tier 3 entries start
at D-155 so the two branches do not collide.

---

### D-156 · 2026-10-30 · Tier 3 conditions: T0 and T5 (both adapters) on all three backbones, seeds 0–2; no "best configuration" selection; T1 and T2 later

**Decision:**
- **Topologies in this phase:** T0 and T5. T1 and T2 are added later through the same code path,
  after their RAGTruth prompts are decided.
- **Backbones:** all three (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct).
  No backbone or configuration is selected per topology.
- **T5 adapters:** both, on all three RAGTruth tasks. These are the XSum adapters (`checkpoint-625`,
  D-111, D-114) and the ASQA adapters (`final`, D-141, D-142).
- **Seeds:** 0, 1, 2 (D-003, D-007).
- **Run count:** 81 generation runs of 150 items each: T0 3 tasks × 3 backbones × 3 seeds = 27, and
  T5 2 adapters × 27 = 54. The GPU-hour estimate is made from the pilot, before the `run_matrix.csv`
  rows are written.

This departs from the proposal's Week 11 wording, "Evaluate the best-performing configuration per
topology".

**Reason:**
- *No selection.* Choosing a "best" configuration requires a selection rule, and choosing one after
  seeing results selects on them. Running every backbone needs no rule. It also keeps the same
  paired comparison against T0 per backbone as Tier 1 and 2 (D-115).
- *T0 and T5 first.* The proposal gives Tier 3 "extra weight for T5", which "inherit[s] the risk of
  overfitting to the reward metric" (Week 11). T0 and T5 need only the generation prompt, which
  RAGTruth supplies (D-159). T1 and T2 need new feedback and critique prompts for each RAGTruth task,
  each a separate decision. Train-plus-orchestrate combinations stay future work (D-047).
- *Both adapters on every task.* This avoids a hand-made matching of adapter to task. Summary is
  near-domain for the XSum adapter and QA for the ASQA adapter; Data2txt is out-of-domain for both.

**Source:** proposal Week 11; D-003, D-007, D-047, D-111, D-114, D-115, D-141, D-142
**Approved:** Sameer, 2026-10-30
**Config:** none yet (`run_matrix.csv` rows after the pilot)
**Supersedes:** — (departs from proposal Week 11's "best-performing configuration")

---

### D-157 · 2026-10-30 · RAGTruth and HaluEval data: pinned commits and SHA-256 checks; RAGTruth matches its paper's counts

**Decision:** Both datasets are read from their authors' GitHub repositories at fixed commits, as
`data.source_files` entries checked against SHA-256 on every load (the D-132 method).

| Dataset | Repository @ commit (date) | File | SHA-256 |
| --- | --- | --- | --- |
| RAGTruth | `ParticleMedia/RAGTruth` @ `c103204b9ce28d6bbad859304bf30de72b8ed8fe` (2024-12-02) | `dataset/response.jsonl` | `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073` |
| | | `dataset/source_info.jsonl` | `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b` |
| HaluEval | `RUCAIBox/HaluEval` @ `b7253db3cdaa0ab2c382f92b26b390109174f77e` (2024-02-12) | `data/qa_data.json` | `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88` |
| | | `data/summarization_data.json` | `86d2561eeb4271eb50b6bd90c9fd38636d30eaf4e98b648045f221adb1c2c758` |

Both commits were the head of `main` when downloaded (2026-10-30). Files are fetched from
`raw.githubusercontent.com/<repository>/<commit>/<file>`. Both repositories are MIT-licensed.

**Checked on the downloaded files (2026-10-30):**
- **RAGTruth sources:** 2,965 sources, all IDs unique. QA 989 (MS MARCO), Data2txt 1,033 (Yelp),
  Summary 943 (628 CNN/DM, 315 Recent News), as in Niu et al. 2024, Table 2.
- **RAGTruth responses:** 17,790, six per source, one from each of `gpt-4-0613`,
  `gpt-3.5-turbo-0613`, `mistral-7B-instruct`, `llama-2-7b-chat`, `llama-2-13b-chat` and
  `llama-2-70b-chat`.
- **RAGTruth splits:** the split is stored per response; no source has responses in two splits.
- **RAGTruth sampling and quality flags:** temperatures 0.7, 0.775, 0.85, 0.925 and 1.0 (field
  `temperature`); `incorrect_refusal` on 144 and `truncated` on 29 of the 17,790 responses.
- **HaluEval:** each file is JSON lines with 10,000 rows. QA rows carry `knowledge`, `question`,
  `right_answer` and `hallucinated_answer`; summarization rows carry `document`, `right_summary` and
  `hallucinated_summary`. In no row is the right output identical to the hallucinated one.

**Reason:** a pinned commit and a checksum make the exact data reproducible and refuse a changed
file, as for ASQA's training data (D-132).

**Source:** the two repositories at the commits above; Niu et al. 2024, Table 2; Li et al. 2023,
§2.3; checks run 2026-10-30
**Approved:** Sameer, 2026-10-30
**Config:** `src/configs/data/tier3_ragtruth.yaml` (to be created); HaluEval pins in the metric
validation config (D-163)
**Supersedes:** —

---

### D-158 · 2026-10-30 · RAGTruth sample: all 150 test sources per task, as three datasets analysed separately

**Decision:**
- **Datasets:** `tier3_ragtruth_qa`, `tier3_ragtruth_data2txt` and `tier3_ragtruth_summary`, each
  analysed on its own.
- **Items:** every source in RAGTruth's test split, 150 per task. The Summary test set is 100 CNN/DM
  articles and 50 Recent News articles.
- **No sampling:** this is a census, so no sampling seed or strata apply (`n_strata: 1`). The 150 IDs
  are still frozen in a sample file, as for every dataset.
- **Split of a source:** taken from its responses, which must all carry the same split; the loader
  refuses a source whose responses disagree.

**Reason:**
- *The official split.* Niu et al. 2024 define it: "450 instances in total, derived by randomly
  selecting 150 instances from each task type" (§5.2). RAG-HAT evaluates on it (§5.2, Table 3), and
  FaithJudge takes "up to 150 sources" per task, "primarily from the test set" (§7). Using it whole
  leaves no sampling choice to defend.
- *Per task.* RAGTruth reports results per task (Table 3), and the task types differ in kind.
  Godbole & Jia 2025 split RAGTruth's LLM-AggreFact examples into its four subsets "since they have
  qualitatively different inputs and task types" (§2.1), and find that "the magnitude of
  quantification error varies widely across 4 subsets of RAGTruth" (§3.3).
- *Smaller than 500.* D-002's 500 items are not available in the test split. Topping up from the
  train split would depart from the precedent above. The cost is wider confidence intervals, which
  are reported as they are.

**Source:** Niu et al. 2024, §5.2, Table 3; Song et al. 2024, §5.2; Tamber et al. 2025, §7; Godbole
& Jia 2025 (Findings of ACL), §2.1, §3.3; counts checked 2026-10-30 (D-157)
**Approved:** Sameer, 2026-10-30
**Config:** `src/configs/data/tier3_ragtruth_{qa,data2txt,summary}.yaml` and their `_sample.yaml`
files (to be created)
**Supersedes:** — (D-002's 500 does not apply to Tier 3)

---

### D-159 · 2026-10-30 · RAGTruth prompt: the released prompt, verbatim, as one user turn; the project's decoding

**Decision:**
- **Prompt:** the generator sends each source's RAGTruth `prompt` field unchanged, as one user turn
  through the backbone's chat template. There is no system message and there are no worked examples.
- **Template:** one template serves all three tasks, label `ragtruth_released_prompt@v1`.
- **Decoding:** D-011 (each backbone's shipped defaults; only the seed and `max_tokens` are sent),
  not RAGTruth's sampling temperatures.

**Reason:**
- *Released wording.* RAGTruth gives the prompt used for each source ("The prompt we used to generate
  responses", repository README; Niu et al. 2024, App. B, Table 9). Using it verbatim leaves no
  wording of our own to defend, and it is the prompt RAGTruth's annotated responses answered.
- *No wrapper to remove.* The README says Llama and Mistral prompts were wrapped in
  `<s>[INST] … [/INST]`. None of the 2,965 released `prompt` fields contains `[INST]` (checked
  2026-10-30), so nothing is stripped. The prompts end with "output:" (QA, Summary) or "Overview:"
  (Data2txt), and are kept so.
- *Decoding.* RAGTruth's responses were sampled at 0.7 to 1.0 (D-157). Our comparisons are within
  our own runs (each topology against T0 on the same backbone), so the project's setting applies, as
  for ASQA (D-121).

**Source:** RAGTruth repository README at `c103204`; Niu et al. 2024, App. B, Table 9; check of the
`prompt` field, 2026-10-30; D-011, D-121
**Approved:** Sameer, 2026-10-30
**Config:** template in `agents/generator.py` (to be added); `src/configs/data/tier3_ragtruth.yaml`
**Supersedes:** —

---

### D-160 · 2026-10-30 · RAGTruth evidence and reference: the source context each annotator judged against; no reference text

**Decision:**
- **Evidence** (what the judges score outputs against):
  - QA: the three passages (`source_info.passages`);
  - Data2txt: the structured business record exactly as it appears in the prompt;
  - Summary: the article (`source_info`).
- **Reference:** the empty string. RAGTruth has no reference outputs, so ROUGE-L and BERTScore are
  not run on Tier 3.
- **Question:** the QA question (`source_info.question`) fills `Example.question`; it is empty for
  the other two tasks.

**Reason:** RAGTruth's annotators marked the spans of each response that contain hallucinated
information, given the context the response was generated from; information not in that context is
labelled a hallucination even if it may be true (Niu et al. 2024, §3.3–3.4). The same text is the
evidence here (AGENTS §4.2:
"Every factuality metric … checks against `evidence`"). Writing the Data2txt record as the prompt
shows it means the model and the judges see the same text.

**Source:** Niu et al. 2024, §3.2–3.4; AGENTS §4.2; RAGTruth `source_info` fields checked 2026-10-30
**Approved:** Sameer, 2026-10-30
**Config:** `src/configs/data/tier3_ragtruth.yaml` (to be created)
**Supersedes:** —

---

### D-161 · 2026-10-30 · RAGTruth completion cap: 512 tokens for all three tasks

**Decision:** RAGTruth generation requests set `max_tokens` to **512** on all three tasks. As D-012
requires, each run reports the share of outputs stopped by the cap.

**Reason:** measured on RAGTruth's own responses to the test sources (900 per task, six models),
counted in each backbone's tokenizer (local copies at the `model_registry.csv` revisions):

| Task | p50 / p99 tokens (Qwen · Mistral · Llama) | Over 256 | Over 300 | Over 400 | Over 512 |
| --- | --- | --- | --- | --- | --- |
| QA | 128 / 356 · 142 / 409 · 126 / 350 | 65 · 106 · 62 | 30 · 56 · 28 | 3 · 13 · 3 | 0 · 1 · 0 |
| Data2txt | 182 / 352 · 202 / 401 · 180 / 349 | 190 · 260 · 176 | 66 · 151 · 56 | 1 · 11 · 1 | 0 · 0 · 0 |
| Summary | 126 / 318 · 139 / 355 · 124 / 318 | 38 · 66 · 36 | 18 · 32 · 16 | 1 · 4 · 1 | 0 · 0 · 0 |

- *Why not 256 or 300* (D-012, D-119): up to 260 of 900 Data2txt responses (29%) would exceed 256,
  and up to 151 (17%) would exceed 300. The Data2txt prompt itself asks for "100 - 200 words", and
  the Summary prompts ask for up to 200 words (word limits on the test sources: 33 to 200, median
  102).
- *Why 512:* across all splits (17,790 responses) at most one response per task and tokenizer
  exceeds 512 (QA: 1 under Mistral; Summary: 1 under each tokenizer).
- *Trade-off:* a higher cap gives verbose outputs more room. The cap share and the mean output length
  are reported (D-012, D-126), so any length effect is visible.
- *Fit:* the longest test prompt is 2,320 tokens (Summary, Mistral tokenizer), well within the
  32,768-token serving context (D-039).

**Source:** lengths measured 2026-10-30 on the D-157 files; RAGTruth prompts (App. B, Table 9);
D-012, D-039, D-119, D-126
**Approved:** Sameer, 2026-10-30
**Config:** `src/configs/data/tier3_ragtruth.yaml` (to be created; overrides `sampling.max_tokens`)
**Supersedes:** — (D-012 and D-119 stand for Tier 1 and ASQA)

---

### D-162 · 2026-10-30 · RAGTruth metrics: MiniCheck headline, AlignScore second; SummaC and FactCC on Summary only; outputs scored in full; QA refusals counted, and the QA headline paired on items both outputs answer

**Decision:**
- **All three tasks:**
  - MiniCheck-FT5 (D-024) is the headline metric (D-115);
  - AlignScore-large (D-023) is the second primary metric, labelled as the T5 reward judge (D-048,
    D-135).
- **Summary only:** SummaC-ZS and SummaC-Conv (D-018) and FactCC (D-025) as secondary metrics.
- **Not run:** ROUGE-L and BERTScore (no reference, D-160).
- **Outputs are scored in full**, all lines included, and each run reports the share of outputs that
  contain a line break, as for ASQA (D-125).
- **QA refusals.** RAGTruth's QA prompt asks the model to reply "Unable to answer based on given
  passages." when the passages lack the answer. Data2txt and Summary prompts have no such reply.
  - **Two counts are reported beside MiniCheck:**
    - `refusal`: the share of outputs that are, as a whole, a refusal;
    - `refusal_mention`: the share of outputs that contain the fallback sentence anywhere.
  - **The rule for `refusal` is fixed after the pilot**, before the 81 runs. Every refusal-like
    output of the three backbones in the pilot is read, and the rule and the counts are recorded in
    the pilot's entry.
  - **QA headline:** MiniCheck's paired gain over T0 (D-115) is computed on the items where
    **neither** output is a refusal under `refusal`, and that number of items is reported with it.
    The paired gain over all 150 items is reported as a secondary figure. AlignScore is reported the
    same way, so that the two judges are compared on the same items (D-165).

**Reason:**
- *Headline unchanged.* One headline metric across tiers is D-115's rule. MiniCheck was not used in
  training, so it is the held-out judge for T5 (D-048).
- *Summarization metrics on Summary only.* SummaC and FactCC were built and validated for
  summarization (D-018, D-022, D-025). They have no validation on QA or data-to-text.
- *Scored in full.* RAGTruth answers are often lists over several lines, and the annotators labelled
  the whole response (Niu et al. 2024, §3.3). Keeping only the first line could leave a heading such
  as "Here are some tips … based on the provided passages:" as the whole output.
- *MiniCheck scores a correct refusal as unsupported.* MiniCheck-FT5 scored the fallback sentence
  against RAGTruth QA passages (12 sources per group, drawn with seed 0; CPU; 2026-10-30):
  - where RAGTruth marks the refusal as warranted (`good`): 0.05–0.19, median 0.10;
  - where it marks it unwarranted (`incorrect_refusal`): 0.07–0.13, median 0.09;
  - a paraphrase, "I am unable to answer the question based on the given passages.", 0.08–0.47.

  All scores are below the 0.5 threshold. Left in the headline, a warranted refusal would count as a
  hallucination, and a model that never refuses would gain. RAGTruth itself treats an unwarranted
  refusal as a quality flag, not a hallucination (`incorrect_refusal`: 128 of the 235 responses that
  are exactly the fallback).
- *Paired on answered items.* Restricting both outputs to the same items keeps the item-by-item
  pairing with T0 (D-115). Refusals do not vanish from the record: `refusal` reports how often each
  topology refuses, and the all-items figure shows the effect of including them. A fixed score for a
  refusal (for example 1.0) was rejected, because a model that always refuses would score perfectly.
- *Why two counts, and why the rule waits for the pilot.* Exact matching and "contains" both fail
  on RAGTruth's own QA responses (989 per model):

  | Model | Exactly the fallback | Fallback, then more text | Fallback inside an answer | Other refusal wording |
  | --- | --- | --- | --- | --- |
  | GPT-4 | 60 | 0 | 0 | 9 |
  | GPT-3.5 | 100 | 0 | 21 | 4 |
  | Mistral-7B | 75 | 0 | 8 | 2 |
  | Llama-2-7B | 0 | 0 | 370 | 11 |
  | Llama-2-13B | 0 | 0 | 29 | 12 |
  | Llama-2-70B | 0 | 67 | 32 | 6 |

  - Normalized: lower-cased, punctuation removed.
  - "Other refusal wording" is a keyword search in the first 200 characters. It is a rough count,
    not a rule.
  - Exact matching misses refusals written in the model's own words (Mistral: "I am unable to answer
    the question based on the given passages.").
  - "Contains" would count Llama-2-7B's 370 answers that carry the fallback sentence inside real
    content.
  - Our own Mistral has already failed fixed-string rules. It wrote "no problems" in its own words
    instead of `NONE` (D-152), and wrote "`NONE` followed by more text" 383 times on ASQA (D-153).
  - It samples at temperature 1.0 with no top-k (D-011).
  - RAGTruth's Mistral is an older version run with other sampling, so the rule is set on our
    backbones' pilot outputs.
- *Mistral's other past errors do not apply here.* Copying worked examples (D-109, D-153) and
  reproducing the loop's turn format (D-153) need worked examples or a T1 loop. Tier 3 T0 and T5 make
  one call with neither (D-159).

**Source:** D-011, D-018, D-022, D-023, D-024, D-025, D-048, D-109, D-115, D-125, D-135, D-152, D-153;
Niu et al. 2024, §3.3, App. B, Table 9; refusal counts and MiniCheck scores measured 2026-10-30 on the
D-157 files (scratchpad scripts, not a results run)
**Approved:** Sameer, 2026-10-30
**Config:** `metrics.suite` in `src/configs/data/tier3_ragtruth*.yaml`; the `refusal` and
`refusal_mention` metrics in `metrics/` (to be created); the answered-items pairing in the Tier 3
analysis (D-166)
**Supersedes:** —

---

### D-163 · 2026-10-30 · Before the first Tier 3 run, MiniCheck and AlignScore are checked on RAGTruth's and HaluEval's human or constructed labels

**Decision:** three checks, run before any Tier 3 generation, with results in their own entry (as
D-124 led to D-129).

1. **LLM-AggreFact, RAGTruth subset.** MiniCheck-FT5 and AlignScore-large are scored on the
   `RAGTruth` rows of the pinned `lytang/LLM-AggreFact` test file (D-129's pin; 16,371 rows, counted
   2026-10-30). Balanced accuracy at threshold 0.5. The LLM-AggreFact leaderboard lists 78.0 for
   MiniCheck-Flan-T5-L in its RAGTruth column (accessed 2026-10-30; the page does not name the
   metric). It is the comparison point, and any gap is reported, as in D-129.
2. **RAGTruth's own labelled test responses, per task.** The 2,700 responses to the 450 test sources
   (six models × 450), scored against their sources:
   - A response counts as hallucinated if **any of its sentences scores below 0.5**. Sentences are
     split as the metric splits them (D-024).
   - The human label is "hallucinated" if the response has at least one annotated span.
   - Balanced accuracy is reported per task: hallucinated responses are 160 of 900 (QA), 579 of 900
     (Data2txt) and 204 of 900 (Summary), counted 2026-10-30.
3. **HaluEval, QA and summarization.** Every item is used, both outputs, giving 20,000 pairs per
   subset.
   - **Summarization:** `document` is paired with `right_summary` (label: supported) and with
     `hallucinated_summary` (label: not supported). A summary counts as unsupported under the same
     any-sentence rule as check 2.
   - **QA:** the claim is **`{question}: {answer}`**, the question and the answer joined by a colon
     and a space. It is checked against `knowledge`, for `right_answer` (supported) and
     `hallucinated_answer` (not supported), as one claim.
   - Accuracy follows Li et al. 2023 (§3.2.1). The classes are exactly balanced, so accuracy equals
     balanced accuracy.
   - **Reported beside the accuracy:**
     - QA: the share of right answers accepted, separately for answers of three words or fewer and
       for longer answers;
     - summarization: the length difference between right and hallucinated summaries.
   - HaluEval dialogue and general are not used. General has no grounding document. Dialogue's
     knowledge is a knowledge-graph string with no counterpart in Tier 1 or 2.

**Reason:**
- *Threshold 0.5.* It is the project's MiniCheck threshold (D-024), taken from Tang et al. 2024:
  "Unless otherwise specified, we set t = 0.5" (§3.3). The package also uses `prob > 0.5`. Their
  LLM-AggreFact results are reported "without per-dataset threshold tuning" (Table 2 caption). At 0.5
  our ports reproduce the published LFQA and ClaimVerify values (D-129). Tuning a threshold on
  RAGTruth's labels and then reporting accuracy on the same labels would overstate the judge.
- *Any sentence.* RAGTruth annotates hallucinations as spans (§3.3) and describes response-level
  detection as "assessing whether a given reference-response pair contains hallucinated content"
  (§5.1). It does not state the rule in so many words; we count a response with at least one
  annotated span as hallucinated, which matches that description. One unsupported part makes the
  response hallucinated, so the matching judge rule is any sentence below the threshold, not the
  average.
- *Balanced accuracy.* The share of hallucinated responses differs widely by task (above). Tang et
  al. 2024 also "evaluate the performance of fact-checkers using balanced accuracy (BAcc)" (§4.1).
- *Per task.* Godbole & Jia 2025 find that factuality metrics "consistently overestimate the system
  error rates on the RAGTruth dataset" and that "the magnitude of quantification error varies widely
  across 4 subsets of RAGTruth" (§3.3), and that the misestimate is larger on small (7B) systems
  (§3.4). A per-task check tells us how far each Tier 3 MiniCheck score can be trusted.
- *HaluEval.* Its recognition protocol is published and reproducible (Li et al. 2023, §3.2.1, Table
  5). It is a check of the judges, not a generation test (D-155). Its synthetic errors (Tang et al.
  2024, App. C.3) mean results there are read as a secondary check.
- *HaluEval QA differs in form between its two answers.* Over all 10,000 rows:
  - right answers have a median of 2 words; 88.6% have three words or fewer, and 124 end with a full
    stop;
  - hallucinated answers have a median of 9 words, and 9,513 end with a full stop;
  - the hallucinated answer is the longer one in 9,579 rows.

  A judge given the answer alone can separate the two partly by form (a bare name against a full
  sentence) rather than by content.
- *Why `{question}: {answer}`.* A diagnostic run of MiniCheck-FT5 on 200 HaluEval QA items (drawn
  with seed 0; both answers; threshold 0.5; GPU; 2026-10-30; not a results run) compared four claim
  formats:

  | Claim | Accuracy | Right accepted | Hallucinated rejected | Right accepted, ≤ 3 words (n = 174) vs longer (n = 26) |
  | --- | --- | --- | --- | --- |
  | answer alone | 0.738 | 0.735 | 0.740 | 0.707 vs 0.923 |
  | `{question}: {answer}` | 0.730 | 0.760 | 0.700 | 0.764 vs 0.731 |
  | `{question}, {answer}` | 0.695 | 0.685 | 0.705 | 0.678 vs 0.731 |
  | `Question: {question} Answer: {answer}` | 0.733 | 0.770 | 0.695 | 0.770 vs 0.769 |

  - With the answer alone, short right answers are accepted less often than longer ones. With the
    colon format the gap disappears, at the same overall accuracy.
  - The comma format is lower on every column.
  - The labelled format performs like the colon format; the difference is within what 200 items can
    resolve. The colon format is the simpler of the two.
  - Only 26 sampled items have longer right answers, so the full run reports the split again (above).
  - Other approaches were considered and not used. Rewriting each question-answer pair as a
    declarative sentence (QA2D; Demszky et al. 2018) is how Chen, Choi & Durrett 2021 check QA
    answers with NLI models. It would add a converter model, with its own errors, to a secondary
    check. Dropping HaluEval QA was also considered; with the colon format it is no longer needed.
- *Summary length.* Hallucinated summaries are longer than right ones in 7,754 of the 10,000 rows
  (median 71 against 47 words). Under the any-sentence rule a longer summary has more chances to
  fail, so the difference is reported with the result.

**Source:** Tang et al. 2024, §3.3, §4.1, Table 2, App. C.3; LLM-AggreFact leaderboard (llm-aggrefact.github.io,
accessed 2026-10-30); Niu et al. 2024, §3.3, §5.1; Godbole & Jia 2025, §3.3–3.4; Li et al. 2023, §3.2.1,
Table 5; Demszky, Guu & Liang 2018 (arXiv 1809.02922); Chen, Choi & Durrett 2021 (Findings of EMNLP);
D-024, D-124, D-129; label counts, HaluEval form statistics and the format diagnostic measured
2026-10-30 on the D-157 files (scratchpad scripts)
**Approved:** Sameer, 2026-10-30
**Config:** `metrics.validation` fields for the RAGTruth and HaluEval checks (to be added)
**Supersedes:** —

---

### D-164 · 2026-10-30 · Overlap checks: no Tier 3 text in either T5 training set; RAGTruth's CNN/DM articles come from the CNN/DM test split; three are shared with our CNN/DM sample and kept

**Decision:**
- **"Never trained on" holds for both T5 adapters:** no RAGTruth test text appears in either T5
  training set.
- **Three shared articles are kept.** They are RAGTruth sources `11991`, `15613` and `15739`, and
  each is also in our CNN/DM evaluation sample (D-035). They stay in Tier 3 and are listed here.

**Method** (D-068): identical text after lower-casing and collapsing whitespace; and the share of a
text's 13-word sequences found in the other set, in both directions.

| Check | Tier 3 texts | Compared with | Identical | ≥ 50% shared 13-grams | Highest share |
| --- | --- | --- | --- | --- | --- |
| RAGTruth Summary test vs the XSum training pool | 150 | 10,000 (`grpo_xsum_pool.yaml`) | 0 | 0 | 0.014 |
| RAGTruth QA test passages vs the ASQA training passages | 450 (3 × 150) | 21,745 (5 × 4,349, `grpo_asqa_pool.yaml`) | 0 | 0 | 0.000 |
| RAGTruth Summary test vs our CNN/DM evaluation sample | 150 | 500 (`tier1_cnndm_sample.yaml`) | 1 | 3 | 1.000 |

**Finding: RAGTruth's CNN/DM articles come from the CNN/DM test split.** Niu et al. 2024 write that
they "randomly selected documents from the training set of the well-known CNN/Daily Mail dataset"
(§3.2, arXiv v2). Our check of `abisee/cnn_dailymail` 3.0.0 at `96df5e6` finds the following.

- **Scope.** Each RAGTruth CNN/DM article was compared with its best match in each CNN/DM split
  (train 287,113, validation 13,368, test 11,490).

| | 100 RAGTruth test articles | 528 RAGTruth train articles |
| --- | --- | --- |
| Best 13-gram share in CNN/DM train | median 0.000, highest 0.193 | median 0.000, highest 0.757 |
| Best 13-gram share in CNN/DM validation | median 0.000, highest 0.524 | median 0.000, highest 0.160 |
| Best 13-gram share in CNN/DM test | median 0.996; ≥ 0.9 for 98 | median 0.996; ≥ 0.9 for 504 |
| Character similarity to the best test match (difflib ratio, leading "(CNN)" removed) | ≥ 0.998 for all 100 | ≥ 0.98 for all 528 |

- **Lower shares.** The two test articles below 0.9 share (0.758, 0.836) differ in spacing and
  punctuation only. Their character similarity is above 0.998.
- **Not a coincidence of wording.** No article has a near-copy in CNN/DM train, and the test split
  is 4% of CNN/DM.
- **Prior reports.** We found no published report of this in a web search on 2026-10-30.
- **How to word it.** In the paper this is reported as what our check finds, with the method.

**Also checked: RAGTruth's own train and test splits.**
- No test source is identical to a train source in any task.
- No QA question and no Data2txt business (name and address) recurs.
- Two Summary test articles share at least 50% of their 13-word sequences with a train article
  (highest 0.649).
- This matters only to systems trained on RAGTruth's train split; none of ours is.

**Reason:**
- *Training sets.* "Never trained on" (D-155) has to hold for both T5 adapters, so both training sets
  were checked.
- *Keeping the three shared articles.* The overlap is between two evaluation sets. No model was
  trained on either, so there is no training exposure. Keeping them keeps Tier 3 as the whole
  official test split (D-158). Removing them would leave Summary with 147 items and depart from the
  precedent in D-158. They mean the CNN/DM transfer results (D-131) and Tier 3 Summary are not fully
  independent; that is stated wherever both are reported.

**Source:** checks run 2026-10-30 on the D-157 files and the pinned XSum (`7d4d486`), CNN/DM
(`96df5e6`) and ASQA training data (D-132); Niu et al. 2024, §3.2; D-035, D-068, D-131
**Approved:** Sameer, 2026-10-30
**Config:** none
**Supersedes:** —

---

### D-165 · 2026-10-30 · Tier 3 read-out for reward exploitation: training-task gain against Tier 3 gain, and the reward judge against the held-out judge; exploratory

**Decision:** for each T5 adapter and backbone, two comparisons are reported. Both are
**exploratory**: no new significance test and no threshold for "exploitation".

1. **Training task against Tier 3.** T5's paired gain over T0 on its training task (XSum, D-114;
   ASQA, D-141, D-142) is set beside its paired gain on each RAGTruth task. Each gain is the D-115
   paired gain with its 95% bootstrap interval.
2. **Reward judge against held-out judge.** On each RAGTruth task, T5's paired AlignScore gain is set
   beside its paired MiniCheck gain.

How the second comparison is read:

| On RAGTruth | Reading |
| --- | --- |
| AlignScore and MiniCheck both rise | consistent with a real gain: the held-out judge agrees |
| AlignScore rises, MiniCheck is flat or falls | consistent with reward exploitation: the model learned the reward judge, not faithfulness |
| Neither rises | the training gain does not carry over to new data |

**Limit for the ASQA adapters.** The second comparison is direct for the XSum adapters, whose
reward was AlignScore on the whole summary (D-048). The ASQA adapters' reward used AlignScore only to
judge citations, alongside exact-match recall (D-133, D-135). RAGTruth outputs carry no citations,
so for the ASQA adapters the second comparison is weaker evidence and is labelled so.

**Reason:**
- *Why compare the two judges.* Optimizing a learned scorer can raise that scorer without raising
  what it stands for. Gao et al. 2023: "Because the reward model is an imperfect proxy, optimizing its
  value too much can hinder ground truth performance" (abstract). AlignScore was T5's reward judge;
  MiniCheck is a separately trained model never used in training (D-048). A gain on AlignScore that
  MiniCheck does not show is the signature named in Q-009.
- *Why Tier 3.* The proposal calls Tier 3 "the primary instrument for detecting reward overfitting"
  (§2). A gain that holds on the training task but not on unseen data is that overfitting.
- *Why exploratory.* The multiple-comparisons correction for the confirmatory set is not yet decided
  (D-115). The two judges also agree at r ≈ 0.7 on ASQA training answers (D-140), so they are not
  independent; a divergence is a signal to examine, not a test result.

**Source:** proposal §2 and Week 11; Gao, Schulman & Hilton 2023 (ICML), abstract; D-048, D-114,
D-115, D-133, D-135, D-140, D-141, D-142; `questions_to_ask.md` Q-009
**Approved:** Sameer, 2026-10-30
**Config:** Tier 3 analysis configs (to be created, D-166)
**Supersedes:** —

---

### D-166 · 2026-10-30 · Tier 3 code structure: one RAGTruth reader shared by the adapters and the judge checks, one verbatim template, config composition; four stacked branches

**Decision (design, no code written yet):**
1. **One reader.** `datasets/ragtruth.py` holds a small `RAGTruthFiles` class that reads and joins
   the two pinned files (D-157) and derives each source's split (D-158). It has two users:
   - `RAGTruthAdapter(DatasetAdapter)`, with `TIER = 3`, overrides only `_load_split` and
     `_map_row`. The training gate, freezing, logging and schema checks of the base class are used
     unchanged.
   - Three registered subclasses each set only their task: `tier3_ragtruth_qa`,
     `tier3_ragtruth_data2txt` and `tier3_ragtruth_summary`.
2. **One template.** `VerbatimPromptTemplate` in `agents/generator.py` sends `example.source`
   unchanged. One instance, `ragtruth_released_prompt@v1`, serves all three tasks (D-159), and T1 and
   T2 reuse it later as their first prompt.
3. **Judge checks through the existing reproducer.** `metrics/validation.py` gains loaders for
   RAGTruth's labelled responses (built on `RAGTruthFiles`) and for HaluEval. They return the same
   `{document, claim, label}` records as `load_aggrefact`, so `metrics/reproduce.py` scores them with
   its existing code. The any-sentence rule (D-163) is added once. HaluEval QA claims are built as
   `{question}: {answer}` in the loader (D-163), with the joining text taken from config.
4. **Refusal metrics.** `refusal` and `refusal_mention` (D-162) are registered `Metric` subclasses in
   one module. The fallback sentence and the `refusal` rule come from config, not literals.
5. **Config composition.** One shared `src/configs/data/tier3_ragtruth.yaml` holds the pins, split,
   cap and validation settings. The three task configs list it under `defaults:` and add only their
   name and metric suite. Analysis configs are split by T5 adapter, because the analysis groups runs
   by (topology, backbone) and would refuse two T5 variants in one config.
6. **One analysis extension.** The QA headline is paired on items where neither output is a refusal
   (D-162). The analysis pairs over all items today, so it gains one optional config setting: a
   per-item metric that excludes an item from the pairing when either run marks it. Everything else
   in the analysis code is unchanged. It lands on the runs branch, after `asqa-analysis` is merged.
7. **Small shared-core changes** (review required, AGENTS §10):
   - the sample-file header written by `datasets/freeze.py` also records `source_files`;
   - the dataset list in `test_registry.py` is updated.
8. **Not changed:** the near-identical Tier 1 adapters and the five dataset-keyed template tables.
   They are listed as follow-ups, not refactored here (AGENTS §10).

**Branches** (created by Sameer, AGENTS §12.3). Four stacked branches, one PR each; the first,
`tier3-ragtruth`, starts from `main` at `e851141`, without `asqa-analysis`:

| Branch | Contents | Starts from |
| --- | --- | --- |
| `tier3-ragtruth` | decisions D-155 to D-166 | `main` |
| `tier3-ragtruth-data` | adapter, template, data configs, frozen IDs | `tier3-ragtruth` |
| `tier3-judge-validation` | validation loaders, configs, results entry | `tier3-ragtruth-data` |
| `tier3-ragtruth-runs` | `run_matrix.csv` rows, `results/`, analysis configs, results entry | `tier3-judge-validation` |

- The analysis package exists so far only on `asqa-analysis`. The Tier 3 analysis therefore waits
  until that branch is merged into `main` and `main` is merged into the runs branch.
- Both branches append to `decisions.md` and `run_matrix.csv`. The merge keeps both blocks in number
  order.

**Reason:** AGENTS §0.4 and §7.1 (no forked files; shared logic in one place); §4 (new behaviour as
registered subclasses); §10 (one PR per extension point; no drive-by refactoring). Reading the
RAGTruth files in one class means the generation data and the judge check cannot diverge.

**Source:** AGENTS §0, §4, §7.1, §10, §12.3; code read 2026-10-30 (`datasets/base.py`,
`agents/generator.py`, `metrics/validation.py`, `metrics/reproduce.py`, `analysis/diagnostics.py` on
`asqa-analysis`)
**Approved:** Sameer, 2026-10-30
**Config:** none
**Supersedes:** —
### D-154 · 2026-10-29 · ASQA diagnostics: Mistral's T1 loss comes with its rewrites, whatever the form of the feedback; answer length explains little of the exact-match changes under T1 and T5

**Decision:** the analysis run `20261029_185305_asqa_diagnostics_seed0` (commit `3a90f60`, not
dirty, 4.6 s on CPU) is the record for two points left open by D-153 and D-142. It reads the 27
ASQA runs of D-130 (T0), D-153 (T1), D-141 and D-142 (T5), generates nothing, and compares each
topology with T0 on the same backbone by D-115's paired method. Its paired gains reproduce the means
of D-153 and D-142. Interval ends match D-142 and differ from D-153 by at most 0.0009, because each
interval now restarts the resampling generator at seed 123.

**Analysis settings** (`src/configs/analysis/asqa_diagnostics.yaml`; config group
`config/analysis.py`):

| Parameter | Value | Reason | Source |
| --- | --- | --- | --- |
| `analysis.bootstrap.n_resamples`, `.seed`, `.ci_level` | 10,000, 123, 0.95 | values already fixed for every interval | D-115 |
| `analysis.baseline` | `t0_single_agent` | every topology is compared with T0 | D-115 |
| `analysis.metrics` | MiniCheck, AlignScore, citation recall, citation precision, `str_em` | the headline, second primary and ASQA metrics | D-115, D-120 |
| `analysis.length_metrics` | `str_em` | exact-match recall rises with answer length | D-119 |
| `analysis.runs` | the 27 run IDs above | T0, T1, T5 × 3 backbones × seeds 0–2 | D-130, D-153, D-141, D-142 |

**Definitions:**
- **Feedback reply kind:** *stop* by D-101's rule; *bullets* when every non-blank line starts with
  `- ` (D-150's format); *stop word then text* when the reply opens with `NONE` and goes on;
  otherwise *prose*. A reply cut at the 300-token cap is counted separately.
- **Uncited sentence:** an NLTK sentence with no `[n]` marker, the citation metric's own rule
  (D-128); such a sentence scores 0 in citation recall.
- **Answer length:** completion tokens of the final answer, from `trace.jsonl` (D-119's unit).
- **Length-adjusted change** (a new method choice): for each question, the topology's score minus
  T0's, seeds averaged, is regressed on the same question's change in answer length by ordinary
  least squares. The intercept, the change expected with no change in length, is bootstrapped over
  questions as in D-115. This assumes the effect of length is linear; no resample was left out.

**1. Mistral under T1.** Answers grouped by the kind of their first feedback reply; each change is
the answer's score minus T0's score on that question, averaged over T0's seeds:

| First feedback | Answers (question × seed) | Answers written, mean | MiniCheck vs T0 | Citation recall vs T0 |
| --- | --- | --- | --- | --- |
| Stop (no rewrite) | 420 | 1.00 | +0.013 | −0.007 |
| Bullets | 448 | 4.29 | −0.049 | −0.085 |
| Prose | 481 | 4.06 | −0.045 | −0.093 |
| Stop word then text | 151 | 3.68 | −0.055 | −0.129 |

Questions rewritten at least once, first draft against final answer:

| | Qwen (337) | Mistral (1,080) | Llama (1,496) |
| --- | --- | --- | --- |
| Answer length, tokens | 66.6 → 55.2 | 107.3 → 124.8 | 67.6 → 65.1 |
| Share of sentences with no citation | 0.181 → 0.175 | 0.166 → 0.223 | 0.093 → 0.066 |
| Answers at the 300-token cap | 0.0% → 0.0% | 0.5% → 8.0% | 0.0% → 0.2% |

All feedback replies (stop / bullets / stop word then text / prose; at the cap): Qwen 1,269 / 1,203 /
0 / 92 of 2,564, 22 at the cap; Mistral 931 / 1,846 / 383 / 1,689 of 4,849, 595 at the cap; Llama
11 / 7,451 / 0 / 10 of 7,472, 67 at the cap.

Examples of Mistral's final answers (question ID, run):
- `-8730155953034932662` (`20261027_000618`, seed 2): the answer reports a debate between sources,
  "according to one search result"; citation recall 0.00, against 0.61 for T0 on that question.
- `-3298679965487154416` (`20261026_201800`, seed 0): the final answer contains a line "Answer 2:",
  the loop's turn format; citation recall 0.42, against 0.11 for T0.
- `-2934971361413993618` (same run): the final answer repeats the feedback and a line "Answer 3:"
  (also D-153's flagged answer).

**2. Exact-match recall with answer length held fixed:**

| | Observed change | Length change, tokens | Slope per 10 tokens | Change at fixed length [95% CI] |
| --- | --- | --- | --- | --- |
| T1 Qwen | −0.0054 | −2.6 | +0.0076 | −0.0034 [−0.0070, +0.0001] |
| T1 Mistral | −0.0160 | +12.2 | +0.0043 | −0.0212 [−0.0306, −0.0120] |
| T1 Llama | −0.0124 | −2.5 | +0.0078 | −0.0105 [−0.0193, −0.0015] |
| T5 Qwen | +0.0004 | −3.4 | +0.0142 | +0.0052 [−0.0046, +0.0149] |
| T5 Mistral | +0.0181 | +5.5 | +0.0083 | +0.0135 [+0.0003, +0.0267] |
| T5 Llama | −0.0183 | −9.7 | +0.0046 | −0.0138 [−0.0269, −0.0011] |

**Reading:**
- **Mistral's T1 loss comes with rewriting, not with the form of the feedback.** Answers rewritten
  after well-formed bullets score as low as those rewritten after prose (MiniCheck −0.049 and
  −0.045); answers the checker stopped at once are at T0's level. The grouping is observational:
  questions the checker stops at once are not a random subset, so this does not show that a
  different feedback format or stop rule would remove the loss; that was not tested.
- **Mistral's rewrites grow and lose citations.** They get 17 tokens longer, the share of sentences
  with no citation rises from 0.166 to 0.223, and 8% end at the cap. Qwen's rewrites get shorter
  with no change in that share; Llama's get slightly shorter and cite more (0.093 → 0.066).
- **Length explains little of the exact-match changes.** Holding length fixed moves each change by
  at most 0.0052. With length fixed, T1 still lowers exact-match recall on Mistral and Llama, and T5
  still lowers it on Llama and raises it on Mistral, whose interval now only just excludes 0. D-142's
  line that these changes "may be partly length effects" is tested here: length accounts for about a
  quarter of the T5 changes on Mistral and Llama, not most of them.
- **Still open:** why Mistral's replies deviate more on ASQA than on XSum (D-153). This entry
  compares ASQA runs only.

**Source:** run `20261029_185305_asqa_diagnostics_seed0` (`summary.json`, `metrics.csv`, and the
SHA-256 of every source file in `manifest.json`); the 27 runs listed in its config; D-101, D-115,
D-119, D-120, D-128, D-130, D-141, D-142, D-150, D-153
**Approved:** Sameer, 2026-10-29
**Config:** `src/configs/analysis/asqa_diagnostics.yaml`; `src/component/factagent_bench/config/analysis.py`
(D-115's `analysis.bootstrap` values, in code for the first time); `src/run_matrix.csv` (row
`asqa_diagnostics`)
**Supersedes:** D-142, the Reading line on length effects ("this was not tested"); D-153's open point
on Mistral's replies stays open

### D-167 · 2026-11-06 · The project is framed as agentic AI evaluation; the package is renamed `agent_evals`; an agentic eval scorecard is added

**Decision:**
1. **Framing.** The project is presented as an evaluation study of agentic LLM systems. The agent
   architectures (T0 single agent, T1 Self-Refine, T2 Critique-Revise with and without the Critic's
   NLI tool, T5 RL-trained agent) are the systems under evaluation; factual consistency on grounded
   generation tasks (XSum, CNN/DM, SAMSum, ASQA) is the measured outcome. No run, prompt, metric,
   config value or earlier decision changes.
2. **Rename.** The Python package `factagent_bench` is renamed `agent_evals` (distribution name
   `agent-evals`). Every import, `python -m` path, logger name and default cache directory
   (`~/.cache/agent_evals/{data,metrics,adapters}`) follows. Entries D-001 to D-166 are not edited
   (append-only, AGENTS §12.2): where they name `factagent_bench.<module>` or
   `~/.cache/factagent_bench/...`, read `agent_evals.<module>` and `~/.cache/agent_evals/...`.
   Run folders in `results/` are records and are not rewritten; their manifests keep the old paths.
   On a machine with cached data or adapters: `mv ~/.cache/factagent_bench ~/.cache/agent_evals`.
3. **Scorecard.** `agent_evals.scorecard` (entry point `src/run_evals.py`) reads `results/` and
   scores every agent configuration (architecture × backbone × task, over seeds) on five dimensions:
   - *outcome*: the task's headline judge (MiniCheck, D-115, D-120) and its secondary metrics;
   - *reliability*: pass@1, pass@k and pass^k over seeds (an item passes at MiniCheck ≥ 0.5), and the
     share of launched runs that completed;
   - *trajectory*: LLM calls and rounds per task, the share of loops that stopped on their own stop
     signal, the completion-cap rate and the blank-output rate;
   - *efficiency*: tokens and wall-clock seconds per task, and the GPU-hours of the training run
     behind an RL adapter (from the adapter path recorded in the manifest);
   - *value over baseline*: D-115's paired per-item gain over T0 on the same backbone and task, with
     its 95% bootstrap interval (10,000 resamples, seed 123), the token multiplier, and the gain per
     1k extra tokens.
   It also reports the T2 tool ablation (T2 against T2 without NLI, paired), the inference-cost
   Pareto frontier per task and backbone, and the request volume at which T5's training cost breaks
   even with each inference-time loop (wall clock per request taken as GPU time).
4. **Which runs are scored.** For each (task, topology, backbone, seed) the latest successful run is
   scored. Earlier successful runs of the same cell (superseded by D-106, D-109) and failed or
   out-of-memory attempts are not scored; all attempts count towards the run-completion rate.
5. **Outputs.** `evals/SCORECARD.md`, `evals/leaderboard.csv`, `evals/scorecard.json` (with the run
   ids and the SHA-256 of every `metrics.csv` read) and `evals/fig/*.svg`. They are generated; they
   are never edited by hand.

**Reason:** The work already measures what agentic evaluation asks of an agent beyond its final
answer: the steps it took, what they cost, whether it succeeds consistently, and whether a tool or a
training run earns its cost. The scorecard puts those measurements side by side for every
configuration from the recorded runs, without new generation. pass^k follows tau-bench (Yao et al.,
2024), which separates agents that *can* succeed from agents that *reliably* do.

**Exploratory, not confirmatory:** the pass threshold (0.5), the break-even arithmetic and the
Pareto reading are descriptive summaries. The confirmatory comparisons remain D-115's paired gains.

**Source:** the 147 agent-evaluation attempts and 57 training runs in `results/` as of 2026-11-06
**Approved:** pending team review
**Config:** `src/component/agent_evals/scorecard/compute.py` (`AGENT_PROFILES`, `TASKS`,
`ScorecardConfig` defaults); `src/run_evals.py` command-line flags
**Supersedes:** —
