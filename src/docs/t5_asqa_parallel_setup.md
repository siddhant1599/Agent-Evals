# T5 ASQA fine-tunes on three instances (D-139)

Each backbone trains on its own instance, at the same time:

| Backbone | Instance |
| --- | --- |
| Qwen2.5-7B-Instruct | this instance (the worktree `~/fall-2026-group1-t5`) |
| Llama-3.1-8B-Instruct | new instance A |
| Mistral-7B-Instruct-v0.3 | new instance B |

The three runs must be comparable, so every instance matches this one:

| Must match | This instance |
| --- | --- |
| Instance type | `g5.2xlarge` (one A10G, 8 vCPU, 30 GiB RAM) |
| OS | Ubuntu 26.04 LTS |
| Python | 3.14.4 |
| Packages | `requirements.txt`, exact pins |
| Code | the same pushed commit of `t5-asqa`, clean tree |

The NVIDIA driver here is 595.99.02. A different driver does not stop a run, but it is recorded in
every `manifest.json` (`gpu`), so the paper can state it.

## 0. On this instance, before anything else

1. Commit everything, then push the branch so the new instances can check it out:
   ```bash
   cd ~/fall-2026-group1-t5 && git push -u origin t5-asqa
   ```
2. Note the commit (`git rev-parse --short HEAD`). All three runs must record it.

## 1. New instance: system

Launch a `g5.2xlarge` from the same AMI as this instance if you can, with a disk of at least
100 GB (one backbone and the judge models need about 25 GB of model cache; adapters and
checkpoints add a few GB). Then:

```bash
nvidia-smi                      # must show an A10G; if it fails, the NVIDIA driver does not match the kernel
python3.14 --version            # 3.14.4
```

## 2. Code and environment

```bash
git clone https://github.com/sameerbatra1/fall-2026-group1.git ~/fall-2026-group1
cd ~/fall-2026-group1 && git checkout t5-asqa
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m nltk.downloader -d .venv/nltk_data punkt_tab
```

## 3. Secrets and monitor (copy them yourself; never commit them)

From this instance, copy to the new one:

| File | Why |
| --- | --- |
| `~/fall-2026-group1/.env` → `~/fall-2026-group1/.env` | Hugging Face token; Llama is gated |
| `~/.local/bin/atlas` → `~/.local/bin/atlas` (`chmod +x`) | the run monitor |
| `~/.config/atlas/worker.env` → same path (`chmod 600`) | Atlas credentials |

For example, from this instance: `scp ~/.local/bin/atlas ubuntu@<new-ip>:~/.local/bin/`.

## 4. Models and data, at the pinned revisions

On the new instance (with the token from `.env` exported as `HF_TOKEN`):

```bash
cd ~/fall-2026-group1 && source .venv/bin/activate
# Instance A (Llama):
hf download meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659
# Instance B (Mistral):
hf download mistralai/Mistral-7B-Instruct-v0.3 --revision c170c708c41dac9275d15a8fff4eca08d52bab71
# Both: the reward's judge, AlignScore, and its base model
hf download yzha/AlignScore AlignScore-large.ckpt --revision 8509e78d25bb914939fc585c626500c9b2944249
hf download FacebookAI/roberta-large --revision 722cf37b1afa9454edce342e7895e588b6ff1d59
```

The training questions and passages (D-132) download on first use and are checked against their
SHA-256. Copying them is faster and avoids Google Drive's download limits:

```bash
# from this instance
rsync -a ~/.cache/agent_evals/data/ ubuntu@<new-ip>:~/.cache/agent_evals/data/
```

## 5. Check before launching

```bash
cd ~/fall-2026-group1 && source .venv/bin/activate
git status --short              # must print nothing: a dirty tree makes the run uncitable
python -m pytest -q src/tests   # all pass, CPU only
```

## 6. Launch (one per instance)

Instance A:
```bash
tmux new-session -d -s t5_asqa_llama 'cd ~/fall-2026-group1 && export HF_HUB_OFFLINE=1; ~/.local/bin/atlas run --job t5-asqa-llama --stale-after 3600 -- .venv/bin/python src/run_training.py --config src/configs/rl/grpo_asqa.yaml --backbone llama-3.1-8b-instruct --set run.owner=sameer 2>&1 | tee -a ~/t5_asqa_llama.log'
```

Instance B:
```bash
tmux new-session -d -s t5_asqa_mistral 'cd ~/fall-2026-group1 && export HF_HUB_OFFLINE=1; ~/.local/bin/atlas run --job t5-asqa-mistral --stale-after 3600 -- .venv/bin/python src/run_training.py --config src/configs/rl/grpo_asqa.yaml --backbone mistral-7b-instruct-v0.3 --set run.owner=sameer 2>&1 | tee -a ~/t5_asqa_mistral.log'
```

This instance (Qwen), from the worktree:
```bash
tmux new-session -d -s t5_asqa_qwen 'cd ~/fall-2026-group1-t5 && export PYTHONPATH=$PWD/src/component HF_HUB_OFFLINE=1; ~/.local/bin/atlas run --job t5-asqa-qwen --stale-after 3600 -- ~/fall-2026-group1/.venv/bin/python src/run_training.py --config src/configs/rl/grpo_asqa.yaml --backbone qwen2.5-7b-instruct --set run.owner=sameer 2>&1 | tee -a ~/t5_asqa_qwen.log'
```

One optimizer step is 16 questions, so the progress bar moves about once every 7 to 13 minutes;
`--stale-after 3600` leaves room for a checkpoint save on top.

If a run stops (out of memory, instance restart), continue it on the same instance:
```bash
.venv/bin/python src/run_training.py --resume results/<run_id>
```
wrapped in the same `tmux` and `atlas run` line, with a new `--job` id such as
`t5-asqa-llama-resume1` (D-059).

## 7. While they run

- Edit no tracked file on any instance: a later continuation would be marked dirty.
- Each run writes `results/<run_id>/` (with `metrics.csv` per step) and its adapter and
  checkpoints under `~/.cache/agent_evals/adapters/<run_id>/`.

## 8. After they finish: bring everything to this instance

```bash
# on this instance, for each new instance
rsync -a ubuntu@<ip>:~/fall-2026-group1/results/<run_id>/ ~/fall-2026-group1-t5/results/<run_id>/
rsync -a ubuntu@<ip>:~/.cache/agent_evals/adapters/<run_id>/ ~/.cache/agent_evals/adapters/<run_id>/
```

Then commit the three `results/` folders from here, and back up the adapters before stopping the
new instances: the adapters exist nowhere else.
