# ELT Bench with Tinker

A recipe for one official ELT-Bench task, **trains**: combine 20 train CSV records and 63 car records into the required seven-column table.

[Google Doc](https://docs.google.com/document/d/1ZVNwi6D-32vbBchQZjh0oppmZ3nUOGTGuvuFk__QVLw/edit) · [Setup and run](docs/reproduction.md)

## Commands and implementation

ELT-Bench supplies requirements, data, reference output and the comparator. We implement connectors, commands, warehouse adapters, execution rewards and the training loop.

| Command | Purpose |
| --- | --- |
| `inspect` | Show source columns, connectors and sample records. |
| `load` | Read a source and write its raw warehouse table. |
| `sql` | Execute permitted SQL and return results or errors. |
| `submit` | Replay the pipeline in a fresh namespace and grade its output. |

The car connector reads official data through a local HTTP replay endpoint; `load` writes to **Snowflake**. Episodes retain messages and tables for up to 14 actions or 16,000 generated tokens. SQL guards and isolated namespaces restrict execution; credentials and references stay outside model access. SQLite supports local tests. Other destinations implement `Warehouse`.

## Model and reward

**Tinker** supplies inference and rank-16 LoRA training for **Qwen/Qwen3.6-35B-A3B**. **Tinker Cookbook** supplies rendering, rollouts, centered advantages and training batches.

Submission reward:

```text
R = 0.8 A + 0.2 B, if valid; otherwise 0
A = sum(a_j) / max(6, C - 1)
B = 1 for a complete official pass with the exact schema; otherwise 0
```

Each `a_j` is the correct-cell proportion in one of six fields, aligned by `train_id`. A wholly wrong field scores 0. Missing fields score 0; extra fields increase the denominator. `C` includes the ID. Comparisons preserve official numeric, NULL and tolerance rules.

Validity requires faithful loading, fresh execution, unique column names and exactly the expected unique, non-NULL IDs. Loading and IDs earn no points. Infrastructure failures abort the run. Ordinary steps score 0; unsubmitted episodes earn no correctness reward. Cookbook framing and length/context penalties remain -1/-0.1; neither occurred here.

## Training and results

**16 rollouts total:** four evaluations before training, two batches of four training rollouts, then four evaluations after training. Both batches produced real updates; only the eight training trajectories entered optimization. There were no intermediate evaluations or replacement samples.

We start from fresh LoRA weights, using importance-sampling loss and Adam at `1e-5`. Training/evaluation temperatures are 0.8/0.3. Both evaluations use the same four seeds, separate from training. Environment, reward and code remain fixed; prompt/tool tokens are masked. Equal-reward batches skip the optimizer without resampling.

| Policy | Fully correct rollouts | Mean reward |
| --- | ---: | ---: |
| Before training | 3/4 | 0.91 |
| After training | 1/4 | 0.58 |

Performance decreased. One trained-policy rollout exhausted its action limit without submitting. The experiment demonstrates execution and optimization, without evidence of a training gain.

### Possible Explanation

- **Outcome-only feedback:** one trajectory advantage reaches every generated token. Without process rewards, successful trajectories can also reinforce incorrect attempts and redundant queries.
- **Small rollout groups:** four training samples give noisy advantages; four evaluations cannot establish a reliable performance change or generalization.
- **Possible policy drift:** LoRA updates may alter prior reasoning behavior. Broader evaluations are needed to test whether general reasoning was affected.

## Files and verification

```text
src/elt_tinker/
  task.py          Official task, connectors and comparator
  environment.py   Commands, state and reward
  warehouse.py     Snowflake/SQLite adapters
  train.py         Rollouts, updates and evaluation
  budget.py        Request cost reservations
  config.py        Credential checks
  provenance.py    Official input hashes
scripts/           Preparation, setup and verification
tests/             Credential-free integration tests
results/           Complete experiment records
```

In `results/`, `summary.json` records settings and metrics; `training.json.gz` retains all eight training trajectories, tensors and two update receipts; `validation.json.gz` retains all eight evaluation trajectories, including failures. `source.json.gz` preserves execution code; `checksums.json` covers these files.

```bash
uv run --extra dev pytest -q
uv run python scripts/verify_results.py
```

Tests cover extraction, SQL feedback, fresh replay, grading and Cookbook/Tinker integration. Verification is offline. Cached sources and concurrent rollouts reduce overhead; request reservations and warehouse auto-suspend/credit monitoring control costs.
