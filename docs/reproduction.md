# Setup and run

Python 3.11+ and uv are required. Tinker supplies remote GPU computation; the local machine needs no GPU or Docker.

## Prepare

```bash
git clone https://github.com/CedricPei/ELT-task.git
cd ELT-task
uv sync --extra dev --locked
git clone https://github.com/uiuc-kang-lab/ELT-Bench.git vendor/ELT-Bench
git -C vendor/ELT-Bench checkout fcf3129df49ce5ce63ff46b2bbbbba1eaf9ec055
uv run python scripts/prepare_trains.py
```

Preparation downloads the official train CSV, the API archive and the evaluator reference. It extracts only the car source and checks pinned hashes. References remain private to the grader.

## Credentials

```bash
umask 077
mkdir -p secrets
cp config/credentials.example.json secrets/credentials.local.json
```

Fill in the Tinker API key, Snowflake account and human login name. Keep this file private. Generate the service-account setup SQL:

```bash
uv run python scripts/bootstrap_snowflake.py --export-sql
```

Run `secrets/bootstrap_snowflake.sql` in Snowsight as ACCOUNTADMIN, then verify the service key:

```bash
uv run python scripts/bootstrap_snowflake.py --verify-key
```

This creates a dedicated service account, isolated project database, X-Small warehouse with 60-second auto-suspend, and a three-credit resource monitor. Configuration is in `config/project.json`. Model request reservations and Snowflake credits are separate controls, not an account-wide dollar cap.

## Train

Run the recorded small protocol from fresh LoRA weights: four evaluations, two batches of four training rollouts, then four evaluations. There are no intermediate evaluations or replacement samples.

```bash
uv run python -m elt_tinker.train --warehouse snowflake \
  --steps 2 --group-size 4 --eval-group-size 4 --seed 20261020 \
  --checkpoint-ttl-seconds 604800 --output runs/trains
```

The model is Qwen/Qwen3.6-35B-A3B with rank-16 LoRA, learning rate `1e-5`, training temperature 0.8, evaluation temperature 0.3 and concurrency 4. Evaluation uses the same four seeds before and after training. Separate training seeds and source hashes are recorded before sampling; source changes abort the run. The command explicitly selects the small experiment rather than the CLI's larger defaults.

Each training batch supplies centered execution rewards to importance-sampling loss and Adam. If all rewards in a batch are equal, its optimizer call is skipped and recorded; no extra trajectories are generated. A runtime failure aborts the run without replacing the missing trajectory.

Use a new output directory for each run. Output includes the frozen source snapshot, every model action, tool response, token trajectory, training batch, optimizer receipt and sampler path. This command keeps checkpoints for seven days.

Run the two local verification commands in the README before using paid services. Stored results use the documented `0.8 A + 0.2 B` submission reward and include all 16 rollouts. The verifier checks metrics, training tensors, policy bindings and the saved source snapshot without service credentials.
