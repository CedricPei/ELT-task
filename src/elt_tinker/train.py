"""ELT training with Cookbook rendering, rollouts and training batches."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import tinker
from tinker.lib.base_url import DEFAULT_BASE_URL
from tinker_cookbook import hyperparam_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.rollout_limits import RolloutLimits
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.rl.types import TrajectoryGroup

from elt_tinker.budget import Budget
from elt_tinker.environment import REWARD_FORMULA, ELTEnv
from elt_tinker.task import Sources, official_trains
from elt_tinker.warehouse import SnowflakeWarehouse, SQLiteWarehouse

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3.6-35B-A3B"
# Published USD / million tokens, checked 2026-10-20. Model-specific and fail closed.
RATE_CARDS = {
    MODEL: (0.45, 1.335, 1.177, 65536),
}


class SeededClient:
    """Use a reproducible, distinct sampling seed for each model turn."""

    def __init__(self, client, seed: int):
        self.client, self.seed, self.calls = client, seed, 0

    async def sample_async(self, *, sampling_params, **kwargs):
        params = sampling_params.model_copy(update={"seed": self.seed + self.calls})
        self.calls += 1
        return await self.client.sample_async(sampling_params=params, **kwargs)


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src/elt_tinker").glob("*.py"))
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def write_private_errors(path: Path, errors) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as handle:
        for error in errors:
            handle.write("".join(traceback.format_exception(error)))


def private_failure(error: Exception) -> None:
    directory = ROOT / "secrets"
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    write_private_errors(directory / "latest-run-error.log", [error])
    print(f"Run failed ({type(error).__name__}); details saved privately.", file=sys.stderr)


class BudgetedPolicy(TokenCompleter):
    def __init__(self, client, gate: Budget, temperature: float, max_tokens: int, model=MODEL):
        prefill, sample, _, context = RATE_CARDS[model]
        self.prefill, self.sample = prefill / 1e6, sample / 1e6
        self.inner = TinkerTokenCompleter(client, max_tokens, temperature, context_window=context)
        self.gate = gate

    async def __call__(self, model_input, stop, *, max_tokens=None):
        cap = min(max_tokens or self.inner.max_tokens, self.inner.max_tokens)
        # Reserve 50% above the full uncached-input + maximum-output rate-card cost.
        ticket = self.gate.reserve(
            "sampling", 1.5 * (model_input.length * self.prefill + cap * self.sample)
        )
        response = await self.inner(model_input, stop, max_tokens=cap)
        self.gate.settle(
            ticket, 1.25 * (model_input.length * self.prefill + len(response.tokens) * self.sample)
        )
        return response


def verified_price_card(model: str) -> dict:
    prefill_rate, sample_rate, train_rate, _ = RATE_CARDS[model]
    request = Request(
        "https://tinker-docs.thinkingmachines.ai/tinker/models.json",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        price_card = next(item for item in json.load(response) if item["tinker_id"] == model)
    for field, expected in (
        ("prefill", prefill_rate),
        ("sample", sample_rate),
        ("train", train_rate),
    ):
        if float(price_card[field].removeprefix("$")) != expected:
            raise ValueError("Published model pricing changed; review rates before running")
    return price_card


async def optimize_update(
    trainer,
    group,
    gate: Budget,
    out: Path,
    model: str,
    training_rate: float,
    iteration: int = 0,
    checkpoint_ttl_seconds: int = 86400,
    learning_rate: float | None = None,
) -> dict:
    lr = hyperparam_utils.get_lr(model) if learning_rate is None else learning_rate
    if not math.isfinite(lr) or not 0 < lr <= 0.01:
        raise ValueError("Learning rate must be finite and in (0, 0.01]")
    advantages = compute_advantages([group])
    if not bool(advantages[0].abs().any()):
        write_json(
            out / "no_update.json",
            {"reason": "All execution rewards equal", "rewards": group.get_total_rewards()},
        )
        raise RuntimeError("No nonzero RL signal; optimizer was not called")
    data, metadata = assemble_training_data([group], advantages)
    data = [
        tinker.Datum(
            model_input=d.model_input,
            loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"},
        )
        for d in data
    ]
    tokens = sum(d.model_input.length for d in data)
    ticket = gate.reserve("training", max(0.000001, 1.5 * tokens * training_rate))
    write_json(
        out / f"step_{iteration}_batch.json",
        {
            "advantages": advantages[0].tolist(),
            "metadata": metadata,
            "data": [
                {
                    "model_input": datum.model_input.model_dump(mode="json"),
                    "loss_fn_inputs": {
                        name: {
                            "data": tensor.tolist(),
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.to_numpy().dtype),
                        }
                        for name, tensor in datum.loss_fn_inputs.items()
                    },
                }
                for datum in data
            ],
        },
    )
    # Queue ordered requests before waiting, following the cookbook loop.
    forward = await trainer.forward_backward_async(data, loss_fn="importance_sampling")
    optim = await trainer.optim_step_async(tinker.AdamParams(learning_rate=lr))
    forward_result, optim_result = await asyncio.gather(
        forward.result_async(), optim.result_async()
    )
    gate.settle(ticket, 1.25 * tokens * training_rate)
    save = await trainer.save_state_async(
        f"elt-step-{iteration + 1}", ttl_seconds=checkpoint_ttl_seconds
    )
    saved = await save.result_async()
    receipt = {
        "rewards": group.get_total_rewards(),
        "advantages": advantages[0].tolist(),
        "training_tokens": tokens,
        "datums": len(data),
        "forward_backward_metrics": forward_result.metrics,
        "optimizer_metrics": optim_result.metrics,
        "checkpoint": saved.model_dump(mode="json"),
        "checkpoint_ttl_seconds": checkpoint_ttl_seconds,
        "learning_rate": lr,
        "loss": "importance_sampling",
        "optimizer_called": True,
    }
    write_json(
        out / f"step_{iteration}_update.json",
        receipt,
    )
    print(
        f"RL update {iteration + 1} completed: {len(data)} datums, {tokens} tokens.",
        flush=True,
    )
    return receipt


async def bind_sampler(service, trainer, out: Path, name: str, model: str, ttl: int):
    """Bind an explicitly saved policy; the recorded path is the one actually sampled."""
    future = await trainer.save_weights_for_sampler_async(name, ttl_seconds=ttl)
    saved = await future.result_async()
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    receipt = {
        "base_model": model,
        "checkpoint": saved.model_dump(mode="json"),
        "checkpoint_ttl_seconds": ttl,
        "bound_model_path": saved.path,
        "utc": datetime.now(UTC).isoformat(),
    }
    write_json(out / f"{name}_sampler.json", receipt)
    return sampler, receipt


async def sample_episode(
    env, policy, renderer, receipt, out: Path, name: str, context_window: int, max_tokens: int,
    *, seed: int | None = None,
):
    """Sample and record an episode; callers create and close environments."""
    wrapped = EnvFromMessageEnv(
        renderer, env, max_trajectory_tokens=context_window, max_generation_tokens=max_tokens
    )
    trajectory = await run_rollout(
        policy,
        wrapped,
        limits=RolloutLimits(
            max_turns=14, max_turn_tokens=max_tokens,
            max_trajectory_tokens=context_window, max_sampled_tokens=16000,
        ),
    )
    evidence = {
        "policy": receipt,
        **({"seed": seed} if seed is not None else {}),
        "events": env.events,
        "pipeline": env.pipeline,
        "result": env.result,
        "rewards": [t.reward for t in trajectory.transitions],
        "stop_reason": trajectory.stop_reason,
        "query_ids": list(getattr(env.backend, "query_ids", [])),
    }
    write_json(out / f"{name}.json", evidence)
    write_json(out / f"{name}_trajectory.json", [
        {
            "observation": t.ob.model_dump(mode="json"),
            "tokens": t.ac.tokens,
            "logprobs": t.ac.logprobs,
            "reward": t.reward,
            **({"episode_done": t.episode_done} if seed is not None else {}),
        }
        for t in trajectory.transitions
    ])
    return trajectory, evidence


async def recipe(args) -> None:
    if not 1 <= args.group_size <= 8 or not 0 <= args.steps <= 5:
        raise ValueError("Use group-size 1..8 and steps 0..5 for this starter recipe")
    if args.steps and args.group_size < 2:
        raise ValueError("Centered RL requires at least two independent rollouts")
    if args.sampler_path and (args.steps or not args.sampler_path.startswith("tinker://")):
        raise ValueError("--sampler-path evaluates a Tinker checkpoint with --steps 0 only")
    if args.model not in RATE_CARDS:
        raise ValueError("Update and verify the price card before enabling a different model")
    if not 512 <= args.max_tokens <= 4096:
        raise ValueError("Use 512..4096 generation tokens per turn")
    if not 1 <= args.eval_group_size <= 16 or not 1 <= args.concurrency <= 4:
        raise ValueError("Use eval-group-size 1..16 and concurrency 1..4")
    if not 0 <= args.seed < 2**31 - 200000:
        raise ValueError("Seed must leave room for distinct rollout seed ranges")
    if not 3600 <= args.checkpoint_ttl_seconds <= 604800:
        raise ValueError("Checkpoint retention must be between one hour and seven days")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.01
    ):
        raise ValueError("Learning rate must be finite and in (0, 0.01]")
    out = Path(args.output).resolve()
    if (out / "run.json").exists():
        raise ValueError("Use a new output directory for each run")
    out.mkdir(parents=True, exist_ok=True)
    prefill_rate, sample_rate, train_rate, context_window = RATE_CARDS[args.model]
    training_rate = train_rate / 1e6
    price_card = verified_price_card(args.model)
    write_json(out / "verified_price_card.json", price_card)
    gate = Budget(ROOT / "runs/budget.json", args.model_budget_usd)
    config = json.loads((ROOT / "config/project.json").read_text())
    frozen_source = source_hashes()
    write_json(out / "source.json", {
        name: (ROOT / name).read_bytes().decode() for name in frozen_source
    })
    evaluation_seeds = [args.seed + 1000 * i for i in range(args.eval_group_size)]
    training_seeds = [
        [args.seed + 100000 + 10000 * step + 1000 * i for i in range(args.group_size)]
        for step in range(args.steps)
    ]
    write_json(
        out / "run.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "model": args.model,
            "warehouse": args.warehouse,
            "group_size": args.group_size,
            "steps": args.steps,
            "max_tokens": args.max_tokens,
            "max_turns": 14,
            "temperature": args.temperature,
            "eval_temperature": args.eval_temperature,
            "eval_group_size": args.eval_group_size,
            "concurrency": args.concurrency,
            "evaluation_seeds": evaluation_seeds,
            "training_seeds": training_seeds,
            "source_sha256": frozen_source,
            "schedule": "evaluate once, fixed training batches, evaluate once; no resampling",
            "zero_signal_rule": "Record the batch and skip its optimizer call without resampling",
            "checkpoint_ttl_seconds": args.checkpoint_ttl_seconds,
            "lora_rank": 16 if args.steps else None,
            "lora_seed": 7 if args.steps else None,
            "learning_rate": args.learning_rate or hyperparam_utils.get_lr(args.model),
            "sampler_path": args.sampler_path,
            "source_revisions": config["source_revisions"],
            "task": "official ELT-Bench trains",
            "reward": REWARD_FORMULA,
            "rl_loss": "importance_sampling",
            "advantage": "within-group centered execution reward",
            "rate_card_usd_per_million": {
                "prefill": prefill_rate,
                "sample": sample_rate,
                "train": train_rate,
            },
            "price_source": "https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/",
        },
    )
    if args.warehouse == "snowflake":

        def factory():
            return SnowflakeWarehouse(ROOT / "secrets/snowflake.runtime.json", out.name)
    else:
        factory = SQLiteWarehouse
    key = json.loads((ROOT / "secrets/credentials.local.json").read_text())["tinker_api_key"]
    service = tinker.ServiceClient(api_key=key, base_url=DEFAULT_BASE_URL)
    sources = None
    status = "errored"
    try:
        task = official_trains(ROOT)
        sources = Sources(task)
        if args.steps:
            trainer = await service.create_lora_training_client_async(
                base_model=args.model, rank=16, seed=7
            )
            sampler, policy_receipt = await bind_sampler(
                service, trainer, out, "initial", args.model, args.checkpoint_ttl_seconds
            )
            tokenizer = trainer.get_tokenizer()
        else:
            trainer = None
            if args.sampler_path:
                sampler = await service.create_sampling_client_async(model_path=args.sampler_path)
                if await sampler.get_base_model_async() != args.model:
                    raise ValueError("Checkpoint base model differs from the reviewed price card")
            else:
                sampler = await service.create_sampling_client_async(base_model=args.model)
            policy_receipt = {"base_model": args.model, "bound_model_path": args.sampler_path}
            tokenizer = sampler.get_tokenizer()
        recommended = model_info.get_recommended_renderer_names(args.model)
        renderer_name = next(
            (name for name in recommended if name.endswith("disable_thinking")),
            model_info.get_recommended_renderer_name(args.model),
        )
        if args.thinking:
            renderer_name = model_info.get_recommended_renderer_name(args.model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        write_json(out / "renderer.json", {"renderer": renderer_name, "recommended": recommended})

        async def rollout(name: str, policy, receipt: dict, seed: int):
            env = await asyncio.to_thread(ELTEnv, task, sources, factory)
            env.trace_path = out / f"{name}_events.json"
            try:
                trajectory, evidence = await sample_episode(
                    env, policy, renderer, receipt, out, name, context_window, args.max_tokens,
                    seed=seed,
                )
                print(
                    f"{name}: reward={sum(evidence['rewards']):.4f}, "
                    f"pass={env.result.get('pass', False)}, "
                    f"turns={len(trajectory.transitions)}",
                    flush=True,
                )
                return trajectory, evidence
            finally:
                await asyncio.to_thread(env.close)

        async def collect(phase: str, seeds: list[int], temperature: float):
            if source_hashes() != frozen_source:
                raise RuntimeError("Recipe source changed during the frozen experiment")
            semaphore = asyncio.Semaphore(args.concurrency)

            async def episode(i, seed):
                async with semaphore:
                    policy = BudgetedPolicy(
                        SeededClient(sampler, seed), gate, temperature, args.max_tokens, args.model
                    )
                    return await rollout(f"{phase}_rollout_{i}", policy, policy_receipt, seed)

            results = await asyncio.gather(
                *[episode(i, seed) for i, seed in enumerate(seeds)],
                return_exceptions=True,
            )
            failed_results = [result for result in results if isinstance(result, Exception)]
            failures = [type(result).__name__ for result in failed_results]
            if failures:
                write_json(out / f"{phase}_failures.json", {"error_types": failures})
                error_path = ROOT / "secrets" / (out.name + "-" + phase + "-errors.log")
                write_private_errors(error_path, failed_results)
                # Runtime errors are excluded, never relabeled as model failures.
                results = [result for result in results if not isinstance(result, Exception)]
            summary = {
                "phase": phase,
                "policy": policy_receipt,
                "temperature": temperature,
                "requested": len(seeds),
                "seeds": seeds,
                "viable": len(results),
                "runtime_failures": len(failures),
                "rewards": [sum(r[1]["rewards"]) for r in results],
                "passes": [r[1]["result"].get("pass", False) for r in results],
            }
            summary["pass_count"] = sum(summary["passes"])
            summary["mean_reward"] = (
                sum(summary["rewards"]) / len(results) if results else None
            )
            write_json(out / f"{phase}_summary.json", summary)
            if failures or not results:
                raise RuntimeError("Incomplete batch; no replacement rollouts are sampled")
            return results, summary

        updates = []
        skipped_batches = []
        if args.steps:
            _, before = await collect("eval_before", evaluation_seeds, args.eval_temperature)
            for iteration in range(args.steps):
                results, batch_summary = await collect(
                    f"step_{iteration}", training_seeds[iteration], args.temperature
                )
                group = TrajectoryGroup(
                    [r[0] for r in results], [0.0] * len(results), [{}] * len(results)
                )
                if not bool(compute_advantages([group])[0].abs().any()):
                    skipped_batches.append(iteration)
                    write_json(out / f"step_{iteration}_skipped.json", {
                        "reason": "All execution rewards equal; no optimizer call",
                        "rewards": batch_summary["rewards"],
                    })
                    print(f"Batch {iteration + 1}: equal rewards; optimizer skipped.", flush=True)
                    continue
                update = await optimize_update(
                    trainer,
                    group,
                    gate,
                    out,
                    args.model,
                    training_rate,
                    iteration,
                    args.checkpoint_ttl_seconds,
                    args.learning_rate,
                )
                updates.append(update)
                sampler, policy_receipt = await bind_sampler(
                    service,
                    trainer,
                    out,
                    f"step_{iteration + 1}",
                    args.model,
                    args.checkpoint_ttl_seconds,
                )
            _, after = await collect("eval_after", evaluation_seeds, args.eval_temperature)
        else:
            phase = "checkpoint_eval" if args.sampler_path else "baseline"
            seeds = [args.seed + 1000 * i for i in range(args.group_size)]
            _, after = await collect(phase, seeds, args.temperature)
            before = None
        if source_hashes() != frozen_source:
            raise RuntimeError("Recipe source changed during the frozen experiment")
        summary = {
            "base_model": args.model,
            "warehouse": args.warehouse,
            "requested_updates": args.steps,
            "actual_updates": len(updates),
            "training_batches": args.steps,
            "training_rollouts": args.steps * args.group_size,
            "evaluation_rollouts": 2 * args.eval_group_size if args.steps else args.group_size,
            "skipped_batches": skipped_batches,
            "eval_before": before,
            "eval_after": after,
            "same_model_throughout": True,
            "updated_policy_sampled": bool(updates),
            "source_unchanged": True,
            "performance_improvement_claim": False,
        }
        status = "success"
    finally:
        if sources:
            sources.close()
        if args.warehouse == "snowflake":
            # All gathered episodes have finished; stop idle compute immediately.
            import snowflake.connector

            connection = snowflake.connector.connect(
                **json.loads((ROOT / "secrets/snowflake.runtime.json").read_text())
            )
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SHOW WAREHOUSES LIKE 'ELT_TINKER_WH'")
                    row = cursor.fetchone()
                    names = [column[0] for column in cursor.description]
                    if dict(zip(names, row, strict=True))["state"] != "SUSPENDED":
                        cursor.execute("ALTER WAREHOUSE ELT_TINKER_WH SUSPEND")
                    cursor.execute("SHOW RESOURCE MONITORS LIKE 'ELT_TINKER_MONITOR'")
                    monitor = dict(
                        zip([c[0] for c in cursor.description], cursor.fetchone(), strict=True)
                    )
                    write_json(
                        out / "warehouse_usage.json",
                        {
                            k: str(monitor[k])
                            for k in (
                                "credit_quota",
                                "used_credits",
                                "remaining_credits",
                                "frequency",
                            )
                        },
                    )
            finally:
                connection.close()
        service.close(status).result()
    # A completion marker is emitted only after warehouse and API cleanup succeeds.
    write_json(out / "completed.json", {"utc": datetime.now(UTC).isoformat(), **summary})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warehouse", choices=["sqlite", "snowflake"], default="sqlite")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--eval-temperature", type=float, default=0.3)
    parser.add_argument("--eval-group-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=600000)
    parser.add_argument("--checkpoint-ttl-seconds", type=int, default=86400)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--sampler-path", help="Evaluate saved sampler weights without training")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--model-budget-usd", type=float, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asyncio.run(recipe(args))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        private_failure(error)
        raise SystemExit(1) from None
