"""Offline SDK compatibility: real cookbook batch assembly and native Tinker tensor types."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

from elt_tinker.budget import Budget
from elt_tinker.train import MODEL, bind_sampler, optimize_update


def group(rewards):
    trajectories = [
        Trajectory(
            [
                Transition(
                    tinker.ModelInput.from_ints([10, 20]),
                    TokensWithLogprobs([30, 40], [-0.2, -0.3]),
                    reward,
                    True,
                )
            ],
            tinker.ModelInput.empty(),
        )
        for reward in rewards
    ]
    return TrajectoryGroup(trajectories, [0.0] * len(rewards), [{}] * len(rewards))


class OfflineTrainer:
    def __init__(self):
        self.calls = []
        self.data = []
        self.learning_rate = None

    def future(self, name, value):
        async def result_async():
            self.calls.append("await-" + name)
            return value

        return SimpleNamespace(result_async=result_async)

    async def forward_backward_async(self, data, loss_fn):
        assert loss_fn == "importance_sampling"
        self.calls.append("forward")
        self.data = data
        return self.future("forward", SimpleNamespace(metrics={"offline_test": 1}))

    async def optim_step_async(self, params):
        assert params.learning_rate > 0
        self.learning_rate = params.learning_rate
        self.calls.append("optim")
        return self.future("optim", SimpleNamespace(metrics={}))

    async def save_state_async(self, name, ttl_seconds):
        assert ttl_seconds == 86400
        self.calls.append("save")
        return self.future("save", SimpleNamespace(model_dump=lambda **kwargs: {"path": "fixture"}))


def test_batch_masks_context_serializes_native_sdk_types_and_orders_updates(tmp_path):
    trainer = OfflineTrainer()
    asyncio.run(
        optimize_update(
            trainer,
            group([0.0, 1.0]),
            Budget(tmp_path / "ledger.json"),
            tmp_path,
            MODEL,
            1.177 / 1e6,
        )
    )
    assert trainer.calls.index("optim") < trainer.calls.index("await-forward")
    assert len(trainer.data) == 2
    for datum, advantage in zip(trainer.data, [-0.5, 0.5], strict=True):
        assert "mask" not in datum.loss_fn_inputs
        assert datum.loss_fn_inputs["advantages"].tolist() == [0.0, advantage, advantage]
    batch = json.loads((tmp_path / "step_0_batch.json").read_text())
    assert batch["advantages"] == [-0.5, 0.5]
    assert batch["data"][0]["loss_fn_inputs"]["target_tokens"]["data"] == [20, 30, 40]


def test_equal_rewards_never_claim_an_optimizer_update(tmp_path):
    trainer = OfflineTrainer()
    with pytest.raises(RuntimeError, match="No nonzero RL signal"):
        asyncio.run(
            optimize_update(
                trainer,
                group([1.0, 1.0]),
                Budget(tmp_path / "ledger.json"),
                tmp_path,
                MODEL,
                1.177 / 1e6,
            )
        )
    assert not trainer.calls
    assert not (tmp_path / "step_0_update.json").exists()


def test_updated_sampler_uses_exact_saved_weights_path(tmp_path):
    calls = []

    class Trainer:
        async def save_weights_for_sampler_async(self, name, ttl_seconds):
            calls.append(("save", name, ttl_seconds))
            saved = tinker.types.SaveWeightsForSamplerResponse(
                path="tinker://fixture/weights/updated", type="save_weights_for_sampler"
            )

            async def result_async():
                return saved

            return SimpleNamespace(result_async=result_async)

    class Service:
        async def create_sampling_client_async(self, **kwargs):
            calls.append(("bind", kwargs))
            return "updated-sampler"

    sampler, receipt = asyncio.run(
        bind_sampler(Service(), Trainer(), tmp_path, "step_1", MODEL, 86400)
    )
    assert sampler == "updated-sampler"
    assert calls == [
        ("save", "step_1", 86400),
        ("bind", {"model_path": "tinker://fixture/weights/updated"}),
    ]
    assert receipt["bound_model_path"] == "tinker://fixture/weights/updated"
    assert json.loads((tmp_path / "step_1_sampler.json").read_text()) == receipt


@pytest.mark.parametrize("flat_training", [False, True])
@pytest.mark.parametrize("runtime_failure", [False, True])
def test_recipe_collects_execution_rewards_trains_and_evaluates_new_policy(
    tmp_path, monkeypatch, flat_training, runtime_failure
):
    """Real local ELT episodes and cookbook batches, with only remote model I/O replaced."""
    from elt_tinker import train
    from elt_tinker.task import fixture_task

    (tmp_path / "src/elt_tinker").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/project.json").write_text('{"source_revisions": {}}')
    (tmp_path / "secrets").mkdir(mode=0o700)
    (tmp_path / "secrets/credentials.local.json").write_text('{"tinker_api_key":"offline"}')
    output = tmp_path / "output"
    sampled_paths = []
    trainer = OfflineTrainer()

    async def save_weights_for_sampler_async(name, ttl_seconds):
        saved = tinker.types.SaveWeightsForSamplerResponse(
            path=f"tinker://fixture/weights/{name}", type="save_weights_for_sampler"
        )
        return trainer.future("sampler-" + name, saved)

    trainer.save_weights_for_sampler_async = save_weights_for_sampler_async
    trainer.get_tokenizer = lambda: None

    class Service:
        async def create_lora_training_client_async(self, **kwargs):
            assert kwargs == {"base_model": MODEL, "rank": 16, "seed": 7}
            return trainer

        async def create_sampling_client_async(self, **kwargs):
            assert set(kwargs) == {"model_path"}  # Never silently revert to base weights.
            return kwargs["model_path"]

        def close(self, status):
            assert status == ("errored" if runtime_failure else "success")
            assert not (output / "completed.json").exists()
            return SimpleNamespace(result=lambda: None)

    async def run_rollout(policy, env, limits):
        sampled_paths.append(policy)
        if runtime_failure and len(sampled_paths) == 2:
            raise ConnectionError("offline injected infrastructure failure")
        await env.initial_observation()
        columns = "id AS train_id" + (
            ", direction" if flat_training or len(sampled_paths) != 2 else ""
        )
        sql = f"CREATE TABLE trains AS SELECT {columns} FROM raw_train"
        loads = [
            {"source": "train", "connector": "csv", "table": "raw_train", "mode": "replace"},
            {"source": "cars", "connector": "api", "table": "raw_cars", "mode": "replace"},
        ]
        actions = [
            {"action": "inspect"},
            *[{"action": "load", **load} for load in loads],
            {"action": "sql", "sql": sql},
            {
                "action": "submit",
                "pipeline": {"target": "trains", "loads": loads, "transforms": [sql]},
            },
        ]
        transitions = []
        for action in actions:
            result = await env.step({"role": "assistant", "content": json.dumps(action)})
            transitions.append(
                Transition(
                    tinker.ModelInput.from_ints([10, 20]),
                    TokensWithLogprobs([30, 40], [-0.2, -0.3]),
                    result.reward,
                    result.episode_done,
                )
            )
        assert env.result["replayed"]
        return Trajectory(transitions, tinker.ModelInput.empty())

    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "verified_price_card", lambda _: {})
    monkeypatch.setattr(train, "official_trains", lambda _: fixture_task())
    monkeypatch.setattr(train.tinker, "ServiceClient", lambda **kwargs: Service())
    monkeypatch.setattr(train.renderers, "get_renderer", lambda *args: None)
    monkeypatch.setattr(train, "EnvFromMessageEnv", lambda renderer, env, **kwargs: env)
    monkeypatch.setattr(train, "BudgetedPolicy", lambda sampler, *args: sampler.client)
    monkeypatch.setattr(train, "run_rollout", run_rollout)
    args = SimpleNamespace(
        group_size=2,
        steps=1,
        model=MODEL,
        max_tokens=2048,
        eval_group_size=1,
        concurrency=1,
        seed=600000,
        checkpoint_ttl_seconds=86400,
        output=str(output),
        model_budget_usd=10,
        warehouse="sqlite",
        thinking=False,
        temperature=0.8,
        eval_temperature=0.3,
        sampler_path=None,
        learning_rate=0.00005,
    )
    if runtime_failure:
        with pytest.raises(RuntimeError, match="Incomplete batch"):
            asyncio.run(train.recipe(args))
        assert len(sampled_paths) == 3  # One evaluation and the fixed pair; no replacement.
        assert "forward" not in trainer.calls and "optim" not in trainer.calls
        assert not (output / "completed.json").exists()
        summary = json.loads((output / "step_0_summary.json").read_text())
        assert summary["runtime_failures"] == 1 and summary["viable"] == 1
        return
    asyncio.run(train.recipe(args))
    assert sampled_paths == ["tinker://fixture/weights/initial"] * 3 + [
        "tinker://fixture/weights/initial" if flat_training
        else "tinker://fixture/weights/step_1"
    ]
    complete = json.loads((output / "completed.json").read_text())
    assert complete["training_batches"] == 1
    assert complete["training_rollouts"] == complete["evaluation_rollouts"] == 2
    plan = json.loads((output / "run.json").read_text())
    assert plan["reward"] == train.REWARD_FORMULA
    assert (output / "source.json").exists()
    assert plan["evaluation_seeds"] == [600000]
    assert plan["training_seeds"] == [[700000, 701000]]
    if flat_training:
        assert complete["actual_updates"] == 0 and complete["skipped_batches"] == [0]
        assert not complete["updated_policy_sampled"]
        assert "forward" not in trainer.calls and "optim" not in trainer.calls
        assert not (output / "step_0_update.json").exists()
        return
    receipt = json.loads((output / "step_0_update.json").read_text())
    assert len(set(receipt["rewards"])) == 2
    assert len(receipt["advantages"]) == 2  # Evaluation data never enters the update.
    assert complete["actual_updates"] == 1 and complete["updated_policy_sampled"]
    assert complete["eval_before"]["policy"] != complete["eval_after"]["policy"]
    assert trainer.learning_rate == receipt["learning_rate"] == 0.00005


def test_seeded_sampling_preserves_parameters_and_uses_distinct_turn_seeds():
    from elt_tinker.train import SeededClient

    calls = []

    class Client:
        async def sample_async(self, **kwargs):
            calls.append(kwargs)

    async def run():
        client = SeededClient(Client(), 600000)
        params = tinker.SamplingParams(max_tokens=100, temperature=0.3)
        for _ in range(2):
            await client.sample_async(sampling_params=params, num_samples=1, prompt="fixture")
        assert params.seed is None

    asyncio.run(run())
    assert [c["sampling_params"].seed for c in calls] == [600000, 600001]
    assert all(c["num_samples"] == 1 and c["prompt"] == "fixture" for c in calls)


@pytest.mark.parametrize("seed", [None, 100000])
def test_episode_records_policy_tokens_and_limits(tmp_path, monkeypatch, seed):
    """The recipe records the sampled policy, limits and native token traces."""
    from elt_tinker import train

    trajectory = group([0.75]).trajectories_G[0]
    env = SimpleNamespace(
        events=[{"action": "submit"}], pipeline={"target": "trains"},
        result={"pass": False}, backend=SimpleNamespace(query_ids=["query-1"]),
    )
    receipt = {"bound_model_path": "tinker://fixture/sampler_weights/step_1"}

    def wrap(renderer, environment, **kwargs):
        assert renderer == "renderer" and environment is env
        assert kwargs == {"max_trajectory_tokens": 65536, "max_generation_tokens": 2048}
        return environment

    async def rollout(policy, environment, limits):
        assert policy == "policy" and environment is env
        assert limits.max_turns == 14 and limits.max_turn_tokens == 2048
        assert limits.max_trajectory_tokens == 65536 and limits.max_sampled_tokens == 16000
        return trajectory

    monkeypatch.setattr(train, "EnvFromMessageEnv", wrap)
    monkeypatch.setattr(train, "run_rollout", rollout)
    actual, detail = asyncio.run(train.sample_episode(
        env, "policy", "renderer", receipt, tmp_path, "episode", 65536, 2048, seed=seed
    ))
    assert actual is trajectory
    assert detail == {
        "policy": receipt, **({"seed": seed} if seed is not None else {}),
        "events": env.events, "pipeline": env.pipeline, "result": env.result,
        "rewards": [0.75], "stop_reason": trajectory.stop_reason, "query_ids": ["query-1"],
    }
    assert json.loads((tmp_path / "episode.json").read_text()) == detail
    assert json.loads((tmp_path / "episode_trajectory.json").read_text()) == [{
        "observation": trajectory.transitions[0].ob.model_dump(mode="json"),
        "tokens": [30, 40], "logprobs": [-0.2, -0.3], "reward": 0.75,
        **({"episode_done": True} if seed is not None else {}),
    }]
