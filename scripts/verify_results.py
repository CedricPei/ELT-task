"""Verify the complete recorded experiment offline, without cloud credentials."""

import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads(gzip.decompress((ROOT / 'results' / name).read_bytes()))


def check_rollout(record):
    episode, trajectory = record['episode'], record['trajectory']
    assert [t['reward'] for t in trajectory] == episode['rewards']
    assert len(episode['events']) <= len(trajectory) <= 14
    assert all(len(t['tokens']) == len(t['logprobs']) for t in trajectory)
    assert episode['query_ids']
    if episode['result']:
        assert episode['pipeline'] and episode['result']['replayed']
        assert episode['result']['query_ids']
        assert episode['result']['pass'] == (sum(episode['rewards']) == 1)
        parts = episode['result']['reward_components']
        assert parts['full_task_success'] == float(episode['result']['pass'])
        assert math.isclose(sum(episode['rewards']),
                            0.8 * parts['partial_accuracy'] + 0.2 * parts['full_task_success'])
    else:
        assert sum(episode['rewards']) <= 0


def main():
    directory = ROOT / 'results'
    for name, digest in json.loads((directory / 'checksums.json').read_text()).items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
    summary = json.loads((directory / 'summary.json').read_text())
    training, validation = load('training.json.gz'), load('validation.json.gz')
    plan = training['plan']
    recorded_source = load('source.json.gz')
    assert set(recorded_source) == set(plan['source_sha256'])
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256(recorded_source[name].encode()).hexdigest() == digest, name
    current_matches = all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for name, digest in plan['source_sha256'].items()
    )
    assert summary['source_sha256'] == plan['source_sha256']
    assert summary['source_unchanged'] and summary['all_rollout_records_retained']
    assert summary['intermediate_evaluations'] == summary['replacement_rollouts'] == 0
    assert summary['extra_posthoc_cloud_execution_replays'] == 0
    group_size, steps, eval_size = plan['group_size'], plan['steps'], plan['eval_group_size']
    assert summary['reward'] == plan['reward']
    assert len(training['batches']) == summary['training']['batches'] == steps
    policy = training['initial_sampler']['bound_model_path']
    updates, train_seeds = 0, []
    for index, batch in enumerate(training['batches']):
        assert batch['input_policy']['bound_model_path'] == policy
        records = batch['rollouts']
        assert len(records) == group_size
        assert [r['episode']['seed'] for r in records] == plan['training_seeds'][index]
        train_seeds.extend(r['episode']['seed'] for r in records)
        for record in records:
            check_rollout(record)
            assert record['episode']['policy']['bound_model_path'] == policy
        rewards = [sum(r['episode']['rewards']) for r in records]
        receipt = batch['update_receipt']
        if receipt:
            updates += 1
            assert receipt['optimizer_called'] and receipt['rewards'] == rewards
            assert receipt['learning_rate'] == plan['learning_rate'] == 0.00001
            advantages = receipt['advantages']
            assert any(abs(a) > 1e-7 for a in advantages)
            mean = sum(rewards) / len(rewards)
            assert all(math.isclose(a, r - mean, abs_tol=1e-6) for a, r in zip(
                advantages, rewards, strict=True
            ))
            data = batch['training_batch']
            assert data['advantages'] == advantages and len(data['data']) == receipt['datums']
            for datum in data['data']:
                inputs = datum['loss_fn_inputs']
                assert 'mask' not in inputs
                assert len({len(t['data']) for t in inputs.values()}) == 1
        else:
            assert batch['skip'] and len(set(rewards)) == 1
            assert batch['training_batch'] is None
            assert batch['output_policy']['bound_model_path'] == policy
        policy = batch['output_policy']['bound_model_path']
    assert updates == summary['training']['optimizer_updates'] >= 1
    assert summary['training']['total_rollouts'] == len(train_seeds) == steps * group_size
    assert len(set(train_seeds)) == len(train_seeds)
    assert policy == summary['training']['final_sampler']
    assert set(train_seeds).isdisjoint(plan['evaluation_seeds'])
    for phase, expected_policy in [('before', training['initial_sampler']['bound_model_path']),
                                   ('after', policy)]:
        recorded = validation['phases'][phase]
        detail, records = recorded['summary'], recorded['rollouts']
        assert detail['runtime_failures'] == 0
        assert detail['viable'] == detail['requested'] == len(records) == eval_size
        assert [r['episode']['seed'] for r in records] == plan['evaluation_seeds']
        for record in records:
            check_rollout(record)
            assert record['episode']['policy']['bound_model_path'] == expected_policy
        rewards = [sum(r['episode']['rewards']) for r in records]
        passes = [r['episode']['result'].get('pass', False) for r in records]
        assert detail['rewards'] == rewards and detail['passes'] == passes
        assert summary['evaluation'][phase]['passed'] == sum(passes)
        assert math.isclose(summary['evaluation'][phase]['mean_reward'], sum(rewards) / eval_size)
    assert summary['evaluation']['total_rollouts'] == 2 * eval_size
    total = len(train_seeds) + 2 * eval_size
    assert summary['total_model_rollouts'] == total
    print(f'Verified {total} recorded model rollouts, {updates} effective Tinker updates, '
          'and their recorded source snapshot.')
    if not current_matches:
        print('Current code differs from the recorded run; '
              'its metrics do not evaluate these changes.')


if __name__ == '__main__':
    main()
