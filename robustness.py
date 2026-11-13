"""Extra studies for the training-rationale and overfitting sections.

    python robustness.py        (run experiment.py first; uses the selected alpha in evidence/summary.json)

1. Hyperparameter sweep  - discount (gamma), training length (episodes), hazard penalty.
                           Judged on validation missions only, never on test missions.
2. Shifted-map test      - overfitting: the trained table on a map with different survivor chances.
3. Forgetting test       - catastrophic forgetting: keep training on stress missions only,
                           then re-test on normal missions.
4. Loop-trap test        - repeated re-sending of rejected commands, with and without a retry guard.

Every comparison uses the same seeds, so differences come from the change being tested.
"""
import copy
import json
import statistics
import time

from experiment import (CONFIG, EVIDENCE, allowed_actions, evaluate, n_sectors, paired_ci, train, write_csv,
                        write_json)


def standard_return(row, cfg=CONFIG):
    """Mission reward recomputed with the ORIGINAL reward weights, so runs trained with
    a different hazard penalty can still be compared fairly."""
    r = cfg['reward']
    surveyed = round(row['coverage_pct'] / 100 * n_sectors(cfg))
    return (r['detection'] * row['detections'] + r['coverage'] * surveyed + r['energy'] * row['energy']
            + r['hazard'] * row['hazards'] + r['api_failure'] * row['api_failures'])


def mean_of(rows, key):
    return statistics.mean(r[key] for r in rows)


def summarise(rows, cfg=CONFIG):
    steps = sum(r['steps'] for r in rows)
    return {'standard_return': statistics.mean(standard_return(r, cfg) for r in rows),
            'detections': mean_of(rows, 'detections'), 'hazards': mean_of(rows, 'hazards'),
            'coverage_pct': mean_of(rows, 'coverage_pct'), 'api_failures': mean_of(rows, 'api_failures'),
            'repeated_retries': mean_of(rows, 'repeated_retries'),
            'hit_attempt_cap_pct': 100 * mean_of(rows, 'hit_attempt_cap'),
            'stopped_by_guard_pct': 100 * mean_of(rows, 'stopped_by_guard'),
            'fallback_pct': 100 * sum(r['fallback_steps'] for r in rows) / steps if steps else 0.0}


def selected_alpha(out, cfg):
    path = out / 'summary.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))['selected_alpha']
    return cfg['alpha_candidates'][0]


def variant(cfg, gamma=None, episodes=None, hazard_penalty=None, presence=None):
    v = copy.deepcopy(cfg)
    if gamma is not None:
        v['gamma'] = gamma
    if episodes is not None:
        v['episodes'] = episodes
    if hazard_penalty is not None:
        v['reward']['hazard'] = hazard_penalty
    if presence is not None:
        v['sector_presence'] = list(presence)
    return v


class TableCache:
    """Trains each (setting, seed) combination once and reuses it - shared by every study run in
    the same process (robustness.py and extra_experiments.py), so tables are never trained twice."""
    _shared = {}

    def __init__(self, alpha):
        self.alpha = alpha

    def get(self, cfg, seed, label):
        key = (self.alpha, label, seed, cfg['episodes'], tuple(cfg['sector_presence']))
        if key not in self._shared:
            self._shared[key] = train(self.alpha, seed, cfg=cfg)[0]
        return self._shared[key]


def hyperparameter_sweep(cfg, cache, out):
    print('1/4  Hyperparameter sweep (validation missions only)')
    base = (cfg['gamma'], cfg['episodes'], cfg['reward']['hazard'])
    rows, summary, all_runs, default_runs = [], {}, {}, None
    metric = lambda row: standard_return(row, cfg)  # noqa: E731
    settings = ([('gamma', g) for g in cfg['sweep']['gamma']]
                + [('episodes', e) for e in cfg['sweep']['episodes']]
                + [('hazard_penalty', p) for p in cfg['sweep']['hazard_penalty']])
    for parameter, value in settings:
        g, e, p = base
        g, e, p = (value, e, p) if parameter == 'gamma' else (g, value, p) if parameter == 'episodes' else (g, e, value)
        v = variant(cfg, gamma=g, episodes=e, hazard_penalty=p)
        per_seed, runs = [], []
        for seed in cfg['training_seeds']:
            q_table = cache.get(v, seed, (g, e, p))
            val_rows, _ = evaluate(q_table, cfg['validation_seed_start'], cfg['validation_episodes'], cfg=v)
            runs.append(val_rows)
            s = summarise(val_rows, cfg)
            per_seed.append(s)
            rows.append({'parameter': parameter, 'value': value, 'is_default': int((g, e, p) == base),
                         'training_seed': seed, 'validation_return': s['standard_return'],
                         'detections': s['detections'], 'hazards': s['hazards'],
                         'coverage_pct': s['coverage_pct'], 'states': len(q_table)})
        mean = {k: statistics.mean(s[k] for s in per_seed) for k in ['standard_return', 'detections', 'hazards']}
        mean['seed_spread'] = max(s['standard_return'] for s in per_seed) - min(s['standard_return'] for s in per_seed)
        summary[f'{parameter}={value}'] = {'parameter': parameter, 'value': value,
                                           'is_default': (g, e, p) == base, **mean}
        all_runs[parameter, value] = runs
        if (g, e, p) == base:
            default_runs = runs
        print(f"  {parameter:<15}{value:<7} validation return {mean['standard_return']:6.2f}  "
              f"survivors {mean['detections']:.3f}  hazards {mean['hazards']:.3f}")
    # Paired comparison of every value against the setting actually used (same validation missions).
    for item in summary.values():
        item['diff_vs_used'] = paired_ci(all_runs[item['parameter'], item['value']], default_runs, metric)
    write_csv(out / 'hyperparameter_sweep.csv', rows)
    return summary


def shifted_map_test(cfg, cache, alpha, out):
    print('2/4  Shifted-map test (overfitting)')
    shifted = variant(cfg, presence=cfg['robustness']['shifted_presence'])
    base_label = (cfg['gamma'], cfg['episodes'], cfg['reward']['hazard'])
    approaches = {'learned_original_map': 'Q-table trained on the original map (not updated)',
                  'greedy_updated_map': 'greedy rule given the new survivor chances',
                  'learned_retrained': 'Q-table retrained on the new map'}
    rows, results, runs_by = [], {}, {}
    for name in approaches:
        per_seed, runs_by[name] = [], []
        seeds = cfg['training_seeds'] if name != 'greedy_updated_map' else [cfg['representative_seed']]
        for seed in seeds:
            if name == 'learned_original_map':
                q_table = cache.get(cfg, seed, base_label)
            elif name == 'learned_retrained':
                q_table = cache.get(shifted, seed, ('shifted',) + base_label)
            else:
                q_table = {}
            test_rows, _ = evaluate(q_table, cfg['test_seed_start'], cfg['test_episodes'], 'nominal',
                                    'greedy' if name == 'greedy_updated_map' else 'learned', cfg=shifted)
            runs_by[name].append(test_rows)
            s = summarise(test_rows, cfg)
            per_seed.append(s)
            rows.append({'approach': name, 'training_seed': seed, **s})
        results[name] = {'description': approaches[name],
                         **{k: statistics.mean(s[k] for s in per_seed) for k in per_seed[0]}}
        print(f"  {name:<22} return {results[name]['standard_return']:6.2f}  "
              f"survivors {results[name]['detections']:.3f}")
    # The same Q-table on its own map, for reference.
    original = []
    for seed in cfg['training_seeds']:
        test_rows, _ = evaluate(cache.get(cfg, seed, base_label), cfg['test_seed_start'], cfg['test_episodes'], cfg=cfg)
        original.append(summarise(test_rows, cfg))
    results['learned_on_training_map'] = {'description': 'same Q-table on the map it was trained on',
                                          **{k: statistics.mean(s[k] for s in original) for k in original[0]}}
    write_csv(out / 'shifted_map_test.csv', rows)
    metric = lambda row: standard_return(row, cfg)  # noqa: E731
    comparisons = {
        'old_table_minus_retrained': paired_ci(runs_by['learned_original_map'], runs_by['learned_retrained'], metric),
        'old_table_minus_greedy': paired_ci(runs_by['learned_original_map'], runs_by['greedy_updated_map'], metric)}
    return {'original_presence': cfg['sector_presence'], 'shifted_presence': shifted['sector_presence'],
            'results': results, 'comparisons': comparisons}


def forgetting_test(cfg, cache, alpha, out):
    print('3/4  Forgetting test (continue training on stress missions only)')
    base_label = (cfg['gamma'], cfg['episodes'], cfg['reward']['hazard'])
    extra = cfg['robustness']['forgetting_episodes']
    rows = []
    runs = {(c, w): [] for c in ('nominal', 'stress') for w in ('before', 'after')}
    for seed in cfg['training_seeds']:
        before = cache.get(cfg, seed, base_label)
        after = copy.deepcopy(before)
        train(alpha, seed, cfg=cfg, condition='stress', q_table=after, episodes=extra,
              mission_seed_offset=cfg['robustness']['forgetting_seed_offset'], fixed_epsilon=cfg['epsilon_end'])
        changed = 0
        for state, row in before.items():
            allowed = allowed_actions(state, True, cfg)
            if allowed:
                best_before = max(allowed, key=lambda a: (row[a], -a))
                best_after = max(allowed, key=lambda a: (after[state][a], -a))
                changed += best_before != best_after
        record = {'training_seed': seed, 'states_before': len(before), 'states_after': len(after),
                  'best_move_changed_pct': 100 * changed / len(before)}
        for condition in ['nominal', 'stress']:
            for label, table in [('before', before), ('after', after)]:
                test_rows, _ = evaluate(table, cfg['test_seed_start'], cfg['test_episodes'], condition, cfg=cfg)
                runs[condition, label].append(test_rows)
                s = summarise(test_rows, cfg)
                record[f'{condition}_return_{label}'] = s['standard_return']
                record[f'{condition}_detections_{label}'] = s['detections']
                record[f'{condition}_hazards_{label}'] = s['hazards']
        rows.append(record)
    write_csv(out / 'forgetting_test.csv', rows)
    result = {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k != 'training_seed'}
    result['extra_stress_episodes'] = extra
    metric = lambda row: standard_return(row, cfg)  # noqa: E731
    result['comparisons'] = {f'{c}_after_minus_before': paired_ci(runs[c, 'after'], runs[c, 'before'], metric)
                             for c in ('nominal', 'stress')}
    print(f"  nominal return {result['nominal_return_before']:.2f} -> {result['nominal_return_after']:.2f}, "
          f"stress return {result['stress_return_before']:.2f} -> {result['stress_return_after']:.2f}, "
          f"best move changed in {result['best_move_changed_pct']:.1f}% of situations")
    return result


def loop_trap_test(cfg, cache, out):
    print('4/4  Loop-trap test (random rejections vs a broken drone link, with and without retry guard)')
    base_label = (cfg['gamma'], cfg['episodes'], cfg['reward']['hazard'])
    rows, results, runs = [], {}, {}
    for condition in ['stress', 'link_fault']:
        for label, limit in [('no_guard', None), ('retry_guard', cfg['retry_limit'])]:
            per_seed, runs[condition, label] = [], []
            for seed in cfg['training_seeds']:
                test_rows, _ = evaluate(cache.get(cfg, seed, base_label), cfg['test_seed_start'],
                                        cfg['test_episodes'], condition, cfg=cfg, retry_limit=limit)
                runs[condition, label].append(test_rows)
                s = summarise(test_rows, cfg)
                per_seed.append(s)
                rows.append({'condition': condition, 'setting': label, 'retry_limit': limit,
                             'training_seed': seed, **s})
            key = f'{condition}_{label}'
            results[key] = {k: statistics.mean(s[k] for s in per_seed) for k in per_seed[0]}
            r = results[key]
            print(f"  {condition:<11}{label:<12} rejected commands {r['api_failures']:.2f}  "
                  f"identical re-sends {r['repeated_retries']:.2f}  hit command limit {r['hit_attempt_cap_pct']:5.1f}%  "
                  f"stopped by guard {r['stopped_by_guard_pct']:5.1f}%  "
                  f"survivors {r['detections']:.3f}  return {r['standard_return']:.2f}")
    write_csv(out / 'loop_trap_test.csv', rows)
    metric = lambda row: standard_return(row, cfg)  # noqa: E731
    results['comparisons'] = {}
    for condition in ['stress', 'link_fault']:
        guard, none = runs[condition, 'retry_guard'], runs[condition, 'no_guard']
        results['comparisons'][f'{condition}_guard_minus_no_guard_return'] = paired_ci(guard, none, metric)
        results['comparisons'][f'{condition}_guard_minus_no_guard_rejections'] = paired_ci(
            guard, none, lambda r: r['api_failures'])
        results['comparisons'][f'{condition}_guard_minus_no_guard_survivors'] = paired_ci(
            guard, none, lambda r: r['detections'])
    results['retry_limit'] = cfg['retry_limit']
    return results


def run_all_robustness(cfg=CONFIG, out=EVIDENCE):
    started = time.time()
    alpha = selected_alpha(out, cfg)
    cache = TableCache(alpha)
    report = {'alpha_used': alpha,
              'hyperparameter_sweep': hyperparameter_sweep(cfg, cache, out),
              'shifted_map': shifted_map_test(cfg, cache, alpha, out),
              'forgetting': forgetting_test(cfg, cache, alpha, out),
              'loop_trap': loop_trap_test(cfg, cache, out)}
    report['elapsed_seconds'] = round(time.time() - started, 1)
    write_json(out / 'robustness.json', report)
    print(f"  done in {report['elapsed_seconds']} s")
    return report


if __name__ == '__main__':
    run_all_robustness()
