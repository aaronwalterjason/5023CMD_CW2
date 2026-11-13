"""Experiments added ON TOP of the baseline. The baseline results are not changed.

    python extra_experiments.py      (run experiment.py first)

1. Optimal benchmark   The best possible policy, computed exactly by dynamic programming
                       (value iteration over the 12-command horizon). Also the EXACT expected
                       reward of the learned, greedy and random policies - no simulation noise.
2. Mixed training      Train on a mix of nominal and stress missions (CONFIG['extensions']) and
                       compare with the baseline table, which only practised nominal missions.
3. Weather-aware filter  The risk filter judges hazard x the current condition's multiplier,
                       so under stress it blocks sectors whose real hazard exceeds the limit.

Why the exact calculation is valid: in the simulator each sector is surveyed at most once and
every command's rejection is drawn independently, so the expected reward of a move depends only
on the current state. The mission is therefore a small finite Markov decision process.
"""
import copy
import statistics
import time

from experiment import (CONFIG, EVIDENCE, allowed_actions, energy_cost, evaluate, greedy_action, mean_ci,
                        n_actions, paired_ci, split_action, train, write_csv, write_json)
from robustness import TableCache, selected_alpha


# ---------------------------------------------------------------------------
# 1. Exact solution
# ---------------------------------------------------------------------------
def outcomes(state, action, condition, cfg=CONFIG):
    """Every possible result of one command: list of (probability, expected reward, next state)."""
    c, r = cfg[condition], cfg['reward']
    surveyed, energy_a, energy_b, windy, attempts = state
    drone, sector = split_action(action, cfg)
    p_reject = 1.0 if c.get('broken_drone') == drone else c['api_failure']
    result = []
    if p_reject > 0:
        result.append((p_reject, r['api_failure'], (surveyed, energy_a, energy_b, windy, attempts + 1)))
    if p_reject < 1:
        cost = energy_cost(drone, sector, windy, cfg)
        energy = [energy_a, energy_b]
        energy[drone] -= cost
        expected = (r['detection'] * cfg['sector_presence'][sector] * c['sensitivity'] + r['coverage']
                    + r['energy'] * cost + r['hazard'] * min(1, cfg['sector_hazard'][sector] * c['hazard_multiplier']))
        result.append((1 - p_reject, expected, (surveyed | (1 << sector), energy[0], energy[1], windy, attempts + 1)))
    return result


def start_states(condition, cfg=CONFIG):
    e, p = cfg['usable_energy_each'], cfg[condition]['wind_probability']
    return [(1 - p, (0, e, e, 0, 0)), (p, (0, e, e, 1, 0))]


class ExactSolver:
    """Exact expected total reward (no discount - the same measure reported for the tests)."""

    def __init__(self, condition, cfg=CONFIG, shield=True):
        self.condition, self.cfg, self.shield = condition, cfg, shield
        self.best, self.q = {}, {}

    def action_value(self, state, action, value_fn):
        return sum(p * (reward + value_fn(nxt)) for p, reward, nxt in outcomes(state, action, self.condition, self.cfg))

    def optimal_value(self, state):
        """Value iteration: V*(s) = max over allowed moves of [expected reward + V*(next)]."""
        if state not in self.best:
            allowed = allowed_actions(state, self.shield, self.cfg)
            row = [0.0] * n_actions(self.cfg)
            for a in allowed:
                row[a] = self.action_value(state, a, self.optimal_value)
            self.q[state] = row
            self.best[state] = max((row[a] for a in allowed), default=0.0)
        return self.best[state]

    def policy_value(self, policy, state, memo):
        """Exact value of a fixed policy. policy(state, allowed) -> list of (probability, action)."""
        if state not in memo:
            allowed = allowed_actions(state, self.shield, self.cfg)
            memo[state] = sum(p * self.action_value(state, a, lambda s: self.policy_value(policy, s, memo))
                              for p, a in policy(state, allowed)) if allowed else 0.0
        return memo[state]

    def expected(self, value_fn):
        return sum(p * value_fn(s) for p, s in start_states(self.condition, self.cfg))


def learned_policy(q_table, cfg=CONFIG):
    def choose(state, allowed):
        if state in q_table:
            row = q_table[state]
            return [(1.0, max(allowed, key=lambda a: (row[a], -a)))]
        return [(1.0, greedy_action(state, allowed, cfg))]
    return choose


def greedy_policy(cfg=CONFIG):
    return lambda state, allowed: [(1.0, greedy_action(state, allowed, cfg))]


def random_policy(state, allowed):
    return [(1 / len(allowed), a) for a in allowed]


def optimal_benchmark(cfg, cache, out):
    print('1/3  Optimal benchmark (exact dynamic programming)')
    label = (cfg['gamma'], cfg['episodes'], cfg['reward']['hazard'])
    tables = {seed: cache.get(cfg, seed, label) for seed in cfg['training_seeds']}
    rep = cfg['representative_seed']
    results, rows = {}, []
    for condition in ['nominal', 'stress']:
        solver = ExactSolver(condition, cfg)
        optimal = solver.expected(solver.optimal_value)
        exact = {'optimal': optimal,
                 'greedy': solver.expected(lambda s: solver.policy_value(greedy_policy(cfg), s, {})),
                 'random': solver.expected(lambda s: solver.policy_value(random_policy, s, {}))}
        learned_by_seed = {seed: solver.expected(lambda s, q=q: solver.policy_value(learned_policy(q, cfg), s, {}))
                           for seed, q in tables.items()}
        exact['learned'] = learned_by_seed[rep]
        exact['learned_mean_of_seeds'] = statistics.mean(learned_by_seed.values())
        # Simulate the optimal policy on the same test missions, for a like-for-like paired comparison.
        optimal_table = {s: row for s, row in solver.q.items()}
        opt_rows, _ = evaluate(optimal_table, cfg['test_seed_start'], cfg['test_episodes'], condition, cfg=cfg)
        learned_rows, _ = evaluate(tables[rep], cfg['test_seed_start'], cfg['test_episodes'], condition, cfg=cfg)
        greedy_rows, _ = evaluate({}, cfg['test_seed_start'], cfg['test_episodes'], condition, 'greedy', cfg=cfg)
        sim_mean, sim_ci = mean_ci(r['return'] for r in opt_rows)
        span = optimal - exact['random']
        results[condition] = {
            'exact_expected_return': exact,
            'learned_return_by_seed': {str(k): v for k, v in learned_by_seed.items()},
            'learned_pct_of_optimal': 100 * exact['learned'] / optimal,
            'greedy_pct_of_optimal': 100 * exact['greedy'] / optimal,
            'learned_gap_closed_pct': 100 * (exact['learned'] - exact['random']) / span,
            'greedy_gap_closed_pct': 100 * (exact['greedy'] - exact['random']) / span,
            'optimal_simulated_test': {'mean': sim_mean, 'ci95_halfwidth': sim_ci},
            'optimal_minus_learned_test': paired_ci([opt_rows], [learned_rows]),
            'optimal_minus_greedy_test': paired_ci([opt_rows], [greedy_rows]),
            'states_solved': len(solver.q)}
        for name, value in exact.items():
            rows.append({'condition': condition, 'policy': name, 'exact_expected_return': value,
                         'pct_of_optimal': 100 * value / optimal})
        res = results[condition]
        print(f"  {condition:<8} optimal {optimal:6.2f} | learned {exact['learned']:6.2f} "
              f"({res['learned_pct_of_optimal']:.1f}% of optimal) | greedy {exact['greedy']:6.2f} "
              f"({res['greedy_pct_of_optimal']:.1f}%) | random {exact['random']:6.2f}")
    write_csv(out / 'optimal_benchmark.csv', rows)
    results['note'] = ('Optimal = best possible policy when the true probabilities of that condition are known. '
                       'It maximises total (undiscounted) reward, the measure reported for all tests; '
                       'the Q-table was trained with gamma = %s.' % cfg['gamma'])
    return results


# ---------------------------------------------------------------------------
# 2. Mixed-condition training
# ---------------------------------------------------------------------------
def runs_for(tables, condition, cfg, policy='learned'):
    return [evaluate(q, cfg['test_seed_start'], cfg['test_episodes'], condition, policy, cfg=cfg)[0] for q in tables]


def summarise_runs(runs):
    rows = [r for run in runs for r in run]
    steps = sum(r['steps'] for r in rows)
    return {'return': statistics.mean(r['return'] for r in rows),
            'detections': statistics.mean(r['detections'] for r in rows),
            'hazards': statistics.mean(r['hazards'] for r in rows),
            'fallback_pct': 100 * sum(r['fallback_steps'] for r in rows) / steps if steps else 0.0}


def mixed_training(cfg, cache, alpha, out):
    mix = cfg['extensions']['mixed_training']
    print(f'2/3  Mixed-condition training {mix}')
    label = (cfg['gamma'], cfg['episodes'], cfg['reward']['hazard'])
    baseline = [cache.get(cfg, seed, label) for seed in cfg['training_seeds']]
    mixed = [train(alpha, seed, cfg=cfg, condition=mix)[0] for seed in cfg['training_seeds']]
    results, rows = {'mix': mix}, []
    for condition in ['nominal', 'stress']:
        base_runs, mixed_runs = runs_for(baseline, condition, cfg), runs_for(mixed, condition, cfg)
        results[condition] = {'baseline': summarise_runs(base_runs), 'mixed': summarise_runs(mixed_runs),
                              'mixed_minus_baseline_return': paired_ci(mixed_runs, base_runs),
                              'mixed_minus_baseline_hazards': paired_ci(mixed_runs, base_runs, lambda r: r['hazards']),
                              'situations_learned_baseline': statistics.mean(len(q) for q in baseline),
                              'situations_learned_mixed': statistics.mean(len(q) for q in mixed)}
        for name, s in [('baseline', results[condition]['baseline']), ('mixed', results[condition]['mixed'])]:
            rows.append({'tested_on': condition, 'training': name, **s})
        d = results[condition]['mixed_minus_baseline_return']
        print(f"  tested {condition:<8} baseline {results[condition]['baseline']['return']:6.2f}  mixed "
              f"{results[condition]['mixed']['return']:6.2f}  difference {d['diff']:+.2f} "
              f"[{d['low']:+.2f}, {d['high']:+.2f}]  fallback {results[condition]['baseline']['fallback_pct']:.1f}% -> "
              f"{results[condition]['mixed']['fallback_pct']:.1f}%")
    write_csv(out / 'mixed_training_test.csv', rows)
    return results


# ---------------------------------------------------------------------------
# 3. Weather-aware risk filter
# ---------------------------------------------------------------------------
def weather_aware_filter(cfg, cache, out):
    print('3/3  Weather-aware risk filter (stress missions)')
    label = (cfg['gamma'], cfg['episodes'], cfg['reward']['hazard'])
    tables = [cache.get(cfg, seed, label) for seed in cfg['training_seeds']]
    aware = copy.deepcopy(cfg)
    aware['filter_hazard_multiplier'] = cfg['stress']['hazard_multiplier']
    blocked = [s for s, h in enumerate(cfg['sector_hazard']) if h * aware['filter_hazard_multiplier'] > cfg['risk_mask_threshold']]
    base_runs, aware_runs = runs_for(tables, 'stress', cfg), runs_for(tables, 'stress', aware)
    results = {'filter_multiplier': aware['filter_hazard_multiplier'], 'sectors_blocked_when_aware': blocked,
               'baseline': summarise_runs(base_runs), 'aware': summarise_runs(aware_runs)}
    for metric in ['return', 'detections', 'hazards']:
        results[f'aware_minus_baseline_{metric}'] = paired_ci(aware_runs, base_runs, lambda r, m=metric: r[m])
    write_csv(out / 'weather_aware_filter_test.csv',
              [{'filter': name, **results[name]} for name in ('baseline', 'aware')])
    h = results['aware_minus_baseline_hazards']
    print(f"  blocks sectors {blocked} under stress | hazards {results['baseline']['hazards']:.3f} -> "
          f"{results['aware']['hazards']:.3f} ({h['diff']:+.3f} [{h['low']:+.3f}, {h['high']:+.3f}]) | survivors "
          f"{results['baseline']['detections']:.3f} -> {results['aware']['detections']:.3f}")
    return results


def run_all_extra(cfg=CONFIG, out=EVIDENCE):
    started = time.time()
    alpha = selected_alpha(out, cfg)
    cache = TableCache(alpha)
    report = {'alpha_used': alpha,
              'optimal_benchmark': optimal_benchmark(cfg, cache, out),
              'mixed_training': mixed_training(cfg, cache, alpha, out),
              'weather_aware_filter': weather_aware_filter(cfg, cache, out)}
    report['elapsed_seconds'] = round(time.time() - started, 1)
    write_json(out / 'extra_experiments.json', report)
    print(f"  done in {report['elapsed_seconds']} s")
    return report


if __name__ == '__main__':
    run_all_extra()
