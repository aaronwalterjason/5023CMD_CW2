"""Automatic correctness checks for the simulator, the learning rule and the approval gate.
They test that the code behaves as described - not how well the policy performs.

    python check_experiment.py
"""
import copy
import json
import random
import sys

import itertools
from pathlib import Path

from experiment import (CONFIG, EVIDENCE, REJECTED, Mission, allowed_actions, block_reason, evaluate, mean_ci,
                        n_actions, n_sectors, paired_ci, q_update, split_action, train, write_json)


def run_checks(cfg=CONFIG, out=EVIDENCE):
    checks = {}

    def check(name, condition):
        checks[name] = 'PASS' if condition else 'FAIL'

    sectors, actions, full_energy = n_sectors(cfg), n_actions(cfg), cfg['usable_energy_each']
    cap = cfg['max_dispatch_attempts']
    start = (0, full_energy, full_energy, 0, 0)
    risky = [s for s, h in enumerate(cfg['sector_hazard']) if h > cfg['risk_mask_threshold']]
    check('risk_filter_blocks_sectors_above_limit_only_when_enabled',
          all(a % sectors not in risky for a in allowed_actions(start, True, cfg))
          and any(a % sectors in risky for a in allowed_actions(start, False, cfg)))

    full = (2 ** sectors - 1, full_energy, full_energy, 0, 0)
    check('mission_ends_when_all_surveyed_no_energy_or_command_limit',
          not allowed_actions(full, True, cfg) and not allowed_actions((0, 0, 0, 0, 0), True, cfg)
          and not allowed_actions((0, full_energy, full_energy, 0, cap), True, cfg))

    aware = copy.deepcopy(cfg)
    aware['filter_hazard_multiplier'] = cfg['stress']['hazard_multiplier']
    consistent = all(allowed_actions(st, sh, c) == [a for a in range(actions) if block_reason(st, a, sh, c) is None]
                     for c in (cfg, aware) for sh in (True, False)
                     for st in itertools.product(range(2 ** sectors), range(full_energy + 1), range(full_energy + 1),
                                                 range(2), range(0, cap + 1, 3)))
    check('allowed_actions_and_block_reason_apply_the_same_rules', consistent)
    expected_blocked = {s for s, h in enumerate(cfg['sector_hazard'])
                        if h * aware['filter_hazard_multiplier'] > cfg['risk_mask_threshold']}
    check('weather_aware_filter_blocks_sectors_whose_scaled_hazard_exceeds_limit',
          {a % sectors for a in range(actions)} - {a % sectors for a in allowed_actions(start, True, aware)}
          == expected_blocked and len(expected_blocked) > len(risky))

    mission = Mission(1, cfg=cfg)
    before = mission.state
    mission.command_rejected[0][0] = True
    reward, event = mission.step(0)
    check('rejected_command_uses_no_energy_and_surveys_nothing',
          event == REJECTED and mission.state[:4] == before[:4] and mission.state[4] == 1
          and reward == cfg['reward']['api_failure'])

    ok = True
    for seed in range(1000):
        m, rng = Mission(seed, 'stress', cfg), random.Random(seed)
        while (allowed := allowed_actions(m.state, True, cfg)):
            previous = m.state
            m.step(rng.choice(allowed))
            ok &= min(m.state[1:3]) >= 0 and m.state[4] <= cfg['max_dispatch_attempts']
            ok &= (m.state[0] | previous[0]) == m.state[0]
            ok &= all(not m.state[0] & (1 << (b % sectors)) for b in allowed_actions(m.state, True, cfg))
    check('1000_random_stress_missions_never_break_energy_or_coverage_rules', ok)

    a, b = Mission(7, cfg=cfg), Mission(7, cfg=cfg)
    check('same_seed_gives_identical_mission',
          a.state == b.state and a.command_rejected == b.command_rejected and a.survivor_present == b.survivor_present)

    check('q_update_matches_worked_example',
          abs(q_update(5, 2, 6, 0.05, 0.95) - 5.135) < 1e-12 and abs(q_update(5, 2, 6, 0.05, 0.95, terminal=True) - 4.85) < 1e-12)

    q1, _, _ = train(0.05, 11, cfg=cfg, episodes=300)
    q2, _, _ = train(0.05, 11, cfg=cfg, episodes=300)
    check('training_is_reproducible', q1 == q2)

    other = copy.deepcopy(cfg)
    other['gamma'] = 0.5
    q3, _, _ = train(0.05, 11, cfg=other, episodes=300)
    capped = copy.deepcopy(cfg)
    capped['max_dispatch_attempts'] = 5
    check('config_values_drive_the_code',
          q3 != q1 and not allowed_actions((0, 4, 4, 0, 5), True, capped) and allowed_actions((0, 4, 4, 0, 5), True, cfg))

    mix = cfg['extensions']['mixed_training']
    m1, _, _ = train(0.05, 11, cfg=cfg, condition=mix, episodes=300)
    m2, _, _ = train(0.05, 11, cfg=cfg, condition=mix, episodes=300)
    check('mixed_condition_training_is_reproducible_and_differs_from_nominal', m1 == m2 and m1 != q1)

    same, _ = evaluate(q1, cfg['test_seed_start'], 40, cfg=cfg)
    zero = paired_ci([same], [same])
    check('paired_ci_gives_zero_difference_for_identical_runs', zero['diff'] == 0 and not zero['significant'])

    broken = Mission(3, 'link_fault', cfg)
    check('link_fault_rejects_every_command_to_broken_drone',
          all(row[act] for row in broken.command_rejected for act in range(actions)
              if split_action(act, cfg)[0] == cfg['link_fault']['broken_drone']))

    rows, _ = evaluate(q1, cfg['test_seed_start'], 50, 'link_fault', cfg=cfg, retry_limit=cfg['retry_limit'])
    check('retry_guard_prevents_command_limit_loops_with_broken_link', all(r['hit_attempt_cap'] == 0 for r in rows))

    try:
        import explain
        expl = explain.explain_decision(q1, None, (0, 4, 4, 0, 0), cfg)
        blocked = {b['move'] for b in expl['blocked']}
        check('explanation_lists_risky_sectors_as_blocked', all(f'A->{s}' in blocked and f'B->{s}' in blocked for s in risky))
        flagged = explain.risk_flags((0, 4, 4, 0, 0), 4, q1, None, cfg)   # A->4, hazard 0.18
        check('approval_gate_flags_high_hazard_moves', bool(flagged) and not explain.risk_flags((0, 4, 4, 0, 0), 0, q1, None, cfg))
        learned, _ = evaluate(q1, cfg['test_seed_start'], 30, 'stress', cfg=cfg)
        gated = [explain.run_supervised_mission(q1, None, cfg['test_seed_start'] + k, 'stress', 'autonomous', cfg)[0]
                 for k in range(30)]
        check('no_gate_mode_reproduces_the_learned_policy',
              [r['return'] for r in learned] == [r['return'] for r in gated])

        # Human mode with scripted keyboard answers (output hidden).
        import builtins
        import contextlib
        import io
        real_input = builtins.input
        try:
            refused = []
            with contextlib.redirect_stdout(io.StringIO()):
                builtins.input = lambda prompt='': 'n'
                for k in range(10):
                    refused.append(explain.run_supervised_mission(q1, None, cfg['test_seed_start'] + k, 'stress',
                                                                  'human', cfg)[0])
                builtins.input = lambda prompt='': 'q'
                stopped = explain.run_supervised_mission(q1, None, cfg['test_seed_start'], 'stress', 'human', cfg)[0]
        finally:
            builtins.input = real_input
        rejecting = [explain.run_supervised_mission(q1, None, cfg['test_seed_start'] + k, 'stress', 'always_reject',
                                                    cfg)[0] for k in range(10)]
        check('human_refusals_lead_to_next_best_move_or_abstain',
              [r['return'] for r in refused] == [r['return'] for r in rejecting] and any(r['refused'] for r in refused))
        check('human_can_stop_the_mission', stopped['abstained'] == 1 and stopped['approved'] == 0)

        # Structured feedback: a refusal with reason 'h' is stored as a category, with flag types.
        answers = iter(['n', 'h'] + ['n', 'o'] * 50)
        try:
            builtins.input = lambda prompt='': next(answers)
            with contextlib.redirect_stdout(io.StringIO()):
                _, feedback, _ = explain.run_supervised_mission(q1, None, cfg['test_seed_start'], 'stress', 'human', cfg)
        finally:
            builtins.input = real_input
        check('refusal_reasons_are_recorded_as_structured_categories',
              bool(feedback) and feedback[0]['reason_category'] == 'hazard_concern'
              and 'hazard' in feedback[0]['flag_types'])
        expl = explain.explain_decision(q1, None, (0, 4, 4, 1, 0), cfg, 0, cfg['stress']['hazard_multiplier'])
        check('explanations_open_with_a_one_line_summary_and_field_report',
              explain.format_explanation(expl).startswith('>> ') and 'field report' in expl['summary'])

        # The feedback review suggests changes but never applies them.
        import tempfile
        import feedback_review
        from experiment import write_csv
        before_cfg = copy.deepcopy(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            log = []
            for k in range(40):
                log += explain.run_supervised_mission(q1, None, cfg['test_seed_start'] + k, 'stress', 'vigilant', cfg)[1]
            write_csv(tmp / 'approval_log.csv', log)
            with contextlib.redirect_stdout(io.StringIO()):
                review = feedback_review.run_feedback_review(cfg, tmp)
            files_ok = (tmp / 'feedback_review.json').exists() and (tmp / 'feedback_suggestions.md').exists()
        check('feedback_review_suggests_changes_but_never_applies_them',
              files_ok and bool(review['suggestions']) and cfg == before_cfg
              and all(s['status'] == 'PENDING HUMAN SIGN-OFF' and not s['applied'] for s in review['suggestions']))
    except ImportError:
        checks['explain_module_checks'] = 'SKIPPED (explain.py missing)'

    try:
        import edge_cases as edge
        naive = [edge.run_edge_mission(q1, cfg['test_seed_start'] + k, False, cfg) for k in range(60)]
        handled = [edge.run_edge_mission(q1, cfg['test_seed_start'] + k, True, cfg) for k in range(60)]
        check('lost_acknowledgements_never_cause_duplicate_flights_when_handled',
              all(r['duplicate_flights'] == 0 and r['battery_overruns'] == 0 for r in handled)
              and any(r['duplicate_flights'] or r['battery_overruns'] for r in naive))
        check('closed_sectors_are_never_dispatched_to_when_handled',
              all(r['closure_violations'] == 0 for r in handled) and any(r['closure_violations'] for r in naive))
        check('bayesian_confirmation_reduces_false_survivor_reports',
              sum(r['confirmed_false'] for r in handled) < sum(r['confirmed_false'] for r in naive))
        fuzz = edge.fuzz_validation(cfg, n=1000)
        check('invalid_states_are_always_held_and_valid_states_accepted',
              fuzz['all_valid_states_accepted'] and fuzz['malformed_states_held'] == fuzz['malformed_states_tested'])
    except ImportError:
        checks['edge_case_checks'] = 'SKIPPED (edge_cases.py missing)'

    try:
        import random as _random
        import stress_test
        import edge_cases as edge_mod
        small = copy.deepcopy(cfg)
        small['stress_test'] = dict(cfg['stress_test'], scenarios=15, missions_per_scenario=4)

        def mini_run(c):
            rng = _random.Random(c['stress_test']['seed'])
            counts, violations = {k: 0 for k in stress_test.INVARIANTS}, []
            for sid in range(c['stress_test']['scenarios']):
                scenario = stress_test.random_scenario(rng, c)
                for k in range(c['stress_test']['missions_per_scenario']):
                    stress_test.dispatch_loop(q1, 900000 + sid * 100 + k, scenario, rng, counts, violations, sid)
            return violations

        check('randomised_stress_scenarios_break_no_safety_rule', not mini_run(small))
        # Mutation test: a deliberately broken dispatcher that ignores closures MUST be caught.
        real = edge_mod.safe_dispatch
        edge_mod.safe_dispatch = lambda q, state, c, closed=frozenset(): (
            ('act', allowed_actions(state, True, c)[-1]) if not edge_mod.validate_state(state, c)
            and allowed_actions(state, True, c) else ('done', []))
        try:
            caught = any(v[2] == 'I2' for v in mini_run(small))
        finally:
            edge_mod.safe_dispatch = real
        check('stress_test_detects_a_deliberately_broken_dispatcher', caught)
    except ImportError:
        checks['stress_test_checks'] = 'SKIPPED (stress_test.py missing)'

    try:
        import extra_experiments as extra
        solver = extra.ExactSolver('nominal', cfg)
        exact_greedy = solver.expected(lambda s: solver.policy_value(extra.greedy_policy(cfg), s, {}))
        exact_learned = solver.expected(lambda s: solver.policy_value(extra.learned_policy(q1, cfg), s, {}))
        optimal = solver.expected(solver.optimal_value)
        sim, _ = evaluate({}, 700000, 4000, 'nominal', 'greedy', cfg=cfg)
        sim_mean, sim_ci = mean_ci(r['return'] for r in sim)
        # Tolerance of two 95% half-widths so a different CONFIG cannot fail this check by chance alone.
        check('exact_calculation_matches_4000_simulated_missions', abs(exact_greedy - sim_mean) < 2 * sim_ci)
        check('optimal_policy_is_at_least_as_good_as_greedy_and_learned',
              optimal >= exact_greedy - 1e-9 and optimal >= exact_learned - 1e-9)
    except ImportError:
        checks['extra_experiment_checks'] = 'SKIPPED (extra_experiments.py missing)'

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'checks.json', checks)
    for name, result in checks.items():
        print(f'  {result:<5} {name}')
    return checks


if __name__ == '__main__':
    results = run_checks()
    sys.exit(0 if all(v != 'FAIL' for v in results.values()) else 1)
