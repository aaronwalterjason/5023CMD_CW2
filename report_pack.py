"""Build report_pack/RESULTS.md: every number, table and figure the report writers need,
organised by the four Task 2 sections of the brief.

    python report_pack.py      (run after the experiment and the extra studies)

The pack is regenerated from evidence/ on every run, so its numbers always match the code.
It gives facts, tables and pointers - not finished report prose.
"""
import csv
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def ci(res):
    return f"{res['mean']:.2f} ± {res['ci95_halfwidth']:.2f}"


def diff(d, digits=2):
    """Paired difference with its 95% interval, e.g. '+1.02 [+0.60, +1.43]'."""
    return f"{d['diff']:+.{digits}f} [{d['low']:+.{digits}f}, {d['high']:+.{digits}f}]"


def sig(d):
    return 'yes' if d['significant'] else 'no'


def approval_excerpt(path):
    """First decision step in a transcript that needed human approval."""
    if not path.exists():
        return None
    blocks, current = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('--- Step'):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    for block in blocks:
        if any('NEEDS HUMAN APPROVAL' in line for line in block):
            return '\n'.join(line for line in block if not line.startswith('==='))
    return '\n'.join(blocks[0]) if blocks else None


CI_NOTE = ('Intervals here are paired 95% confidence intervals over missions (same missions for both sides; each mission '
           'averaged over all training runs). They show mission-to-mission uncertainty; variation between training runs '
           'is reported separately where marked "across training runs". "Significant" = the interval excludes 0.')


def section_glance(s, sup, rob, extra):
    nl, ng, nr = (s['results'][f'nominal_{p}']['return']['mean'] for p in ('learned', 'greedy', 'random'))
    pn, ps = s['paired_comparisons']
    rows = [['Selected learning rate', s['selected_alpha'], 'validation.csv'],
            ['Nominal test reward: learned / greedy / random', f'{nl:.2f} / {ng:.2f} / {nr:.2f}', 'summary.json'],
            ['Learned − greedy, nominal (95% CI)',
             f"{pn['learned_minus_greedy']:+.2f} [{pn['learned_minus_greedy'] - pn['ci95_halfwidth']:+.2f}, "
             f"{pn['learned_minus_greedy'] + pn['ci95_halfwidth']:+.2f}]", 'summary.json'],
            ['Learned − greedy, stress (95% CI)',
             f"{ps['learned_minus_greedy']:+.2f} [{ps['learned_minus_greedy'] - ps['ci95_halfwidth']:+.2f}, "
             f"{ps['learned_minus_greedy'] + ps['ci95_halfwidth']:+.2f}]", 'summary.json']]
    if s.get('paired_across_runs'):
        for cond in ('nominal', 'stress'):
            a = s['paired_across_runs'][cond]
            if a['runs'] > 1:
                rows.append([f"Learned − greedy, {cond}, across {a['runs']} training runs (95% CI)",
                             f"{a['mean']:+.2f} [{a['low']:+.2f}, {a['high']:+.2f}]", 'summary.json'])
    if extra:
        b = extra['optimal_benchmark']
        rows.append(['Learned policy as % of best possible (nominal / stress)',
                     f"{b['nominal']['learned_pct_of_optimal']:.1f}% / {b['stress']['learned_pct_of_optimal']:.1f}%",
                     'extra_experiments.json'])
    if sup:
        a, v = sup['results']['stress_always_approve'], sup['results']['stress_vigilant']
        rows.append(['Stress hazards per mission: always-approve vs vigilant supervisor',
                     f"{a['hazards']:.2f} vs {v['hazards']:.2f}", 'supervisor_study.json'])
    if rob:
        d = rob['shifted_map']['comparisons']['old_table_minus_retrained']
        rows.append(['New survivor map: old Q-table − retrained (95% CI)', diff(d), 'robustness.json'])
    return '## At a glance\n\n' + table(['Fact', 'Value', 'Source file'], rows)


def section_model(s, cfg, q_states, extra):
    n = len(cfg['sector_presence'])
    combos = 2 ** n * (cfg['usable_energy_each'] + 1) ** 2 * 2 * (cfg['max_dispatch_attempts'] + 1)
    res = s['results']
    rows = [[f"{cond.capitalize()} – {name}", ci(res[f'{cond}_{p}']['return']),
             f"{res[f'{cond}_{p}']['detections']['mean']:.3f}", f"{res[f'{cond}_{p}']['hazards']['mean']:.3f}",
             f"{res[f'{cond}_{p}']['fallback_pct']:.2f}%"]
            for cond in ('nominal', 'stress')
            for p, name in [('learned', 'learned Q-table'), ('greedy', 'greedy rule'), ('random', 'random'),
                            ('unshielded', 'learned, no risk filter')]]
    out = [f"""## Section 1 – Choice of ML model

**What the model is**
- Tabular Q-learning: one central dispatcher controls {len(cfg['base_energy_cost'])} simulated drones over {n} sectors.
- Observation (state), 5 values: surveyed sectors (bitmask), energy of drone A, energy of drone B, wind (0/1), commands sent so far.
- Actions: {2 * n} (drone A or B × sector 0–{n - 1}). Output: one predicted value per action; the highest *allowed* action is executed.
- Possible state combinations: {combos:,}. Situations actually reached in training (representative table): {q_states}.
- Safety rules (already surveyed, not enough energy, hazard above {cfg['risk_mask_threshold']}) remove actions *before* the model chooses.

**Points that support the choice** (justify in your own words)
- The state and action spaces are small and discrete, so a table can hold a value for every situation reached – no neural network or dataset needed.
- Every decision can be traced to a number in `evidence/q_table.json` (see Figure 8), which suits explainability and the viva.
- The reward turns mission goals (find survivors, cover ground, save energy, avoid hazards) into one learning signal.
- It learns from simulated experience, so no labelled real-world data is required.

**Limits to state honestly**
- It decides *where to search*; it does not see images or fly the drones. Detection is simulated with a probability.
- The survivor and hazard chances are fixed inside the simulator, not part of the observation, so the table is tied to one map (see the shifted-map test in Section 2).
- It is a central policy for two drones, not multi-agent reinforcement learning.

**Test results (500 unseen missions per condition, representative seed {cfg['representative_seed']})**

{table(['Condition – policy', 'Reward (95% CI)', 'Survivors found', 'Hazard events', 'Fallback use'], rows)}
"""]
    across = s.get('across_training_runs', {})
    if across:
        arows = [[key.replace('_', ' – ', 1), f"{v['mean']:.2f} ± {v['ci95']:.2f}", f"{v['min']:.2f} – {v['max']:.2f}"]
                 for key, v in across.items()]
        pa = s.get('paired_across_runs', {})
        out.append(f"""**Across all {len(cfg['training_seeds'])} training runs** (one test mean per run; 95% interval uses the t-distribution, so it includes training-run variability)

{table(['Condition – policy', 'Mean reward ± 95% CI', 'Range over runs'], arows)}

- Learned − greedy across training runs: nominal {pa['nominal']['mean']:+.2f} [{pa['nominal']['low']:+.2f}, {pa['nominal']['high']:+.2f}] (significant: {'yes' if pa['nominal']['significant_vs_zero'] else 'no'}); stress {pa['stress']['mean']:+.2f} [{pa['stress']['low']:+.2f}, {pa['stress']['high']:+.2f}] (significant: {'yes' if pa['stress']['significant_vs_zero'] else 'no'}). This interval shows the result is not training luck, but it is measured on the same 500 missions.
""")
        ml = s.get('paired_all_runs_mission_level')
        if ml:
            out.append(f"""- Same comparison with the policy averaged over all runs, interval over missions: nominal {diff(ml['nominal'])} (significant: {sig(ml['nominal'])}); stress {diff(ml['stress'])} (significant: {sig(ml['stress'])}).
- How to read the three views: single trained table (seed {cfg['representative_seed']}) vs greedy, interval over missions; difference across training runs; and the run-averaged policy, interval over missions. Claim an advantage only where the relevant interval excludes 0.
""")
    if extra:
        b = extra['optimal_benchmark']
        brows = []
        for cond in ('nominal', 'stress'):
            e = b[cond]['exact_expected_return']
            brows += [[cond, 'Random (baseline)', f"{e['random']:.2f}", f"{100 * e['random'] / e['optimal']:.1f}%"],
                      [cond, 'Greedy rule', f"{e['greedy']:.2f}", f"{b[cond]['greedy_pct_of_optimal']:.1f}%"],
                      [cond, 'Learned Q-table', f"{e['learned']:.2f}", f"{b[cond]['learned_pct_of_optimal']:.1f}%"],
                      [cond, 'Exact planner (benchmark)', f"{e['optimal']:.2f}", '100%']]
        nb, sb = b['nominal'], b['stress']
        out.append(f"""**How close to the best possible policy?** (`extra_experiments.py`, exact dynamic programming)

The mission is small enough to compute the best possible decision for every situation exactly. The table gives the *exact* long-run expected reward of each policy – no simulation noise.

{table(['Condition', 'Policy', 'Exact expected reward', '% of best possible'], brows)}

- The learned policy closes {nb['learned_gap_closed_pct']:.0f}% of the gap between random and best possible in nominal conditions (greedy: {nb['greedy_gap_closed_pct']:.0f}%); under stress {sb['learned_gap_closed_pct']:.0f}% (greedy: {sb['greedy_gap_closed_pct']:.0f}%).
- On the 500 test missions, best possible − learned = {diff(nb['optimal_minus_learned_test'])} (nominal) and {diff(sb['optimal_minus_learned_test'])} (stress).
- "Best possible" knows the true probabilities of each condition, which a real disaster would not provide – that is why it is a benchmark, not the agent.
- Why the test numbers are lower than these exact values: the 500 test missions happen to be harder than average (exact learned {nb['exact_expected_return']['learned']:.2f} vs test mean {s['results']['nominal_learned']['return']['mean']:.2f}). Every policy faces the same missions, so comparisons between policies are unaffected.
""")
    out.append('Figures: `figures/fig3_policy_comparison.png`, `figures/fig4_safety_tradeoff.png`, '
               '`figures/fig8_q_values_first_decision.png`, `figures/fig9_optimal_benchmark.png`.')
    return '\n'.join(out)


def section_training(s, cfg, rob, extra, ev):
    validation = read_csv(ev / 'validation.csv')
    vba = s.get('validation_by_alpha', {})
    val_rows = []
    for a in cfg['alpha_candidates']:
        runs = [r for r in validation if float(r['alpha']) == float(a)]
        v = vba.get(str(a))
        states = sum(int(r['states']) for r in runs) / len(runs) if runs else 0
        val_rows.append([f"{a}" + (' (selected)' if float(a) == float(s['selected_alpha']) else ''),
                         f"{v['mean']:.2f} ± {v['ci95']:.2f}" if v else '–',
                         f"{v['min']:.2f} – {v['max']:.2f}" if v else '–', len(runs), f'{states:.0f}'])
    means = ', '.join(f'{a}: {m:.2f}' for a, m in s['validation_means'].items())
    best_v = vba.get(str(s['selected_alpha']))
    overlap = [a for a in cfg['alpha_candidates'] if float(a) != float(s['selected_alpha']) and best_v and vba.get(str(a))
               and vba[str(a)]['high'] >= best_v['low']]
    curve = read_csv(ev / 'learning_curve.csv')
    rw = cfg['reward']
    hyper = [
        ['Learning rate (alpha)', s['selected_alpha'], '`alpha_candidates`', f"Tuned: best mean validation reward of {cfg['alpha_candidates']}"],
        ['Discount (gamma)', cfg['gamma'], '`gamma`', f"Fixed design choice; tested in sweep {cfg['sweep']['gamma']}"],
        ['Training missions per run', f"{cfg['episodes']:,}", '`episodes`', f"Fixed; tested in sweep {cfg['sweep']['episodes']}"],
        ['Exploration (epsilon)', f"{cfg['epsilon_start']} → {cfg['epsilon_end']} over first {int(cfg['decay_fraction'] * 100)}%", '`epsilon_*`, `decay_fraction`', 'Fixed design choice'],
        ['Training seeds', f"{len(cfg['training_seeds'])} runs", '`training_seeds`',
         'Independent runs, so results and the learning-rate choice can be checked against training luck'],
        ['Validation / test missions', f"{cfg['validation_episodes']} / {cfg['test_episodes']}", '`validation_*`, `test_*`', 'Separate seed ranges, no overlap'],
        ['Batch size / epochs', 'not applicable', '–', 'Tabular Q-learning updates one cell after every move (online); there are no batches or epochs'],
    ]
    reward_rows = [['Survivor detected', rw['detection']], ['Sector surveyed', rw['coverage']],
                   ['Per energy unit used', rw['energy']], ['Hazard event', rw['hazard']],
                   ['Command rejected', rw['api_failure']]]
    out = [f"""## Section 2 – Training and hyperparameter rationale

**Update rule (the whole learning step)**
- target = reward + gamma × (best predicted value among allowed moves in the next situation); just the reward if the mission ended
- new value = old value + alpha × (target − old value)
- Worked example used in the code docs: old 5, reward 2, best next 6, alpha 0.05, gamma 0.95 → target 7.7, new 5.135 (`q_update()` in experiment.py).

**Hyperparameters** (all set in `CONFIG` at the top of experiment.py)

{table(['Setting', 'Value', 'CONFIG key', 'How it was chosen'], hyper)}

**Reward structure**

{table(['Event', 'Reward'], reward_rows)}

**Learning-rate selection (validation missions {cfg['validation_seed_start']}–{cfg['validation_seed_start'] + cfg['validation_episodes'] - 1})**

{table(['Alpha', 'Validation reward, mean ± 95% CI across runs', 'Range over runs', 'Runs', 'Situations learned (mean)'], val_rows)}

Selected: **{s['selected_alpha']}** (highest mean). The test missions were not used for this choice. Per-run values are in `evidence/validation.csv`.
- {('Learning rates whose interval overlaps the selected one: ' + ', '.join(str(a) for a in overlap) + ' – so the choice among them is not clear-cut; report it as "highest mean", not "proven best".') if overlap else 'No other learning rate overlaps the selected one, so the choice is clear.'}
"""]
    if curve:
        out.append(f"Learning curve (selected alpha, seed {cfg['representative_seed']}): mean training reward "
                   f"{float(curve[0]['mean_training_return']):.2f} in the first 1,000 missions → "
                   f"{float(curve[-1]['mean_training_return']):.2f} in the last 1,000.\n")
    if rob:
        sweep = rob['hyperparameter_sweep']
        srows = [[v['parameter'], f"{v['value']:g}" + (' (used)' if v['is_default'] else ''),
                  f"{v['standard_return']:.2f}", '–' if v['is_default'] else diff(v['diff_vs_used']),
                  '–' if v['is_default'] else sig(v['diff_vs_used']), f"{v['detections']:.3f}", f"{v['hazards']:.3f}"]
                 for v in sweep.values()]
        better = [f"{v['parameter']} = {v['value']:g}" for v in sweep.values()
                  if not v['is_default'] and v['diff_vs_used']['significant'] and v['diff_vs_used']['diff'] > 0]
        worse = [f"{v['parameter']} = {v['value']:g}" for v in sweep.values()
                 if not v['is_default'] and v['diff_vs_used']['significant'] and v['diff_vs_used']['diff'] < 0]
        sm, f, lt = rob['shifted_map'], rob['forgetting'], rob['loop_trap']
        smr, smc, fc, lc = sm['results'], sm['comparisons'], f['comparisons'], lt['comparisons']
        out.append(f"""**Sweep of the other settings** (validation missions only; reward recomputed with the original weights so all rows are comparable)

{table(['Parameter', 'Value', 'Validation reward', 'Difference vs used [95% CI]', 'Significant?', 'Survivors', 'Hazards'], srows)}

- Significantly better than the value used: {', '.join(better) if better else 'none'}. Significantly worse: {', '.join(worse) if worse else 'none'}.
- {CI_NOTE}
- Report this honestly: the values used were design choices; the sweep shows how much they matter, not that they are optimal.

**Risks of an overfitted policy – measured, not just described**

All numbers in A–C are means over the {len(cfg['training_seeds'])} training seeds, so they differ slightly from the seed-{cfg['representative_seed']} test results in Section 1.

*A. Overfitting to one map.* Survivor chances changed from {sm['original_presence']} to {sm['shifted_presence']}; test missions, nominal conditions.

{table(['Approach', 'Reward', 'Survivors', 'Fallback use'], [[v['description'], f"{v['standard_return']:.2f}", f"{v['detections']:.3f}", f"{v['fallback_pct']:.2f}%"] for v in smr.values()])}

- Old Q-table on the new map − retrained table: {diff(smc['old_table_minus_retrained'])} (significant: {sig(smc['old_table_minus_retrained'])}).
- Old Q-table − the simple greedy rule given the new chances: {diff(smc['old_table_minus_greedy'])} (significant: {sig(smc['old_table_minus_greedy'])}).
- Compare rows on the *new* map only. The last row is on the original map, which offers different total reward, so it is a reference, not a like-for-like comparison.

*B. Catastrophic forgetting.* The trained table was trained for {f['extra_stress_episodes']:,} more missions on stress conditions only (5% exploration), then re-tested.

{table(['Tested on', 'Reward before', 'Reward after', 'After − before [95% CI]', 'Significant?'], [['Nominal missions', f"{f['nominal_return_before']:.2f}", f"{f['nominal_return_after']:.2f}", diff(fc['nominal_after_minus_before']), sig(fc['nominal_after_minus_before'])], ['Stress missions', f"{f['stress_return_before']:.2f}", f"{f['stress_return_after']:.2f}", diff(fc['stress_after_minus_before']), sig(fc['stress_after_minus_before'])]])}

- The preferred move changed in {f['best_move_changed_pct']:.1f}% of previously learned situations.

*C. Agent loop traps.* Retry guard: after {lt['retry_limit']} rejections in a row for one drone, avoid that drone; if only that drone could act, stop and hand back to the supervisor.

{table(['Condition', 'Guard', 'Rejected commands', 'Identical re-sends', 'Hit command limit', 'Stopped by guard', 'Survivors', 'Reward'], [[k.split('_')[0] if not k.startswith('link') else 'link fault (drone B broken)', 'on' if k.endswith('retry_guard') else 'off', f"{v['api_failures']:.2f}", f"{v['repeated_retries']:.2f}", f"{v['hit_attempt_cap_pct']:.1f}%", f"{v['stopped_by_guard_pct']:.1f}%", f"{v['detections']:.3f}", f"{v['standard_return']:.2f}"] for k, v in lt.items() if isinstance(v, dict) and k != 'comparisons'])}

{table(['Guard on − off', 'Reward [95% CI]', 'Rejected commands [95% CI]', 'Survivors [95% CI]'], [[label, diff(lc[f'{c}_guard_minus_no_guard_return']), diff(lc[f'{c}_guard_minus_no_guard_rejections']), diff(lc[f'{c}_guard_minus_no_guard_survivors'], 3)] for c, label in [('stress', 'random rejections (stress)'), ('link_fault', 'broken link')]])}

- With a broken link and no guard, the policy keeps re-sending to the broken drone until the command limit; the guard breaks that loop.
- With random (temporary) rejections the guard raises false alarms and stops missions early – a trade-off worth discussing.
""")
    if extra:
        mt = extra['mixed_training']
        mrows = [[cond, f"{mt[cond]['baseline']['return']:.2f}", f"{mt[cond]['mixed']['return']:.2f}",
                  diff(mt[cond]['mixed_minus_baseline_return']), sig(mt[cond]['mixed_minus_baseline_return']),
                  f"{mt[cond]['baseline']['fallback_pct']:.1f}% → {mt[cond]['mixed']['fallback_pct']:.1f}%"]
                 for cond in ('nominal', 'stress')]
        mix = ', '.join(f'{int(v * 100)}% {k}' for k, v in mt['mix'].items())
        out.append(f"""**Side experiment – mixed-condition training** (baseline unchanged; `extra_experiments.py`)

The baseline table practised nominal missions only. A second set of tables practised {mix} missions.

{table(['Tested on', 'Nominal-only training', 'Mixed training', 'Mixed − nominal-only [95% CI]', 'Significant?', 'Fallback use'], mrows)}

- Practising some stress missions means far fewer unseen situations under stress (fallback use above).
- The weather severity is still not part of the state, so one table must serve both conditions.
""")
    out.append('Figures: `figures/fig1_learning_curve.png`, `figures/fig2_learning_rate_selection.png`, '
               '`figures/fig5_hyperparameter_sweep.png`, `figures/fig6_overfitting_tests.png`, '
               '`figures/fig10_side_experiments.png` (panel A), `figures/fig11_paired_differences.png`.')
    return '\n'.join(out)


def section_stress(st):
    if not st:
        return ''
    rows = [[k, st['invariants'][k], f"{st['checks_by_invariant'][k]:,}",
             'PASS' if not any(v['invariant'] == k for v in st['violations']) else 'FAIL'] for k in st['invariants']]
    rng = st['ranges_tested']
    return f"""### Randomised stress test of the safety rules (`stress_test.py`)

{st['scenarios']} random scenarios × {st['settings']['missions_per_scenario']} missions × 4 modes = **{st['missions_total']:,} missions** and **{st['checks_total']:,} safety checks**, with **{st['violations_total']} violations**. Each scenario draws: command-rejection rate {rng['api_failure'][0]}–{rng['api_failure'][1]}, sensor detection {rng['sensitivity'][0]}–{rng['sensitivity'][1]}, wind {rng['wind_probability'][0]}–{rng['wind_probability'][1]}, hazard multiplier {rng['hazard_multiplier'][0]}–{rng['hazard_multiplier'][1]}, a broken drone link in {int(rng['broken_drone_share'] * 100)}% of scenarios, lost acknowledgements {rng['ack_loss'][0]}–{rng['ack_loss'][1]}, false detections {rng['false_positive'][0]}–{rng['false_positive'][1]}, a random closure time, and corrupted telemetry on {rng['corrupt_prob_per_step']:.0%} of steps.

{table(['Invariant', 'Rule that must never break', 'Checks run', 'Result'], rows)}

- A mutation check confirms the test can fail: a deliberately broken dispatcher that ignores closed sectors is caught (`checks.json`).
- Limits: it explores combinations of the fault types written into the simulator; it does not test fault types nobody modelled, and it checks safety, not performance.
"""


def section_edge_cases(s, cfg, edge, rob, ev):
    if not edge:
        return ''
    n, h, c, val = edge['summary']['naive'], edge['summary']['handled'], edge['comparisons'], edge['input_validation']
    ec = edge['settings']
    fallback = s['results']['stress_learned']['fallback_pct']
    lt = (rob or {}).get('loop_trap', {})
    link_n, link_g = lt.get('link_fault_no_guard'), lt.get('link_fault_retry_guard')
    rows = [
        ['Situation never seen in training', 'Greedy fallback; flagged for human approval',
         f'Fallback used in {fallback:.1f}% of stress decisions', 'check: approval gate flags'],
        ['Command rejected before execution', 'No energy used, no coverage; attempt counted', 'Every mission',
         'check: rejected command'],
        ['Drone radio link permanently broken', 'Retry guard avoids the drone; stops and hands back if it is the only option',
         (f"Command-limit hits {link_n['hit_attempt_cap_pct']:.0f}% → {link_g['hit_attempt_cap_pct']:.0f}%" if link_n else '–'),
         'check: retry guard'],
        ['Acknowledgement lost after a completed survey', 'Command marked PENDING; telemetry queried; view corrected',
         f"Duplicate flights {n['duplicate_flights']:.2f} → {h['duplicate_flights']:.2f}; battery overruns "
         f"{n['battery_overruns']:.2f} → {h['battery_overruns']:.2f} per mission", 'check: lost acknowledgements'],
        ['False survivor detection', f"Bayesian posterior; below {ec['confirm_threshold']:.2f} the report is escalated for verification",
         f"False confirmations {n['confirmed_false']:.3f} → {h['confirmed_false']:.3f} per mission "
         f"({n['false_reports_share_pct']:.1f}% → {h['false_reports_share_pct']:.1f}% of confirmed reports)", 'check: Bayesian confirmation'],
        ['Supervisor closes a sector mid-mission', 'Sector removed from allowed moves immediately',
         f"Flights into closed sector {n['closure_violations']:.2f} → {h['closure_violations']:.2f} per mission", 'check: closed sectors'],
        ['Corrupted or out-of-range state', 'validate_state() refuses to act; holds for the supervisor',
         f"{val['malformed_states_held']}/{val['malformed_states_tested']} malformed states held; all valid states accepted",
         'check: invalid states'],
        ['Risky move refused by supervisor', 'Next-best move offered; abstains if none acceptable', 'Supervisor study', 'check: human refusals'],
        ['Supervisor gives no answer', 'Treated as not approved', 'By design', 'code: ask_human()'],
        ['No safe or affordable move left', 'Mission ends cleanly', 'Every mission', 'check: mission ends'],
        ['Hazard estimate out of date (storm)', 'Approval gate, or weather-aware filter', 'See Section 3', 'supervisor study, filter test'],
    ]
    return f"""### Edge-case handling (`edge_cases.py`, `explain.py`, `experiment.py`)

Each edge case has a defined response and is verified by an automatic check. The four new cases were measured on the same {cfg['test_episodes']} test missions ({ec['base_condition']} conditions, {ec['ack_loss']:.0%} of acknowledgements lost, {ec['false_positive']:.0%} false detections, one sector closed after {ec['closure_attempt']} commands) for a naive dispatcher and for the handled one.

{table(['Edge case', 'System response', 'Measured effect (naive → handled)', 'Verified by'], rows)}

Paired differences, handled − naive (95% CI): reward {diff(c['handled_minus_naive_return'])} (significant: {sig(c['handled_minus_naive_return'])}); survivors found {diff(c['handled_minus_naive_detections'], 3)}; true survivors confirmed directly {diff(c['handled_minus_naive_confirmed_true'], 3)}.

- **Cascading error, shown in `evidence/edge_case_example.txt`:** one lost acknowledgement → the naive dispatcher believes the battery is still full → it re-sends the same flight → the drone aborts repeatedly until the command limit. Reconciliation breaks the chain.
- **Trade-off:** Bayesian confirmation removes most false reports, but {h['escalated_true']:.2f} true survivors per mission are sent for verification instead of being confirmed at once – safer, but slower. Reconciliation also costs a command slot, so slightly fewer survivors are found.
- **Still not handled:** moving survivors, simultaneous flights and collisions, communication loss between drones, and verification flights themselves (escalated detections are counted, not re-flown).

Figure: `figures/fig12_edge_cases.png`."""


def section_xai(cfg, sup, extra, ev):
    excerpt = approval_excerpt(ev / 'explanation_stress_vigilant.txt')
    rules = cfg['approval']
    out = [f"""## Section 3 – Human-agent collaboration (explainable AI)

**What the agent shows before it acts** (implemented in explain.py, simulated – there is no real user interface)
- The situation in plain words, and whether the decision comes from the Q-table or the fallback rule.
- How often that situation was seen in training (low count = low confidence).
- The recommended move with its predicted mission value, expected reward this step, energy cost and hazard estimate.
- The next-best alternatives, and every blocked move with the reason it is blocked.
- The multi-step plan (the moves it intends to make next if every command succeeds).
- A risk check: proceed automatically, or **needs human approval** with the reasons.

**When approval is required** (`CONFIG['approval']`)

{table(['Rule', 'Setting'], [['Hazard estimate of the chosen sector at or above', rules['hazard_flag']], ['Situation seen in training fewer than (times)', rules['min_visits']], ['Rejected commands in this mission at least', rules['rejections_flag']], ['Situation never seen in training', 'always flagged']])}

If the supervisor refuses, the agent offers its next-best move; if nothing acceptable remains it abstains and reports back. The thresholds are design choices, not tuned values.

**One-line summary first:** every explanation opens with a single line giving the move, whether it needs approval and why, and (where an operations-centre report exists) the current hazard beside the agent's estimate, e.g. `>> B->3 NEEDS APPROVAL - hazard estimate 0.12 >= approval level 0.10 | field report: current hazard 0.24 ABOVE safety limit`.

**Live human-in-the-loop mode:** `python explain.py --condition stress --supervisor human` pauses at every flagged move and waits for a real person to type y (approve), n (refuse) or q (stop the mission). After a refusal it asks for a reason (hazard concern, priority change, doubt about detection, other). Sessions are saved to `evidence/human_session_*.txt` and each decision is appended to `evidence/feedback_log.jsonl`. This mode is for demonstration only; it was not part of the measured study.
"""]
    fb = read_json(ev / 'feedback_review.json')
    if fb:
        srows = [[s['type'], s['evidence'], s['status']] for s in fb['suggestions']]
        out.append(f"""**Structured feedback, parsed safely** (`feedback_review.py`): {fb['records']:,} logged decisions ({', '.join(f'{k}: {v:,}' for k, v in fb['sources'].items())}) were summarised by flag type, refusal reason and sector. The tool only suggests rule reviews; it never changes the model or the rules, because learning automatically from overrides could absorb human mistakes.

{table(['Suggestion', 'Evidence', 'Status'], srows) if srows else 'No suggestions were supported by the evidence.'}
""")
    if excerpt:
        out.append(f"**Real excerpt** (`evidence/explanation_stress_vigilant.txt`):\n\n```text\n{excerpt}\n```\n")
    if sup:
        res, comp = sup['results'], sup['comparisons']
        names = {'autonomous': 'No approval gate', 'always_approve': 'Always approves (over-trust)',
                 'vigilant': 'Checks current hazard report', 'always_reject': 'Always refuses (under-trust)'}
        rows = [[cond, names[s], ci({'mean': r['return'], 'ci95_halfwidth': r['return_ci95']}),
                 f"{r['detections']:.3f}", f"{r['hazards']:.3f}", f"{r['approval_requests_per_mission']:.2f}",
                 r['approved_unsafe_total'], r['abstained_missions']]
                for key, r in res.items() for cond, s in [key.split('_', 1)]]
        crows = [[label, diff(comp[f'stress_{key}_hazards'], 3), diff(comp[f'stress_{key}_detections'], 3),
                  diff(comp[f'stress_{key}_return'])]
                 for key, label in [('vigilant_minus_always_approve', 'Checks evidence − always approves'),
                                    ('always_reject_minus_always_approve', 'Always refuses − always approves'),
                                    ('always_approve_minus_autonomous', 'Always approves − no gate')]]
        a, v, auto = res['stress_always_approve'], res['stress_vigilant'], res['stress_autonomous']
        out.append(f"""**Over-trust (automation bias) study** – the same {a['missions']} test missions for each supervisor type.
Under stress the real hazard is {cfg['stress']['hazard_multiplier']}× higher, but the agent still uses normal-weather estimates – so some of its recommendations are faulty.

{table(['Condition', 'Supervisor', 'Reward (95% CI)', 'Survivors', 'Hazards', 'Requests / mission', 'Unsafe requests approved', 'Missions abstained'], rows)}

Paired differences under stress (95% CI):

{table(['Comparison', 'Hazards', 'Survivors', 'Reward'], crows)}

- Always-approve under stress approved {a['approved_unsafe_total']:,} unsafe requests and ended with exactly the same hazards as having no gate ({a['hazards']:.3f} vs {auto['hazards']:.3f}): a rubber-stamp gate adds no safety.
- Checking the evidence cut hazards from {a['hazards']:.3f} to {v['hazards']:.3f} per mission, but survivors found fell from {a['detections']:.3f} to {v['detections']:.3f} – safety versus effectiveness.
- Always refusing (under-trust) gives the lowest hazards but also the fewest survivors.
- Limits: supervisors are simulated rules, not people; the vigilant supervisor is assumed to know the true current hazard level; no user study was run.
""")
    if extra:
        w = extra['weather_aware_filter']
        out.append(f"""**Side experiment – weather-aware risk filter** (baseline unchanged; `extra_experiments.py`)

Instead of relying on a human to catch out-of-date hazard estimates, the filter itself multiplies hazards by the current condition ({w['filter_multiplier']}× under stress), which blocks sectors {w['sectors_blocked_when_aware']}.

{table(['Stress missions', 'Normal-weather filter', 'Weather-aware filter', 'Aware − normal [95% CI]'], [['Hazard events', f"{w['baseline']['hazards']:.3f}", f"{w['aware']['hazards']:.3f}", diff(w['aware_minus_baseline_hazards'], 3)], ['Survivors found', f"{w['baseline']['detections']:.3f}", f"{w['aware']['detections']:.3f}", diff(w['aware_minus_baseline_detections'], 3)], ['Reward', f"{w['baseline']['return']:.2f}", f"{w['aware']['return']:.2f}", diff(w['aware_minus_baseline_return'])]])}

- It automates what the vigilant supervisor did by hand, with a similar result: far fewer hazards, fewer survivors.
- It needs a reliable, current hazard report – the simulation hands it the true value, which is optimistic.
""")
    out.append('Figures: `figures/fig13_decision_explanation.png` (the explanation shown before a high-risk action, '
               'rebuilt from a real stress mission), `figures/fig7_supervisor_study.png`, '
               '`figures/fig10_side_experiments.png` (panel B). '
               'Full logs: `evidence/approval_log.csv`, `evidence/supervisor_missions.csv`.')
    return '\n'.join(out)


def section_evidence(ev, checks):
    files = [
        ('config.json', 'Every setting used for the run', 'Yes – configuration'),
        ('run_output.log', 'Console output of the latest full run, with date and settings', 'Yes – execution log'),
        ('validation.csv', 'Learning-rate selection (9 training runs)', 'Yes'),
        ('summary.json', 'All test results, paired comparison, code fingerprint', 'Yes – evaluation'),
        ('q_table.json', 'The trained model itself', 'Yes – open and point at a value'),
        ('visit_counts.json', 'How often each situation was seen in training', 'Optional'),
        ('learning_curve.csv / learning_curves_all.csv', 'Training progress', 'Via Figure 1'),
        ('trace_nominal.json / trace_stress.json', 'Every decision of test mission 20000', 'Yes – execution log'),
        ('explanation_*.txt', 'Readable decision transcripts with approval requests', 'Yes – XAI'),
        ('explanation_first_decision.json', 'Machine-readable explanation of one decision', 'Optional'),
        ('supervisor_study.json, approval_log.csv', 'Over-trust study results and every approval request', 'Via Figure 7'),
        ('hyperparameter_sweep.csv', 'Extra settings tested', 'Via Figure 5'),
        ('shifted_map_test.csv, forgetting_test.csv, loop_trap_test.csv, robustness.json', 'Overfitting / forgetting / loop-trap tests', 'Via Figure 6'),
        ('optimal_benchmark.csv, extra_experiments.json', 'Best-possible benchmark and side experiments', 'Via Figures 9–10'),
        ('mixed_training_test.csv, weather_aware_filter_test.csv', 'Side-experiment results', 'Optional'),
        ('edge_cases.json, edge_case_missions.csv', 'Edge-case study: naive vs handled', 'Via Figure 12'),
        ('edge_case_example.txt', 'One mission showing a cascading error and its handling', 'Yes – edge cases'),
        ('stress_test.json, stress_test_scenarios.csv', 'Randomised stress test: checks run, violations, scenarios', 'Yes – robustness'),
        ('feedback_review.json, feedback_suggestions.md', 'Parsed supervisor feedback and suggested rule reviews (never applied)', 'Yes – human feedback'),
        ('feedback_log.jsonl', 'Decisions from live human sessions, with refusal reasons (only after a human session)', 'Optional'),
        ('nominal_*.csv / stress_*.csv', 'One row per test mission', 'Optional'),
        ('checks.json', 'Automatic correctness checks', 'Yes'),
    ]
    out = [f"""## Section 4 – Evidence (configuration, logs, evaluation)

{table(['File in evidence/', 'What it shows', 'Screenshot?'], [[f'`{a}`', b, c] for a, b, c in files])}

**Reproduce everything:** `python run_all.py` (about 1–2 minutes). **Fast live demo:** `python run_all.py --quick` (writes to `quick_demo/`, official evidence untouched). **Explain one mission live:** `python explain.py --condition stress`.

Figure 11 (`figures/fig11_paired_differences.png`) summarises every key comparison with its 95% confidence interval on one chart.
"""]
    if checks:
        out.append(table(['Automatic check', 'Result'], [[k.replace('_', ' '), v] for k, v in checks.items()]))
    return '\n'.join(out)


def section_claims():
    return """## What the report can and cannot claim

**Can claim (implemented and measured in simulation)**
- A tabular Q-learning dispatcher was trained, tuned on validation missions and tested on separate missions.
- It beat the greedy rule in nominal conditions; under stress the advantage was not statistically established (check the CI in Section 1).
- Its exact expected reward is close to, but below, the best possible policy (see the benchmark in Section 1).
- Before each move the code produces an explanation, a multi-step plan and a risk check; flagged moves go to a (simulated) supervisor.
- Over-trust, overfitting to one map, forgetting and loop traps were each demonstrated with measured numbers and confidence intervals.
- Lost acknowledgements, false detections, mid-mission closures and corrupted input are each handled by a defined strategy, verified by automatic checks and measured against a naive dispatcher.
- No safety rule was broken across a randomised stress test of tens of thousands of missions mixing every modelled fault type.
- Results were validated across multiple independent training runs, with confidence intervals that include training-run variability.

**Cannot claim**
- Real drones, flight control, collision avoidance, image-based detection, or a working user interface.
- That the explanations were tested with real people, or that the supervisor rules model real human behaviour.
- That the hyperparameters or approval thresholds are optimal.
- That edge-case handling is complete or "flawless": moving survivors, collisions and drone-to-drone communication loss are not modelled.
- That results generalise beyond this small simulated scenario.
- Multi-agent reinforcement learning (it is one central policy)."""


def build_report_pack(base=ROOT):
    ev = base / 'evidence'
    s, cfg = read_json(ev / 'summary.json'), read_json(ev / 'config.json')
    if s is None or cfg is None:
        raise SystemExit('Run experiment.py first.')
    sup, rob = read_json(ev / 'supervisor_study.json'), read_json(ev / 'robustness.json')
    extra, checks = read_json(ev / 'extra_experiments.json'), read_json(ev / 'checks.json')
    edge, stress = read_json(ev / 'edge_cases.json'), read_json(ev / 'stress_test.json')
    q_states = len(read_json(ev / 'q_table.json') or {})
    header = f"""# Results pack for the Task 2 report

Generated {datetime.date.today().isoformat()} from `evidence/` (code fingerprint `{s['source_sha256'][:12]}`).
Regenerate with `python run_all.py` – never type numbers in by hand.

How to use this pack
- It lists facts, numbers, tables and figure files for each section of the Task 2 brief. Write the report in your own words.
- Every number comes from a file in `evidence/`; cite that file (or a screenshot of it) in the appendix.
- Figures are in `figures/`. Square brackets after a difference are its 95% confidence interval.
- Remember to declare AI-tool use as the brief requires."""
    parts = [header, section_glance(s, sup, rob, extra), section_model(s, cfg, q_states, extra),
             section_training(s, cfg, rob, extra, ev), section_edge_cases(s, cfg, edge, rob, ev), section_stress(stress),
             section_xai(cfg, sup, extra, ev),
             section_evidence(ev, checks), section_claims()]
    folder = base / 'report_pack'
    folder.mkdir(exist_ok=True)
    path = folder / 'RESULTS.md'
    path.write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
    print(f'  wrote {path.relative_to(base)}')
    return path


if __name__ == '__main__':
    build_report_pack()
