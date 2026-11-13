"""Draw report-ready charts (PNG) from the files in evidence/.

    python make_figures.py      (needs matplotlib:  pip install matplotlib)

Writes figures/fig1_... to figures/fig12_... . Run after experiment.py, explain.py, robustness.py,
extra_experiments.py and edge_cases.py (figures for missing studies are skipped).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e34948', '#898781'
POLICY_STYLE = {'learned': ('Learned (Q-table)', BLUE), 'greedy': ('Greedy rule', ORANGE),
                'random': ('Random', GRAY), 'unshielded': ('Learned, no risk filter', VIOLET)}
SUPERVISOR_STYLE = {'autonomous': ('No\napproval\ngate', GRAY),
                    'always_approve': ('Always\napproves\n(over-trust)', RED),
                    'vigilant': ('Vigilant\n(checks field\nreport)', AQUA),
                    'always_reject': ('Always\nrefuses\n(under-trust)', YELLOW)}


def read_csv(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def style(ax, title=None, ylabel=None, xlabel=None):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e', labelsize=9)
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, loc='left', color='#0b0b0b')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color='#52514e')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color='#52514e')


def save(fig, folder, name):
    path = folder / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor='white')
    print(f'  wrote {path.relative_to(folder.parent)}')


def fig_learning_curve(plt, ev, folder, summary):
    alpha = str(summary['selected_alpha'])
    rows = [r for r in read_csv(ev / 'learning_curves_all.csv') if r['alpha'] == alpha]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    seeds = sorted({r['training_seed'] for r in rows}, key=int)
    episodes = sorted({int(r['episode']) for r in rows})
    for i, seed in enumerate(seeds):
        pts = [r for r in rows if r['training_seed'] == seed]
        ax1.plot([int(p['episode']) for p in pts], [float(p['mean_training_return']) for p in pts],
                 color='#b5d4f4', linewidth=1, label='individual training runs' if i == 0 else None)
    mean = [sum(float(r['mean_training_return']) for r in rows if int(r['episode']) == e) /
            sum(1 for r in rows if int(r['episode']) == e) for e in episodes]
    ax1.plot(episodes, mean, color=BLUE, marker='o', markersize=4, linewidth=2.5,
             label=f'mean of {len(seeds)} training runs')
    style(ax1, f'Training progress (learning rate {alpha})', 'Mean reward per mission')
    ax1.legend(frameon=False, fontsize=8)
    first = [r for r in rows if r['training_seed'] == rows[0]['training_seed']]
    ax2.plot([int(p['episode']) for p in first], [float(p['epsilon']) for p in first], color=GRAY, linewidth=2)
    style(ax2, None, 'Exploration rate', 'Training missions completed')
    save(fig, folder, 'fig1_learning_curve.png')


def fig_alpha_selection(plt, ev, folder, summary):
    rows = read_csv(ev / 'validation.csv')
    alphas = sorted({r['alpha'] for r in rows}, key=float)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, a in enumerate(alphas):
        vals = [float(r['validation_return']) for r in rows if r['alpha'] == a]
        mean = sum(vals) / len(vals)
        chosen = float(a) == summary['selected_alpha']
        ax.scatter([i] * len(vals), vals, color=GRAY, s=18, zorder=3)
        ax.scatter([i], [mean], color=BLUE if chosen else '#85b7eb', marker='D', s=70, zorder=4,
                   edgecolor='white', linewidth=1.5)
        ax.text(i + 0.12, mean, f'mean {mean:.2f}' + (' (selected)' if chosen else ''), va='center', fontsize=8)
    ax.set_xticks(range(len(alphas)), [f'alpha = {a}' for a in alphas])
    ax.set_xlim(-0.5, len(alphas) - 0.2)
    style(ax, 'Learning-rate selection on validation missions', 'Validation reward (diamond = mean, dots = seeds)')
    save(fig, folder, 'fig2_learning_rate_selection.png')


def fig_policy_comparison(plt, ev, folder, summary):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, condition in zip(axes, ['nominal', 'stress']):
        for i, policy in enumerate(POLICY_STYLE):
            res = summary['results'][f'{condition}_{policy}']['return']
            label, colour = POLICY_STYLE[policy]
            ax.bar(i, res['mean'], width=0.6, color=colour, yerr=res['ci95_halfwidth'], capsize=3,
                   error_kw={'ecolor': '#52514e', 'linewidth': 1})
            ax.text(i, res['mean'] + res['ci95_halfwidth'] + 0.3, f"{res['mean']:.1f}", ha='center', fontsize=8)
        ax.set_xticks(range(4), ['Learned\n(Q-table)', 'Greedy\nrule', 'Random', 'Learned,\nno risk filter'], fontsize=8)
        style(ax, f'{condition.capitalize()} conditions', 'Mean reward per mission (95% CI)' if condition == 'nominal' else None)
    save(fig, folder, 'fig3_policy_comparison.png')


def fig_safety_tradeoff(plt, ev, folder, summary):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    markers = {'learned': 'o', 'greedy': 's', 'random': '^', 'unshielded': 'D'}
    for ax, condition in zip(axes, ['nominal', 'stress']):
        for policy, (label, colour) in POLICY_STYLE.items():
            res = summary['results'][f'{condition}_{policy}']
            ax.scatter(res['hazards']['mean'], res['detections']['mean'], s=60, color=colour,
                       marker=markers[policy], label=label, edgecolor='white', linewidth=1.5, zorder=3)
        style(ax, f'{condition.capitalize()} conditions', 'Survivors found per mission', 'Hazard events per mission')
        ax.grid(axis='x', color='#e1e0d9', linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=7, loc='lower right')
    save(fig, folder, 'fig4_safety_tradeoff.png')


def fig_sweep(plt, ev, folder, robustness):
    sweep = robustness['hyperparameter_sweep']
    rows = read_csv(ev / 'hyperparameter_sweep.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    titles = {'gamma': 'Discount factor (gamma)', 'episodes': 'Training missions', 'hazard_penalty': 'Hazard penalty'}
    for ax, parameter in zip(axes, titles):
        items = [v for v in sweep.values() if v['parameter'] == parameter]
        for i, item in enumerate(items):
            vals = [float(r['validation_return']) for r in rows
                    if r['parameter'] == parameter and float(r['value']) == float(item['value'])]
            ax.scatter([i] * len(vals), vals, color=GRAY, s=16, zorder=3)
            ax.scatter([i], [item['standard_return']], color=BLUE if item['is_default'] else '#85b7eb',
                       marker='D', s=60, zorder=4, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(items)), [f"{item['value']:g}" + ('\n(used)' if item['is_default'] else '') for item in items])
        ax.set_xlim(-0.5, len(items) - 0.5)
        style(ax, titles[parameter], 'Validation return' if parameter == 'gamma' else None)
    runs = len({r['training_seed'] for r in rows})
    fig.suptitle(f'Other settings tested on validation missions (diamond = mean of {runs} training runs, '
                 'dots = individual runs)',
                 fontsize=10, x=0.01, ha='left', color='#52514e')
    save(fig, folder, 'fig5_hyperparameter_sweep.png')


def fig_overfitting(plt, ev, folder, robustness):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    shifted = robustness['shifted_map']['results']
    # Only approaches tested on the NEW map: scores on different maps are not comparable.
    names = [('learned_original_map', 'Original Q-table', BLUE),
             ('greedy_updated_map', 'Greedy rule\n(new probabilities)', ORANGE),
             ('learned_retrained', 'Retrained\nQ-table', AQUA)]
    for i, (key, label, colour) in enumerate(names):
        axes[0].bar(i, shifted[key]['standard_return'], color=colour, width=0.6)
        axes[0].text(i, shifted[key]['standard_return'] + 0.3, f"{shifted[key]['standard_return']:.1f}", ha='center', fontsize=8)
    axes[0].set_xticks(range(len(names)), [n[1] for n in names], fontsize=8)
    axes[0].set_ylim(0, max(shifted[n[0]]['standard_return'] for n in names) * 1.15)
    style(axes[0], 'A. Changed survivor map', 'Mean return per mission')

    f = robustness['forgetting']
    x = [0, 1]
    width = 0.35
    before = [f['nominal_return_before'], f['stress_return_before']]
    after = [f['nominal_return_after'], f['stress_return_after']]
    axes[1].bar([i - width / 2 for i in x], before, width, color=BLUE, label='before')
    axes[1].bar([i + width / 2 for i in x], after, width,
                color=ORANGE, label=f"after {f['extra_stress_episodes']:,} stress-only missions", hatch='//', edgecolor='white')
    for i in x:
        axes[1].text(i - width / 2, before[i] + 0.3, f'{before[i]:.1f}', ha='center', fontsize=8)
        axes[1].text(i + width / 2, after[i] + 0.3, f'{after[i]:.1f}', ha='center', fontsize=8)
    axes[1].set_xticks(x, ['Tested: nominal', 'Tested: stress'], fontsize=8)
    axes[1].set_ylim(0, max(before + after) * 1.35)
    axes[1].legend(frameon=False, fontsize=7, loc='upper right')
    style(axes[1], 'B. Forgetting after retraining', 'Mean return per mission')

    loop = robustness['loop_trap']
    groups = [('stress', 'Random\nrejections'), ('link_fault', 'Drone B link\nbroken')]
    for i, (cond, label) in enumerate(groups):
        none, guard = loop[f'{cond}_no_guard']['api_failures'], loop[f'{cond}_retry_guard']['api_failures']
        axes[2].bar(i - width / 2, none, width, color=RED, label='no guard')
        axes[2].bar(i + width / 2, guard, width, color=AQUA,
                    label='retry guard', hatch='//', edgecolor='white')
        axes[2].text(i - width / 2, none + 0.15, f'{none:.1f}', ha='center', fontsize=8)
        axes[2].text(i + width / 2, guard + 0.15, f'{guard:.1f}', ha='center', fontsize=8)
    axes[2].set_xticks(range(2), [g[1] for g in groups], fontsize=8)
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles[:2], labels[:2], frameon=False, fontsize=7)
    style(axes[2], 'C. Loop trap: wasted commands', 'Rejected commands per mission')
    save(fig, folder, 'fig6_overfitting_tests.png')


def fig_supervisors(plt, ev, folder):
    study = read_json(ev / 'supervisor_study.json')['results']
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    metrics = [('hazards', 'Hazard events per mission'), ('detections', 'Survivors found per mission'),
               ('approved_unsafe_total', 'Unsafe requests approved (total)')]
    for ax, (metric, title) in zip(axes, metrics):
        for i, sup in enumerate(SUPERVISOR_STYLE):
            label, colour = SUPERVISOR_STYLE[sup]
            value = study[f'stress_{sup}'][metric]
            ax.bar(i, value, color=colour, width=0.6)
            ax.text(i, value, f'{value:.2f}' if metric != 'approved_unsafe_total' else f'{value:,}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(4), [SUPERVISOR_STYLE[s][0] for s in SUPERVISOR_STYLE], fontsize=7.5)
        style(ax, title)
    fig.suptitle('Stress conditions: effect of supervisor behaviour at the approval gate', fontsize=11, x=0.01, ha='left')
    save(fig, folder, 'fig7_supervisor_study.png')


def fig_q_values(plt, ev, folder, cfg):
    q = read_json(ev / 'q_table.json')
    n = len(cfg['sector_presence'])
    starts = [(f"0,{cfg['usable_energy_each']},{cfg['usable_energy_each']},0,0", 'Start, calm'),
              (f"0,{cfg['usable_energy_each']},{cfg['usable_energy_each']},1,0", 'Start, windy')]
    data = [q.get(key, [0.0] * 2 * n) for key, _ in starts]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    image = ax.imshow(data, cmap='Blues', aspect='auto')
    for r, row in enumerate(data):
        best = max(range(len(row)), key=lambda a: row[a])
        for c, value in enumerate(row):
            blocked = cfg['sector_hazard'][c % n] > cfg['risk_mask_threshold']
            text = 'blocked' if blocked else f'{value:.1f}'
            ax.text(c, r, text, ha='center', va='center', fontsize=8,
                    color='white' if value > max(row) * 0.6 else '#0b0b0b',
                    fontweight='bold' if c == best else 'normal')
    ax.set_xticks(range(2 * n), [f"{'AB'[a // n]}->{a % n}" for a in range(2 * n)], fontsize=8)
    ax.set_yticks(range(2), [s[1] for s in starts], fontsize=9)
    ax.set_title('Learned Q-values for the first decision (bold = chosen move)', fontsize=11, loc='left')
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    save(fig, folder, 'fig8_q_values_first_decision.png')


def fig_optimal(plt, ev, folder, extra):
    bench = extra['optimal_benchmark']
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    order = [('random', 'Random\n(baseline)', GRAY), ('greedy', 'Greedy rule', ORANGE),
             ('learned', 'Learned\n(Q-table)', BLUE), ('optimal', 'Exact planner\n(benchmark)', AQUA)]
    top = max(bench[c]['exact_expected_return']['optimal'] for c in ('nominal', 'stress'))
    for ax, condition in zip(axes, ['nominal', 'stress']):
        exact = bench[condition]['exact_expected_return']
        for i, (key, label, colour) in enumerate(order):
            ax.bar(i, exact[key], color=colour, width=0.6, hatch='//' if key == 'optimal' else None, edgecolor='white')
            pct = 100 * exact[key] / exact['optimal']
            ax.text(i, exact[key] + 0.3, f'{exact[key]:.1f}\n({pct:.0f}%)', ha='center', fontsize=8)
        ax.text(3, exact['optimal'] + 0.12 * top + 1.2, 'assumes known\nprobabilities', ha='center', fontsize=7,
                color='#52514e', style='italic')
        ax.set_xticks(range(4), [o[1] for o in order], fontsize=8)
        ax.set_ylim(0, top * 1.45)
        style(ax, f'{condition.capitalize()} conditions', 'Exact expected reward per mission' if condition == 'nominal' else None)
    fig.suptitle('Learned policy vs baselines and the exact planner (exact calculation, % of optimal in brackets)',
                 fontsize=10, x=0.01, ha='left', color='#52514e')
    save(fig, folder, 'fig9_optimal_benchmark.png')


def fig_side_experiments(plt, ev, folder, extra):
    mixed, aware = extra['mixed_training'], extra['weather_aware_filter']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    width = 0.35
    for i, condition in enumerate(['nominal', 'stress']):
        base, mix = mixed[condition]['baseline'], mixed[condition]['mixed']
        axes[0].bar(i - width / 2, base['return'], width, color=BLUE, label='trained on nominal only')
        axes[0].bar(i + width / 2, mix['return'], width, color=AQUA, hatch='//', edgecolor='white',
                    label='trained on mixed conditions')
        axes[0].text(i - width / 2, base['return'] + 0.3, f"fallback\n{base['fallback_pct']:.1f}%", ha='center', fontsize=7)
        axes[0].text(i + width / 2, mix['return'] + 0.3, f"fallback\n{mix['fallback_pct']:.1f}%", ha='center', fontsize=7)
    axes[0].set_xticks(range(2), ['Tested: nominal', 'Tested: stress'], fontsize=8)
    axes[0].set_ylim(0, max(mixed['nominal']['baseline']['return'], mixed['nominal']['mixed']['return']) * 1.35)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:2], labels[:2], frameon=False, fontsize=7, loc='upper right')
    style(axes[0], 'A. Mixed-condition training', 'Mean reward per mission')
    metrics = [('hazards', 'Hazard events'), ('detections', 'Survivors found')]
    for i, (metric, label) in enumerate(metrics):
        axes[1].bar(i - width / 2, aware['baseline'][metric], width, color=RED, label='normal-weather filter')
        axes[1].bar(i + width / 2, aware['aware'][metric], width, color=AQUA, hatch='//', edgecolor='white',
                    label='weather-aware filter')
    axes[1].set_xticks(range(2), [m[1] for m in metrics], fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:2], labels[:2], frameon=False, fontsize=7)
    style(axes[1], 'B. Weather-aware risk filter (stress)', 'Per mission')
    save(fig, folder, 'fig10_side_experiments.png')


def fig_edge_cases(plt, ev, folder, edge):
    naive, handled = edge['summary']['naive'], edge['summary']['handled']
    metrics = [('duplicate_flights', 'Duplicate\nflights'), ('battery_overruns', 'Battery\noverruns'),
               ('closure_violations', 'Flights into\nclosed sector'), ('confirmed_false', 'False survivor\nreports')]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    width = 0.36
    for i, (key, label) in enumerate(metrics):
        ax.bar(i - width / 2, naive[key], width, color=RED, label='naive dispatcher')
        ax.bar(i + width / 2, handled[key], width, color=AQUA, hatch='//', edgecolor='white', label='edge cases handled')
        ax.text(i - width / 2, naive[key] + 0.01, f'{naive[key]:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width / 2, handled[key] + 0.01, f'{handled[key]:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics], fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, fontsize=8)
    s = edge['settings']
    style(ax, f"Edge cases: lost acknowledgements ({s['ack_loss']:.0%}), false detections ({s['false_positive']:.0%}), "
              'mid-mission closure', 'Per mission')
    save(fig, folder, 'fig12_edge_cases.png')


def fig_paired_differences(plt, ev, folder, summary, robustness, extra, supervisor):
    """Forest plot: every key A-vs-B comparison with its 95% confidence interval."""
    items = []
    for p in summary['paired_comparisons']:
        items.append((f"Learned vs greedy ({p['condition']})", p['learned_minus_greedy'], p['ci95_halfwidth']))
    if extra:
        for c in ('nominal', 'stress'):
            d = extra['optimal_benchmark'][c]['optimal_minus_learned_test']
            items.append((f'Exact planner vs learned ({c})', d['diff'], d['ci95']))
        for c in ('nominal', 'stress'):
            d = extra['mixed_training'][c]['mixed_minus_baseline_return']
            items.append((f'Mixed vs nominal-only training ({c})', d['diff'], d['ci95']))
        d = extra['weather_aware_filter']['aware_minus_baseline_return']
        items.append(('Weather-aware vs normal filter (stress)', d['diff'], d['ci95']))
    if robustness:
        d = robustness['shifted_map']['comparisons']['old_table_minus_retrained']
        items.append(('New map: old table vs retrained', d['diff'], d['ci95']))
        d = robustness['forgetting']['comparisons']['nominal_after_minus_before']
        items.append(('After stress-only retraining (nominal)', d['diff'], d['ci95']))
        for c, label in [('stress', 'random rejections'), ('link_fault', 'broken link')]:
            d = robustness['loop_trap']['comparisons'][f'{c}_guard_minus_no_guard_return']
            items.append((f'Retry guard on vs off ({label})', d['diff'], d['ci95']))
    if supervisor:
        d = supervisor['comparisons']['stress_vigilant_minus_always_approve_return']
        items.append(('Vigilant vs always-approve supervisor (stress)', d['diff'], d['ci95']))
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(items) + 1.4))
    for i, (label, diff, half) in enumerate(reversed(items)):
        colour = BLUE if diff - half > 0 else RED if diff + half < 0 else GRAY
        ax.errorbar(diff, i, xerr=half, fmt='o', color=colour, ecolor=colour, capsize=3, markersize=6)
        ax.text(diff + half + 0.15, i, f'{diff:+.2f}', va='center', fontsize=8, color='#52514e')
    ax.axvline(0, color='#c3c2b7', linewidth=1)
    ax.set_yticks(range(len(items)), [i[0] for i in reversed(items)], fontsize=8)
    style(ax, 'Paired differences in mean reward (A minus B), 95% CI', None,
          'Difference in reward per mission\nblue = A higher   red = A lower   grey = not significant')
    ax.grid(axis='x', color='#e1e0d9', linewidth=0.8)
    ax.grid(axis='y', visible=False)
    save(fig, folder, 'fig11_paired_differences.png')


def decision_card_data(ev, cfg, seed, condition='stress', supervisor='vigilant'):
    """The first high-risk decision of a real mission, rebuilt with the system's own functions."""
    import explain
    from experiment import load_table, move_name, parse_state_key
    q_table = load_table(ev / 'q_table.json')
    visits = load_table(ev / 'visit_counts.json') if (ev / 'visit_counts.json').exists() else None
    _, requests, _ = explain.run_supervised_mission(q_table, visits, seed, condition, supervisor, cfg)
    if not requests:
        return None
    step = requests[0]['step']
    asked = [r for r in requests if r['step'] == step]
    state = parse_state_key(asked[0]['state'])
    rejections = asked[0]['rejections_so_far']
    expl = explain.explain_decision(q_table, visits, state, cfg, rejections, cfg[condition]['hazard_multiplier'])
    decided = {r['move']: r for r in asked}
    executed = None
    allowed = explain.allowed_actions(state, True, cfg)
    for action in explain.ranked_actions(q_table, state, allowed, cfg)[0]:
        name = move_name(action, cfg)
        if name in decided:
            if decided[name]['approved']:
                executed = name
                break
            continue
        executed = name
        break
    return {'seed': seed, 'condition': condition, 'step': step, 'expl': expl, 'decisions': asked,
            'executed': executed}


def fig_decision_card(plt, ev, folder, cfg, seed):
    from matplotlib.patches import FancyBboxPatch
    data = decision_card_data(ev, cfg, seed)
    if data is None:
        return
    e, rec = data['expl'], data['expl']['recommended']
    fig = plt.figure(figsize=(9.6, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ink, muted, border = '#0b0b0b', '#52514e', '#c3c2b7'

    def box(x, y, w, h, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                    facecolor=face, edgecolor=edge, linewidth=1.2))

    box(2, 3, 96, 94, '#ffffff', border)
    ax.text(5, 92.5, f"Decision explanation  |  mission {data['seed']}, {data['condition']} conditions, step {data['step']}",
            fontsize=11, color=muted, va='center')
    # 1. Summary line
    approval = e['needs_approval']
    box(5, 80, 90, 8, '#fdf3dc' if approval else '#e6f4ee', YELLOW if approval else AQUA)
    ax.text(7, 85.6, 'Summary', fontsize=8, color=muted, va='center')
    head, _, field = e['summary'].partition(' | ')
    ax.text(7, 82.3, head, fontsize=10.5, color=ink, va='center', fontweight='bold')
    if field:
        ax.text(93, 85.6, field, fontsize=9, color=RED if 'ABOVE' in field else muted, va='center', ha='right',
                fontweight='bold')
    # 2. Reasoning details
    groups = {}
    for b in e['blocked']:
        groups.setdefault(b['reason'], []).append(b['move'])
    alts = ', '.join(f"{a['move']} ({a['predicted_value']})" for a in e['alternatives'])
    rows = [('Situation', e['situation']),
            ('Source', f"{e['decision_source']}, seen {e['times_seen_in_training']:,} times in training"),
            ('Recommend', f"{rec['move']}: predicted mission value {rec['predicted_value']}, expected reward "
                          f"{rec['expected_step_reward']}, energy {rec['energy_cost']}, hazard estimate {rec['hazard_estimate']:.2f}"),
            ('Next best', alts),
            ('Blocked', '; '.join(f"{', '.join(m)}: {r}" for r, m in groups.items())),
            ('Plan', ' then '.join(p['move'] for p in e['plan']) + '  (if every command succeeds)')]
    y = 75
    for label, text in rows:
        ax.text(7, y, label, fontsize=9, color=muted, va='top', fontweight='bold')
        wrapped = __import__('textwrap').wrap(text, 92)
        ax.text(22, y, '\n'.join(wrapped), fontsize=9, color=ink, va='top', linespacing=1.35)
        y -= 4.6 * len(wrapped) + 1.6
    # 3. Risk check
    risk_y = y - 7
    box(5, risk_y, 90, 6.5, '#fbe9e9' if approval else '#e6f4ee', RED if approval else AQUA)
    ax.text(7, risk_y + 3.3, 'Risk check', fontsize=9, color=muted, va='center', fontweight='bold')
    ax.text(22, risk_y + 3.3, ('NEEDS HUMAN APPROVAL: ' + '; '.join(e['risk_flags'])) if approval
            else 'Low risk: proceeds automatically', fontsize=9, color=ink, va='center')
    # 4. Supervisor decisions
    y = risk_y - 6
    ax.text(7, y, 'Supervisor', fontsize=9, color=muted, va='top', fontweight='bold')
    for d in data['decisions']:
        verdict = 'APPROVED' if d['approved'] else 'REFUSED'
        reason = d['reason'].split(': ', 1)[-1] if d['reason'].startswith(('refused:', 'approved')) else d['reason']
        ax.text(22, y, f"{d['move']} {verdict}: {reason}", fontsize=9, va='top',
                color=AQUA if d['approved'] else RED)
        y -= 4.6
    if data['executed']:
        ax.text(22, y, f"Executed {data['executed']} instead: the highest-ranked action that was acceptable"
                if data['executed'] != rec['move'] else f"Executed {data['executed']}", fontsize=9, va='top', color=ink)
    save(fig, folder, 'fig13_decision_explanation.png')


def make_all_figures(base=ROOT):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib is not installed - skipping figures (pip install matplotlib)')
        return False
    ev, folder = base / 'evidence', base / 'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    summary, cfg = read_json(ev / 'summary.json'), read_json(ev / 'config.json')
    fig_learning_curve(plt, ev, folder, summary)
    fig_alpha_selection(plt, ev, folder, summary)
    fig_policy_comparison(plt, ev, folder, summary)
    fig_safety_tradeoff(plt, ev, folder, summary)
    if (ev / 'robustness.json').exists():
        robustness = read_json(ev / 'robustness.json')
        fig_sweep(plt, ev, folder, robustness)
        fig_overfitting(plt, ev, folder, robustness)
    if (ev / 'supervisor_study.json').exists():
        fig_supervisors(plt, ev, folder)
    fig_q_values(plt, ev, folder, cfg)
    extra = read_json(ev / 'extra_experiments.json') if (ev / 'extra_experiments.json').exists() else None
    robustness = read_json(ev / 'robustness.json') if (ev / 'robustness.json').exists() else None
    supervisor = read_json(ev / 'supervisor_study.json') if (ev / 'supervisor_study.json').exists() else None
    if extra:
        fig_optimal(plt, ev, folder, extra)
        fig_side_experiments(plt, ev, folder, extra)
    fig_paired_differences(plt, ev, folder, summary, robustness, extra, supervisor)
    if (ev / 'edge_cases.json').exists():
        fig_edge_cases(plt, ev, folder, read_json(ev / 'edge_cases.json'))
    if (ev / 'q_table.json').exists():
        fig_decision_card(plt, ev, folder, cfg, cfg['test_seed_start'])
    plt.close('all')
    return True


if __name__ == '__main__':
    make_all_figures()
