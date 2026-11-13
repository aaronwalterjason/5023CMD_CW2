"""Explainable decisions, a human approval gate, and an over-trust (automation bias) study.

Uses the trained Q-table in evidence/. Nothing here changes the model; it only
explains, gates and logs the decisions the frozen Q-table makes.

Live demo (good for the viva):
    python explain.py                                   # mission 20000, nominal, vigilant supervisor
    python explain.py --condition stress --supervisor always_approve
    python explain.py --mission 20003 --condition stress
    python explain.py --condition stress --supervisor human      # YOU approve or refuse each risky move

Supervisor types
    autonomous      no approval gate at all (the plain learned policy)
    always_approve  approves every request without checking  -> over-trust / automation bias
    vigilant        checks the current hazard report before approving
    always_reject   refuses every flagged action               -> under-trust
    human           a real person types y / n / q at each request (live demo only, not in the study)
"""
import argparse
import datetime
import json
import statistics
from pathlib import Path

from experiment import (CONFIG, EVIDENCE, REJECTED, Mission, allowed_actions, block_reason, describe_state,
                        energy_cost, expected_step_reward, greedy_score, load_table, move_name, n_actions,
                        paired_ci, split_action, write_csv, write_json)
from edge_cases import validate_state

SUPERVISORS = ['autonomous', 'always_approve', 'vigilant', 'always_reject']   # used in the automated study
ALL_SUPERVISORS = SUPERVISORS + ['human']                                     # 'human' = interactive demo


def load_trained(evidence=EVIDENCE):
    q_table = load_table(evidence / 'q_table.json')
    visits_path = evidence / 'visit_counts.json'
    visits = load_table(visits_path) if visits_path.exists() else None
    return q_table, visits


def ranked_actions(q_table, state, allowed, cfg=CONFIG):
    """Allowed actions, best first. Uses the Q-table if the situation was seen in
    training, otherwise the greedy fallback rule. Returns (ranked, used_fallback)."""
    if state in q_table:
        row = q_table[state]
        return sorted(allowed, key=lambda a: (-row[a], a)), False
    return sorted(allowed, key=lambda a: (-greedy_score(state, a, cfg), a)), True


def plan_preview(q_table, state, cfg=CONFIG, max_steps=6):
    """The multi-step plan shown to the supervisor: the moves the agent intends to
    make, assuming every command succeeds."""
    plan = []
    for _ in range(max_steps):
        allowed = allowed_actions(state, True, cfg)
        if not allowed:
            break
        action = ranked_actions(q_table, state, allowed, cfg)[0][0]
        drone, sector = split_action(action, cfg)
        cost = energy_cost(drone, sector, state[3], cfg)
        plan.append({'move': move_name(action, cfg), 'energy_cost': cost,
                     'expected_reward': round(expected_step_reward(state, action, cfg), 2)})
        energy = [state[1], state[2]]
        energy[drone] -= cost
        state = (state[0] | (1 << sector), energy[0], energy[1], state[3], state[4] + 1)
    return plan


REASON_CODES = {'h': 'hazard_concern', 'p': 'priority_change', 'd': 'detection_doubt', 'o': 'other'}


def flag_types(flags):
    """Short category for each approval flag, for structured feedback logs."""
    types = []
    for f in flags:
        if f.startswith('hazard estimate'):
            types.append('hazard')
        elif 'training' in f:
            types.append('unfamiliar')
        elif 'rejected commands' in f:
            types.append('repeated_rejections')
    return types


def risk_flags(state, action, q_table, visits, cfg=CONFIG, rejections_so_far=0):
    """Reasons an action needs human approval. An empty list means it may run automatically."""
    rules = cfg['approval']
    flags = []
    hazard = cfg['sector_hazard'][split_action(action, cfg)[1]]
    if hazard >= rules['hazard_flag']:
        flags.append(f"hazard estimate {hazard:.2f} >= approval level {rules['hazard_flag']:.2f}")
    if state not in q_table:
        flags.append('situation never seen in training (fallback rule in use)')
    elif visits is not None and visits.get(state, 0) < rules['min_visits']:
        flags.append(f'situation seen only {visits.get(state, 0)} times in training')
    if rejections_so_far >= rules['rejections_flag']:
        flags.append(f'{rejections_so_far} rejected commands this mission: conditions may differ from training')
    return flags


def explain_decision(q_table, visits, state, cfg=CONFIG, rejections_so_far=0, field_multiplier=None):
    """Everything a supervisor needs to judge one recommendation.
    field_multiplier: if the operations centre has a current hazard report (the condition's hazard
    multiplier), it is shown beside the agent's own estimate as contradicting evidence."""
    allowed = allowed_actions(state, True, cfg)
    ranked, used_fallback = ranked_actions(q_table, state, allowed, cfg)
    row = q_table.get(state)

    def option(a):
        drone, sector = split_action(a, cfg)
        return {'move': move_name(a, cfg), 'sector_name': cfg['sector_names'][sector],
                'predicted_value': None if row is None else round(row[a], 2),
                'expected_step_reward': round(expected_step_reward(state, a, cfg), 2),
                'hazard_estimate': cfg['sector_hazard'][sector],
                'energy_cost': energy_cost(drone, sector, state[3], cfg)}

    blocked = [{'move': move_name(a, cfg), 'reason': block_reason(state, a, True, cfg)}
               for a in range(n_actions(cfg)) if a not in allowed]
    flags = risk_flags(state, ranked[0], q_table, visits, cfg, rejections_so_far) if ranked else []
    # One-line summary first, so the key decision and its risk can be checked at a glance.
    if not ranked:
        summary = 'No allowed moves: mission complete'
    else:
        rec = option(ranked[0])
        field = ''
        if field_multiplier is not None:
            current = rec['hazard_estimate'] * field_multiplier
            field = (f" | field report: current hazard {current:.2f}"
                     + (' ABOVE safety limit' if current > cfg['risk_mask_threshold'] else ''))
        summary = (f"{rec['move']} NEEDS APPROVAL - {flags[0]}{field}" if flags else
                   f"{rec['move']} proceeds automatically (hazard {rec['hazard_estimate']:.2f}, "
                   f"energy {rec['energy_cost']}){field}")
    return {'summary': summary, 'state': list(state), 'situation': describe_state(state, cfg),
            'decision_source': 'fallback rule (situation not in Q-table)' if used_fallback else 'Q-table',
            'times_seen_in_training': None if visits is None else visits.get(state, 0),
            'recommended': option(ranked[0]) if ranked else None,
            'alternatives': [option(a) for a in ranked[1:4]],
            'blocked': blocked, 'plan': plan_preview(q_table, state, cfg),
            'risk_flags': flags, 'needs_approval': bool(flags)}


def format_explanation(expl):
    """Turn explain_decision() output into the text a supervisor would read: summary first, details below."""
    lines = [f">> {expl['summary']}",
             f"Situation : {expl['situation']}",
             f"Source    : {expl['decision_source']}"
             + (f", seen {expl['times_seen_in_training']} times in training"
                if expl['times_seen_in_training'] is not None else '')]
    rec = expl['recommended']
    if rec is None:
        return '\n'.join(lines + ['No allowed moves: mission complete.'])
    value = '' if rec['predicted_value'] is None else f"predicted mission value {rec['predicted_value']}, "
    lines.append(f"Recommend : {rec['move']} ({value}expected reward this step {rec['expected_step_reward']}, "
                 f"energy {rec['energy_cost']}, hazard {rec['hazard_estimate']:.2f})")
    if expl['alternatives']:
        alts = ', '.join(f"{a['move']} ({a['predicted_value'] if a['predicted_value'] is not None else a['expected_step_reward']})"
                         for a in expl['alternatives'])
        lines.append(f'Next best : {alts}')
    groups = {}
    for b in expl['blocked']:
        groups.setdefault(b['reason'], []).append(b['move'])
    for reason, moves in groups.items():
        lines.append(f"Blocked   : {', '.join(moves)} - {reason}")
    lines.append('Plan      : ' + ' then '.join(p['move'] for p in expl['plan'])
                 + '  (if every command succeeds)')
    if expl['needs_approval']:
        lines.append('Risk check: NEEDS HUMAN APPROVAL - ' + '; '.join(expl['risk_flags']))
    else:
        lines.append('Risk check: low - proceeds automatically')
    return '\n'.join(lines)


def ask_reason():
    """Structured feedback: why did the supervisor refuse? Unknown or missing answers become 'other'."""
    try:
        answer = input('Reason?  [h = hazard concern, p = priority change, d = doubt about detection, o = other]: ')
    except EOFError:
        return 'other'
    return REASON_CODES.get(answer.strip().lower()[:1], 'other')


def ask_human(action, condition, cfg=CONFIG):
    """Ask a real person at the keyboard. Returns (approved, reason, reason_category);
    approved is None if they stop the mission. The person sees the same field report the vigilant
    supervisor uses, and decides for themselves. No answer counts as NOT approved."""
    sector = split_action(action, cfg)[1]
    current_hazard = cfg['sector_hazard'][sector] * cfg[condition]['hazard_multiplier']
    print(f"Field report: current hazard in sector {sector} is {current_hazard:.2f} "
          f"(agent's estimate {cfg['sector_hazard'][sector]:.2f}, safety limit {cfg['risk_mask_threshold']:.2f})")
    while True:
        try:
            answer = input(f'Approve {move_name(action, cfg)}?  [y = approve, n = refuse, q = stop mission]: ')
        except EOFError:
            print()
            return False, 'no answer received - not approved', 'no_answer'
        answer = answer.strip().lower()
        if answer in ('y', 'yes'):
            return True, 'approved by the human supervisor', 'approved'
        if answer in ('n', 'no'):
            category = ask_reason()
            return False, f'refused by the human supervisor ({category})', category
        if answer in ('q', 'quit', 'stop'):
            return None, 'the human supervisor stopped the mission', 'stopped'
        print('Please type y, n or q.')


def supervisor_decides(kind, action, condition, cfg=CONFIG):
    """Supervisor decision. Returns (approved, reason, reason_category).
    The vigilant supervisor knows the current hazard level (the agent only knows nominal estimates).
    'human' asks a real person; it may return approved=None, meaning stop the mission."""
    sector = split_action(action, cfg)[1]
    current_hazard = cfg['sector_hazard'][sector] * cfg[condition]['hazard_multiplier']
    if kind == 'human':
        return ask_human(action, condition, cfg)
    if kind == 'always_approve':
        return True, 'approved without checking', 'approved'
    if kind == 'always_reject':
        return False, 'refused: flagged actions are never approved', 'blanket_refusal'
    if current_hazard > cfg['risk_mask_threshold']:
        return (False, f"refused: current hazard {current_hazard:.2f} exceeds limit {cfg['risk_mask_threshold']:.2f}",
                'hazard_concern')
    return True, f'approved after check: current hazard {current_hazard:.2f} within limit', 'approved'


def run_supervised_mission(q_table, visits, seed, condition='nominal', supervisor='vigilant',
                           cfg=CONFIG, explain=False):
    """One mission with the approval gate. If a request is refused, the agent offers its
    next-best move; if nothing acceptable is left, it abstains (stops the mission).
    With supervisor='human' every line is printed as it happens and a real person decides.
    Returns (metrics_row, approval_requests, transcript_lines)."""
    mission = Mission(seed, condition, cfg)
    rejections = steps = 0
    requests, lines = [], []
    abstained = stopped_by_human = False
    live = supervisor == 'human'
    explain = explain or live

    def emit(text):
        lines.append(text)
        if live:
            print(text)

    while True:
        problems = validate_state(mission.state, cfg)
        if problems:                        # corrupted input: never act on it, hand back to the supervisor
            abstained = True
            if explain:
                emit('HOLD      : invalid state - ' + '; '.join(problems))
            break
        allowed = allowed_actions(mission.state, True, cfg)
        if not allowed:
            break
        state = mission.state
        if explain:
            emit(f'--- Step {steps + 1} ---')
            field = None if supervisor == 'autonomous' else cfg[condition]['hazard_multiplier']
            emit(format_explanation(explain_decision(q_table, visits, state, cfg, rejections, field)))
        chosen = None
        for action in ranked_actions(q_table, state, allowed, cfg)[0]:
            flags = [] if supervisor == 'autonomous' else risk_flags(state, action, q_table, visits, cfg, rejections)
            if not flags:
                chosen = action
                break
            if live and requests and requests[-1]['step'] == steps + 1:
                emit(f'Agent     : next-best option is {move_name(action, cfg)} - it also needs approval '
                     f"({'; '.join(flags)})")
            approved, reason, category = supervisor_decides(supervisor, action, condition, cfg)
            sector = split_action(action, cfg)[1]
            current_hazard = cfg['sector_hazard'][sector] * cfg[condition]['hazard_multiplier']
            requests.append({'seed': seed, 'condition': condition, 'supervisor': supervisor, 'step': steps + 1,
                             'state': ','.join(map(str, state)), 'rejections_so_far': rejections,
                             'move': move_name(action, cfg), 'sector': sector,
                             'flags': '; '.join(flags), 'flag_types': '|'.join(flag_types(flags)),
                             'approved': int(bool(approved)), 'reason': reason, 'reason_category': category,
                             'agent_hazard_estimate': cfg['sector_hazard'][sector],
                             'current_hazard': round(current_hazard, 3),
                             'unsafe': int(current_hazard > cfg['risk_mask_threshold'])})
            if approved is None:            # the human pressed q
                stopped_by_human = True
                break
            if explain:
                emit(f"Supervisor: {move_name(action, cfg)} {'APPROVED' if approved else 'REFUSED'} - {reason}")
            if approved:
                chosen = action
                break
        if chosen is None:
            abstained = True
            if explain:
                emit('Supervisor: mission stopped by the human supervisor' if stopped_by_human else
                     'Agent     : no acceptable move left - abstains and reports back to the supervisor')
            break
        reward, event = mission.step(chosen)
        steps += 1
        rejections += event == REJECTED
        if explain:
            outcome = 'command REJECTED (no energy used)' if event == REJECTED else 'survey completed'
            emit(f'Executed  : {move_name(chosen, cfg)} -> {outcome}, reward {round(reward, 2)}')
    row = {'seed': seed, 'condition': condition, 'supervisor': supervisor,
           'return': mission.total_reward, 'detections': mission.detections, 'hazards': mission.hazards,
           'coverage_pct': mission.surveyed / len(cfg['sector_presence']) * 100,
           'approval_requests': len(requests), 'approved': sum(r['approved'] for r in requests),
           'refused': sum(1 - r['approved'] for r in requests),
           'approved_unsafe': sum(r['approved'] and r['unsafe'] for r in requests),
           'abstained': int(abstained)}
    if explain:
        emit(f"=== Mission total: reward {round(mission.total_reward, 2)}, survivors found "
             f"{mission.detections}, hazards {mission.hazards}, approval requests {len(requests)}, "
             f"unsafe approvals {row['approved_unsafe']} ===")
    return row, requests, lines


def supervisor_study(q_table, visits, cfg=CONFIG, out=EVIDENCE):
    """Same 500 test missions for every supervisor type, in both conditions."""
    summary, all_rows, all_requests, rows_by = {}, [], [], {}
    for condition in ['nominal', 'stress']:
        for supervisor in SUPERVISORS:
            rows = []
            for k in range(cfg['test_episodes']):
                row, requests, _ = run_supervised_mission(q_table, visits, cfg['test_seed_start'] + k,
                                                          condition, supervisor, cfg)
                rows.append(row)
                all_requests += requests
            all_rows += rows
            rows_by[condition, supervisor] = rows
            n = len(rows)
            result = {m: statistics.mean(r[m] for r in rows) for m in ['return', 'detections', 'hazards', 'coverage_pct']}
            for m in ['return', 'detections', 'hazards']:
                vals = [r[m] for r in rows]
                result[m + '_ci95'] = 1.96 * statistics.stdev(vals) / n ** 0.5 if n > 1 else 0.0
            result.update({'approval_requests_per_mission': sum(r['approval_requests'] for r in rows) / n,
                           'approved_total': sum(r['approved'] for r in rows),
                           'refused_total': sum(r['refused'] for r in rows),
                           'approved_unsafe_total': sum(r['approved_unsafe'] for r in rows),
                           'abstained_missions': sum(r['abstained'] for r in rows), 'missions': n})
            summary[f'{condition}_{supervisor}'] = result
            print(f"  {condition:<8}{supervisor:<15} return {result['return']:6.2f}  survivors "
                  f"{result['detections']:.3f}  hazards {result['hazards']:.3f}  "
                  f"unsafe approvals {result['approved_unsafe_total']}")
    # Paired differences on the same missions, with 95% confidence intervals.
    comparisons = {}
    for condition in ['nominal', 'stress']:
        for a, b in [('vigilant', 'always_approve'), ('always_reject', 'always_approve'),
                     ('always_approve', 'autonomous')]:
            for metric in ['hazards', 'detections', 'return']:
                comparisons[f'{condition}_{a}_minus_{b}_{metric}'] = paired_ci(
                    [rows_by[condition, a]], [rows_by[condition, b]], lambda r, m=metric: r[m])
    write_csv(out / 'supervisor_missions.csv', all_rows)
    if all_requests:
        write_csv(out / 'approval_log.csv', all_requests)
    write_json(out / 'supervisor_study.json', {'approval_rules': cfg['approval'],
                                               'risk_mask_threshold': cfg['risk_mask_threshold'],
                                               'results': summary, 'comparisons': comparisons})
    return summary


def write_explanation_examples(q_table, visits, cfg=CONFIG, out=EVIDENCE):
    """Readable step-by-step transcripts for the report and the viva."""
    seed = cfg['test_seed_start']
    examples = [('nominal', 'vigilant'), ('stress', 'vigilant'), ('stress', 'always_approve')]
    for condition, supervisor in examples:
        _, _, lines = run_supervised_mission(q_table, visits, seed, condition, supervisor, cfg, explain=True)
        header = [f'Decision transcript - mission {seed}, {condition} conditions, supervisor: {supervisor}',
                  'Generated by explain.py from the trained Q-table (evidence/q_table.json).', '']
        path = out / f'explanation_{condition}_{supervisor}.txt'
        path.write_text('\n'.join(header + lines) + '\n', encoding='utf-8')
        print(f'  wrote {path.name}')
    # Machine-readable explanation of the very first decision, for anyone building a UI mock-up.
    first = Mission(seed, 'nominal', cfg).state
    write_json(out / 'explanation_first_decision.json', explain_decision(q_table, visits, first, cfg))


def run_all_explain(cfg=CONFIG, out=EVIDENCE):
    q_table, visits = load_trained(out)
    print('Decision transcripts')
    write_explanation_examples(q_table, visits, cfg, out)
    print('Supervisor (over-trust) study')
    return supervisor_study(q_table, visits, cfg, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explain one mission decision by decision.')
    parser.add_argument('--mission', type=int, default=CONFIG['test_seed_start'], help='mission seed')
    parser.add_argument('--condition', choices=['nominal', 'stress'], default='nominal')
    parser.add_argument('--supervisor', choices=ALL_SUPERVISORS, default='vigilant',
                        help="'human' = you approve or refuse each risky move at the keyboard")
    parser.add_argument('--evidence', type=Path, default=EVIDENCE, help='folder with q_table.json')
    args = parser.parse_args()
    q, v = load_trained(args.evidence)
    if args.supervisor == 'human':
        print(f'Human-in-the-loop demo - mission {args.mission}, {args.condition} conditions.')
        print('Risky moves wait for your decision. The field report shows the CURRENT hazard level;')
        print("the agent itself only knows normal-weather estimates.\n")
    row, requests, transcript = run_supervised_mission(q, v, args.mission, args.condition, args.supervisor,
                                                       explain=True)
    if args.supervisor == 'human':
        now = datetime.datetime.now()
        stamp = now.strftime('%Y%m%d_%H%M%S')
        path = args.evidence / f'human_session_{args.mission}_{args.condition}_{stamp}.txt'
        header = [f'Human-in-the-loop session - mission {args.mission}, {args.condition} conditions, '
                  f"{now.isoformat(timespec='seconds')}", '']
        path.write_text('\n'.join(header + transcript) + '\n', encoding='utf-8')
        # Structured feedback log, one JSON record per decision (read by feedback_review.py).
        with (args.evidence / 'feedback_log.jsonl').open('a', encoding='utf-8') as log:
            for r in requests:
                log.write(json.dumps({'session': stamp, 'timestamp': now.isoformat(timespec='seconds'), **r}) + '\n')
        print(f'\nSession saved to {path}')
        print(f"{len(requests)} decision(s) added to {args.evidence / 'feedback_log.jsonl'} "
              f"- run 'python feedback_review.py' to summarise all feedback")
    else:
        print('\n'.join(transcript))
