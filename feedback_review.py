"""Parse supervisor feedback and suggest rule reviews - WITHOUT changing anything.

    python feedback_review.py          (run after explain.py; also reads live human sessions)

Inputs (in evidence/):
    approval_log.csv     every approval request from the simulated supervisor study
    feedback_log.jsonl   every decision made by a real person in the live mode (explain.py --supervisor human)

Outputs:
    feedback_review.json     summary per supervisor type: approval rates by flag type, refusal reasons,
                             most-refused sectors, unsafe approvals
    feedback_suggestions.md  suggested rule reviews, each marked PENDING HUMAN SIGN-OFF

Safety by design: this script never edits CONFIG, the Q-table or any rule. Learning automatically from
overrides could absorb human mistakes, so every suggestion must be reviewed and applied by a person.
"""
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiment import CONFIG, EVIDENCE, write_json

# Supervisors whose refusals reflect judgement. Blanket policies (always approve / always refuse)
# are summarised but excluded from suggestions, because their decisions carry no information.
JUDGING = {'vigilant', 'human'}
MIN_REQUESTS = 20          # minimum evidence before a suggestion is made
HIGH_REFUSAL = 0.8         # a sector refused at least this often is flagged for review
LOW_VALUE_APPROVAL = 0.95  # a flag type approved at least this often may be too sensitive


def load_records(out):
    records = []
    path = out / 'approval_log.csv'
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                r['source'] = 'simulated study'
                records.append(r)
    human = out / 'feedback_log.jsonl'
    if human.exists():
        for line in human.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                r['source'] = 'live human session'
                records.append(r)
    for r in records:
        r['approved'] = int(r['approved'])
        r['unsafe'] = int(r['unsafe'])
        r['sector'] = int(r.get('sector', r['move'].split('->')[1]))
        r['current_hazard'] = float(r['current_hazard'])
        r['agent_hazard_estimate'] = float(r.get('agent_hazard_estimate', CONFIG['sector_hazard'][r['sector']]))
        r['flag_types'] = [t for t in str(r.get('flag_types', '')).split('|') if t]
        r['reason_category'] = r.get('reason_category') or ('approved' if r['approved'] else 'unrecorded')
    return records


def rate(rows, key='approved'):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def summarise(records):
    by_group = defaultdict(list)
    for r in records:
        by_group[(r['supervisor'], r['condition'])].append(r)
    summary = {}
    for (supervisor, condition), rows in sorted(by_group.items()):
        refused = [r for r in rows if not r['approved']]
        flag_rows = defaultdict(list)
        for r in rows:
            for t in r['flag_types']:
                flag_rows[t].append(r)
        summary[f'{supervisor}_{condition}'] = {
            'supervisor': supervisor, 'condition': condition, 'requests': len(rows),
            'approval_rate': rate(rows),
            'approval_rate_by_flag_type': {t: {'requests': len(v), 'approval_rate': rate(v)}
                                           for t, v in sorted(flag_rows.items())},
            'refusal_reasons': dict(Counter(r['reason_category'] for r in refused)),
            'most_refused_sectors': dict(Counter(r['sector'] for r in refused).most_common(3)),
            'unsafe_approvals': sum(r['approved'] and r['unsafe'] for r in rows)}
    return summary


def suggest(records):
    """Rule reviews supported by the evidence. Nothing here is ever applied automatically."""
    suggestions = []
    judged = [r for r in records if r['supervisor'] in JUDGING]
    # 1. Sectors that judging supervisors keep refusing on hazard grounds -> hazard estimate may be stale.
    groups = defaultdict(list)
    for r in judged:
        if 'hazard' in r['flag_types']:
            groups[(r['condition'], r['sector'])].append(r)
    for (condition, sector), rows in sorted(groups.items()):
        refusal = 1 - rate(rows)
        if len(rows) >= MIN_REQUESTS and refusal >= HIGH_REFUSAL:
            suggestions.append({
                'type': 'review hazard estimate', 'condition': condition, 'sector': sector,
                'evidence': (f'{refusal:.0%} of {len(rows)} hazard-flagged requests for sector {sector} were refused '
                             f'under {condition} conditions; agent estimate {rows[0]["agent_hazard_estimate"]:.2f}, '
                             f'mean field report {statistics.mean(r["current_hazard"] for r in rows):.2f}'),
                'possible_action': f'raise the hazard estimate for sector {sector} in {condition} conditions, '
                                   'or adopt a weather-aware risk filter',
                'status': 'PENDING HUMAN SIGN-OFF', 'applied': False})
    # 2. Flag types that judging supervisors almost always approve -> the approval rule may be too sensitive.
    by_flag = defaultdict(list)
    for r in judged:
        for t in r['flag_types']:
            by_flag[(r['condition'], t)].append(r)
    for (condition, flag), rows in sorted(by_flag.items()):
        unsafe = sum(r['approved'] and r['unsafe'] for r in rows)
        if len(rows) >= MIN_REQUESTS and rate(rows) >= LOW_VALUE_APPROVAL and unsafe == 0:
            suggestions.append({
                'type': 'review approval threshold', 'condition': condition, 'flag_type': flag,
                'evidence': f'{rate(rows):.0%} of {len(rows)} "{flag}" requests were approved under {condition} '
                            'conditions with no unsafe approvals',
                'possible_action': f'consider whether "{flag}" requests in {condition} conditions need approval, '
                                   'weighing supervisor workload against safety',
                'status': 'PENDING HUMAN SIGN-OFF', 'applied': False})
    # 3. Any supervisor type that approved unsafe requests -> automation-bias warning.
    unsafe_by = Counter(f"{r['supervisor']} ({r['condition']})" for r in records if r['approved'] and r['unsafe'])
    for who, count in unsafe_by.most_common():
        suggestions.append({
            'type': 'automation-bias warning', 'supervisor': who,
            'evidence': f'{count} requests were approved although the current hazard exceeded the safety limit',
            'possible_action': 'review supervisor training and make the field report more prominent',
            'status': 'PENDING HUMAN SIGN-OFF', 'applied': False})
    return suggestions


def write_markdown(path, summary, suggestions, n_records, sources):
    lines = ['# Feedback review - suggested rule reviews', '',
             f'Generated from {n_records} supervisor decisions ({", ".join(f"{k}: {v}" for k, v in sources.items())}).',
             'Nothing below has been applied. Each item needs review and sign-off by a person.', '']
    for i, s in enumerate(suggestions, 1):
        lines += [f"## {i}. {s['type'].capitalize()}", f"- Evidence: {s['evidence']}",
                  f"- Possible action: {s['possible_action']}", f"- Status: **{s['status']}**", '']
    if not suggestions:
        lines.append('No suggestions: the evidence does not support any rule change.')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_feedback_review(cfg=CONFIG, out=EVIDENCE):
    records = load_records(out)
    sources = dict(Counter(r['source'] for r in records))
    summary = summarise(records)
    suggestions = suggest(records)
    write_json(out / 'feedback_review.json', {'records': len(records), 'sources': sources,
                                              'summary': summary, 'suggestions': suggestions,
                                              'note': 'Suggestions are never applied automatically.'})
    write_markdown(out / 'feedback_suggestions.md', summary, suggestions, len(records), sources)
    print(f"Feedback review: {len(records)} decisions ({', '.join(f'{k}: {v}' for k, v in sources.items())})")
    for s in suggestions:
        print(f"  [{s['status']}] {s['type']}: {s['evidence']}")
    return {'summary': summary, 'suggestions': suggestions}


if __name__ == '__main__':
    run_feedback_review()
