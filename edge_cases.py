"""Edge-case handling: situations the baseline simulator does not model, each handled
by a defined strategy and measured against a naive dispatcher on the SAME missions.

    python edge_cases.py       (run experiment.py first; uses evidence/q_table.json)

Edge cases added here (CONFIG['edge_cases']):
1. Lost acknowledgement  The survey happened but the confirmation never arrived.
                         naive   : treats it as a failure -> re-sends -> duplicate flight, wrong battery belief
                         handled : marks the command PENDING, queries telemetry (costs one command slot),
                                   then updates its view from the drone's real state
2. False detection       The sensor reports a survivor where there is none.
                         naive   : reports every detection to rescue teams as confirmed
                         handled : Bayesian posterior (Task 1, section 1.3); below the threshold the
                                   detection is escalated for verification instead of confirmed
3. Sector closed by a human mid-mission (e.g. responders entered the area)
                         naive   : closure message not applied -> may still dispatch there
                         handled : sector removed from the allowed moves immediately (replanning)
4. Invalid input state   Corrupted or out-of-range telemetry.
                         handled : validate_state() refuses to act and holds for the supervisor
                         (also used inside explain.py's supervised loop)

The baseline results in summary.json are not affected by anything in this file.
"""
import copy
import itertools
import random
import statistics
import time

from experiment import (CONFIG, EVIDENCE, DRONES, Mission, allowed_actions, choose_action, energy_cost, load_table,
                        mean_ci, move_name, n_actions, n_sectors, paired_ci, split_action, write_csv, write_json)


# ---------------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------------
def validate_state(state, cfg=CONFIG):
    """Problems with a state, or [] if it is valid. The dispatcher must not act on an invalid state."""
    problems = []
    if not isinstance(state, tuple) or len(state) != 5:
        return ['state must be a tuple of 5 values']
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in state):
        return ['state values must be whole numbers']
    surveyed, energy_a, energy_b, windy, attempts = state
    if not 0 <= surveyed < 2 ** n_sectors(cfg):
        problems.append(f'surveyed mask {surveyed} outside 0..{2 ** n_sectors(cfg) - 1}')
    for name, e in (('A', energy_a), ('B', energy_b)):
        if not 0 <= e <= cfg['usable_energy_each']:
            problems.append(f"energy of drone {name} = {e} outside 0..{cfg['usable_energy_each']}")
    if windy not in (0, 1):
        problems.append(f'wind flag {windy} is not 0 or 1')
    if not 0 <= attempts <= cfg['max_dispatch_attempts']:
        problems.append(f"command count {attempts} outside 0..{cfg['max_dispatch_attempts']}")
    return problems


def safe_dispatch(q_table, state, cfg=CONFIG, closed=frozenset()):
    """Validated dispatch decision. Returns ('act', action) | ('hold', reasons) | ('done', [])."""
    problems = validate_state(state, cfg)
    if problems:
        return 'hold', problems
    allowed = [a for a in allowed_actions(state, True, cfg) if split_action(a, cfg)[1] not in closed]
    if not allowed:
        return 'done', []
    return 'act', choose_action('learned', q_table, state, allowed, cfg)[0]


# ---------------------------------------------------------------------------
# 2. Bayesian confirmation of detections
# ---------------------------------------------------------------------------
def posterior_survivor(sector, condition, cfg=CONFIG):
    """P(survivor | positive detection) - the Bayesian update described in Task 1."""
    prior = cfg['sector_presence'][sector]
    sensitivity = cfg[condition]['sensitivity']
    fp = cfg['edge_cases']['false_positive']
    return sensitivity * prior / (sensitivity * prior + fp * (1 - prior))


# ---------------------------------------------------------------------------
# Mission with a separate "true world" and "agent's view"
# ---------------------------------------------------------------------------
def run_edge_mission(q_table, seed, handled, cfg=CONFIG, log=None):
    """One mission with lost acknowledgements, false detections and a mid-mission closure.
    The agent decides from its own view; the true world decides what really happens."""
    ec = cfg['edge_cases']
    condition = ec['base_condition']
    world = Mission(seed, condition, cfg)                 # same base outcomes as the main experiment
    extra = random.Random(seed + 31337)                   # separate stream: baseline draws unchanged
    sectors, cap = n_sectors(cfg), cfg['max_dispatch_attempts']
    ack_lost = [extra.random() < ec['ack_loss'] for _ in range(cap)]
    false_positive = [extra.random() < ec['false_positive'] for _ in range(sectors)]
    closable = [s for s in range(sectors) if cfg['sector_hazard'][s] <= cfg['risk_mask_threshold']]
    closed_sector = extra.choice(closable)

    true_state = world.state            # what is really true
    view = world.state                  # what the agent believes
    closed, closure_known = set(), False
    r = cfg['reward']
    m = {'seed': seed, 'policy': 'handled' if handled else 'naive', 'return': 0.0, 'detections': 0,
         'hazards': 0, 'commands': 0, 'acks_lost': 0, 'reconciliations': 0, 'duplicate_flights': 0,
         'battery_overruns': 0, 'closure_violations': 0, 'confirmed_true': 0, 'confirmed_false': 0,
         'escalated_true': 0, 'escalated_false': 0, 'held_invalid_state': 0}

    def note(text):
        if log is not None:
            log.append(text)

    while true_state[4] < cap:
        attempt = true_state[4]
        if attempt >= ec['closure_attempt'] and not closure_known:
            closure_known = True
            note(f'Supervisor closes sector {closed_sector} (responders entering).')
            if handled:
                closed.add(closed_sector)
        decision, payload = safe_dispatch(q_table, view, cfg, frozenset(closed))
        if decision == 'hold':
            m['held_invalid_state'] += 1
            note(f'HOLD - invalid state: {payload}')
            break
        if decision == 'done':
            break
        action = payload
        drone, sector = split_action(action, cfg)
        m['commands'] += 1
        surveyed, ea, eb, windy, _ = true_state
        if sector == closed_sector and closure_known:
            m['closure_violations'] += 1
        if world.command_rejected[attempt][action]:                 # clean rejection: both views agree
            m['return'] += r['api_failure']
            true_state = (surveyed, ea, eb, windy, attempt + 1)
            view = (view[0], view[1], view[2], view[3], attempt + 1)
            note(f'{move_name(action, cfg)}: rejected before execution')
            continue
        cost = energy_cost(drone, sector, windy, cfg)
        energy = [ea, eb]
        if energy[drone] < cost:                                    # agent's battery belief was wrong
            m['battery_overruns'] += 1
            m['return'] += r['api_failure']
            true_state = (surveyed, ea, eb, windy, attempt + 1)
            view = (view[0], view[1], view[2], view[3], attempt + 1)
            note(f'{move_name(action, cfg)}: drone {DRONES[drone]} lacks battery - aborted on board')
            continue
        energy[drone] -= cost
        duplicate = bool(surveyed & (1 << sector))
        hazard = int(world.hazard_occurs[sector])
        reward = r['energy'] * cost + r['hazard'] * hazard
        if duplicate:
            m['duplicate_flights'] += 1
            note(f'{move_name(action, cfg)}: DUPLICATE flight - sector already surveyed')
        else:
            detected = int(world.survivor_present[sector] and world.sensor_detects[sector])
            positive = detected or (not world.survivor_present[sector] and false_positive[sector])
            reward += r['detection'] * detected + r['coverage']
            m['detections'] += detected
            if positive:
                confirmed = (not handled) or posterior_survivor(sector, condition, cfg) >= ec['confirm_threshold']
                truth = bool(world.survivor_present[sector])
                key = ('confirmed_' if confirmed else 'escalated_') + ('true' if truth else 'false')
                m[key] += 1
        m['hazards'] += hazard
        m['return'] += reward
        true_state = (surveyed | (1 << sector), energy[0], energy[1], windy, attempt + 1)
        if ack_lost[attempt]:
            m['acks_lost'] += 1
            if handled:
                # Pending -> query telemetry (one command slot, no energy) -> adopt the real state.
                m['reconciliations'] += 1
                true_state = (true_state[0], true_state[1], true_state[2], windy, min(cap, true_state[4] + 1))
                view = true_state
                note(f'{move_name(action, cfg)}: acknowledgement lost -> PENDING -> telemetry confirms survey')
            else:
                view = (view[0], view[1], view[2], view[3], attempt + 1)   # believes nothing happened
                note(f'{move_name(action, cfg)}: acknowledgement lost -> assumed failed')
        else:
            view = true_state
            note(f'{move_name(action, cfg)}: survey completed')
    return m


METRICS = ['return', 'detections', 'hazards', 'duplicate_flights', 'battery_overruns', 'closure_violations',
           'confirmed_false', 'confirmed_true', 'escalated_true', 'escalated_false', 'reconciliations']


def fuzz_validation(cfg=CONFIG, n=5000, seed=99):
    """Every reachable-range state is accepted; malformed states are always held, never acted on."""
    rng = random.Random(seed)
    accepted_valid = all(not validate_state(s, cfg) for s in itertools.product(
        range(2 ** n_sectors(cfg)), range(cfg['usable_energy_each'] + 1), range(cfg['usable_energy_each'] + 1),
        range(2), range(cfg['max_dispatch_attempts'] + 1)))
    # Out-of-range values specific to each field (a value valid for one field may be invalid for another).
    invalid = [[-1, 2 ** n_sectors(cfg), 999], [-1, cfg['usable_energy_each'] + 1, 99],
               [-1, cfg['usable_energy_each'] + 1, 99], [-1, 2, 5], [-1, cfg['max_dispatch_attempts'] + 1, 99]]
    held = 0
    for _ in range(n):
        bad = [rng.randint(0, 2 ** n_sectors(cfg) - 1), rng.randint(0, cfg['usable_energy_each']),
               rng.randint(0, cfg['usable_energy_each']), rng.randint(0, 1), rng.randint(0, cfg['max_dispatch_attempts'])]
        field = rng.randrange(5)
        bad[field] = rng.choice(invalid[field])
        decision, _ = safe_dispatch({}, tuple(bad), cfg)
        held += decision == 'hold'
    extra_cases = [None, (1, 2, 3), (0, 4, 4, 0, '1'), (0.5, 4, 4, 0, 0), (0, True, 4, 0, 0)]
    held_extra = sum(safe_dispatch({}, s, cfg)[0] == 'hold' for s in extra_cases)
    return {'all_valid_states_accepted': accepted_valid, 'malformed_states_tested': n + len(extra_cases),
            'malformed_states_held': held + held_extra}


def edge_case_study(cfg=CONFIG, out=EVIDENCE):
    q_table = load_table(out / 'q_table.json')
    ec = cfg['edge_cases']
    print(f"Edge-case study ({cfg['test_episodes']} test missions, {ec['base_condition']} conditions, "
          f"ack loss {ec['ack_loss']:.0%}, false detections {ec['false_positive']:.0%})")
    runs = {}
    for handled in (False, True):
        runs[handled] = [run_edge_mission(q_table, cfg['test_seed_start'] + k, handled, cfg)
                         for k in range(cfg['test_episodes'])]
    summary = {}
    for handled, label in ((False, 'naive'), (True, 'handled')):
        rows = runs[handled]
        summary[label] = {m: statistics.mean(r[m] for r in rows) for m in METRICS}
        summary[label]['missions_with_duplicates_pct'] = 100 * statistics.mean(r['duplicate_flights'] > 0 for r in rows)
        summary[label]['false_reports_share_pct'] = (
            100 * sum(r['confirmed_false'] for r in rows)
            / max(1, sum(r['confirmed_false'] + r['confirmed_true'] for r in rows)))
    comparisons = {f'handled_minus_naive_{m}': paired_ci([runs[True]], [runs[False]], lambda r, k=m: r[k])
                   for m in ['return', 'duplicate_flights', 'battery_overruns', 'closure_violations',
                             'confirmed_false', 'confirmed_true', 'detections']}
    posteriors = {str(s): posterior_survivor(s, ec['base_condition'], cfg) for s in range(n_sectors(cfg))}
    validation = fuzz_validation(cfg)
    for label in ('naive', 'handled'):
        s = summary[label]
        print(f"  {label:<8} duplicates/mission {s['duplicate_flights']:.3f}  battery overruns {s['battery_overruns']:.3f}  "
              f"closure violations {s['closure_violations']:.3f}  false confirmations {s['confirmed_false']:.3f}  "
              f"return {s['return']:.2f}")
    print(f"  input validation: {validation['malformed_states_held']}/{validation['malformed_states_tested']} "
          f"malformed states held; all valid states accepted: {validation['all_valid_states_accepted']}")
    sample = []
    run_edge_mission(q_table, _first_seed_with_lost_ack(q_table, cfg), True, cfg, log=sample)
    naive_sample = []
    run_edge_mission(q_table, _first_seed_with_lost_ack(q_table, cfg), False, cfg, log=naive_sample)
    (out / 'edge_case_example.txt').write_text(
        'Edge-case example - same mission, naive vs handled dispatcher\n\n--- naive ---\n' + '\n'.join(naive_sample)
        + '\n\n--- handled ---\n' + '\n'.join(sample) + '\n', encoding='utf-8')
    write_csv(out / 'edge_case_missions.csv', runs[False] + runs[True])
    report = {'settings': ec, 'summary': summary, 'comparisons': comparisons,
              'posterior_by_sector': posteriors, 'input_validation': validation}
    write_json(out / 'edge_cases.json', report)
    return report


def _first_seed_with_lost_ack(q_table, cfg):
    for k in range(cfg['test_episodes']):
        m = run_edge_mission(q_table, cfg['test_seed_start'] + k, False, cfg)
        if m['duplicate_flights'] > 0:
            return cfg['test_seed_start'] + k
    return cfg['test_seed_start']


def run_all_edge(cfg=CONFIG, out=EVIDENCE):
    started = time.time()
    report = edge_case_study(cfg, out)
    print(f'  done in {time.time() - started:.1f} s')
    return report


if __name__ == '__main__':
    run_all_edge()
