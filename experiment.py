"""Drone-swarm sector dispatcher: a small, reproducible Q-learning experiment.

Educational simulation only - NOT flight-control software. One central
dispatcher learns which drone (A or B) should survey which of six sectors.

    python run_all.py        # everything: experiment, checks, extra studies, figures, report pack
    python experiment.py     # only the main experiment (train, validate, test)

The experiment itself needs only the Python standard library.

Vocabulary used throughout the code
    state   = (surveyed_mask, energy_a, energy_b, windy, attempts)
              surveyed_mask: sector k surveyed -> bit k set (sector 2 adds 4)
    action  = a number 0-11. 0-5 send drone A to sector 0-5, 6-11 send drone B.
    q_table = {state: [12 predicted values, one per action]}
"""
import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'

# ---------------------------------------------------------------------------
# CONFIG - every setting the experiment uses. Change values here only.
# ---------------------------------------------------------------------------
CONFIG = {
    # --- Training and data split ---------------------------------------
    'training_seeds': [11, 22, 33, 44, 55, 66, 77, 88, 99, 110],   # ten independent training runs
    'representative_seed': 11,           # run reported in tables, fixed in advance
    'episodes': 12000,                   # practice missions per training run
    'alpha_candidates': [0.02, 0.05, 0.1, 0.2, 0.3],   # learning rates compared on validation
    'gamma': 0.95,                       # discount: weight of future reward
    'epsilon_start': 1.0,                # exploration: 100% random moves at first
    'epsilon_end': 0.05,                 # ... falling to 5% random moves
    'decay_fraction': 0.8,               # ... over the first 80% of episodes
    'validation_seed_start': 10000, 'validation_episodes': 200,
    'test_seed_start': 20000, 'test_episodes': 500,

    # --- Simulated world -------------------------------------------------
    'sector_names': ['Sector 0', 'Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Sector 5'],
    'sector_presence': [0.15, 0.45, 0.6, 0.8, 0.35, 0.9],     # chance a survivor is there
    'sector_hazard': [0.02, 0.04, 0.08, 0.12, 0.18, 0.30],    # chance of a hazard event
    'base_energy_cost': [[1, 1, 2, 2, 3, 2],    # drone A, sectors 0-5
                         [3, 2, 2, 1, 1, 2]],   # drone B, sectors 0-5
    'wind_extra_cost': 1,            # wind adds this much energy ...
    'wind_min_cost': 2,              # ... to trips whose base cost is at least this
    'usable_energy_each': 4,         # energy units per drone per mission
    'max_dispatch_attempts': 12,     # mission ends after this many commands

    # --- Safety ----------------------------------------------------------
    'risk_mask_threshold': 0.2,      # sectors with hazard above this are blocked
    'filter_hazard_multiplier': 1,   # the filter judges hazard x this. 1 = normal-weather estimates
                                     # (baseline); extra_experiments.py sets it to the current
                                     # condition's multiplier to test a weather-aware filter

    # --- Reward ------------------------------------------------------------
    'reward': {'detection': 10, 'coverage': 1, 'energy': -0.4, 'hazard': -6, 'api_failure': -0.5},

    # --- Operating conditions ----------------------------------------------
    'nominal': {'api_failure': 0.05, 'sensitivity': 0.85, 'wind_probability': 0.5, 'hazard_multiplier': 1},
    'stress': {'api_failure': 0.35, 'sensitivity': 0.5, 'wind_probability': 0.9, 'hazard_multiplier': 2},
    # Loop-trap test only: normal weather, but drone B's radio link is down for the whole
    # mission, so every command sent to drone B is rejected.
    'link_fault': {'api_failure': 0.05, 'sensitivity': 0.85, 'wind_probability': 0.5, 'hazard_multiplier': 1,
                   'broken_drone': 1},

    # --- Human approval gate (used by explain.py) ---------------------------
    'approval': {
        'hazard_flag': 0.10,     # ask a human before sending a drone to a sector this risky
        'min_visits': 30,        # ... or when the situation was rarely seen in training
        'rejections_flag': 2,    # ... or after this many rejected commands in one mission
    },
    'retry_limit': 2,            # loop guard: after this many rejections in a row for one drone,
                                 # treat its link as suspect and use the other drone

    # --- Edge cases (used by edge_cases.py; baseline unaffected) --------------
    'edge_cases': {
        'base_condition': 'nominal',
        'ack_loss': 0.15,           # chance a completed survey's acknowledgement is lost
        'false_positive': 0.10,     # chance the sensor reports a survivor where there is none
        'confirm_threshold': 0.90,  # Bayesian posterior needed to report a survivor as confirmed
        'closure_attempt': 2,       # after this many commands the supervisor closes one sector
    },

    # --- Randomised stress test (used by stress_test.py) ---------------------
    'stress_test': {'scenarios': 400, 'missions_per_scenario': 25, 'seed': 2026, 'corrupt_prob': 0.03},

    # --- Side experiments (used by extra_experiments.py; baseline unaffected) ---
    'extensions': {'mixed_training': {'nominal': 0.7, 'stress': 0.3}},   # share of training missions

    # --- Extra studies (used by robustness.py) ------------------------------
    'sweep': {'gamma': [0.9, 0.95, 0.99], 'episodes': [3000, 6000, 12000], 'hazard_penalty': [-6, -10]},
    'robustness': {
        'shifted_presence': [0.9, 0.35, 0.8, 0.6, 0.45, 0.15],  # survivor map reversed
        'forgetting_episodes': 6000,          # extra stress-only training
        'forgetting_seed_offset': 500000,     # keeps those missions separate from earlier ones
    },
}

DRONES = ('A', 'B')
REJECTED = 'api_rejected_before_execution'
COMPLETED = 'survey_complete'
POLICIES = ['learned', 'greedy', 'random', 'unshielded']


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def n_sectors(cfg=CONFIG):
    return len(cfg['sector_presence'])


def n_actions(cfg=CONFIG):
    return len(DRONES) * n_sectors(cfg)


def split_action(action, cfg=CONFIG):
    """Action number -> (drone index, sector index)."""
    return divmod(action, n_sectors(cfg))


def move_name(action, cfg=CONFIG):
    drone, sector = split_action(action, cfg)
    return f'{DRONES[drone]}->{sector}'


def state_key(state):
    """State tuple -> text key used in the JSON files, e.g. '0,4,4,1,0'."""
    return ','.join(map(str, state))


def parse_state_key(key):
    return tuple(int(x) for x in key.split(','))


def describe_state(state, cfg=CONFIG):
    """Plain-English description of a state, for logs and explanations."""
    surveyed, energy_a, energy_b, windy, attempts = state
    done = [str(s) for s in range(n_sectors(cfg)) if surveyed & (1 << s)]
    return (f"surveyed: {', '.join(done) if done else 'none'} | energy A={energy_a}, B={energy_b} | "
            f"{'windy' if windy else 'calm'} | commands sent: {attempts}")


# ---------------------------------------------------------------------------
# World rules
# ---------------------------------------------------------------------------
def energy_cost(drone, sector, windy, cfg=CONFIG):
    """Energy a drone needs for one sector survey (round trip included)."""
    base = cfg['base_energy_cost'][drone][sector]
    if windy and base >= cfg['wind_min_cost']:
        return base + cfg['wind_extra_cost']
    return base


def block_reason(state, action, shield=True, cfg=CONFIG):
    """Why an action is not allowed in a state, or None if it is allowed."""
    surveyed, energy_a, energy_b, windy, attempts = state
    drone, sector = split_action(action, cfg)
    if attempts >= cfg['max_dispatch_attempts']:
        return 'command limit reached'
    if surveyed & (1 << sector):
        return 'sector already surveyed'
    cost = energy_cost(drone, sector, windy, cfg)
    have = (energy_a, energy_b)[drone]
    if cost > have:
        return f'needs {cost} energy, drone {DRONES[drone]} has {have}'
    hazard = cfg['sector_hazard'][sector] * cfg.get('filter_hazard_multiplier', 1)
    if shield and hazard > cfg['risk_mask_threshold']:
        return f"hazard {hazard:.2f} above safety limit {cfg['risk_mask_threshold']:.2f}"
    return None


def allowed_actions(state, shield=True, cfg=CONFIG):
    """Actions the dispatcher may choose. shield=True applies the risk filter.
    Same rules as block_reason(), written out directly because training calls it millions of times."""
    surveyed, energy_a, energy_b, windy, attempts = state
    if attempts >= cfg['max_dispatch_attempts']:
        return []
    sectors = n_sectors(cfg)
    energy_left = (energy_a, energy_b)
    hazard_scale = cfg.get('filter_hazard_multiplier', 1)
    allowed = []
    for drone in range(len(DRONES)):
        for sector in range(sectors):
            if surveyed & (1 << sector):
                continue
            if energy_cost(drone, sector, windy, cfg) > energy_left[drone]:
                continue
            if shield and cfg['sector_hazard'][sector] * hazard_scale > cfg['risk_mask_threshold']:
                continue
            allowed.append(drone * sectors + sector)
    return allowed


def expected_step_reward(state, action, cfg=CONFIG):
    """Average reward of one survey, using nominal estimates (what the agent believes)."""
    drone, sector = split_action(action, cfg)
    r = cfg['reward']
    return (r['detection'] * cfg['sector_presence'][sector] * cfg['nominal']['sensitivity']
            + r['coverage'] + r['energy'] * energy_cost(drone, sector, state[3], cfg)
            + r['hazard'] * cfg['sector_hazard'][sector])


def greedy_score(state, action, cfg=CONFIG):
    """Greedy rule: expected benefit per unit of energy (nominal estimates only)."""
    drone, sector = split_action(action, cfg)
    r = cfg['reward']
    benefit = (r['detection'] * cfg['sector_presence'][sector] * cfg['nominal']['sensitivity']
               + r['coverage'] + r['hazard'] * cfg['sector_hazard'][sector])
    return benefit / energy_cost(drone, sector, state[3], cfg) + r['energy']


def greedy_action(state, allowed, cfg=CONFIG):
    """Baseline policy, and the fallback for situations the Q-table has never seen."""
    return max(allowed, key=lambda a: (greedy_score(state, a, cfg), -a))


class Mission:
    """One simulated mission. All random outcomes are drawn up front from the seed,
    so every policy evaluated on the same seed faces exactly the same mission."""

    def __init__(self, seed, condition='nominal', cfg=CONFIG):
        rng = random.Random(seed)
        self.cfg = cfg
        conditions = cfg[condition]
        start = cfg['usable_energy_each']
        windy = int(rng.random() < conditions['wind_probability'])
        self.state = (0, start, start, windy, 0)
        # Hidden from the dispatcher: where survivors are, whether the sensor
        # spots them, where hazards occur, and which commands get rejected.
        self.survivor_present = [rng.random() < p for p in cfg['sector_presence']]
        self.sensor_detects = [rng.random() < conditions['sensitivity'] for _ in cfg['sector_presence']]
        self.hazard_occurs = [rng.random() < min(1, h * conditions['hazard_multiplier'])
                              for h in cfg['sector_hazard']]
        self.command_rejected = [[rng.random() < conditions['api_failure'] for _ in range(n_actions(cfg))]
                                 for _ in range(cfg['max_dispatch_attempts'])]
        broken = conditions.get('broken_drone')
        if broken is not None:          # persistent fault: that drone rejects everything
            for attempt in self.command_rejected:
                for action in range(n_actions(cfg)):
                    if split_action(action, cfg)[0] == broken:
                        attempt[action] = True
        self.total_reward = 0
        self.detections = self.hazards = self.surveyed = self.api_failures = self.energy_used = 0

    def step(self, action):
        """Carry out one command. Returns (reward, event)."""
        surveyed, energy_a, energy_b, windy, attempts = self.state
        drone, sector = split_action(action, self.cfg)
        r = self.cfg['reward']
        if self.command_rejected[attempts][action]:
            # Rejected before execution: costs an attempt, uses no energy, surveys nothing.
            reward = r['api_failure']
            self.api_failures += 1
            self.state = (surveyed, energy_a, energy_b, windy, attempts + 1)
            event = REJECTED
        else:
            cost = energy_cost(drone, sector, windy, self.cfg)
            energy = [energy_a, energy_b]
            energy[drone] -= cost
            assert min(energy) >= 0 and not surveyed & (1 << sector)
            detected = int(self.survivor_present[sector] and self.sensor_detects[sector])
            hazard = int(self.hazard_occurs[sector])
            reward = (r['detection'] * detected + r['coverage']
                      + r['energy'] * cost + r['hazard'] * hazard)
            self.detections += detected
            self.hazards += hazard
            self.surveyed += 1
            self.energy_used += cost
            self.state = (surveyed | (1 << sector), energy[0], energy[1], windy, attempts + 1)
            event = COMPLETED
        self.total_reward += reward
        return reward, event


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------
def q_update(old, reward, best_next, alpha, gamma, terminal=False):
    """The Q-learning rule.
        target = reward + gamma * best_next     (just reward if the mission ended)
        new    = old + alpha * (target - old)
    Example: q_update(5, 2, 6, 0.05, 0.95) -> target 7.7, new 5.135."""
    target = reward if terminal else reward + gamma * best_next
    return old + alpha * (target - old)


def epsilon_at(episode, episodes, cfg=CONFIG):
    """Exploration rate: falls linearly from epsilon_start to epsilon_end, then stays."""
    start, end = cfg['epsilon_start'], cfg['epsilon_end']
    return max(end, start - (start - end) * episode / (episodes * cfg['decay_fraction']))


def train(alpha, seed, shield=True, cfg=CONFIG, condition='nominal', q_table=None,
          episodes=None, mission_seed_offset=0, fixed_epsilon=None):
    """Train a Q-table. Returns (q_table, visit_counts, learning_curve).
    condition: 'nominal', 'stress', or a mix such as {'nominal': 0.7, 'stress': 0.3}
               (each mission's condition is then drawn from a separate random stream).
    q_table / fixed_epsilon / mission_seed_offset allow continued training (forgetting test)."""
    episodes = episodes or cfg['episodes']
    rng = random.Random(seed)
    mix_rng = random.Random(seed + 424242) if isinstance(condition, dict) else None
    q_table = {} if q_table is None else q_table
    visits, curve, window = {}, [], []
    blank = [0.0] * n_actions(cfg)
    for episode in range(episodes):
        mission_condition = condition
        if mix_rng is not None:
            mission_condition = mix_rng.choices(list(condition), weights=list(condition.values()))[0]
        mission = Mission(seed * 1_000_000 + mission_seed_offset + episode, mission_condition, cfg)
        epsilon = fixed_epsilon if fixed_epsilon is not None else epsilon_at(episode, episodes, cfg)
        while True:
            allowed = allowed_actions(mission.state, shield, cfg)
            if not allowed:
                break
            state = mission.state
            row = q_table.setdefault(state, list(blank))
            # Explore (random move) or exploit (highest predicted value).
            if rng.random() < epsilon:
                action = rng.choice(allowed)
            else:
                action = max(allowed, key=lambda a: (row[a], -a))
            reward, _ = mission.step(action)
            next_allowed = allowed_actions(mission.state, shield, cfg)
            if next_allowed:
                next_row = q_table.get(mission.state, blank)
                best_next = max(next_row[a] for a in next_allowed)
                row[action] = q_update(row[action], reward, best_next, alpha, cfg['gamma'])
            else:
                row[action] = q_update(row[action], reward, 0, alpha, cfg['gamma'], terminal=True)
            visits[state] = visits.get(state, 0) + 1
        window.append(mission.total_reward)
        if (episode + 1) % 1000 == 0:
            curve.append({'episode': episode + 1, 'mean_training_return': statistics.mean(window),
                          'epsilon': epsilon})
            window = []
    return q_table, visits, curve


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def choose_action(policy, q_table, state, allowed, cfg=CONFIG, rng=None):
    """Returns (action, used_fallback)."""
    if policy == 'random':
        return rng.choice(allowed), False
    if policy == 'greedy':
        return greedy_action(state, allowed, cfg), False
    if state not in q_table:          # never seen in training -> greedy fallback
        return greedy_action(state, allowed, cfg), True
    row = q_table[state]
    return max(allowed, key=lambda a: (row[a], -a)), False


def evaluate(q_table, first_seed, n_missions, condition='nominal', policy='learned', shield=True,
             trace=False, cfg=CONFIG, retry_limit=None):
    """Run a frozen policy on missions first_seed .. first_seed+n_missions-1.
    retry_limit (loop guard): once a drone has had this many commands rejected in a row,
    its link is treated as suspect and only the other drone is used. If only suspect drones
    have moves left, the mission stops and is handed back to the supervisor instead of looping.
    Returns (rows, trace_log)."""
    rows, logs = [], []
    for k in range(n_missions):
        seed = first_seed + k
        mission = Mission(seed, condition, cfg)
        rng = random.Random(seed + 777)
        fallback_steps = steps = repeated_retries = stopped_by_guard = 0
        last_rejected = None                       # last command that was rejected
        consecutive = [0] * len(DRONES)            # rejections in a row, per drone
        suspect = set()                            # drones the loop guard avoids
        while True:
            allowed = allowed_actions(mission.state, shield, cfg)
            if not allowed:
                break
            state = mission.state
            options = allowed
            if retry_limit is not None and suspect:
                options = [a for a in allowed if split_action(a, cfg)[0] not in suspect]
                if not options:            # only the suspect drone could act: stop, don't loop
                    stopped_by_guard = 1
                    break
            action, used_fallback = choose_action(policy, q_table, state, options, cfg, rng)
            fallback_steps += used_fallback
            if action == last_rejected:            # identical command re-sent straight after rejection
                repeated_retries += 1
            reward, event = mission.step(action)
            steps += 1
            drone = split_action(action, cfg)[0]
            if event == REJECTED:
                last_rejected = action
                consecutive[drone] += 1
                if retry_limit is not None and consecutive[drone] >= retry_limit:
                    suspect.add(drone)
            else:
                last_rejected = None
                consecutive[drone] = 0
            if trace and k == 0:
                logs.append({'seed': seed, 'state_before': state, 'allowed_actions': allowed,
                             'action': action, 'move': move_name(action, cfg),
                             'drone': action // n_sectors(cfg), 'sector': action % n_sectors(cfg),
                             'q_value': q_table.get(state, [0.0] * n_actions(cfg))[action],
                             'used_fallback': used_fallback, 'event': event, 'reward': reward,
                             'state_after': mission.state})
        rows.append({'seed': seed, 'return': mission.total_reward, 'detections': mission.detections,
                     'coverage_pct': mission.surveyed / n_sectors(cfg) * 100, 'hazards': mission.hazards,
                     'api_failures': mission.api_failures, 'energy': mission.energy_used,
                     'fallback_steps': fallback_steps, 'steps': steps,
                     'repeated_retries': repeated_retries,
                     'hit_attempt_cap': int(mission.state[4] >= cfg['max_dispatch_attempts']),
                     'stopped_by_guard': stopped_by_guard})
    return rows, logs


def mean_ci(values):
    """(mean, 95% confidence half-width) using 1.96 x standard error."""
    values = list(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return statistics.mean(values), 1.96 * se


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045}


def across_runs_ci(values):
    """Mean and 95% confidence half-width ACROSS training runs (one value per seed), using the
    t-distribution because the number of runs is small. This captures training-run variability."""
    values = list(values)
    n = len(values)
    if n < 2:
        return {'mean': values[0] if values else 0.0, 'ci95': 0.0, 'runs': n, 'min': min(values, default=0.0),
                'max': max(values, default=0.0)}
    t = T95.get(n - 1) or next((T95[k] for k in sorted(T95) if k >= n - 1), 1.96)
    mean = statistics.mean(values)
    half = t * statistics.stdev(values) / math.sqrt(n)
    return {'mean': mean, 'ci95': half, 'low': mean - half, 'high': mean + half, 'runs': n,
            'min': min(values), 'max': max(values), 'significant_vs_zero': bool(mean - half > 0 or mean + half < 0)}


def paired_ci(a_runs, b_runs, metric=lambda row: row['return']):
    """Paired comparison of A minus B on the SAME missions.
    a_runs / b_runs: one list of mission rows per training seed (same missions, same order).
    Each mission is first averaged over the seeds, then the mean difference and its 95%
    confidence interval are taken over missions. 'significant' = the interval excludes 0."""
    n = len(a_runs[0])
    a = [statistics.mean(metric(run[i]) for run in a_runs) for i in range(n)]
    b = [statistics.mean(metric(run[i]) for run in b_runs) for i in range(n)]
    diff, half = mean_ci(x - y for x, y in zip(a, b))
    return {'diff': diff, 'ci95': half, 'low': diff - half, 'high': diff + half,
            'significant': bool(diff - half > 0 or diff + half < 0), 'missions': n,
            'seeds_a': len(a_runs), 'seeds_b': len(b_runs)}


METRICS = ['return', 'detections', 'coverage_pct', 'hazards', 'api_failures', 'energy', 'repeated_retries']


def aggregate(rows):
    """Mean and 95% confidence half-width (1.96 x standard error) for each metric."""
    out = {}
    for key in METRICS:
        values = [r[key] for r in rows]
        se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        out[key] = {'mean': statistics.mean(values), 'ci95_halfwidth': 1.96 * se}
    total_steps = sum(r['steps'] for r in rows)
    out['fallback_pct'] = 100 * sum(r['fallback_steps'] for r in rows) / total_steps if total_steps else 0.0
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding='utf-8')


def save_table(path, table):
    """Save a {state: value} dict with text keys, e.g. the Q-table or visit counts."""
    write_json(path, {state_key(s): v for s, v in table.items()}, indent=None)


def load_table(path):
    return {parse_state_key(k): v for k, v in json.loads(path.read_text(encoding='utf-8')).items()}


# ---------------------------------------------------------------------------
# Main experiment: train -> validate (choose alpha) -> test -> save evidence
# ---------------------------------------------------------------------------
def run_main_experiment(cfg=CONFIG, out=EVIDENCE):
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', cfg)
    rep = cfg['representative_seed']

    # 1. Train one table per (learning rate, seed) and score it on validation missions.
    print('Step 1/3  Training and validation')
    tuning, models, all_curves = [], {}, []
    for alpha in cfg['alpha_candidates']:
        for seed in cfg['training_seeds']:
            q_table, visits, curve = train(alpha, seed, cfg=cfg)
            models[alpha, seed] = (q_table, visits, curve)
            rows, _ = evaluate(q_table, cfg['validation_seed_start'], cfg['validation_episodes'], cfg=cfg)
            score = statistics.mean(r['return'] for r in rows)
            tuning.append({'alpha': alpha, 'training_seed': seed, 'validation_return': score,
                           'states': len(q_table)})
            all_curves += [{'alpha': alpha, 'training_seed': seed, **c} for c in curve]
            print(f'  alpha={alpha:<5} seed={seed}  validation return={score:7.3f}  states={len(q_table)}')
    validation_means = {str(a): statistics.mean(t['validation_return'] for t in tuning if t['alpha'] == a)
                        for a in cfg['alpha_candidates']}
    validation_by_alpha = {str(a): across_runs_ci(t['validation_return'] for t in tuning if t['alpha'] == a)
                           for a in cfg['alpha_candidates']}
    best = max(cfg['alpha_candidates'], key=lambda a: validation_means[str(a)])
    for a in cfg['alpha_candidates']:
        v = validation_by_alpha[str(a)]
        print(f"  alpha={a:<5} mean validation return {v['mean']:7.3f} +/- {v['ci95']:.3f} across "
              f"{v['runs']} seeds (range {v['min']:.2f}-{v['max']:.2f})")
    print(f'  -> selected alpha = {best} (highest mean validation return)')
    write_csv(out / 'validation.csv', tuning)
    write_csv(out / 'learning_curves_all.csv', all_curves)

    # 2. Test every policy on the same unseen missions, in both conditions.
    print('Step 2/3  Testing on unseen missions')
    summary = {'selected_alpha': best, 'validation_means': validation_means,
               'validation_by_alpha': validation_by_alpha,
               'results': {}, 'training_seed_returns': {}, 'across_training_runs': {}}
    test_rows = {}
    for condition in ['nominal', 'stress']:
        for policy in POLICIES:
            seeds = cfg['training_seeds'] if policy in ('learned', 'unshielded') else [rep]
            all_rows, means = [], []
            for seed in seeds:
                if policy == 'unshielded':
                    if ('unshielded', seed) not in models:
                        models['unshielded', seed] = train(best, seed, shield=False, cfg=cfg)
                    q_table = models['unshielded', seed][0]
                else:
                    q_table = models[best, seed][0]
                rows, logs = evaluate(q_table, cfg['test_seed_start'], cfg['test_episodes'], condition,
                                      'learned' if policy == 'unshielded' else policy,
                                      shield=policy != 'unshielded',
                                      trace=(seed == rep and policy == 'learned'), cfg=cfg)
                means.append(statistics.mean(r['return'] for r in rows))
                for r in rows:
                    r['training_seed'] = seed
                all_rows += rows
                if logs:
                    write_json(out / f'trace_{condition}.json', logs)
            write_csv(out / f'{condition}_{policy}.csv', all_rows)
            # Confidence intervals use the representative seed only: different policies
            # share the same missions, so pooling them would overstate certainty.
            representative = [r for r in all_rows if r['training_seed'] == rep]
            test_rows[condition, policy] = representative
            summary['results'][f'{condition}_{policy}'] = aggregate(representative)
            summary['training_seed_returns'][f'{condition}_{policy}'] = means
            if len(means) > 1:
                summary['across_training_runs'][f'{condition}_{policy}'] = across_runs_ci(means)
            seed_runs = {}
            for r in all_rows:
                seed_runs.setdefault(r['training_seed'], []).append(r)
            test_rows[condition, policy, 'all'] = list(seed_runs.values())
            res = summary['results'][f'{condition}_{policy}']['return']
            spread = (f"   | across {len(means)} runs {statistics.mean(means):.3f} +/- "
                      f"{summary['across_training_runs'][f'{condition}_{policy}']['ci95']:.3f}") if len(means) > 1 else ''
            print(f'  {condition:<8}{policy:<11} return {res["mean"]:7.3f} +/- {res["ci95_halfwidth"]:.3f}{spread}')

    # 3. Save the representative model and the paired learned-vs-greedy comparison.
    print('Step 3/3  Saving evidence')
    q_table, visits, curve = models[best, rep]
    save_table(out / 'q_table.json', q_table)
    save_table(out / 'visit_counts.json', visits)
    write_csv(out / 'learning_curve.csv', curve)
    paired = []
    for condition in ['nominal', 'stress']:
        diffs = [a['return'] - b['return']
                 for a, b in zip(test_rows[condition, 'learned'], test_rows[condition, 'greedy'])]
        paired.append({'condition': condition, 'learned_minus_greedy': statistics.mean(diffs),
                       'ci95_halfwidth': 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs))})
    summary['paired_comparisons'] = paired
    # Learned minus greedy for EVERY training run (greedy needs no training, so it is the same in each).
    greedy_means = {c: statistics.mean(r['return'] for r in test_rows[c, 'greedy']) for c in ('nominal', 'stress')}
    summary['paired_across_runs'] = {
        c: across_runs_ci(m - greedy_means[c] for m in summary['training_seed_returns'][f'{c}_learned'])
        for c in ('nominal', 'stress')}
    # Mission-level interval for the policy AVERAGED over all training runs (each mission averaged over runs).
    summary['paired_all_runs_mission_level'] = {
        c: paired_ci(test_rows[c, 'learned', 'all'], [test_rows[c, 'greedy']]) for c in ('nominal', 'stress')}
    summary['elapsed_seconds'] = round(time.time() - started, 2)
    summary['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(out / 'summary.json', summary)
    for p in paired:
        lo, hi = p['learned_minus_greedy'] - p['ci95_halfwidth'], p['learned_minus_greedy'] + p['ci95_halfwidth']
        a = summary['paired_across_runs'][p['condition']]
        across = (f"   | across {a['runs']} training runs {a['mean']:.3f} [{a.get('low', a['mean']):.3f}, "
                  f"{a.get('high', a['mean']):.3f}]") if a['runs'] > 1 else ''
        print(f"  learned - greedy ({p['condition']}): {p['learned_minus_greedy']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]{across}")
        m = summary['paired_all_runs_mission_level'][p['condition']]
        print(f"      all-runs average, mission-level CI: {m['diff']:.3f} [{m['low']:.3f}, {m['high']:.3f}]")
    print(f"  done in {summary['elapsed_seconds']} s -> {out}")
    return summary


if __name__ == '__main__':
    run_main_experiment()
