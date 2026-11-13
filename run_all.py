"""One command to regenerate all evidence, figures and checks.
(The results pack is no longer built here; run `python report_pack.py` separately if needed.)

    python run_all.py            # full run (a few minutes) -> evidence/, figures/
    python run_all.py --quick    # small fast run for live demos -> quick_demo/ (official files untouched)

The console output is also saved to evidence/run_output.log (or quick_demo/evidence/run_output.log).
"""
import argparse
import copy
import datetime
import hashlib
import json
import platform
import sys
import time

import check_experiment
import edge_cases
import explain
import extra_experiments
import feedback_review
import make_figures
import robustness
import stress_test
from experiment import CONFIG, ROOT, run_main_experiment


class Tee:
    """Write console output to the screen and to a log file at the same time."""

    def __init__(self, path):
        self.file = path.open('w', encoding='utf-8')
        self.screen = sys.stdout

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)

    def flush(self):
        self.screen.flush()
        self.file.flush()


def quick_config():
    cfg = copy.deepcopy(CONFIG)
    cfg.update({'episodes': 2000, 'validation_episodes': 50, 'test_episodes': 100,
                'training_seeds': [11, 22, 33], 'alpha_candidates': [0.05, 0.1, 0.2]})
    cfg['stress_test'] = dict(cfg['stress_test'], scenarios=60)
    cfg['sweep'] = {'gamma': [0.9, 0.95], 'episodes': [1000, 2000], 'hazard_penalty': [-6, -10]}
    cfg['robustness'] = dict(cfg['robustness'], forgetting_episodes=1000)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='small fast run into quick_demo/')
    args = parser.parse_args()
    cfg, base = (quick_config(), ROOT / 'quick_demo') if args.quick else (CONFIG, ROOT)
    evidence = base / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(evidence / 'run_output.log')
    started = time.time()
    print('=' * 72)
    print(f"Drone dispatcher - full run ({'QUICK DEMO' if args.quick else 'official'})")
    print(f"Date: {datetime.datetime.now().isoformat(timespec='seconds')}   Python {platform.python_version()} on {platform.system()}")
    print(f"Config fingerprint: {hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]}   "
          f"episodes={cfg['episodes']} alphas={cfg['alpha_candidates']} gamma={cfg['gamma']} seeds={cfg['training_seeds']}")
    print('=' * 72)

    print('\n[1/8] Main experiment (train, choose learning rate, test)')
    run_main_experiment(cfg, evidence)
    print('\n[2/8] Correctness checks')
    checks = check_experiment.run_checks(cfg, evidence)
    print('\n[3/8] Explanations, supervisor (over-trust) study and feedback review')
    explain.run_all_explain(cfg, evidence)
    feedback_review.run_feedback_review(cfg, evidence)
    print('\n[4/8] Hyperparameter sweep and overfitting tests')
    robustness.run_all_robustness(cfg, evidence)
    print('\n[5/8] Best-possible benchmark and side experiments')
    extra_experiments.run_all_extra(cfg, evidence)
    print('\n[6/8] Edge cases (lost acknowledgements, false detections, closures, invalid input)')
    edge_cases.run_all_edge(cfg, evidence)
    print('\n[7/8] Randomised stress test of the safety rules')
    stress_test.run_stress_test(cfg, evidence)
    print('\n[8/8] Figures')
    make_figures.make_all_figures(base)

    failed = [k for k, v in checks.items() if v == 'FAIL']
    print('\n' + '=' * 72)
    print(f'Finished in {time.time() - started:.0f} s. Checks: {"ALL PASSED" if not failed else "FAILED: " + ", ".join(failed)}')
    print(f'Outputs: {evidence}  |  {base / "figures"}')
    sys.stdout.flush()
    sys.stdout = sys.stdout.screen
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
