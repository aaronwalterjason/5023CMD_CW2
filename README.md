# Drone dispatcher: code, evidence and report pack

This package started as a reference report and simulation generated with OpenAI Codex on 16 October 2026. On 12 November 2026 the code was refactored and extended with Claude Code (Anthropic). The experiment really runs, but every outcome is **simulated**. These are not physical drone trials.

## For the report team: start here

1. Run `python run_all.py` (takes a few minutes).
2. Optional: run `python report_pack.py`, which regenerates **`report_pack/RESULTS.md`** (no longer built by `run_all.py`). It holds every number, table and figure reference, grouped under the four Task 2 sections of the brief, plus a list of what the report can and cannot claim.
3. Put charts from **`figures/`** into the report. Screenshot files from **`evidence/`** for the appendix.

Don't type numbers in by hand. The pack is regenerated from `evidence/` on every run, so it always matches the code.

| Brief section | Where to look |
|---|---|
| 1. Choice of ML model | RESULTS.md Section 1, figures 3, 4, 8 and 9 |
| 2. Training and hyperparameters, overfitting risks, edge cases | RESULTS.md Section 2 and "Edge-case handling", figures 1, 2, 5, 6, 10A, 11 and 12 |
| 3. Explainable AI, approval before risky actions, over-trust | RESULTS.md Section 3, figures 13 (decision explanation), 7 and 10B, `evidence/explanation_*.txt`, `evidence/feedback_suggestions.md` |
| 4. Configuration files, logs, evaluation | RESULTS.md Section 4, `evidence/config.json`, `evidence/run_output.log`, `evidence/summary.json` |

## Commands

Use Python 3.10 or newer. The experiment itself needs only the standard library. The charts also need `matplotlib` (`pip install matplotlib`); without it, everything else still runs.

```text
python run_all.py                      # everything: evidence, checks, studies, figures, report pack
python run_all.py --quick              # short version for live demos, 3 seeds (writes to quick_demo/, official files untouched)
python explain.py                      # explain test mission 20000 decision by decision
python explain.py --condition stress --supervisor always_approve
python explain.py --condition stress --supervisor human     # YOU approve or refuse each risky move (y / n / q)
python check_experiment.py             # correctness checks only
```

`run_all.py` overwrites `evidence/` and `figures/`. The original Codex code and evidence are kept in `_backup_original/`. With the default settings the refactored code reproduces the original results exactly.

## Code map (for the viva)

| File | What it does |
|---|---|
| `experiment.py` | `CONFIG` (every setting), the simulator (`Mission`), Q-learning (`q_update`, `train`), evaluation, and the main train → validate → test pipeline |
| `explain.py` | Decision explanations, the multi-step plan preview, the human approval gate, the simulated supervisors, the over-trust study, and a live mode where a real person approves or refuses (`--supervisor human`) |
| `robustness.py` | Hyperparameter sweep, plus the overfitting tests: shifted map, forgetting, loop traps (all with paired 95% confidence intervals) |
| `extra_experiments.py` | The best-possible-policy benchmark (exact dynamic programming), mixed-condition training, and a weather-aware risk filter. These are added on top; the baseline results don't change. |
| `edge_cases.py` | Edge cases the baseline doesn't model: lost acknowledgements, false detections, a sector closed mid-mission, corrupted input. Each is handled and measured against a naive dispatcher. |
| `feedback_review.py` | Parses every supervisor decision (study logs and live human sessions) by flag type, refusal reason and sector, and writes suggested rule reviews. It never changes the model or the rules. |
| `stress_test.py` | Randomised stress test: hundreds of random fault mixes; checks that no safety rule is ever broken |
| `check_experiment.py` | 30 automatic correctness checks, including a mutation check that the stress test can catch a broken dispatcher; written to `evidence/checks.json` |
| `make_figures.py` | Charts in `figures/` |
| `report_pack.py` | Writes `report_pack/RESULTS.md` (run on its own; not part of `run_all.py`) |
| `run_all.py` | Runs all of the above in order and logs the output |
| `_unused/` | Files no longer used: the old Codex report and its build scripts (`codex_report/`), plus old zips, the 3-seed backup and the outdated results pack (`old_outputs/`) |

Every setting lives in `CONFIG` at the top of `experiment.py`. Change a value there, run `python run_all.py`, and all evidence, figures and the pack update to match. `evidence/config.json` is a copy written at run time; the program never reads it back.

## What the code models, and what it doesn't

**Implemented, in simulation:**
- A central tabular Q-learning dispatcher assigns 6 sectors between 2 drones.
- Survivor detection, hazards and rejected commands are simulated with probabilities, and wind makes trips cost more energy.
- A risk filter blocks dangerous sectors.
- Before each move the code produces an explanation, a plan preview and a risk check.
- Flagged moves go to a simulated supervisor.
- A retry guard stops command loops.

**Not implemented:**
- Real drones, flight control or collision avoidance
- Image-based detection
- Bayesian fusion or a vector database
- A real user interface
- A distributed architecture

**Simplifications:**
- The supervisors are simple rules, not people.
- The "link fault" condition, where drone B rejects every command, exists only to demonstrate loop traps.

Training, validation and test seeds don't overlap: 10 training runs (seeds 11–110), validation 10000–10199, test 20000–20499. Five learning rates are compared, and the choice is made on validation only. Every policy faces the same test missions. Results are reported two ways: mission-level confidence intervals for the representative seed (11), and confidence intervals across all 10 training runs, which include training luck.

## Evidence files (`evidence/`)

- `config.json`: every setting used. `run_output.log`: console log of the latest full run.
- `validation.csv`: learning-rate selection. `summary.json`: all test results, the paired comparison, and a code fingerprint.
- `q_table.json`: the trained model (key format `mask,energyA,energyB,wind,attempts`). `visit_counts.json`: how often each situation was seen in training.
- `learning_curve.csv`, `learning_curves_all.csv`: training progress.
- `nominal_*.csv`, `stress_*.csv`: one row per test mission.
- `trace_nominal.json`, `trace_stress.json`: every decision in test mission 20000.
- `explanation_*.txt`, `explanation_first_decision.json`: readable decision transcripts.
- `supervisor_study.json`, `supervisor_missions.csv`, `approval_log.csv`: the over-trust study.
- `hyperparameter_sweep.csv`, `shifted_map_test.csv`, `forgetting_test.csv`, `loop_trap_test.csv`, `robustness.json`: the extra studies.
- `optimal_benchmark.csv`, `mixed_training_test.csv`, `weather_aware_filter_test.csv`, `extra_experiments.json`: the benchmark and side experiments.
- `edge_cases.json`, `edge_case_missions.csv`, `edge_case_example.txt`: the edge-case study and a sample mission.
- `stress_test.json`, `stress_test_scenarios.csv`: the randomised stress test.
- `feedback_review.json`, `feedback_suggestions.md`: parsed supervisor feedback and suggested rule reviews (pending human sign-off). `feedback_log.jsonl` appears after a live human session.
- `checks.json`: results of the correctness checks.

## Before submission

The assignment requires:
- 1,200–1,800 words per task, with each count stated
- two diagrams
- ML configuration, logs and screenshots
- an AI-use declaration
- a Turnitin plagiarism-rate screenshot

The brief places Task 1 in Amber, with an AI-content limit of 20%. Check how that limit is interpreted with your lecturer. Failing the Task 2 viva gives zero for Task 2.

Before you submit:
- Replace the group-name and date fields.
- Recalculate word counts after editing.
- Check every claim against the "Cannot claim" list in `report_pack/RESULTS.md`.

## Suggested AI declaration

OpenAI Codex was used on 16 October 2026 to locate research sources, generate an initial report draft and diagrams, develop a simplified Python simulation, execute training and evaluation, and organise supporting evidence. Claude Code (Anthropic) was used on 12 November 2026 to refactor and extend the simulation code: the explanations, approval gate, supervisor study, robustness tests, best-possible benchmark, side experiments, edge-case handling, figures and results pack. The reported experiment is a simulation, and the full proposed swarm architecture was not implemented. [Add an accurate account of the group's independent writing, code changes, reproduction, source checking and validation before submission.]

Keep this declaration's substantive disclosure when adapting it. Don't state that your group performed checks it hasn't actually completed.

## Presentation and viva preparation

Everyone should be able to explain the complete experiment, not only their own section.

1. **Why Q-learning?** The state and action space is small and easy to inspect, so deep learning is unnecessary for this prototype. It is not the recommended final solution for raw imagery or continuous control.
1b. **How good is it?** It reaches about 98% of the best possible policy in nominal conditions and 87% under stress. The best possible policy was computed exactly by dynamic programming; see figure 9.
2. **Is it multi-agent reinforcement learning?** No. One central learned dispatcher allocates tasks to two simulated drones.
3. **What is a state?** Surveyed sectors, two energy budgets, wind, and the attempt count. The surveyed sectors are a bitmask: surveying sector 2 sets its bit, which adds 4.
4. **What does a Q-value mean?** An estimated discounted return. It is not a confidence level, a detection probability or a safety certification.
5. **Explain one update.** With Q = 5, reward = 2, best next value = 6, alpha = 0.05 and gamma = 0.95, the target is 7.7 and the new Q is 5.135. A terminal update leaves out the next-state term. See `q_update()` in experiment.py.
6. **Why mask actions?** To prevent known invalid assignments before execution. A reward penalty alone still permits a profitable risky action.
7. **How does the agent explain itself?** Run `python explain.py`. Before each move it shows its recommendation, the alternatives, the blocked moves with reasons, its plan, and a risk check.
8. **When does it ask a human?** When the hazard estimate is at least 0.10, when the situation was rarely or never seen in training, or after 2 rejected commands in one mission. If the supervisor refuses, it offers its next-best move or abstains. To show it live, run `python explain.py --condition stress --supervisor human` and let the examiner type y or n.
9. **What is automation bias here?** Under stress the agent's hazard estimates are out of date. A supervisor who always approves let through more than 1,400 unsafe requests and gained no safety at all. See figure 7.
10. **Why not claim stress superiority?** The paired interval includes zero.
11. **What is the fallback?** In situations not seen in training, the dispatcher uses the greedy rule. It is used more under stress.
11b. **Which edge cases are handled?** Unseen situations, rejected commands, broken links, lost acknowledgements (pending, then telemetry check), false detections (Bayesian confirmation), sectors closed mid-mission, corrupted input, refused or unanswered approvals. Each has an automatic check; see figure 12 and `evidence/edge_case_example.txt`.
12. **How do you show overfitting?** On a new survivor map, the old Q-table scores below a retrained one. Continued stress-only training lowers its nominal performance (forgetting). A broken drone link causes command loops until the retry guard stops them. See figure 6.
13. **How is leakage avoided?** Training, validation and test seeds are separate, and the learning rate is chosen on validation only.
14. **What is the largest missing safety feature?** There is no flight physics, collision model or validated real-world detector.
15. **What would you improve first?** Put survivor and hazard estimates into the state, and test the explanations with real users. Mixed-condition training and a weather-aware filter were already tried as side experiments; see figure 10.

For the live demo:
- run `python run_all.py --quick`
- open `config.json`
- show `validation.csv`
- point at a value in `q_table.json`
- run `python explain.py --condition stress`
- run `python explain.py --condition stress --supervisor human` and approve or refuse the moves yourself (the session is saved to `evidence/human_session_*.txt`)
