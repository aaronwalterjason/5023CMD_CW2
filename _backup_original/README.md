# Report and reproducible evidence

This package was generated with OpenAI Codex on 16 October 2026 for a reference report on autonomous drone swarms. The experiment has actually been run; its outcomes are simulated, not physical drone trials. Your group has not yet independently reproduced or validated this work.

## Before submission

The assignment requires 1,200–1,800 words for each task, with each count stated, two diagrams, ML configuration/logs/screenshots, an AI-use declaration, and a Turnitin plagiarism-rate screenshot. The supplied brief places Task 1 in Amber with an AI-content limit of 20%. Use this as a research and learning reference, develop your own group's analysis, and check the interpretation of that limit with your lecturer. Detector scores cannot be guaranteed. Task 2 permits existing models, but still requires declared tool use and practical understanding. Failing its viva results in zero for Task 2 according to the brief. A report alone cannot secure the 20% presentation/viva marks.

Replace the group-name and date fields. Recalculate counts after edits. Do not claim that the proposed aircraft control, Bayesian fusion, VectorDB, human interface or distributed architecture has been implemented. The code only models central sector allocation, stochastic detection/hazards, wind-dependent costs and requests explicitly rejected before execution.

## Reproduce the experiment

Use Python 3.10 or newer. No third-party packages are required for the simulation.

```text
python experiment.py
python check_experiment.py
```

Running the experiment overwrites its generated evidence files. Save your group's original run and any modified configurations separately. The input settings are the CONFIG dictionary in experiment.py; evidence/config.json is an emitted manifest, not a configuration file read by the program. The nominal and stress dictionaries and learning-rate candidate list are read by the experiment, while several fixed environment values are also explicitly encoded in functions. Keep those definitions consistent when changing the model. The implemented validation range is 10000–10199.

The main report uses training seed 11, fixed before evaluation. Other training seeds are in summary.json and the CSVs. Each policy shares the same 500 scenario seeds; confidence intervals must not pool these dependent repeats as independent missions. The unshielded model is trained separately without the risk filter; it still has battery and visited-sector masks.

## Evidence files

- config.json: training and scenario manifest.
- validation.csv: all nine learning-rate and training-seed comparisons.
- summary.json: selected learning rate, all metrics, paired comparisons and source hash.
- nominal_*.csv and stress_*.csv: individual mission outcomes.
- q_table.json: representative learned table; key format is mask,energyA,energyB,wind,attempts.
- learning_curve.csv: mean training return per thousand episodes for selected alpha and seed 11.
- trace_nominal.json and trace_stress.json: decision-level logs for scenario 20000.
- checks.json: results of simulator invariant checks.

## Suggested AI declaration

OpenAI Codex was used on 16 October 2026 to locate research sources, generate an initial report draft and diagrams, develop a simplified Python simulation, execute training and evaluation, and organise supporting evidence. The reported experiment is a simulation, and the full proposed swarm architecture was not implemented. [Add an accurate account of the group's independent writing, code changes, reproduction, source checking and validation before submission.]

Retain this declaration's substantive disclosure when adapting it. Do not state that your group performed checks it has not actually completed.

## Presentation and viva preparation

Use four speaking roles if your group has four members, adjusted to the lecturer's allocated duration: mission and uncertainty; architecture and memory; ML and evidence; safety, limitations and conclusion. Everyone should be able to explain the complete experiment, not only their allocated section.

1. Why Q-learning? The state/action space is small and inspectable; deep learning is unnecessary for this prototype. It is not the recommended final solution for raw imagery or continuous control.
2. Is it multi-agent reinforcement learning? No. One central learned dispatcher allocates tasks to two simulated drone agents.
3. What is a state? Surveyed sectors, two energy budgets, wind and attempt count. Explain a bitmask using sector 2: setting its bit adds 4.
4. What does a Q-value mean? Estimated discounted return, not confidence, detection probability or safety certification.
5. Explain one update. With Q=5, reward=2, next feasible maximum=6, alpha=0.05 and gamma=0.95, the target is 7.7 and the new Q is 5.135. Terminal updates omit the next-state term.
6. Why mask actions? Prevent known invalid assignments before execution. A reward penalty alone still permits a profitable risky action. The risk mask does not eliminate nonzero residual hazards or incorrect estimates.
7. What counts as ground truth? The simulator internally samples survivor presence; policies cannot read those samples. In reality verified annotations and independent confirmation would be required.
8. Why not claim stress superiority? The paired interval includes zero. The result does not establish an advantage over greedy allocation under that stress distribution.
9. What is the fallback? At unseen states, the dispatcher uses the analytical greedy heuristic. Its use increases under stress, so the evaluated system is a learned policy plus fallback.
10. What did the ablation show? Removing the risk filter increased reward and detections, but also hazards. It changes training and evaluation together and does not isolate only inference-time masking.
11. How is leakage avoided? Training, validation and test seeds are distinct. Alpha is selected using validation only. All conditions still share the same small scenario family, limiting generalisation.
12. What is the largest missing safety feature? There is no flight physics, collision model or validated real-world detector. The prototype cannot establish physical flight safety.
13. Why separate a VectorDB from operational state? Semantic similarity is not a reliable current-status query. Structured versioned state controls dispatch; semantic retrieval supports contextual knowledge.
14. Can retries duplicate a flight? Not in the simplified pre-execution rejection model. Real ambiguous acknowledgements need idempotent commands and telemetry reconciliation, which are proposed rather than implemented.
15. What would you improve first? Independently reproduce results, isolate stressors, add new priors/maps, compare an exact planner, then introduce realistic sensing and flight dynamics under supervision.

For the live demonstration, open config.json, show validation.csv, identify alpha 0.05 in summary.json, explain the first two trace events, and compare nominal versus stress outcomes. Each member should practise locating those files and answering follow-up questions without reading a script.
