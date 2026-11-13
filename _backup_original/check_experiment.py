"""Invariant checks for the simulator, independent of policy performance."""
import json, random
from experiment import Mission, actions, energy, H, OUT

checks = {}
s=(0,4,4,0,0)
assert all(a%6 != 5 for a in actions(s))
assert any(a%6 == 5 for a in actions(s,False))
checks['risk_filter_excludes_sector_5_only_when_enabled']='PASS'
assert not actions((63,4,4,0,0)) and not actions((0,0,0,0,0)) and not actions((0,4,4,0,12))
checks['terminal_conditions']='PASS'
env=Mission(1); before=env.s; env.fail[0][0]=True
r,event=env.step(0)
assert env.s[:4]==before[:4] and env.s[4]==1 and r==-0.5
checks['rejected_request_does_not_consume_energy_or_mark_coverage']='PASS'
for seed in range(1000):
    e=Mission(seed,'stress'); rng=random.Random(seed)
    while aa:=actions(e.s):
        a=rng.choice(aa); prev=e.s; e.step(a)
        assert min(e.s[1:3])>=0 and e.s[4]<=12
        assert e.s[0] | prev[0] == e.s[0]
        assert all(not e.s[0]&(1<<(b%6)) for b in actions(e.s))
checks['1000_stress_missions_preserve_energy_and_unique_coverage']='PASS'
a=Mission(7); b=Mission(7)
assert a.s==b.s and a.fail==b.fail and a.present==b.present
checks['scenario_seed_reproducibility']='PASS'
(OUT/'checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
