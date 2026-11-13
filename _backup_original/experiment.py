"""Reproducible educational dispatch simulator; not flight-control software.
Run: python experiment.py. Requires only the Python standard library.
"""
import csv, hashlib, json, math, random, statistics, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence'
OUT.mkdir(exist_ok=True)
CONFIG = {
    'training_seeds': [11, 22, 33], 'episodes': 12000,
    'alpha_candidates': [0.05, 0.1, 0.2], 'gamma': 0.95,
    'epsilon_start': 1.0, 'epsilon_end': 0.05, 'decay_fraction': 0.8,
    'validation_seed_start': 10000, 'test_seed_start': 20000,
    'validation_episodes': 200, 'test_episodes': 500,
    'sector_presence': [0.15, 0.45, 0.6, 0.8, 0.35, 0.9],
    'sector_hazard': [0.02, 0.04, 0.08, 0.12, 0.18, 0.30],
    'base_energy_cost': [[1, 1, 2, 2, 3, 2], [3, 2, 2, 1, 1, 2]],
    'usable_energy_each': 4, 'max_dispatch_attempts': 12,
    'risk_mask_threshold': 0.2,
    'reward': {'detection':10, 'coverage':1, 'energy':-0.4, 'hazard':-6, 'api_failure':-0.5},
    'nominal': {'api_failure':0.05, 'sensitivity':0.85, 'wind_probability':0.5, 'hazard_multiplier':1},
    'stress': {'api_failure':0.35, 'sensitivity':0.5, 'wind_probability':0.9, 'hazard_multiplier':2},
}
P = CONFIG['sector_presence']; H = CONFIG['sector_hazard']; COST = CONFIG['base_energy_cost']

def energy(d, j, w):
    return COST[d][j] + int(w and COST[d][j] >= 2)

def actions(s, shield=True):
    mask, e0, e1, w, t = s
    if t == 12: return []
    return [d*6+j for d in range(2) for j in range(6)
            if not mask & (1<<j) and energy(d,j,w) <= (e0,e1)[d]
            and (not shield or H[j] <= 0.2)]

def heuristic(s, aa):
    w=s[3]
    return max(aa, key=lambda a: ((10*P[a%6]*0.85+1-6*H[a%6])/energy(a//6,a%6,w)-0.4, -a))

class Mission:
    def __init__(self, seed, condition='nominal'):
        r=random.Random(seed); self.c=CONFIG[condition]
        self.s=(0,4,4,int(r.random()<self.c['wind_probability']),0)
        # Fixed per-sector outcomes and per-time/command failures allow paired policies.
        self.present=[r.random()<p for p in P]
        self.detect=[r.random()<self.c['sensitivity'] for _ in P]
        self.hazard=[r.random()<min(1,h*self.c['hazard_multiplier']) for h in H]
        self.fail=[[r.random()<self.c['api_failure'] for _ in range(12)] for _ in range(12)]
        self.total=0; self.found=0; self.hits=0; self.surveyed=0; self.failures=0; self.used=0
    def step(self,a):
        mask,e0,e1,w,t=self.s; d,j=divmod(a,6)
        if self.fail[t][a]:
            reward=-0.5; self.failures+=1
            self.s=(mask,e0,e1,w,t+1); event='api_rejected_before_execution'
        else:
            cost=energy(d,j,w); es=[e0,e1]; es[d]-=cost
            assert min(es)>=0 and not mask&(1<<j)
            det=int(self.present[j] and self.detect[j]); haz=int(self.hazard[j])
            reward=10*det+1-0.4*cost-6*haz
            self.found+=det; self.hits+=haz; self.surveyed+=1; self.used+=cost
            self.s=(mask|(1<<j),*es,w,t+1); event='survey_complete'
        self.total+=reward
        return reward,event

def train(alpha, seed, shield=True):
    rng=random.Random(seed); q={}; visits={}; curve=[]; window=[]
    for ep in range(CONFIG['episodes']):
        env=Mission(seed*1000000+ep)
        eps=max(0.05,1-0.95*ep/(CONFIG['episodes']*0.8))
        while (aa:=actions(env.s,shield)):
            s=env.s; row=q.setdefault(s,[0.0]*12)
            a=rng.choice(aa) if rng.random()<eps else max(aa,key=lambda x:(row[x],-x))
            reward,_=env.step(a); nxt=actions(env.s,shield)
            target=reward+(0.95*max(q.get(env.s,[0.0]*12)[b] for b in nxt) if nxt else 0)
            row[a]+=alpha*(target-row[a]); visits[s]=visits.get(s,0)+1
        window.append(env.total)
        if (ep+1)%1000==0:
            curve.append({'episode':ep+1,'mean_training_return':statistics.mean(window),'epsilon':eps})
            window=[]
    return q,visits,curve

def evaluate(q, start, n, condition='nominal', policy='learned', shield=True, trace=False):
    rows=[]; logs=[]
    for k in range(n):
        env=Mission(start+k,condition); rng=random.Random(start+k+777); fallback=0; steps=0
        while (aa:=actions(env.s,shield)):
            s=env.s
            if policy=='random': a=rng.choice(aa)
            elif policy=='greedy': a=heuristic(s,aa)
            elif s not in q:
                a=heuristic(s,aa); fallback+=1
            else: a=max(aa,key=lambda x:(q[s][x],-x))
            reward,event=env.step(a); steps+=1
            if trace and k==0: logs.append({'seed':start,'state_before':s,'allowed_actions':aa,'action':a,'drone':a//6,'sector':a%6,'q_value':q.get(s,[0.0]*12)[a], 'event':event,'reward':reward,'state_after':env.s})
        rows.append({'seed':start+k,'return':env.total,'detections':env.found,'coverage_pct':env.surveyed/6*100,
                     'hazards':env.hits,'api_failures':env.failures,'energy':env.used,
                     'fallback_steps':fallback,'steps':steps})
    return rows,logs

def aggregate(rows):
    out={}
    for key in ['return','detections','coverage_pct','hazards','api_failures','energy']:
        vals=[r[key] for r in rows]; m=statistics.mean(vals); se=statistics.stdev(vals)/math.sqrt(len(vals))
        out[key]={'mean':m,'ci95_halfwidth':1.96*se}
    out['fallback_pct']=100*sum(r['fallback_steps'] for r in rows)/sum(r['steps'] for r in rows)
    return out

def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    begun=time.time(); (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    tuning=[]; models={}
    for alpha in CONFIG['alpha_candidates']:
        for seed in CONFIG['training_seeds']:
            q,v,curve=train(alpha,seed); models[alpha,seed]=(q,v,curve)
            rows,_=evaluate(q,10000,200)
            tuning.append({'alpha':alpha,'training_seed':seed,'validation_return':statistics.mean(r['return'] for r in rows),'states':len(q)})
    best=max(CONFIG['alpha_candidates'],key=lambda a:statistics.mean(x['validation_return'] for x in tuning if x['alpha']==a))
    write_csv('validation.csv',tuning); summary={'selected_alpha':best,'results':{},'training_seed_returns':{}}
    for condition in ['nominal','stress']:
        for policy in ['learned','greedy','random','unshielded']:
            allrows=[]; means=[]
            seeds=CONFIG['training_seeds'] if policy in ['learned','unshielded'] else [11]
            for seed in seeds:
                if policy=='unshielded':
                    if ('unshielded',seed) not in models: models['unshielded',seed]=train(best,seed,False)
                    q=models['unshielded',seed][0]
                else: q=models[best,seed][0]
                rows,logs=evaluate(q,20000,500,condition,'learned' if policy=='unshielded' else policy,policy!='unshielded',trace=seed==11 and policy=='learned')
                means.append(statistics.mean(r['return'] for r in rows))
                for r in rows: r['training_seed']=seed
                allrows+=rows
                if logs: (OUT/f'trace_{condition}.json').write_text(json.dumps(logs,indent=2))
            write_csv(f'{condition}_{policy}.csv',allrows)
            # CIs for representative seed 11 only: repeated policies on shared missions are dependent.
            representative=[r for r in allrows if r['training_seed']==11]
            summary['results'][f'{condition}_{policy}']=aggregate(representative)
            summary['training_seed_returns'][f'{condition}_{policy}']=means
    q,visits,curve=models[best,11]
    (OUT/'q_table.json').write_text(json.dumps({','.join(map(str,s)):v for s,v in q.items()}))
    write_csv('learning_curve.csv',curve)
    paired=[]
    for cond in ['nominal','stress']:
        learned=list(csv.DictReader((OUT/f'{cond}_learned.csv').open()))[:500]
        greedy=list(csv.DictReader((OUT/f'{cond}_greedy.csv').open()))
        diffs=[float(a['return'])-float(b['return']) for a,b in zip(learned,greedy)]
        paired.append({'condition':cond,'learned_minus_greedy':statistics.mean(diffs),'ci95_halfwidth':1.96*statistics.stdev(diffs)/math.sqrt(500)})
    summary['paired_comparisons']=paired
    summary['elapsed_seconds']=round(time.time()-begun,2)
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
