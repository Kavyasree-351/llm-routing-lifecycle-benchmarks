"""Public cached LLM routing: independent lifecycle deployment assessment.

This is a pilot of an empirical research question, not a new router or theorem.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from data_io import load,SHA256

ROOT=Path(__file__).resolve().parent
REF='gpt-4-1106-preview'
SEEDS=[3101,3102,3103]
BUDGETS=[256,1024]
LAMBDAS=[0.,.02,.05,.1,.2,.5,1.,2.]
WORKLOADS={'math':['grade-school-math'],
 'reasoning':['hellaswag','winogrande','arc-challenge'],
 'law':['mmlu-professional-law']}
EPS=.02


def canonical(text):return ' '.join(str(text).split())


def split_name(text,first):
    if text==first:return 'train'
    u=int(hashlib.sha256(('lifecycle-v1:'+text).encode()).hexdigest()[:16],16)/2**64
    for end,name in [(.30,'train'),(.40,'tune'),(.60,'cal'),(.80,'pilot'),(.90,'fresh'),(1.,'final')]:
        if u<end:return name


def cp_hi(k,n,alpha):return 1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))
def cp_lo(k,n,alpha):return 0. if k==0 else float(beta.ppf(alpha,k,n-k+1))


def quality_upper(d,alpha=.025):
    # Cached scores can be fractional, so do NOT treat these as Bernoulli.
    return eb_upper(d,alpha=alpha,span=2.)


def eb_upper(v,alpha=.025,span=2.):
    n=len(v)
    return float(v.mean()+np.sqrt(2*v.var(ddof=1)*np.log(2/alpha)/n)+7*span*np.log(2/alpha)/(3*(n-1)))


def bootstrap(v,seed,reps=400):
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(v),size=(reps,len(v)))
    means=v[ix].mean(axis=1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def matrix(df,ix,models):
    rows=df.iloc[ix]
    y=rows[models].to_numpy(dtype=float)
    c=rows[[m+'|total_cost' for m in models]].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(c).all():raise ValueError('Missing cached feedback; do not silently complete-case select')
    if np.min(y)<0 or np.max(y)>1:raise ValueError('Selected quality outside documented bounded-score analysis')
    if np.min(c)<0 or np.max(c)>1:raise ValueError('Declared historical per-call support bound violated')
    return y,c


def prediction_families(texts,ys,seed):
    start=time.perf_counter()
    vector=TfidfVectorizer(max_features=12000,ngram_range=(1,2),min_df=2)
    xs={};xs['train']=vector.fit_transform(texts['train'])
    for part in ['tune','cal','pilot']:xs[part]=vector.transform(texts[part])
    metadata={'vocabulary_size':len(vector.vocabulary_),'ridge':[],'knn':[]}
    best=None
    for alpha in [1.,10.,100.]:
        model=Ridge(alpha=alpha,solver='lsqr').fit(xs['train'],ys['train'])
        pred=np.clip(model.predict(xs['tune']),0,1)
        mse=float(np.mean((pred-ys['tune'])**2));metadata['ridge'].append({'alpha':alpha,'mse':mse})
        if best is None or mse<best[0]:best=(mse,alpha,model)
    preds={'ridge':{part:np.clip(best[2].predict(xs[part]),0,1) for part in ['tune','cal','pilot']}}
    metadata['ridge_chosen_alpha']=best[1]
    maxk=min(45,len(texts['train']))
    nn=NearestNeighbors(n_neighbors=maxk,metric='cosine',algorithm='brute',n_jobs=1).fit(xs['train'])
    indexes={part:nn.kneighbors(xs[part],return_distance=False) for part in ['tune','cal','pilot']}
    bestk=None
    for k in [5,15,45]:
        k=min(k,maxk);pred=ys['train'][indexes['tune'][:,:k]].mean(axis=1)
        mse=float(np.mean((pred-ys['tune'])**2));metadata['knn'].append({'k':k,'mse':mse})
        if bestk is None or mse<bestk[0]:bestk=(mse,k)
    preds['knn']={part:ys['train'][indexes[part][:,:bestk[1]]].mean(axis=1) for part in ['tune','cal','pilot']}
    metadata['knn_chosen_k']=bestk[1]
    lengths={part:np.array([len(x) for x in texts[part]]) for part in texts}
    cuts=np.unique(np.quantile(lengths['train'],np.linspace(0,1,7)[1:-1]))
    bins=np.digitize(lengths['train'],cuts)
    means=[]
    for b in range(len(cuts)+1):
        group=ys['train'][bins==b]
        means.append(group.mean(axis=0) if len(group) else ys['train'].mean(axis=0))
    means=np.array(means)
    preds['length_bins']={part:means[np.digitize(lengths[part],cuts)] for part in ['tune','cal','pilot']}
    preds['global_mean']={part:np.tile(ys['train'].mean(axis=0),(len(texts[part]),1)) for part in ['tune','cal','pilot']}
    metadata['fit_and_all_prediction_seconds']=time.perf_counter()-start
    return preds,metadata


def policies(preds,cost_mean,nmodels,texts=None,ref=None):
    result=[]
    for family,p in preds.items():
        for lam in LAMBDAS:
            picks={part:np.argmax(p[part]-lam*cost_mean[None,:]/cost_mean.max(),axis=1) for part in p}
            result.append({'family':family,'lambda':lam,'picks':picks})
    for m in range(nmodels):
        result.append({'family':'fixed_model','lambda':None,'fixed_index':m,
            'picks':{part:np.repeat(m,len(next(iter(preds.values()))[part])) for part in ['tune','cal','pilot']}})
    if texts is not None:
        # Strong query-independent random mixtures (Zero-Router style). The
        # independent hash salt makes the draw stable without seeing outcomes.
        uniform={part:np.array([int(hashlib.sha256(('fixed-mixture-v1:'+s).encode()).hexdigest()[:16],16)/2**64 for s in texts[part]]) for part in ['tune','cal','pilot']}
        for m in range(nmodels):
            if m==ref:continue
            for fraction in [.05,.1,.2,.4,.6,.8]:
                result.append({'family':'fixed_mixture','lambda':None,'mix_fraction':fraction,'mixture_model_index':m,
                    'picks':{part:np.where(uniform[part]<fraction,m,ref) for part in uniform}})
    return result


def select_policy(candidates,y,c,ref):
    feasible=[]
    for index,p in enumerate(candidates):
        pick=p['picks']['tune'];rows=np.arange(len(pick))
        loss=float(np.mean(y[:,ref]-y[rows,pick]));cost=float(c[rows,pick].mean())
        if loss<=EPS:feasible.append((cost,index,loss))
    if not feasible:raise AssertionError('Reference fixed policy must be feasible')
    return min(feasible)


def main(args):
    if args.stage!='pilot':
        raise SystemExit('Fresh/final evaluation closed: requires a passing pilot and a defensible novelty finding.')
    out=ROOT/'results/pilot';out.mkdir(parents=True,exist_ok=True)
    df=load(args.data)
    models=[x for x in df.columns if x not in ['sample_id','prompt','eval_name','oracle_model_to_route_to'] and '|' not in x]
    ref=models.index(REF)
    first=canonical(df.prompt.iloc[0])
    # No conversion, scoring, fitting, or inference on reserved fresh/final rows.
    canonical_prompts=df.prompt.map(canonical)
    assignments=[split_name(x,first) for x in canonical_prompts]
    partitions={part:np.flatnonzero(np.array(assignments)==part).tolist() for part in ['train','tune','cal','pilot','fresh','final']}
    all_rows=[];runs=[];selected=[]
    start=time.perf_counter()
    for workload,names in WORKLOADS.items():
        groups=np.flatnonzero(df.eval_name.isin(names).to_numpy())
        # Duplicate removal is based on prompt only, before touching outcomes.
        seen=set();unique=[]
        for ix in groups:
            prompt=canonical_prompts.iloc[ix]
            if prompt not in seen:seen.add(prompt);unique.append(ix)
        pools={part:np.array([ix for ix in unique if assignments[ix]==part],dtype=int) for part in partitions}
        for seed in SEEDS:
            order=np.random.default_rng(seed).permutation(pools['train'])
            for requested in BUDGETS:
                ix={part:pools[part] for part in ['tune','cal','pilot']}
                ix['train']=order[:min(requested,len(order))]
                ys={};cs={};texts={}
                for part in ['train','tune','cal','pilot']:
                    ys[part],cs[part]=matrix(df,ix[part],models)
                    texts[part]=canonical_prompts.iloc[ix[part]].tolist()
                preds,timing=prediction_families(texts,ys,seed)
                candidates=policies(preds,cs['train'].mean(axis=0),len(models),texts,ref)
                _,index,tune_loss=select_policy(candidates,ys['tune'],cs['tune'],ref)
                policy=candidates[index]
                diffs={}
                for part in ['cal','pilot']:
                    choice=policy['picks'][part];rr=np.arange(len(choice))
                    diffs[part]={'quality':ys[part][:,ref]-ys[part][rr,choice],
                        'cost':cs[part][rr,choice]-cs[part][:,ref]}
                cal_q=diffs['cal']['quality'];cal_c=diffs['cal']['cost']
                dev_q=diffs['pilot']['quality'];dev_c=diffs['pilot']['cost']
                key=seed+requested*100+(list(WORKLOADS).index(workload)+1)*100000
                bounds={'cal_quality_boot':bootstrap(cal_q,key),'cal_cost_boot':bootstrap(cal_c,key+1),
                  'pilot_quality_boot':bootstrap(dev_q,key+2),'pilot_cost_boot':bootstrap(dev_c,key+3),
                  'cal_quality_finite':quality_upper(cal_q),'cal_cost_bernstein':eb_upper(cal_c)}
                reference_only=bool(np.all(policy['picks']['cal']==ref) and policy['family']=='fixed_model')
                if reference_only:
                    bounds['cal_quality_finite']=0.;bounds['cal_cost_bernstein']=0.
                ref_mean=float(cs['train'][:,ref].mean())
                acquisition=float(sum(cs[part].sum() for part in ['train','tune','cal']))
                feedback=sum(cs[part].size for part in ['train','tune','cal'])
                cal_pick=policy['picks']['cal'];cr=np.arange(len(cal_pick));nonref=cal_pick!=ref
                minimal_acquisition=float(cs['train'].sum()+cs['tune'].sum()+cs['cal'][:,ref].sum()+cs['cal'][cr[nonref],cal_pick[nonref]].sum())
                minimal_feedback=cs['train'].size+cs['tune'].size+len(cal_pick)+int(nonref.sum())
                uid=f'{workload}_{seed}_{requested}'
                selected.append({'id':uid,'workload':workload,'seed':seed,'requested_train':requested,
                    'actual_train':len(ix['train']),'family':policy['family'],'lambda':policy['lambda'],
                    'fixed_model':models[policy['fixed_index']] if policy['family']=='fixed_model' else None,
                    'mix_fraction':policy.get('mix_fraction'),
                    'mixture_model':models[policy['mixture_model_index']] if policy['family']=='fixed_mixture' else None,
                    'tune_excess':tune_loss,'cal_excess':float(cal_q.mean()),'pilot_excess':float(dev_q.mean()),
                    'historical_upfront_api_cost':acquisition,'feedback_count':feedback,'reference_mean_cost':ref_mean,
                    'selected_pair_upfront_api_cost':minimal_acquisition,'selected_pair_feedback_count':minimal_feedback,
                    'pilot_cost_difference':float(dev_c.mean()),'bounds':bounds})
                # Every sensitivity is predefined and retained, including bad ones.
                for horizon in [10000,100000]:
                    for overhead in [0.,.01,.1]:
                        for judge in [0.,1.]:
                            for reuse in [0.,1.]:
                                for accounting,base,nfeedback in [('dense',acquisition,feedback),('selected_pair',minimal_acquisition,minimal_feedback)]:
                                    setup=reuse*(base+judge*ref_mean*nfeedback)
                                    fixed_charge=setup/horizon+overhead*ref_mean
                                    point=cal_q.mean()<=EPS and cal_c.mean()+fixed_charge<0
                                    boot=bounds['cal_quality_boot'][1]<=EPS and bounds['cal_cost_boot'][1]+fixed_charge<0
                                    finite=bounds['cal_quality_finite']<=EPS and bounds['cal_cost_bernstein']+fixed_charge<0
                                    observed_bad=dev_q.mean()>EPS or dev_c.mean()+fixed_charge>=0
                                    clear_bad=bounds['pilot_quality_boot'][0]>EPS or bounds['pilot_cost_boot'][0]+fixed_charge>0
                                    clear_good=bounds['pilot_quality_boot'][1]<=EPS and bounds['pilot_cost_boot'][1]+fixed_charge<0
                                    for rule,approved in [('point',point),('paired_bootstrap',boot),('finite_bound',finite)]:
                                        all_rows.append({'id':uid,'workload':workload,'seed':seed,'train_budget':requested,
                                          'horizon':horizon,'overhead_fraction':overhead,'judge_fraction':judge,'fresh_cost_multiplier':reuse,
                                          'calibration_accounting':accounting,
                                          'rule':rule,'approved':bool(approved),'observed_bad':bool(observed_bad),
                                          'clearly_bad':bool(clear_bad),'clearly_good':bool(clear_good),
                                          'pilot_quality_loss':float(dev_q.mean()),
                                          'pilot_amortized_saving_per_query':float(-dev_c.mean()-fixed_charge),
                                          'setup_cost':setup})
                runs.append({'id':uid,'models':models,'splits':{k:v.tolist() for k,v in ix.items()},
                    'fresh_reserved_indices':pools['fresh'].tolist(),'final_reserved_indices':pools['final'].tolist(),
                    'hyperparameters_and_timing':timing,'candidate_count':len(candidates)})
                runs[-1]['selected_cal_model_indices']=policy['picks']['cal'].tolist()
                runs[-1]['selected_pilot_model_indices']=policy['picks']['pilot'].tolist()
                print('completed',uid,policy['family'],flush=True)
    (out/'raw_decisions.json').write_text(json.dumps(all_rows,indent=2))
    (out/'selected_policies.json').write_text(json.dumps(selected,indent=2))
    (out/'manifest.json').write_text(json.dumps({'data_sha256':SHA256,'models':models,'runs':runs,
       'all_partition_indices':partitions,'seconds_excluding_imports':time.perf_counter()-start,
       'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()},indent=2))
    aggregate(out)


def aggregate(out):
    rows=json.loads((out/'raw_decisions.json').read_text())
    primary=[r for r in rows if r['horizon']==10000 and r['overhead_fraction']==0 and r['judge_fraction']==0 and r['fresh_cost_multiplier']==1 and r['calibration_accounting']=='dense']
    results=[]
    for workload in WORKLOADS:
        for rule in ['point','paired_bootstrap','finite_bound']:
            group=[r for r in primary if r['workload']==workload and r['rule']==rule]
            approved=[r for r in group if r['approved']]
            results.append({'workload':workload,'rule':rule,'decisions':len(group),'approvals':len(approved),
                'observed_bad_approvals':sum(r['observed_bad'] for r in approved),
                'clearly_bad_approvals':sum(r['clearly_bad'] for r in approved),
                'clearly_good_approvals':sum(r['clearly_good'] for r in approved),
                'bad_fraction':float(np.mean([r['observed_bad'] for r in approved])) if approved else None})
    qualifying=[]
    for workload in WORKLOADS:
        p=next(r for r in results if r['workload']==workload and r['rule']=='point')
        b=next(r for r in results if r['workload']==workload and r['rule']=='paired_bootstrap')
        retained_seeds={r['seed'] for r in primary if r['workload']==workload and r['rule']=='paired_bootstrap' and r['approved']}
        yes=(p['approvals']>=6 and p['bad_fraction'] is not None and p['bad_fraction']>=.2 and
             b['approvals']>=p['approvals']*.5 and b['bad_fraction'] is not None and b['bad_fraction']<=p['bad_fraction']*.5
             and len(retained_seeds)>=2 and b['clearly_good_approvals']>0)
        if yes:qualifying.append(workload)
    gate={'performance_gate_passed':len(qualifying)>=2,'qualifying_workloads':qualifying,
      'novelty_established':False,'reason':'Lifecycle payback already studied; current tools are established inference. Novel empirical claim requires independent benchmark confirmation and stronger comparisons.',
      'primary_summary':results}
    (out/'gate.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps(gate,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='work/lifecycle-routing/routerbench_0shot.pkl')
    ap.add_argument('--stage',choices=['pilot','fresh','final'],default='pilot')
    main(ap.parse_args())
