"""Independent stream calibration and holdout audit; synthetic only."""
from __future__ import annotations
import argparse, json, math
import numpy as np
from neural_decision import NeuralModel, make_world
from multi_environment_learning import outcomes

WORLDS=('regulation_change','supply_disruption','demand_reversal')

def errors(model,rng,n,world):
    c=rng.uniform(-1,1,(n,3))
    a=rng.integers(0,2,n)
    zero,one=outcomes(c,world)
    observed=np.where(a==1,one,zero)+rng.normal(0,.07,n)
    return np.abs(observed-model.predict(np.column_stack((c,a))))

def scores(err,window,center,allowance):
    rolling=np.convolve(err,np.ones(window)/window,mode='valid')
    cusum=np.empty(len(err))
    state=0.
    for i,value in enumerate(err):
        state=max(0.,state+float(value)-center-allowance)
        cusum[i]=state
    return rolling,cusum

def first_alarm(path,threshold,offset):
    hits=np.flatnonzero(path>threshold)
    return int(hits[0]+offset) if len(hits) else None

def wilson_interval(k,n,z=1.96):
    if n==0: return None
    p=k/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [round(max(0,mid-margin),4),round(min(1,mid+margin),4)]

def evaluate(seed=42,train_size=2500,epochs=100,calibration=1000,
             stable=500,changed_per_world=50,pre=300,post=350,
             window=30,target=.05):
    if train_size<100 or epochs<1 or calibration<20 or stable<1 or changed_per_world<1:
        raise ValueError('Invalid sample sizes')
    if pre<window or post<window or window<2 or not 0<target<1:
        raise ValueError('Invalid window or target')
    x,y=make_world(train_size,seed)
    model=NeuralModel(seed).fit(x,y,epochs=epochs,seed=seed+5)
    length=pre+post
    rng_cal=np.random.default_rng(seed+1000)
    calibration_errors=[errors(model,rng_cal,length,'reference') for _ in range(calibration)]
    center=float(np.mean([v.mean() for v in calibration_errors]))
    allowance=float(np.std(np.concatenate(calibration_errors))*.25)
    maxima=np.array([[max(path) for path in scores(e,window,center,allowance)]
                     for e in calibration_errors])
    rank=math.ceil((calibration+1)*(1-target))
    thresholds=(np.array([float('inf')]*2) if rank>calibration
                else np.sort(maxima,axis=0)[rank-1])
    methods=('rolling_mean','cusum')
    counts={m:{'stable':0,'early':0,'detected':0,'delays':[]} for m in methods}
    stable_rng=np.random.default_rng(seed+2000)
    for _ in range(stable):
        e=errors(model,stable_rng,length,'reference')
        paths=scores(e,window,center,allowance)
        for j,name in enumerate(methods):
            counts[name]['stable']+=first_alarm(paths[j],thresholds[j],window-1 if j==0 else 0) is not None
    for k,world in enumerate(WORLDS):
        rng=np.random.default_rng(seed+3000+k)
        for _ in range(changed_per_world):
            e=np.concatenate((errors(model,rng,pre,'reference'),errors(model,rng,post,world)))
            paths=scores(e,window,center,allowance)
            for j,name in enumerate(methods):
                alarm=first_alarm(paths[j],thresholds[j],window-1 if j==0 else 0)
                item=counts[name]
                if alarm is not None and alarm<pre:
                    item['early']+=1
                elif alarm is not None:
                    item['detected']+=1
                    item['delays'].append(alarm-pre+1)
    total=changed_per_world*len(WORLDS)
    report={}
    for name,item in counts.items():
        report[name]={
            'stable_false_alarms':item['stable'],'stable_streams':stable,
            'stable_false_alarm_rate':round(item['stable']/stable,4),
            'stable_false_alarm_95pct_wilson':wilson_interval(item['stable'],stable),
            'prechange_alarms':item['early'],'changed_streams':total,
            'detected_without_early_alarm':item['detected'],
            'detection_fraction':round(item['detected']/total,4),
            'median_detection_delay_if_detected':(float(np.median(item['delays'])) if item['delays'] else None)
        }
    return {'notice':'Synthetic validation, independent calibration and evaluation streams.',
            'nominal_stream_false_alarm_target':target,
            'calibration_streams':calibration,'stable_evaluation_streams':stable,
            'changed_evaluation_streams':total,'methods':report}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--calibration',type=int,default=1000)
    p.add_argument('--stable',type=int,default=500)
    p.add_argument('--changed-per-world',type=int,default=50)
    a=p.parse_args()
    print(json.dumps(evaluate(calibration=a.calibration,stable=a.stable,
                              changed_per_world=a.changed_per_world),indent=2))
