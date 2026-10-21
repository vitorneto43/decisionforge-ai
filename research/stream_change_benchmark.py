"""Repeated synthetic change-point audit. No real-world validity claims."""
from __future__ import annotations
import argparse, json
import numpy as np
from neural_decision import NeuralModel, make_world
from multi_environment_learning import outcomes

WORLDS=('regulation_change','supply_disruption','demand_reversal')

def absolute_errors(model,rng,n,regime):
    c=rng.uniform(-1,1,(n,3))
    action=rng.integers(0,2,n)
    zero,one=outcomes(c,regime)
    realized=np.where(action==1,one,zero)+rng.normal(0,.07,n)
    predicted=model.predict(np.column_stack([c,action]))
    return np.abs(realized-predicted)

def first_alarm(errors,window,trigger):
    if len(errors)<window: return None
    mean=np.convolve(errors,np.ones(window)/window,mode='valid')
    hit=np.flatnonzero(mean>trigger)
    return int(hit[0]+window) if len(hit) else None

def evaluate(seed=42,trials=8,train_size=2200,epochs=100,
             calibration=450,pre=300,post=350,window=30):
    if min(trials,epochs,window)<1 or train_size<100 or calibration<window or pre<window or post<window:
        raise ValueError('Invalid experiment dimensions')
    stats={name:{'false':0,'detected':0,'delays':[]} for name in ('surprise','mean_error')}
    for trial in range(trials):
        s=seed+101*trial
        x,y=make_world(train_size,s)
        model=NeuralModel(s).fit(x,y,epochs=epochs,seed=s+5)
        ref=absolute_errors(model,np.random.default_rng(s+999),calibration,'reference')
        threshold=float(np.quantile(ref,.9))
        ones=np.ones(window)/window
        surprise_limit=float(np.quantile(np.convolve((ref>threshold).astype(float),ones,mode='valid'),.99))
        mean_limit=float(np.quantile(np.convolve(ref,ones,mode='valid'),.99))
        for i,world in enumerate(WORLDS):
            rng=np.random.default_rng(s+2000+i)
            before=absolute_errors(model,rng,pre,'reference')
            after=absolute_errors(model,rng,post,world)
            stream=np.concatenate([before,after])
            for name,values,limit in (
                ('surprise',(stream>threshold).astype(float),surprise_limit),
                ('mean_error',stream,mean_limit)):
                pre_alarm=first_alarm(values[:pre],window,limit)
                # Incorporate the previous window-1 points so an alarm can be
                # recorded promptly following the change boundary.
                post_alarm=first_alarm(values[pre-window+1:],window,limit)
                absolute_index=pre-window+1+post_alarm if post_alarm is not None else None
                detected=absolute_index is not None and absolute_index>pre
                stats[name]['false']+=int(pre_alarm is not None)
                stats[name]['detected']+=int(detected)
                if detected:stats[name]['delays'].append(absolute_index-pre)
    count=trials*len(WORLDS)
    scores={}
    for name,d in stats.items():
        delays=d['delays']
        scores[name]={
            'streams':count,
            'false_alarm_stream_rate':round(d['false']/count,4),
            'detection_rate':round(d['detected']/count,4),
            'median_delay_detected':float(np.median(delays)) if delays else None,
            'mean_delay_detected':round(float(np.mean(delays)),2) if delays else None}
    return {'note':'Synthetic repeated change-point audit; thresholds are uncalibrated for sequential false alarm probability.',
            'trials':trials,'worlds':list(WORLDS),'prechange_samples':pre,
            'postchange_samples':post,'window':window,'metrics':scores}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--epochs',type=int,default=100)
    a=p.parse_args()
    print(json.dumps(evaluate(seed=a.seed,trials=a.trials,epochs=a.epochs),indent=2))
