"""Synthetic sequential drift study with independently calibrated stream-level alarms."""
from __future__ import annotations
import argparse, json
import numpy as np
from neural_decision import NeuralModel, make_world
from multi_environment_learning import outcomes

WORLDS = ('regulation_change','supply_disruption','demand_reversal')

def errors(model, rng, n, world):
    c = rng.uniform(-1,1,(n,3))
    a = rng.integers(0,2,n)
    b,e = outcomes(c,world)
    y = np.where(a==1,e,b) + rng.normal(0,.07,n)
    return np.abs(y - model.predict(np.column_stack((c,a))))

def rolling_mean(data, window):
    return np.convolve(data,np.ones(window)/window,mode='valid')

def cusum_path(data, center, allowance):
    current = 0.
    out = np.empty(len(data))
    for i, v in enumerate(data):
        current = max(0.,current+float(v)-center-allowance)
        out[i] = current
    return out

def first_alarm(scores, threshold, first_index):
    loc = np.flatnonzero(scores > threshold)
    return int(loc[0]+first_index) if len(loc) else None

def evaluate(seed=42,trials=8,train_size=2200,epochs=100,
             calibration_streams=100,pre=300,post=350,window=30,
             target_false_alarm=.05):
    if trials<1 or epochs<1 or train_size<100 or calibration_streams<20:
        raise ValueError('Invalid trials, epochs, train size or calibration streams')
    if pre<window or post<window or window<2 or not 0<target_false_alarm<1:
        raise ValueError('Invalid stream length, window or alarm target')
    length = pre+post
    metrics={method:{'stable_alarms':0,'prechange_alarms':0,
                     'postchange_detections':0,'delays':[]}
             for method in ('rolling_mean','cusum')}
    for trial in range(trials):
        s=seed+101*trial
        x,y=make_world(train_size,s)
        model=NeuralModel(s).fit(x,y,epochs=epochs,seed=s+5)
        rng_cal=np.random.default_rng(s+10000)
        cal=np.array([errors(model,rng_cal,length,'reference')
                      for _ in range(calibration_streams)])
        center=float(np.mean(cal))
        allowance=max(float(np.std(cal))*.25,1e-8)
        def score_pair(e):
            return rolling_mean(e,window),cusum_path(e,center,allowance)
        maxima=np.array([[float(np.max(a)),float(np.max(b))]
                         for a,b in (score_pair(e) for e in cal)])
        rank=min(calibration_streams-1,
                 int(np.ceil((calibration_streams+1)*(1-target_false_alarm)))-1)
        cut=np.sort(maxima,axis=0)[rank]
        for i,world in enumerate(WORLDS):
            rng=np.random.default_rng(s+20000+i)
            stable=errors(model,rng,length,'reference')
            before=errors(model,rng,pre,'reference')
            after=errors(model,rng,post,world)
            stream=np.concatenate((before,after))
            stable_scores=score_pair(stable)
            stream_scores=score_pair(stream)
            for j,method in enumerate(metrics):
                offset=window-1 if method=='rolling_mean' else 0
                stable_alarm=first_alarm(stable_scores[j],cut[j],offset)
                alarm=first_alarm(stream_scores[j],cut[j],offset)
                item=metrics[method]
                item['stable_alarms']+=int(stable_alarm is not None)
                item['prechange_alarms']+=int(alarm is not None and alarm<pre)
                if alarm is not None and alarm>=pre:
                    item['postchange_detections']+=1
                    item['delays'].append(alarm-pre+1)
    total=trials*len(WORLDS)
    return {
        'warning':'Synthetic only. No certified false-positive guarantee or real-world decision accuracy.',
        'trials':trials,'streams_per_condition':total,
        'calibration_streams_per_trial':calibration_streams,
        'target_stream_false_alarm':target_false_alarm,
        'methods':{name:{
            'stable_false_alarm_rate':round(d['stable_alarms']/total,4),
            'prechange_false_alarm_rate':round(d['prechange_alarms']/total,4),
            'postchange_detection_rate_no_earlier_alarm':round(d['postchange_detections']/total,4),
            'median_delay_when_detected':float(np.median(d['delays'])) if d['delays'] else None
        } for name,d in metrics.items()}
    }

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--epochs',type=int,default=100)
    a=p.parse_args()
    print(json.dumps(evaluate(trials=a.trials,epochs=a.epochs),indent=2))
