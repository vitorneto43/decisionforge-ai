"""Real-data predictive benchmark: UCI Bank Marketing, chronological split.

No pretrained AI, external AI APIs, scikit-learn, or test-set threshold tuning.
This predicts observed subscription among contacted clients; NOT treatment effects.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from neural_decision import NeuralModel

DATA_URL = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
CATEGORICAL = ('job','marital','education','default','housing','loan','contact',
               'month','day_of_week','poutcome')
NUMERIC = ('age','campaign','pdays','previous','emp.var.rate','cons.price.idx',
           'cons.conf.idx','euribor3m','nr.employed')
# 'duration' is intentionally EXCLUDED: the length of a call cannot be
# known before the call is made, and causes outcome leakage.

def find_csv(payload):
    """Extract the original dated-in-order dataset, including nested zip files."""
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        for name in outer.namelist():
            if name.endswith('bank-additional-full.csv'):
                return outer.read(name)
        for name in outer.namelist():
            if name.lower().endswith('.zip'):
                try:
                    return find_csv(outer.read(name))
                except (ValueError, zipfile.BadZipFile):
                    continue
    raise ValueError('bank-additional-full.csv not found in UCI archive')

def get_rows(archive=None, allow_download=False):
    if archive:
        data=Path(archive).read_bytes()
    elif allow_download:
        with urllib.request.urlopen(DATA_URL,timeout=45) as response:
            data=response.read()
    else:
        raise ValueError('Provide --archive FILE or --download')
    raw=find_csv(data)
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')),delimiter=';'))
    if not rows or 'y' not in rows[0]:
        raise ValueError('Invalid dataset format')
    return rows

def split_chronological(rows):
    n=len(rows)
    a=int(n*.70); b=int(n*.85)
    if a<1 or b<=a or b>=n: raise ValueError('Need more observations')
    return rows[:a],rows[a:b],rows[b:]

class Encoder:
    def fit(self,rows):
        self.vocab={col:sorted({str(r[col]) for r in rows}) for col in CATEGORICAL}
        self.means=np.array([np.mean([float(r[c]) for r in rows]) for c in NUMERIC])
        self.stds=np.array([np.std([float(r[c]) for r in rows]) for c in NUMERIC])
        self.stds=np.maximum(self.stds,1e-8)
        self.width=len(NUMERIC)+sum(len(v)+1 for v in self.vocab.values())
        return self
    def transform(self,rows):
        x=np.zeros((len(rows),self.width),dtype=np.float64)
        for i,c in enumerate(NUMERIC):
            x[:,i]=np.clip((np.array([float(r[c]) for r in rows])-self.means[i])/self.stds[i],-8,8)
        offset=len(NUMERIC)
        for c in CATEGORICAL:
            vocabulary={v:j for j,v in enumerate(self.vocab[c])}
            for i,r in enumerate(rows):
                x[i,offset+vocabulary.get(str(r[c]),len(vocabulary))]=1
            offset+=len(vocabulary)+1
        return x

class ScratchMLP(NeuralModel):
    def __init__(self,n_features,seed=42,hidden=32):
        super().__init__(seed=seed,hidden=hidden)
        self.w1=np.random.default_rng(seed).normal(0,np.sqrt(2/n_features),(n_features,hidden))
    def probability(self,x):
        z=np.clip(self.predict(x),-30,30)
        return 1/(1+np.exp(-z))

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))

def logistic_train(x,y,epochs=160,lr=.08):
    x=np.column_stack((np.ones(len(x)),x))
    w=np.zeros(x.shape[1])
    for _ in range(epochs):
        p=sigmoid(x@w)
        w-=lr*(x.T@(p-y)/len(y))
    return w

def confusion(y,p,threshold):
    pred=p>=threshold
    return {'tp':int(np.sum((y==1)&pred)), 'fp':int(np.sum((y==0)&pred)),
            'tn':int(np.sum((y==0)&~pred)), 'fn':int(np.sum((y==1)&~pred))}

def metrics(y,p,threshold):
    c=confusion(y,p,threshold)
    tp,fp,tn,fn=(c[k] for k in ('tp','fp','tn','fn'))
    div=lambda a,b:float(a/b) if b else 0.
    precision=div(tp,tp+fp); recall=div(tp,tp+fn)
    return {**c,'accuracy':round(div(tp+tn,len(y)),4),
            'precision':round(precision,4),'recall':round(recall,4),
            'specificity':round(div(tn,tn+fp),4),
            'balanced_accuracy':round((recall+div(tn,tn+fp))/2,4),
            'f1':round(div(2*precision*recall,precision+recall),4),
            'brier':round(float(np.mean((p-y)**2)),6)}

def choose_threshold(y,p):
    # Validation only: tune threshold for F1; not a reliability certificate.
    choices=np.linspace(.05,.95,91)
    return float(max(choices,key=lambda t:metrics(y,p,t)['f1']))

def evaluate(rows,epochs=100,seed=42):
    if epochs<1:raise ValueError('epochs must be positive')
    train,val,test=split_chronological(rows)
    encode=Encoder().fit(train)
    xtr,xv,xt=(encode.transform(s) for s in (train,val,test))
    ytr,yv,yt=(np.array([int(r['y']=='yes') for r in s],dtype=float)
                for s in (train,val,test))
    linear=logistic_train(xtr,ytr)
    def linear_p(x):return sigmoid(np.column_stack((np.ones(len(x)),x))@linear)
    neural=ScratchMLP(xtr.shape[1],seed=seed).fit(xtr,ytr,epochs=epochs,lr=.005,seed=seed+1)
    # MSE-trained neural scores are mapped with sigmoid; calibration must be audited.
    def neural_p(x):return neural.probability(x)
    result={}
    for name,pv,pt in (('logistic_baseline',linear_p(xv),linear_p(xt)),
                       ('scratch_neural',neural_p(xv),neural_p(xt))):
        threshold=choose_threshold(yv,pv)
        result[name]={'validation_threshold':round(threshold,3),
                      'test':metrics(yt,pt,threshold)}
    return {'notice':'Actual recorded outcomes; not causal effects, not decision reliability.',
            'source':DATA_URL,'split':'70% train / 15% validation / 15% final test, original chronological order',
            'target':'observed term-deposit subscription among contacted clients',
            'excluded_leakage_field':'duration','count':len(rows),
            'positive_rate_test':round(float(yt.mean()),4),
            'always_no_test':metrics(yt,np.zeros(len(yt)),.5),
            'models':result}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive')
    group.add_argument('--download',action='store_true')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    print(json.dumps(evaluate(get_rows(args.archive,args.download),args.epochs,args.seed),indent=2))
