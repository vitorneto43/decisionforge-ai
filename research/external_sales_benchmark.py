"""Independent weekly sales forecasting experiment: NumPy network trained from scratch.
Method replication on a different source; NOT weight transfer or verified 97% reliability.
"""
from __future__ import annotations
import argparse,csv,io,json,zipfile
from pathlib import Path
import numpy as np

def read_weekly(path):
    p=Path(path)
    if p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p) as archive:
            names=[n for n in archive.namelist() if n.endswith('Sales_Transactions_Dataset_Weekly.csv')]
            if len(names)!=1: raise ValueError('Expected exactly one official weekly CSV')
            content=archive.read(names[0]).decode('utf-8-sig')
    else: content=p.read_text(encoding='utf-8-sig')
    rows=list(csv.DictReader(io.StringIO(content)))
    cols=[f'W{i}' for i in range(52)]
    if not rows or not set(cols).issubset(rows[0]):raise ValueError('Missing weekly W0..W51 columns')
    data=np.array([[float(r[c]) for c in cols] for r in rows])
    if not np.isfinite(data).all() or (data<0).any():raise ValueError('Bad demand values')
    return data

def make_examples(sales,start,end,lookback=8):
    if start<lookback or end>sales.shape[1] or end<=start:raise ValueError('Invalid time range')
    xx,yy=[],[]
    for product in sales:
        for week in range(start,end):
            xx.append(np.log1p(product[week-lookback:week]))
            yy.append(np.log1p(product[week]))
    return np.asarray(xx),np.asarray(yy)

class TinyMLP:
    def __init__(self,hidden=24,seed=42):
        rng=np.random.default_rng(seed)
        self.w1=rng.normal(0,np.sqrt(2/8),(8,hidden)); self.b1=np.zeros(hidden)
        self.w2=rng.normal(0,np.sqrt(1/hidden),(hidden,1));self.b2=np.zeros(1)
    def predict(self,x):return (np.tanh(x@self.w1+self.b1)@self.w2+self.b2).ravel()
    def fit(self,x,y,epochs=100,batch=256,lr=.01,seed=42):
        if len(x)!=len(y) or x.shape[1]!=8 or epochs<1:raise ValueError('Bad training inputs')
        rng=np.random.default_rng(seed)
        for _ in range(epochs):
            for ids in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/batch)))):
                xb,yb=x[ids],y[ids]; h=np.tanh(xb@self.w1+self.b1)
                delta=2*((h@self.w2+self.b2).ravel()-yb)[:,None]/len(ids)
                dw2,db2=h.T@delta,delta.sum(axis=0)
                back=(delta@self.w2.T)*(1-h*h)
                dw1,db1=xb.T@back,back.sum(axis=0)
                for key,grad in [('w1',dw1),('b1',db1),('w2',dw2),('b2',db2)]:
                    setattr(self,key,getattr(self,key)-lr*np.clip(grad,-5,5))
        return self

def score(sales,model,start,end):
    x,y=make_examples(sales,start,end);actual=np.expm1(y)
    neural=np.maximum(0,np.expm1(np.clip(model.predict(x),0,15)))
    baseline=np.expm1(x[:,-4:]).mean(axis=1)
    def metric(p):
        diff=p-actual
        return {'mae':round(float(np.mean(np.abs(diff))),5),
                'rmse':round(float(np.sqrt(np.mean(diff**2))),5)}
    return {'n':len(actual),'neural':metric(neural),'mean_4_weeks':metric(baseline)}

def evaluate(sales,epochs=100,seed=42):
    if sales.ndim!=2 or sales.shape[1]!=52:raise ValueError('Expected 52 weeks')
    x,y=make_examples(sales,8,32)
    model=TinyMLP(seed=seed).fit(x,y,epochs=epochs,seed=seed+1)
    return {'status':'external_dataset_method_replication_only',
            'warning':'Fresh weights trained on external source; NOT weight transfer or decision reliability.',
            'products':int(len(sales)),'weeks':52,'train_weeks':[8,32],
            'validation_weeks':[32,42],'test_weeks':[42,52],
            'validation':score(sales,model,32,42),'test':score(sales,model,42,52)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv-or-zip',required=True)
    p.add_argument('--epochs',type=int,default=100);a=p.parse_args()
    print(json.dumps(evaluate(read_weekly(a.csv_or_zip),a.epochs),indent=2))
