"""Cross-world synthetic transfer test. NOT real decision accuracy."""
from __future__ import annotations
import argparse, json
import numpy as np
from neural_decision import NeuralModel, make_world
from safety_gate import SupportGate

def consequences(c, world):
    demand, competition, capacity = c.T
    baseline = .35*demand-.25*competition+.2*capacity
    if world == 'reference':
        uplift = .65*np.sin(2.4*demand)-.55*np.maximum(competition,0)+.6*demand*capacity-.15
    elif world == 'regulation_change':
        uplift = .3*demand-.75*np.maximum(competition,0)-.4*capacity-.3
    elif world == 'supply_disruption':
        uplift = .5*demand-.2*competition-.9*np.maximum(-capacity,0)-.25
    else:
        raise ValueError('unknown world: '+world)
    return baseline,baseline+uplift

def evaluate(seed=42, training=3500, testing=800, epochs=180):
    if training<100 or testing<100 or epochs<1:
        raise ValueError('training/testing >=100, epochs >=1')
    x,y=make_world(training,seed)
    model=NeuralModel(seed).fit(x,y,epochs=epochs,seed=seed+50)
    design=np.column_stack([np.ones(training),x])
    weights=np.linalg.lstsq(design,y,rcond=None)[0]
    gate=SupportGate(x[:,:3])
    worlds={}
    for i,world in enumerate(('reference','regulation_change','supply_disruption')):
        conditions=np.random.default_rng(seed+100+i).uniform(-1,1,(testing,3))
        zero,one=consequences(conditions,world)
        best=np.maximum(zero,one)
        x0=np.column_stack([conditions,np.zeros(testing)])
        x1=np.column_stack([conditions,np.ones(testing)])
        policies={
            'always_maintain':np.zeros(testing,dtype=bool),
            'linear':np.column_stack([np.ones(testing),x1])@weights > np.column_stack([np.ones(testing),x0])@weights,
            'neural':model.predict(x1)>model.predict(x0)
        }
        accepted=gate.accept(conditions)
        scores={}
        for label,choice in policies.items():
            outcome=np.where(choice,one,zero)
            regret=best-outcome
            scores[label]={
                'optimal_choice_rate':round(float(np.isclose(best,outcome).mean()),4),
                'mean_regret':round(float(regret.mean()),6)}
            if label=='neural':
                scores[label]['gate_coverage']=round(float(accepted.mean()),4)
                scores[label]['accepted_mean_regret']=(round(float(regret[accepted].mean()),6) if accepted.any() else None)
        worlds[world]=scores
    return {'warning':'Synthetic worlds only. No demonstrated accuracy in real decisions.',
            'trained_on':'reference','seed':seed,'test_cases_per_world':testing,'worlds':worlds}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=180)
    args=p.parse_args()
    print(json.dumps(evaluate(seed=args.seed,epochs=args.epochs),indent=2))
