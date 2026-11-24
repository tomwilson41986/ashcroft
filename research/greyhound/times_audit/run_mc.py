"""A fitted Monte Carlo race: the run to the first timing point from each dog's early figure, a check for a dog
arriving level with, and behind, the dog drawn next to it, then the run home; the settings fitted by the winners'
likelihood on FIT races, scored on EVAL against the conditional logits (run_shape.py)."""
import sys, json, itertools; from research.greyhound.times_audit.pace_study import *
x=pd.read_pickle('pace_x2.pkl')
x=x[x.groupby('race_id').dog_id.transform('size').between(4,6)].copy()
x['eL']=(x.ew_S*x.sec_par*0.01/L); x['rL']=(x.m3_R*(x.adjusted_time-x.sectional).groupby(x.key).transform('median')*0.01/L)
for c in ('eL','rL'):
    x[c]=x[c].fillna(x.groupby('race_id')[c].transform('mean')).fillna(0)
x=x.sort_values(['race_id','trap'])
def pack(df):
    races=df.race_id.unique(); k=df.groupby('race_id').size().reindex(races).to_numpy(); K=6
    E=np.full((len(races),K),np.nan); R=E.copy(); T=E.copy(); W=np.zeros((len(races),K))
    pos=df.groupby('race_id').cumcount().to_numpy(); ri=pd.factorize(df.race_id)[0]
    E[ri,pos]=df.eL; R[ri,pos]=df.rL; T[ri,pos]=df.trap; W[ri,pos]=df.won
    return E,R,T,W
def sim(E,R,T,se,sr,delta,cost,w,n=400,seed=1):
    rng=np.random.default_rng(seed); valid=~np.isnan(E)
    wins=np.zeros_like(E); leads=np.zeros_like(E); checks=np.zeros_like(E)
    for _ in range(n//50):
        e=E[:,:,None]+se*rng.standard_normal(E.shape+(50,))
        e=np.where(valid[:,:,None],e,-1e9)
        # adjacent traps (sorted by trap within race, so neighbours are index +-1 when traps differ by 1)
        lost=np.zeros_like(e)
        for a in range(5):
            adj=(np.abs(T[:,a+1]-T[:,a])==1)[:,None]
            gap=e[:,a+1]-e[:,a]
            close=adj&(np.abs(gap)<delta)
            lost[:,a]+=np.where(close&(gap>0),cost*(1-np.abs(gap)/delta),0)
            lost[:,a+1]+=np.where(close&(gap<=0),cost*(1-np.abs(gap)/delta),0)
        fin=e-lost+w*R[:,:,None]+sr*rng.standard_normal(e.shape)
        fin=np.where(valid[:,:,None],fin,-1e9)
        wins+=(fin==fin.max(1,keepdims=True)).sum(2); leads+=(e==e.max(1,keepdims=True)).sum(2); checks+=lost.sum(2)
    return wins/n, leads/n, checks/n
def ll(P,W): p=(P*W).sum(1); return -np.log(np.clip(p,1/400/2,1)).mean()
fit=x[(x.day>=FIT_FROM)&(x.day<=TRAIN_END)]; ev=x[x.day>=EVAL_FROM]
fr=fit.race_id.drop_duplicates().sample(6000,random_state=0); Ef,Rf,Tf,Wf=pack(fit[fit.race_id.isin(fr)])
best=None
for se,sr,delta,cost,w in itertools.product((0.8,1.5,2.5),(2.0,3.0,4.5),(0.5,1.0),(0.0,1.0,2.5),(0.5,1.0)):
    P,_,_=sim(Ef,Rf,Tf,se,sr,delta,cost,w,n=200); v=ll(P,Wf)
    if best is None or v<best[0]: best=(v,(se,sr,delta,cost,w))
print('fitted (sd early, sd run home, crowd distance, check cost, run-home weight):',best[1],'fit logloss',round(best[0],5),flush=True)
E,R,T,W=pack(ev); P,PL,CK=sim(E,R,T,*best[1],n=1000)
Wl=np.zeros_like(W); er=ev.groupby('race_id').cumcount().to_numpy(); ri=pd.factorize(ev.race_id)[0]; Wl[ri,er]=ev.lead.to_numpy()
ok=Wl.sum(1)==1
print('EVAL races',len(P),'win logloss',round(ll(P,W),5),'| lead logloss',round(ll(PL[ok],Wl[ok]),5),'top1 lead',round(((PL[ok].argmax(1)==Wl[ok].argmax(1))).mean(),4))
print('uniform', round(np.log((~np.isnan(E)).sum(1)).mean(),5))
json.dump({'params':best[1],'fit_ll':best[0],'eval_win_ll':ll(P,W),'eval_lead_ll':ll(PL[ok],Wl[ok])},open('mc.json','w'))
