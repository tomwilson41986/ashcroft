"""Run style, pace pressure, which setup suits whom, the early-to-finish Markov chain (reads pace_x.pkl from
run_leader.py: the full-sectional races with p_lead, the fitted expected-leader chance, earlier days only)."""
import sys, json; from research.greyhound.times_audit.pace_study import *
x=pd.read_pickle('pace_x.pkl')
fit=(x.day>=FIT_FROM)&(x.day<=TRAIN_END); ev=x.day>=EVAL_FROM
x['won']=(x.position==1).astype(float)
out={}
# --- run styles from history (earlier days): early index and positions gained
x['rstyle']=pd.cut(x.ew_epct,[-0.01,0.2,0.45,0.7,1.01],labels=['leader','prominent','midfield','back'])
st=x[x.rstyle.notna()].groupby('rstyle',observed=True).agg(runs=('won','size'),win=('won','mean'),lead=('lead','mean'),
     erank=('erank','mean'),finish=('position','mean'),gain=('gain','mean'))
print('styles (from earlier days) against what happened today:\n',st.round(3).to_string()); out['styles']=st.round(4).reset_index().to_dict('records')
# stability: style index next window vs this
y=x.sort_values(['dog_id','day']); y['n_ew']=y.groupby('dog_id').ew_epct.shift(-6)
print('early index stability (6 runs apart) corr', round(y[['ew_epct','n_ew']].corr().iloc[0,1],3))
# --- pace pressure: how contested the lead is
grp=x.groupby('race_id').p_lead
x['p_max']=grp.transform('max'); x['p_top2']=grp.transform(lambda s: s.nlargest(2).sum())
x['press']=x.p_top2-x.p_max          # the second-most-likely leader's chance: a duel when high
x['others_early']=grp.transform('sum')-x.p_lead
x['press_c']=x.press-x.press.mean()
# finish model: ability, expected leader, interactions
x['ability']=x.m3_T; x['runin']=x.m3_R; x['closer']=x.m3_gain
x['pl_x_press']=x.p_lead*x.press_c; x['cl_x_press']=x.closer*x.press_c; x['pl_x_trap1']=x.p_lead*(x.trap==1)
x['pl_x_wide']=x.p_lead*(x.trap==6)
M={'ability (whole-race figure, last 3)':['ability'],
   '+ expected leader chance':['ability','p_lead'],
   '+ run-in figure and positions gained':['ability','p_lead','runin','closer'],
   '+ pace pressure interactions':['ability','p_lead','runin','closer','pl_x_press','cl_x_press'],
   '+ leader x end traps':['ability','p_lead','runin','closer','pl_x_press','cl_x_press','pl_x_trap1','pl_x_wide']}
res={}
for k,f in M.items():
    p,b=clogit(x,f,'won',fit); r=evaluate(x,p,'won',ev); r['coef']=b; res[k]=r; print(k,r,flush=True)
out['finish_models']=res
# --- who benefits: the early leader's win rate by how contested the lead is (eval)
e=x[ev].copy(); e['exp_leader']=e.p_lead==e.p_max
q=e[e.exp_leader]; q['press_band']=pd.qcut(q.press,4,labels=['lone','low','mid','duel'])
t=q.groupby('press_band',observed=True).agg(races=('won','size'),won=('won','mean'),led=('lead','mean'),p_lead=('p_lead','mean'))
print("the expected leader by pace pressure (eval):\n",t.round(3).to_string()); out['leader_by_pressure']=t.round(4).reset_index().to_dict('records')
c=e[e.rstyle=='back'].copy(); c['press_band']=pd.qcut(c.press,4,labels=['lone','low','mid','duel'])
t2=c.groupby('press_band',observed=True).agg(runs=('won','size'),won=('won','mean'),gain=('gain','mean'))
print("back-runners by pace pressure (eval):\n",t2.round(3).to_string()); out['closers_by_pressure']=t2.round(4).reset_index().to_dict('records')
# --- Markov chain: early rank -> finish (eval), and by track the leader's win rate
mk=pd.crosstab(e.erank,e.position,normalize='index').round(3); print('early rank -> finish:\n',mk.iloc[:6,:6].to_string())
out['markov']=mk.iloc[:6,:6].to_dict()
tw=e[e.erank==1].groupby('track').won.agg(['mean','size']).sort_values('mean'); print('leader win rate by track:\n',tw.round(3).to_string())
out['leader_win_by_track']=tw.round(4).reset_index().to_dict('records')
x.to_pickle('pace_x2.pkl'); json.dump(out,open('shape.json','w'),indent=1,default=str)
