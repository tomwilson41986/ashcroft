import sys, json, time; from research.greyhound.times_audit.pace_study import *
t=time.time(); g=base(); print('rows',len(g),'races with full sectionals',g[g.sec_ok].race_id.nunique(), round(time.time()-t), flush=True)
H=['S','R','T','epct','lead','c_brk','c_epace','c_led','c_path','gain']
h3=history(g,H,3).add_prefix('m3_'); h1=history(g,['S','epct'],1,kind='last').add_prefix('l1_')
he=history(g,['S','epct','lead'],10,kind='ewm',half=2.0).add_prefix('ew_')
h10=history(g,['lead','c_brk','c_epace'],10).add_prefix('m10_')
g=pd.concat([g,h3,h1,he,h10],axis=1); print('history done',round(time.time()-t),flush=True)
g.to_pickle('pace_base.pkl')
x=g[g.sec_ok].copy()
for tr in range(1,7): x[f'trap{tr}']=(x.trap==tr).astype(float)
# neighbours' early pace (ewm sectional figure of the dogs drawn inside / outside)
x=x.sort_values(['race_id','trap'])
x['in_S']=x.groupby('race_id').ew_S.shift(1); x['out_S']=x.groupby('race_id').ew_S.shift(-1)
x['in_gap']=x.ew_S-x.in_S; x['out_gap']=x.ew_S-x.out_S
fit=(x.day>=FIT_FROM)&(x.day<=TRAIN_END); ev=x.day>=EVAL_FROM
traps=[f'trap{t}' for t in range(2,7)]
M={'last sectional figure':['l1_S'],
   'mean of last 3 sectional figures':['m3_S'],
   'recency-weighted sectional (ewm, half-life 2 runs)':['ew_S'],
   'ewm sectional + trap':['ew_S']+traps,
   '+ early position history (ewm rank, lead rate)':['ew_S','ew_epct','ew_lead']+traps,
   '+ break and early-pace comments':['ew_S','ew_epct','ew_lead','m10_c_brk','m10_c_epace','m3_c_path']+traps,
   '+ neighbours drawn either side':['ew_S','ew_epct','ew_lead','m10_c_brk','m10_c_epace','m3_c_path','in_gap','out_gap']+traps}
out={}
for k,f in M.items():
    p,b=clogit(x,f,'lead',fit); r=evaluate(x,p,'lead',ev); out[k]=r; print(k,r,flush=True)
x['p_lead'],coef=clogit(x,M['+ neighbours drawn either side'],'lead',fit); print(coef)
x.to_pickle('pace_x.pkl'); json.dump(out,open('leader.json','w'),indent=1)
