import sys, json; from research.greyhound.times_audit.figs_study import *
g=base()
V={'live: calc time in lengths':({'unit':'lengths'},1.0),
   'percent':({},1.0),
   'percent + linked + variant':({'link':True,'variant':'eb'},1.0),
   '+ cap 6 L':({'link':True,'variant':'eb','cap':6},1.0),
   '+ cap 4 L':({'link':True,'variant':'eb','cap':4},1.0),
   '+ cap 8 L':({'link':True,'variant':'eb','cap':8},1.0),
   '+ cap 6 L, 0.06 s/L':({'link':True,'variant':'eb','cap':6,'spl':0.06},1.0),
   '+ cap 6 L + field-anchored 0.5':({'link':True,'variant':'eb','cap':6},0.5),
   '+ cap 6 L, 0.06 s/L + field-anchored 0.5':({'link':True,'variant':'eb','cap':6,'spl':0.06},0.5)}
res={}; base_ll=None
for k,(h,a) in V.items():
    F=figure(g,h); F=class_figure(g,F,a) if a<1 else F
    r,b=race_ll(g,predictor(g,F)); res[k]=r
    if base_ll is None: base_ll=r
    print(k,'beta',b,'logloss',round(r.ll.mean(),5),'vs live',paired(base_ll,r),flush=True)
