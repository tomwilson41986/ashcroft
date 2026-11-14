import sys, json; from research.greyhound.times_audit.figs_study import *
g=base()
V={
 'raw time, no going (lengths)': {'raw':True,'unit':'lengths'},
 'calc time (GBGB going), lengths [live tn_/fg_ convention]': {'unit':'lengths'},
 'calc time, percent': {},
 'going x0.5': {'going_alpha':0.5}, 'going x1.25': {'going_alpha':1.25},
 'percent + linked standards': {'link':True},
 'percent + meeting variant': {'variant':'eb'},
 'percent + linked + variant': {'link':True,'variant':'eb'},
 'linked + variant + cap 6 L': {'link':True,'variant':'eb','cap':6},
 'linked + variant + cap 10 L': {'link':True,'variant':'eb','cap':10},
 'linked + variant + 0.06 s/L': {'link':True,'variant':'eb','spl':0.06},
 'linked + variant + 0.10 s/L': {'link':True,'variant':'eb','spl':0.10},
}
out={}
for name,h in V.items():
    F=figure(g,h); X=predictor(g,F); r=score(g,X); out[name]=r; print(name, r, flush=True)
json.dump(out,open('variants.json','w'),indent=1)
