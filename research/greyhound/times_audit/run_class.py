import sys, json; from research.greyhound.times_audit.figs_study import *
g=base()
best={'link':True,'variant':'eb'}
F=figure(g,best)
out={}
for a in (1.0,0.75,0.5,0.25,0.0):
    X=predictor(g,class_figure(g,F,a)) if a<1 else predictor(g,F)
    r=score(g,X); out[f'class blend alpha {a}']=r; print('class blend alpha',a,r,flush=True)
# over/under grade par as the predictor
X=predictor(g,grade_par(g,F)); r=score(g,X); out['figure over grade par']=r; print('over grade par',r,flush=True)
# windows: last 1, 3, 6 and recency
for last in (1,6):
    r=score(g,predictor(g,F,last=last)); out[f'last {last}']=r; print('last',last,r,flush=True)
json.dump(out,open('class.json','w'),indent=1)
