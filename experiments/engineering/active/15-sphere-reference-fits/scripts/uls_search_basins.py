#!/usr/bin/env python3
"""Refine independent local ULS references after all cold search fits finish."""
import argparse
import hashlib
import time
from pathlib import Path
import mpmath as mp
from verify_uls_guard import read, write, yes
from uls_unit_reference import Profile, ReferenceFailure, equilibrated, number
from audit_terminal_reference import binary


class SearchProfile(Profile):
    def __init__(self,sample,units):
        self.scales=[binary(x) for x in units]
        self.mixed=sample
        self.sample=mp.matrix([[sample[i,j]/self.scales[i]/self.scales[j] for j in range(6)] for i in range(6)])
        self.pairs=[(i,j) for i in range(6) for j in range(i)]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--score-digits',type=int,choices=(40,75),default=75)
    parser.add_argument('--output-subdir')
    args=parser.parse_args();out=args.run_dir
    dest=out if args.output_subdir is None else out/args.output_subdir
    if args.output_subdir is not None:
        if not args.output_subdir.replace('-','').isalnum():raise ValueError('invalid output subdir')
        if dest.exists():raise ValueError('fresh output required')
        dest.mkdir()
    if (dest/'basin_references.csv').exists():raise ValueError('fresh basin output required')
    mp.mp.dps=90;clock=time.monotonic()
    cases=read(out/'cases.csv');fits=read(out/'fits.csv');portfolios=read(out/'portfolios.csv');pt=read(out/'points.csv')
    samples={};partables={}
    for r in read(out/'covariances.csv'):samples.setdefault(r['case_id'],mp.matrix(6))[int(r['row'])-1,int(r['col'])-1]=binary(r['value'])
    for r in pt:partables.setdefault(r['point_id'],[]).append(r)
    references=[];attempts=[];matches=[]
    for case in cases:
        cid=case['case_id'];units=(.01,100,2,.3,10,.1) if case['family']=='mixed' else (1,)*6
        model=SearchProfile(samples[cid],units)
        candidates={p['selected_id'] for p in portfolios if p['case_id']==cid and yes(p['qualified'])}
        minima=[]
        for pid in sorted(candidates,key=int):
            rows=partables[pid]
            def get(a,op,b):
                return next(binary(r['est']) for r in rows if r['lhs']==a and r['op']==op and r['rhs']==b)
            loadings=[get('X','=~',f'x{i+1}')/model.scales[i] for i in range(3)]
            loadings += [get('Y','=~',f'y{i-2}')/model.scales[i] for i in range(3,6)]
            vx=get('X','~~','X');beta=get('Y','~','X');vy=get('Y','~~','Y')+beta*beta*vx
            anchors=loadings[1],loadings[4]
            seed=[loadings[0]/anchors[0],loadings[2]/anchors[0],loadings[3]/anchors[1],loadings[5]/anchors[1],
                  vx*anchors[0]**2,beta*vx*anchors[0]*anchors[1],vy*anchors[1]**2]
            try:
                point,history=model.refine(seed,tolerance=mp.power(10,-args.score_digits));objective,g,h=model.evaluate(point);minimum,condition=equilibrated(h)
                status='finite_local_minimum' if minimum>0 else 'stationary_nonminimum'
                if minimum>0:minima.append((objective,point,pid,minimum,condition))
                attempts.append(dict(case_id=cid,point_id=pid,status=status,iterations=len(history)-1,objective=number(objective),
                                     profile_min=number(minimum),scaled_score=history[-1]['scaled_score']))
            except (ReferenceFailure,ZeroDivisionError,ValueError) as error:
                history=getattr(error,'history',[])
                attempts.append(dict(case_id=cid,point_id=pid,status=str(error),iterations=len(history),objective=history[-1]['objective'] if history else 'nan',profile_min='nan',scaled_score=history[-1]['scaled_score'] if history else 'nan'))
        if minima:
            objective,point,pid,minimum,condition=min(minima,key=lambda x:x[0])
            references.append(dict(case_id=cid,family=case['family'],available=True,objective=number(objective),seed_point_id=pid,
                                   profile_min=number(minimum),profile_condition=number(condition),qualified_refinements=len(minima)))
            for fit in fits:
                if fit['case_id']!=cid:continue
                gap=binary(fit['objective'])-objective if fit['objective'] not in ('NA','') else mp.nan
                matches.append(dict(point_id=fit['point_id'],case_id=cid,chart=fit['chart'],start_id=fit['start_id'],qualified=fit['qualified'],
                                    objective_gap=number(gap),reference_match=bool(mp.isfinite(gap) and abs(gap)<=mp.mpf('1e-6')*(1+abs(objective)))))
        else:
            references.append(dict(case_id=cid,family=case['family'],available=False,objective='nan',seed_point_id='',profile_min='nan',profile_condition='nan',qualified_refinements=0))
            for fit in fits:
                if fit['case_id']==cid:matches.append(dict(point_id=fit['point_id'],case_id=cid,chart=fit['chart'],start_id=fit['start_id'],qualified=fit['qualified'],objective_gap='nan',reference_match=False))
        print(f'Local basin reference {cid}/{len(cases)}: {len(minima)} finite refinements; {time.monotonic()-clock:.1f}s',flush=True)
    write(dest/'basin_references.csv',references);write(dest/'basin_matches.csv',matches)
    if attempts:write(dest/'basin_attempts.csv',attempts)
    else:(dest/'basin_attempts.csv').write_text('case_id,point_id,status,iterations,objective,profile_min,scaled_score\n')
    write(dest/'basin_metadata.csv',[dict(digits=90,score_digits=args.score_digits,elapsed_s=time.monotonic()-clock,mpmath=mp.__version__,
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),profile_sha256=hashlib.sha256(Path(__file__).with_name('uls_unit_reference.py').read_bytes()).hexdigest(),
         input_sha256=';'.join(hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ('covariances.csv','points.csv','fits.csv','portfolios.csv')),
         scope='exact binary64 sample, unrestricted diagonal-profile ULS, refine only post-fit qualified portfolio points; positive profile curvature plus independent full endpoint checks; best known local basin, no global reference')])


if __name__=='__main__':main()
