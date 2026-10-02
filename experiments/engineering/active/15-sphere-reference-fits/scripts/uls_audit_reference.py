#!/usr/bin/env python3
"""90-digit local ULS metric distances at saved ordinary parameter vectors.

Uses analytic implied-covariance derivatives, never the C++ factor, its QR,
its gradient or an optimizer. Keep every nonzero residual, including diagonal
perturbations and floating-point transport of the reference parameter vector.
"""
import argparse
import hashlib
from pathlib import Path
import time
import mpmath as mp
from uls_unit_reference import components, equilibrated, number, read, write


def evaluate(sample, rows, retain=False):
    values = {(r['lhs'],r['op'],r['rhs']):mp.mpf(r['est']) for r in rows}
    get = lambda a,op,b:values[a,op,b]
    vx = get('X','~~','X'); beta = get('Y','~','X'); psi = get('Y','~~','Y')
    z = mp.matrix([get('X','=~','x2'),get('X','=~','x3'),get('Y','=~','y2'),get('Y','=~','y3'),
                   vx,beta*vx,psi+beta**2*vx])
    loading,phi,derivatives,second = components(z,markers=(0,3))
    sigma = loading*phi*loading.T
    for i,name in enumerate(('x1','x2','x3','y1','y2','y3')):
        sigma[i,i] += get(name,'~~',name)
        e = mp.matrix(6); e[i,i] = 1; derivatives.append(e)
    pairs = [(i,j) for j in range(6) for i in range(j,6)]
    residual = mp.matrix([sigma[i,j]-sample[i,j] for i,j in pairs])
    J = mp.matrix([[d[i,j] for d in derivatives] for i,j in pairs])
    g = J.T*residual; correction = mp.matrix(13)
    for j in range(7):
        for i in range(j+1):
            v = mp.fsum(residual[k]*second[i,j][p] for k,p in enumerate(pairs))
            correction[i,j] += v
            if i!=j: correction[j,i] += v
    T = mp.eye(13)
    T[5,4]=beta; T[5,5]=vx; T[5,6]=0
    T[6,4]=beta**2; T[6,5]=2*beta*vx; T[6,6]=1
    correction = 100*T.T*correction*T
    correction[4,5] += 100*(g[5]+2*beta*g[6]); correction[5,4]=correction[4,5]
    correction[5,5] += 200*vx*g[6]
    objective_jacobian = 10*J*T
    h = objective_jacobian.T*objective_jacobian + correction
    G = 100*T.T*g
    Gamma = mp.matrix([[sample[i,k]*sample[j,l]+sample[i,l]*sample[j,k] for k,l in pairs] for i,j in pairs])
    omega = 100*T.T*J.T*Gamma*J*T
    distance = mp.sqrt((G.T*mp.lu_solve(omega,G))[0])
    h_min,h_condition = equilibrated(h)
    _,metric_condition = equilibrated(omega)
    result = dict(reference_distance=number(distance),reference_accurate=bool(distance<=mp.mpf('.01') and h_min>0),
        reference_hessian_positive=bool(h_min>0),reference_hessian_condition=number(h_condition),
        reference_metric_condition=number(metric_condition))
    if retain:
        # Independent sampling factor in the same column-major tensor layout
        # as the retained core artifacts. No core derivatives are inputs.
        L = mp.cholesky(sample); F = mp.matrix(36,21)
        for k,(i,j) in enumerate(pairs):
            for col in range(6):
                for row in range(6):
                    F[row+6*col,k] = (L[i,row]*L[j,col]+L[j,row]*L[i,col])/mp.sqrt(2)
        R = sigma-sample; whitened = L**-1*R*(L.T**-1)
        b = mp.matrix([10*whitened[row,col]/mp.sqrt(2) for col in range(6) for row in range(6)])
        result['artifacts'] = dict(factor=10*F*J*T, score_residual=b, hessian=h,
            objective_jacobian=objective_jacobian, correction=correction,
            gradient=G, metric=omega)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args = parser.parse_args(); mp.mp.dps=90; start=time.monotonic()
    if (args.run_dir/'point_references.csv').exists(): raise ValueError('fresh output required')
    covariance=read(args.run_dir/'covariances.csv'); points=read(args.run_dir/'points.csv')
    samples={}
    for r in covariance:
        cid=int(r['case_id']); samples.setdefault(cid,mp.matrix(6))
        samples[cid][int(r['row'])-1,int(r['col'])-1]=mp.mpf(r['value'])
    rows=[]
    for pid in sorted({int(r['point_id']) for r in points}):
        p=[r for r in points if int(r['point_id'])==pid]; cid=int(p[0]['case_id'])
        result=evaluate((samples[cid]+samples[cid].T)/2,p)
        rows.append(dict(point_id=pid,case_id=cid,**result))
    write(args.run_dir/'point_references.csv',rows)
    write(args.run_dir/'point_reference_metadata.csv',[dict(digits=90,mpmath=mp.__version__,elapsed_s=time.monotonic()-start,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        components_sha256=hashlib.sha256(Path(__file__).with_name('uls_unit_reference.py').read_bytes()).hexdigest(),
        inputs_sha256=';'.join(hashlib.sha256((args.run_dir/n).read_bytes()).hexdigest() for n in ('covariances.csv','points.csv')),
        scope='analytic total ULS gradient, observed Hessian, normal-theory Gamma at saved parameter vectors; no refinement')])


if __name__=='__main__': main()
