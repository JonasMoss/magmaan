#!/usr/bin/env python3
"""Independent high-precision SEM and nonlinear-map checks at sampled endpoints."""
import argparse
import hashlib
import time
import sys
from pathlib import Path
import mpmath as mp
from verify_uls_guard import read, write


def binary(x): return mp.mpf(float(x))
def number(x): return mp.nstr(x,90)
def trace(a): return sum(a[i,i] for i in range(a.rows))
def yes(x): return x.lower()=='true'


def moments(rows,point,block):
    """Build LISREL cells from the retained partable, independently of C++ layout."""
    ov=['x1','x2','x3','y1','y2','y3']
    lv=list(dict.fromkeys(r['lhs'] for r in rows if r['op']=='=~'))
    q=len(point); p=len(ov); m=len(lv)
    matrices=[mp.matrix(p,m),mp.matrix(m),mp.matrix(p),mp.matrix(m)]
    derivatives=[[mp.matrix(x.rows,x.cols) for _ in range(q)] for x in matrices]
    for r in rows:
        if int(r['group'])!=block: continue
        a,b,op=r['lhs'],r['rhs'],r['op']; free=int(r['free'])
        if op=='=~': which,i,j=0,ov.index(b),lv.index(a)
        elif op=='~' and a in lv and b in lv: which,i,j=3,lv.index(a),lv.index(b)
        elif op=='~~' and a in lv and b in lv: which,i,j=1,lv.index(a),lv.index(b)
        elif op=='~~' and a in ov and b in ov: which,i,j=2,ov.index(a),ov.index(b)
        elif op in ('==',':='): continue
        else: raise ValueError(f'unsupported independent row {a} {op} {b}')
        value=point[free-1] if free else binary(r['est'])
        matrices[which][i,j]=value
        for k in range(q): derivatives[which][k][i,j]=int(k+1==free)
        if which in (1,2):
            matrices[which][j,i]=value
            for k in range(q): derivatives[which][k][j,i]=int(k+1==free)
    L,P,T,B=matrices; dL,dP,dT,dB=derivatives
    A=(mp.eye(m)-B)**-1; M=L*A; sigma=M*P*M.T+T
    da=[A*x*A for x in dB]; dm=[dL[k]*A+L*da[k] for k in range(q)]
    ds=[dm[k]*P*M.T+M*dP[k]*M.T+M*P*dm[k].T+dT[k] for k in range(q)]
    def second(i,j):
        dd=L*(da[j]*dB[i]*A+A*dB[i]*da[j])+dL[i]*da[j]+dL[j]*da[i]
        return (dd*P*M.T+M*P*dd.T+dm[i]*P*dm[j].T+dm[j]*P*dm[i].T+
                dm[i]*dP[j]*M.T+M*dP[j]*dm[i].T+dm[j]*dP[i]*M.T+M*dP[i]*dm[j].T)
    return sigma,ds,second


def sphere_map(u,artifact):
    point=artifact('map_offset'); K=artifact('map_rest_basis',optional=True)
    q=point.rows; G=mp.matrix(q,len(u)); units=[]; writers={}
    if K is not None:
        point+=K*mp.matrix(list(u)[:K.cols])
        G[:,:K.cols]=K
    k=1
    while artifact(f'sphere_{k}_basis',optional=True) is not None:
        Q=artifact(f'sphere_{k}_basis'); D=mp.diag(list(artifact(f'sphere_{k}_units')))
        ids=artifact(f'sphere_{k}_parameters'); offset=int(artifact(f'sphere_{k}_offset')[0])
        beta=mp.matrix(list(u)[offset:offset+Q.cols]); radius=mp.norm(beta); V=D*Q
        loading=V*beta/radius; jac=V*(mp.eye(Q.cols)/radius-beta*beta.T/radius**3)
        for member in range(ids.cols):
            for j in range(ids.rows):
                row=int(ids[j,member])
                if row<0: continue
                point[row]=loading[j]
                for col in range(G.cols): G[row,col]=0
                for col in range(Q.cols): G[row,offset+col]=jac[j,col]
                writers[row]=(k-1,j)
        units.append((V,beta,radius,offset)); k+=1
    def chain(g):
        h=mp.matrix(len(u))
        for k,(V,b,r,offset) in enumerate(units):
            v=mp.matrix(V.rows,1)
            for row,(owner,component) in writers.items():
                if owner==k: v[component]+=g[row]
            c=V.T*v; radial=(b.T*c)[0]
            correction=-(c*b.T+b*c.T+radial*mp.eye(b.rows))/r**3+3*radial*b*b.T/r**5
            for i in range(b.rows):
                for j in range(b.rows): h[offset+i,offset+j]+=correction[i,j]
        return h
    return point,G,chain


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--diagnostic-out',type=Path,help='fresh output directory; read only points 64/84/92 from run-dir')
    args=parser.parse_args(); inputs=args.run_dir; out=args.diagnostic_out or inputs
    if args.diagnostic_out: out.mkdir(parents=True,exist_ok=False)
    if (out/'comparisons.csv').exists(): raise ValueError('fresh reference output required')
    mp.mp.dps=90; start=time.monotonic()
    all_points=read(inputs/'points.csv'); raw=read(inputs/'artifacts.csv'); intervals=read(inputs/'intervals.csv')
    samples={}; counts={}
    for r in read(inputs/'covariances.csv'):
        key=int(r['case_id']),int(r['block']); counts[key]=int(r['nobs'])
        samples.setdefault(key,mp.matrix(6))[int(r['row'])-1,int(r['col'])-1]=binary(r['value'])
    matrix_rows={}; partables={}
    for r in raw: matrix_rows.setdefault((int(r['point_id']),r['name']),[]).append(r)
    for r in all_points: partables.setdefault(int(r['point_id']),[]).append(r)
    comparisons=[]
    for row in intervals:
        if args.diagnostic_out and int(row['point_id']) not in (64,84,92): continue
        pid=int(row['point_id']); cid=int(row['case_id']); pt=partables[pid]
        def artifact(name,optional=False):
            data=matrix_rows.get((pid,name))
            if data is None:
                if optional: return None
                raise KeyError((pid,name))
            x=mp.matrix(max(int(r['row']) for r in data),max(int(r['col']) for r in data))
            for r in data: x[int(r['row'])-1,int(r['col'])-1]=binary(r['value'])
            return x
        q=max(int(r['free']) for r in pt)
        if row['chart']=='sphere':
            point,map_jac,chain=sphere_map(artifact('point'),artifact)
            basis=artifact('tangent_basis'); stored_gradient=artifact('reduced_gradient')
        else:
            point=mp.matrix(q,1)
            for r in pt:
                if int(r['free']): point[int(r['free'])-1]=binary(r['est'])
            map_jac=mp.eye(q); chain=lambda g:mp.matrix(q)
            basis=artifact('derivative_basis'); stored_gradient=artifact('gradient')
        total_g=mp.matrix(q,1); total_h=mp.matrix(q); factor_blocks=[]; score_blocks=[]; objective=mp.mpf(0); domain=True; total_n=0
        for (case,block),S in samples.items():
            if case!=cid: continue
            n=counts[case,block]; total_n+=n; sigma,ds,second=moments(pt,point,block)
            p=S.rows; positions=[(i,j) for j in range(p) for i in range(j,p)]
            J=mp.matrix([[d[i,j] for d in ds] for i,j in positions])
            residual=mp.matrix([sigma[i,j]-S[i,j] for i,j in positions])
            if row['estimator']=='ULS':
                objective+=n*(residual.T*residual)[0]/2
                total_g+=n*J.T*residual; H=n*J.T*J
                L=mp.cholesky(S); inverse=L**-1; F=mp.matrix(p*p,len(positions))
                for k,(i,j) in enumerate(positions):
                    for col in range(p):
                        for r in range(p): F[r+p*col,k]=(L[i,r]*L[j,col]+L[j,r]*L[i,col])/mp.sqrt(2)
                factor_blocks.append(mp.sqrt(n)*F*J)
                white=mp.sqrt(n/ mp.mpf(2))*inverse*(sigma-S)*inverse.T
                score_blocks.append(mp.matrix([white[i,j] for j in range(p) for i in range(p)]))
            else:
                domain=domain and mp.eigsy(sigma,eigvals_only=True)[0]>0
                W=sigma**-1; K=W*S*W; V=(W-K)/2; H=mp.matrix(q)
                objective+=n*(mp.log(abs(mp.det(sigma)))+trace(W*S)-mp.log(mp.det(S))-p)/2
                for i in range(q): total_g[i]+=n*trace(V*ds[i])
            for j in range(q):
                for i in range(j+1):
                    dd=second(i,j)
                    if row['estimator']=='ULS': H[i,j]+=n*sum(residual[k]*dd[r,c] for k,(r,c) in enumerate(positions))
                    else: H[i,j]=n*trace(((K*ds[j]*W+W*ds[j]*K-W*ds[j]*W)/2)*ds[i]+V*dd)
                    H[j,i]=H[i,j]
            total_h+=H
        reduced=map_jac*basis
        h=reduced.T*total_h*reduced+basis.T*chain(total_g)*basis
        g=reduced.T*total_g
        positive=domain and mp.eigsy(h,eigvals_only=True)[0]>0
        target=mp.nan; matrix_error=vector_error=curvature_error=mp.inf; exact_curvature=None
        if row['estimator']=='ULS':
            A=mp.matrix(sum(x.rows for x in factor_blocks),q); b=mp.matrix(A.rows,1); offset=0
            for x,y in zip(factor_blocks,score_blocks):
                A[offset:offset+x.rows,:]=x; b[offset:offset+x.rows,:]=y; offset+=x.rows
            if positive: target=mp.sqrt((g.T*mp.lu_solve((A*reduced).T*(A*reduced),g))[0])
            D=mp.diag(list(artifact('factor_scale',optional=True) or mp.matrix(0,1)))
            scaled=artifact('equilibrated_factor',optional=True)
            if scaled is not None:
                exact_factor=A*reduced*D
                matrix_error=mp.norm(exact_factor-scaled); vector_error=mp.norm(b-artifact('metric_score_residual'))

            T=artifact('curvature_coordinate_map',optional=True)
            if T is not None: exact_curvature=T.T*h*T
        else:
            if positive: target=mp.sqrt((g.T*mp.lu_solve(h,g))[0])
            scale=artifact('curvature_scale',optional=True)
            if scale is not None:
                D=mp.diag(list(scale)); exact_curvature=D*h*D
                rounded_score=mp.matrix([binary(float(D[i,i])*float(stored_gradient[i])) for i in range(D.rows)])
                vector_error=mp.norm(D*g-rounded_score)
        C=artifact('curvature_equilibrated_hessian',optional=True)
        if C is not None and exact_curvature is not None:
            curvature_error=mp.norm(C-exact_curvature)
            if row['estimator']=='ML': matrix_error=curvature_error
        if args.diagnostic_out:
            eigenvalues, eigenvectors=mp.eigsy(exact_curvature)
            # E = stored minus exact, in the SAME retained scaled coordinates.
            error=C-exact_curvature
            symmetry_error=mp.norm(C-C.T)
            exact_symmetry_error=mp.norm(exact_curvature-exact_curvature.T)
            tolerance=mp.mpf('1e-70')
            if symmetry_error or exact_symmetry_error>tolerance:
                raise ValueError('curvature symmetry check failed')
            rotated=eigenvectors.T*error*eigenvectors
            inverse_root=mp.diag([1/mp.sqrt(x) for x in eigenvalues])
            relative=inverse_root*rotated*inverse_root
            relative_norm=max(abs(x) for x in mp.eigsy(relative,eigvals_only=True))
            coupling=mp.sqrt(sum(rotated[i,0]**2 for i in range(1,C.rows)))
            orthogonality=mp.norm(eigenvectors.T*eigenvectors-mp.eye(C.rows))
            reconstruction=mp.norm(eigenvectors*mp.diag(list(eigenvalues))*eigenvectors.T-exact_curvature)
            norm_invariance=abs(mp.norm(rotated)-mp.norm(error))
            # Independent generalized-Rayleigh whitening with H = L L^T.
            linv=mp.cholesky(exact_curvature)**-1
            alternative=linv*error*linv.T
            alternative=(alternative+alternative.T)/2
            whitening_gap=abs(relative_norm-max(abs(x) for x in mp.eigsy(alternative,eigvals_only=True)))
            if max(orthogonality,reconstruction,norm_invariance,whitening_gap)>tolerance:
                raise ValueError('basis/relative-norm sanity check failed')
            # Ideal inverse-Frobenius certificate used by singular_lower(L)^2:
            # ||L^-1||_F^2 = trace(C^-1). Excludes binary64 factor/solve defects.
            ideal_lower=1/trace(C**-1)
        available=row['source_status']=='available'
        bounds_covers=not available or (matrix_error<=binary(row['matrix_bound']) and
            vector_error<=binary(row['vector_bound']) and curvature_error<=binary(row['curvature_bound']))
        lower_covers=not available or binary(row['curvature_lower'])<=0 or binary(row['curvature_lower'])<=mp.eigsy(exact_curvature,eigvals_only=True)[0]
        decision=row['decision']; budget=binary(.01)
        wrong=decision=='within_budget' and (not positive or not mp.isfinite(target) or target>budget) or decision=='above_budget' and positive and mp.isfinite(target) and target<=budget
        comparisons.append(dict(point_id=pid,case_id=cid,estimator=row['estimator'],chart=row['chart'],target=row['target'],
            reference_distance=number(target),reference_positive=positive,reference_domain=domain,reference_total_objective=number(objective) if domain else 'nan',
            objective_agrees=not domain or abs(objective/total_n-binary(row['recomputed_objective']))<=mp.mpf('1e-8')*(1+abs(objective/total_n)),
            matrix_error=number(matrix_error),vector_error=number(vector_error),curvature_error=number(curvature_error),
            source_status=row['source_status'],bounds_covers=bounds_covers,curvature_lower_covers=lower_covers,
            interval_covers=not mp.isfinite(target) or binary(row['lower'])<=target<=binary(row['upper']),
            wrong_decision=bool(wrong),decision=decision,selected_status=row['selected_status'],
            legacy_passed=row['legacy_passed'],selected_passed=row['selected_passed']))
        if args.diagnostic_out:
            comparisons[-1].update(exact_scaled_min_eigenvalue=number(eigenvalues[0]),
                stored_scaled_min_eigenvalue=number(mp.eigsy(C,eigvals_only=True)[0]),
                ideal_inverse_frobenius_lower=number(ideal_lower),
                ideal_margin_after_construction=number(ideal_lower-binary(row['curvature_bound'])),
                construction_bound=row['curvature_bound'],score_bound=row['vector_bound'],
                retained_lower=row['lower'],retained_upper=row['upper'],retained_curvature_lower=row['curvature_lower'],
                dimension=C.rows,
                weak_direction_error=number(rotated[0,0]),
                weak_direction_relative_error=number(rotated[0,0]/eigenvalues[0]),
                relative_curvature_operator_norm=number(relative_norm),
                weak_offdiagonal_coupling_norm=number(coupling),
                weak_relative_coupling_norm=number(mp.sqrt(sum(relative[i,0]**2 for i in range(1,C.rows)))),
                stored_symmetry_error=number(symmetry_error),exact_symmetry_error=number(exact_symmetry_error),
                basis_orthogonality_error=number(orthogonality),basis_reconstruction_error=number(reconstruction),
                frobenius_invariance_error=number(norm_invariance),cholesky_whitening_norm_gap=number(whitening_gap))
        if pid%8==0: print(f'90-digit terminal case {cid}; {time.monotonic()-start:.1f}s',flush=True)
    write(out/'comparisons.csv',comparisons)
    write(out/'reference_metadata.csv',[dict(digits=90,mpmath=mp.__version__,elapsed_s=time.monotonic()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256=';'.join(hashlib.sha256((inputs/f).read_bytes()).hexdigest() for f in ('points.csv','artifacts.csv','covariances.csv','intervals.csv')),
        command=' '.join(sys.argv),input_dir=str(inputs.resolve()),diagnostic_points='64;84;92' if args.diagnostic_out else 'all',
        convention='exact binary64 partable/point/sample/map/basis; independent LISREL moments and full normalization chain; total objective')])


if __name__=='__main__': main()
