#!/usr/bin/env python3
"""Independent 90-digit fixed-weight SEM, producer and construction-bound checks."""
import argparse
import hashlib
import time
from pathlib import Path
import mpmath as mp
from verify_uls_guard import read, write
from audit_terminal_reference import sphere_map


def binary(x): return mp.mpf(float(x))
def number(x): return mp.nstr(x, 90)
def yes(x): return x.lower() == 'true'


def matrices(rows):
    grouped = {}
    for r in rows: grouped.setdefault((int(r['id']), r['name']), []).append(r)
    def get(key, name, optional=False):
        data = grouped.get((key, name))
        if data is None:
            if optional: return None
            raise KeyError((key, name))
        x = mp.matrix(max(int(r['row']) for r in data), max(int(r['col']) for r in data))
        for r in data: x[int(r['row'])-1, int(r['col'])-1] = binary(r['value'])
        return x
    return get


def moments(rows, point, block, means):
    ov = list(dict.fromkeys(r['rhs'] for r in rows if r['op'] == '=~'))
    lv = list(dict.fromkeys(r['lhs'] for r in rows if r['op'] == '=~'))
    p, m, q = len(ov), len(lv), len(point)
    cells = [mp.matrix(p,m),mp.matrix(m),mp.matrix(p),mp.matrix(m),mp.matrix(p,1),mp.matrix(m,1)]
    d = [[mp.matrix(x.rows,x.cols) for _ in range(q)] for x in cells]
    for r in rows:
        if int(r['group']) != block: continue
        a, b, op, free = r['lhs'], r['rhs'], r['op'], int(r['free'])
        if op == '=~': which,i,j = 0,ov.index(b),lv.index(a)
        elif op == '~' and a in lv and b in lv: which,i,j = 3,lv.index(a),lv.index(b)
        elif op == '~~' and a in lv and b in lv: which,i,j = 1,lv.index(a),lv.index(b)
        elif op == '~~' and a in ov and b in ov: which,i,j = 2,ov.index(a),ov.index(b)
        elif op == '~1' or (op == '~' and b == '1'):
            if not means: continue
            which,i,j = (4,ov.index(a),0) if a in ov else (5,lv.index(a),0)
        elif op in ('==', ':='): continue
        else: raise ValueError(f'unsupported independent row {a} {op} {b}')
        cells[which][i,j] = point[free-1] if free else binary(r['est'])
        for k in range(q): d[which][k][i,j] = int(k+1 == free)
        if which in (1,2):
            cells[which][j,i] = cells[which][i,j]
            for k in range(q): d[which][k][j,i] = int(k+1 == free)
    L,P,T,B,nu,alpha = cells; dL,dP,dT,dB,dnu,dalpha = d
    A = (mp.eye(m)-B)**-1; M = L*A
    sigma = M*P*M.T+T; mu = nu+M*alpha
    da = [A*x*A for x in dB]; dm = [dL[k]*A+L*da[k] for k in range(q)]
    ds = [dm[k]*P*M.T+M*dP[k]*M.T+M*P*dm[k].T+dT[k] for k in range(q)]
    dmu = [dnu[k]+dm[k]*alpha+M*dalpha[k] for k in range(q)]
    positions = [(i,j) for j in range(p) for i in range(j,p)]
    J = mp.matrix(([list(x) for x in zip(*[list(x) for x in dmu])] if means else [])+
                  [[x[i,j] for x in ds] for i,j in positions])
    def second(i,j):
        dd = dL[i]*da[j]+dL[j]*da[i]+L*(da[j]*dB[i]*A+A*dB[i]*da[j])
        cov = (dd*P*M.T+M*P*dd.T+dm[i]*P*dm[j].T+dm[j]*P*dm[i].T+
               dm[i]*dP[j]*M.T+M*dP[j]*dm[i].T+dm[j]*dP[i]*M.T+M*dP[i]*dm[j].T)
        mean = dd*alpha+dm[i]*dalpha[j]+dm[j]*dalpha[i]
        return mp.matrix((list(mean) if means else [])+[cov[r,c] for r,c in positions])
    return sigma, mu, J, second


def retained_weight(get, key, block, p, means):
    prefix = f'weight_{block}_'; kind = get(key,prefix+'kind',True)
    n = p*(p+1)//2+(p if means else 0)
    if kind is None or int(kind[0]) == 1: return mp.eye(n)
    if int(kind[0]) == 2: return mp.diag(list(get(key,prefix+'diagonal')))
    if int(kind[0]) == 3:
        F = get(key,prefix+'factor'); return F*F.T
    L = get(key,prefix+'root'); A = (L*L.T)**-1
    off = p if means else 0; W = mp.matrix(n)
    if means: W[:p,:p] = A
    positions = [(i,j) for j in range(p) for i in range(j,p)]
    E = []
    for i,j in positions:
        e = mp.matrix(p); e[i,j] = e[j,i] = 1; E.append(e)
    for i,a in enumerate(E):
        for j,b in enumerate(E): W[off+i,off+j] = sum((A*a*A*b)[k,k] for k in range(p))/2
    return W


def gammas(X, S, means):
    n,p = X.rows,X.cols; off = p if means else 0
    positions = [(i,j) for j in range(p) for i in range(j,p)]; q = off+len(positions)
    mean = [sum(X[i,j] for i in range(n))/n for j in range(p)]
    Y = mp.matrix([[X[i,j]-mean[j] for j in range(p)] for i in range(n)])
    Z = mp.matrix([(list(Y[i,:]) if means else [])+[Y[i,a]*Y[i,b] for a,b in positions] for i in range(n)])
    center = [sum(Z[i,j] for i in range(n))/n for j in range(q)]
    Z = mp.matrix([[Z[i,j]-center[j] for j in range(q)] for i in range(n)])
    empirical = Z.T*Z/n; nt = mp.matrix(q)
    if means: nt[:p,:p] = S
    for i,(a,b) in enumerate(positions):
        for j,(c,d) in enumerate(positions): nt[off+i,off+j] = S[a,c]*S[b,d]+S[a,d]*S[b,c]
    return nt, empirical


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',type=Path,required=True)
    out = parser.parse_args().run_dir
    if (out/'comparisons.csv').exists(): raise ValueError('fresh output required')
    mp.mp.dps = 90; start = time.monotonic()
    cases = {int(r['case_id']):r for r in read(out/'cases.csv')}
    sample = matrices(read(out/'samples.csv')); raw = matrices(read(out/'raw_rows.csv'))
    artifact = matrices(read(out/'artifacts.csv')); producer = matrices(read(out/'producers.csv'))
    partables = {}
    for r in read(out/'points.csv'): partables.setdefault(int(r['point_id']),[]).append(r)
    cached = {}; recipe_checks = []
    for r in read(out/'recipes.csv'):
        key,cid = int(r['id']),int(r['case_id']); means = yes(cases[cid]['has_means']); g = 1
        while sample(cid,f'S_{g}',True) is not None:
            S = sample(cid,f'S_{g}'); p = S.rows
            if (cid,g) not in cached: cached[cid,g] = gammas(raw(cid,f'X_{g}'),S,means)
            nt,emp = cached[cid,g]
            W = retained_weight(producer,key,g,p,means); method = r['method']
            if method == 'ULS': expected = mp.eye(W.rows)
            elif method == 'DWLS': expected = mp.diag([1/emp[i,i] for i in range(emp.rows)])
            elif method == 'GLS': expected = nt**-1
            elif method == 'WLS': expected = emp**-1
            else:
                a = binary(r['a']); expected = ((1-a)*nt+a*emp)**-1
            # Congruence by moment units keeps mixed-unit errors visible.
            units = mp.diag([mp.sqrt(nt[i,i]) for i in range(nt.rows)])
            error = mp.norm(units*(W-expected)*units)/max(1,mp.norm(units*expected*units))
            supplied = producer(key,f'supplied_{g}')
            representation_error = mp.norm(units*(supplied-W)*units)/max(1,mp.norm(units*W*units))
            recipe_checks.append(dict(case_id=cid,family=cases[cid]['family'],method=method,block=g,
                recipe_relative_error=number(error),representation_relative_error=number(representation_error),
                passed=bool(error<mp.mpf('1e-9') and representation_error<mp.mpf('1e-12'))))
            g += 1
    comparisons = []
    for row in read(out/'intervals.csv'):
        pid,cid = int(row['point_id']),int(row['case_id']); pt = partables[pid]
        means = yes(cases[cid]['has_means']); q = max(int(r['free']) for r in pt)
        get = lambda name,optional=False: artifact(pid,name,optional)
        if row['chart'] == 'sphere':
            point,map_jac,chain = sphere_map(get('point'),get); basis = get('tangent_basis')
        else:
            point = mp.matrix(q,1)
            for r in pt:
                if int(r['free']): point[int(r['free'])-1] = binary(r['est'])
            map_jac = mp.eye(q); chain = lambda g:mp.matrix(q); basis = get('derivative_basis')
        G,H = mp.matrix(q,1),mp.matrix(q); factor_blocks,score_blocks = [],[]
        objective,total_n,g = mp.mpf(0),0,1
        while sample(cid,f'S_{g}',True) is not None:
            S = sample(cid,f'S_{g}'); n = int(sample(cid,f'n_{g}')[0]); total_n += n
            p = S.rows; off = p if means else 0
            sigma,mu,J,second = moments(pt,point,g,means)
            pos = [(i,j) for j in range(p) for i in range(j,p)]
            residual = mp.matrix((list(mu-sample(cid,f'mean_{g}')) if means else [])+[sigma[i,j]-S[i,j] for i,j in pos])
            W = retained_weight(artifact,pid,g,p,means); wr = W*residual
            G += n*J.T*wr; h = n*J.T*W*J; objective += n*(residual.T*wr)[0]/2
            for j in range(q):
                for i in range(j+1):
                    h[i,j] += n*(wr.T*second(i,j))[0]; h[j,i] = h[i,j]
            H += h; L = mp.cholesky(S); inverse = L**-1
            F = mp.matrix(off+p*p,J.rows)
            if means: F[:p,:p] = L.T
            for k,(i,j) in enumerate(pos):
                for col in range(p):
                    for r in range(p): F[off+r+p*col,off+k] = (L[i,r]*L[j,col]+L[j,r]*L[i,col])/mp.sqrt(2)
            factor_blocks.append(mp.sqrt(n)*F*W*J)
            R=sigma-S
            # The objective packs lower-vech, including binary64 asymmetry in
            # supplied S. Its score factor mirrors those same lower entries.
            for i in range(p):
                for j in range(i+1,p): R[i,j]=R[j,i]
            white = mp.sqrt(n/mp.mpf(2))*inverse*R*inverse.T
            score_blocks.append(mp.matrix((list(mp.sqrt(n)*inverse*(mu-sample(cid,f'mean_{g}'))) if means else [])+
                                         [white[i,j] for j in range(p) for i in range(p)]))
            g += 1
        reduced = map_jac*basis; h = reduced.T*H*reduced+basis.T*chain(G)*basis; gradient = reduced.T*G
        positive = mp.eigsy(h,eigvals_only=True)[0]>0
        A = mp.matrix(sum(x.rows for x in factor_blocks),q); b = mp.matrix(A.rows,1); offset = 0
        for x,y in zip(factor_blocks,score_blocks):
            A[offset:offset+x.rows,:] = x; b[offset:offset+x.rows,:] = y; offset += x.rows
        target = mp.sqrt((gradient.T*mp.lu_solve((A*reduced).T*(A*reduced),gradient))[0]) if positive else mp.nan
        errors = [mp.inf]*3; exact_h = None
        scaled = get('equilibrated_factor',True); T = get('curvature_coordinate_map',True)
        if scaled is not None:
            D = mp.diag(list(get('factor_scale')))
            errors[0] = mp.norm(A*reduced*D-scaled); errors[1] = mp.norm(b-get('metric_score_residual'))
        if T is not None:
            exact_h = T.T*h*T; errors[2] = mp.norm(exact_h-get('curvature_equilibrated_hessian'))
        available = row['source_status']=='available'
        bounds = [binary(row[k]) for k in ('matrix_bound','vector_bound','curvature_bound')]
        covers = not available or all(e<=bound for e,bound in zip(errors,bounds))
        lower_covers = not available or binary(row['curvature_lower'])<=0 or binary(row['curvature_lower'])<=mp.eigsy(exact_h,eigvals_only=True)[0]
        interval_covers = row['interval_status']!='available' or (positive and binary(row['lower'])<=target<=binary(row['upper']))
        decision = row['decision']; budget = binary(.01)
        wrong = ((decision=='within_budget' and (not positive or target>budget)) or
                 (decision=='above_budget' and positive and target<=budget))
        objective_agrees = abs(objective/total_n-binary(row['recomputed_objective']))<=mp.mpf('1e-10')*(1+abs(objective/total_n))
        comparisons.append(dict(point_id=pid,case_id=cid,family=cases[cid]['family'],method=row['method'],chart=row['chart'],target=row['target'],
            reference_distance=number(target),reference_positive=bool(positive),source_status=row['source_status'],
            matrix_error=number(errors[0]),vector_error=number(errors[1]),curvature_error=number(errors[2]),bounds_covers=bool(covers),
            curvature_lower_covers=bool(lower_covers),interval_status=row['interval_status'],interval_covers=bool(interval_covers),wrong_decision=bool(wrong),
            reference_smallest_curvature=number(mp.eigsy(h,eigvals_only=True)[0]),reference_gradient_norm=number(mp.norm(gradient)),
            objective_agrees=bool(objective_agrees),decision=decision,selected_passed=row['selected_passed']))
        if pid%8==0: print(f'Weighted reference {pid}; {time.monotonic()-start:.1f}s',flush=True)
    write(out/'producer_checks.csv',recipe_checks); write(out/'comparisons.csv',comparisons)
    gates = [dict(check='weight_recipe',tested=len(recipe_checks),failed=sum(not r['passed'] for r in recipe_checks))]
    failures=read(out/'producer_failures.csv')
    gates.append(dict(check='producer_returned',tested=len(recipe_checks)+len(failures),failed=len(failures)))
    for k in ('bounds_covers','curvature_lower_covers','interval_covers','objective_agrees'):
        applicable=[r for r in comparisons if (r['source_status']=='available' if k in ('bounds_covers','curvature_lower_covers') else
                    r['interval_status']=='available' if k=='interval_covers' else True)]
        gates.append(dict(check=k,tested=len(applicable),failed=sum(not r[k] for r in applicable)))
    gates.append(dict(check='wrong_decision',tested=len(comparisons),failed=sum(r['wrong_decision'] for r in comparisons)))
    controls=[r for r in comparisons if r['target']=='saddle_rank']
    gates.append(dict(check='saddle_rank_rejected',tested=len(controls),failed=sum(r['reference_positive'] or
        float(r['reference_gradient_norm'])>1e-8 or yes(r['selected_passed']) for r in controls)))
    near=[r for r in comparisons if r['target'].startswith('near_')]
    gates.append(dict(check='budget_controls',tested=len(near),failed=sum(r['decision']!=
        ('within_budget' if r['target']=='near_0.0099' else 'above_budget') for r in near)))
    write(out/'verification.csv',gates)
    write(out/'reference_metadata.csv',[dict(digits=90,mpmath=mp.__version__,elapsed_s=time.monotonic()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=';'.join(hashlib.sha256((out/f).read_bytes()).hexdigest() for f in
            ('cases.csv','samples.csv','raw_rows.csv','artifacts.csv','points.csv','intervals.csv','recipes.csv','producers.csv')),
        convention='exact binary64 retained weight representation, SEM/sample/map inputs; producer checked separately from raw moments; Frobenius error norm')])
    if any(r['failed'] for r in gates): raise SystemExit('independent weighted gate failed; evidence retained')


if __name__=='__main__': main()
