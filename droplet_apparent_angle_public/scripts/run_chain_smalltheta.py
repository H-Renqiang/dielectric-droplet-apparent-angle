# -*- coding: utf-8 -*-
"""run_chain_smalltheta.py — 生成 10/15/20/25° 的 BEM 平衡分支（chain_t*_g*.csv）。"""
import os, sys, csv
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from freesurf_checked import CheckedDrop
from freesurf import cap_guess

UMAX = 5000.0
USTEP = 250.0

def main(th):
    out_dir = os.path.join(BASE, 'results_chain_smalltheta')
    os.makedirs(out_dir, exist_ok=True)
    for g in [0.0, 9.8]:
        d = CheckedDrop(th, g=g, N=160, mode='halfspace')
        X = None; prev_U = 0.0
        rows = []
        for U in np.arange(0.0, UMAX + 1e-9, USTEP):
            if U == 0:
                res = d.solve_voltage(0.0, X0=X, strict=True)
            else:
                base_X = X.copy(); base_fn = d.fn_hat.copy(); result = None
                for nstep in [1, 2, 4, 8, 16]:
                    Xtry = base_X.copy(); d.fn_hat = base_fn.copy(); ok = True
                    for j in range(1, nstep + 1):
                        Uj = prev_U + (U - prev_U) * j / nstep
                        cand = d.solve_voltage(float(Uj), X0=Xtry, strict=True)
                        if not cand.get('ok', 0):
                            ok = False; result = cand; break
                        result = cand; Xtry = [cand['h'], cand['dp0'], cand['L']]
                    if ok:
                        break
                if not result or not result.get('ok', 0):
                    rows.append(dict(theta_Y=th, g=g, U=round(U, 1),
                                     theta_c_deg='', h_mm='', r_c_mm='', ok=0))
                    break
                res = result
            X = [res['h'], res['dp0'], res['L']]
            prev_U = float(U)
            thc = th + float(res.get('theta_error_deg', 0.0))
            rows.append(dict(theta_Y=th, g=g, U=round(U, 1),
                             theta_c_deg=round(thc, 6),
                             h_mm=round(float(res['h_mm']), 6),
                             r_c_mm=round(float(res['rc_mm']), 6),
                             ok=int(res.get('ok', 1))))
        gtag = '0' if g == 0 else '9.8'
        fp = os.path.join(out_dir, f'chain_t{th}_g{gtag}.csv')
        with open(fp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['theta_Y', 'g', 'U', 'theta_c_deg', 'h_mm', 'r_c_mm', 'ok'])
            w.writeheader(); w.writerows(rows)
        print(f'[t{th} g{g}] {len(rows)} 个电压 -> {fp}', flush=True)

if __name__ == '__main__':
    main(int(sys.argv[1]))
