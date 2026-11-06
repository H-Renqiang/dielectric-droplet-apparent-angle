# -*- coding: utf-8 -*-
"""l1_chains_paper.py — 论文用 L1（截断椭球单参数族 + 透镜闭式 f）全链数据

对所有 M3 链角度（30°–120°，g=0 与 9.8）：
  在 M3 的电压网格上做 L1 平衡（对 k 一维极小化 E0 - beta*U^2*f_lens），
  落盘 h、r_c 及相对 M3 的误差；汇总最大误差（分支跳变截断口径 8%）。
输出:
  analysis/l1_chains/l1_t{th}_g{g}.csv     每链逐点
  analysis/l1_chains/l1_summary.csv        每链最大误差与截断电压
"""
import csv, os, sys
import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from derive_hu import E0, geom, BETA, M3D
from n_structure_fit import fit_S, N_corr, load_surf, f_of_N

OUT = os.path.join(HERE, 'l1_chains')
JUMP = 8.0   # 分支跳变截断口径（%）

def gtag(g):
    return '0' if g == 0 else '9.8'

def load_m3(th, g):
    fp = os.path.join(M3D, f"chain_t{th}_g{gtag(g)}.csv")
    if not os.path.exists(fp):
        return None
    return [(float(r['U']), float(r['h_mm']), float(r['r_c_mm']))
            for r in csv.DictReader(open(fp)) if r.get('ok', '1') == '1']

def main():
    dat = load_surf(); terms = [(2, 2), (2, 3)]; a = fit_S(dat, terms)
    f_lens = lambda t, k: f_of_N(N_corr(t, k, a, terms))

    # 收集全部 M3 角度
    ths = sorted({int(fn.split('_')[1][1:]) for fn in os.listdir(M3D)
                  if fn.startswith('chain_t')})
    os.makedirs(OUT, exist_ok=True)
    summ = []
    for th in ths:
        for g in (0.0, 9.8):
            m3 = load_m3(th, g)
            if not m3:
                continue
            k = 1.0
            rows, eh, er, ustop = [], 0.0, 0.0, m3[-1][0]
            for U, h3, r3 in m3:
                res = minimize_scalar(
                    lambda kk: E0(th, kk, g) - BETA*U*U*f_lens(th, kk),
                    bounds=(max(0.15, k-0.35), min(2.5, k+0.35)),
                    method='bounded', options=dict(xatol=1e-10))
                k = res.x
                h, rc, _ = geom(th, k)
                h_mm, rc_mm = h*1e3, rc*1e3
                e1 = abs(h_mm-h3)/h3*100; e2 = abs(rc_mm-r3)/r3*100
                if max(e1, e2) > JUMP:
                    ustop = U
                    break
                eh, er = max(eh, e1), max(er, e2)
                rows.append(dict(U=U, k=k, h_mm=h_mm, r_c_mm=rc_mm,
                                 M3_h=h3, M3_rc=r3,
                                 h_err=(h_mm-h3)/h3*100,
                                 rc_err=(rc_mm-r3)/r3*100))
            fp = os.path.join(OUT, f"l1_t{th}_g{gtag(g)}.csv")
            with open(fp, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            summ.append(dict(theta=th, g=g, max_h_err=round(eh, 2),
                             max_rc_err=round(er, 2), U_stop=ustop,
                             M3_end=m3[-1][0], n_pts=len(rows)))
            print(f"t{th:>3} g={g:4.1f}: max|h err|={eh:5.2f}%  "
                  f"max|rc err|={er:5.2f}%  U_stop={ustop:5.0f}  "
                  f"(M3 end {m3[-1][0]:5.0f})", flush=True)
    fp = os.path.join(OUT, 'l1_summary.csv')
    with open(fp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader(); w.writerows(summ)
    allh = max(r['max_h_err'] for r in summ)
    allr = max(r['max_rc_err'] for r in summ)
    print(f"\n全局: max|h err|={allh:.2f}%  max|rc err|={allr:.2f}%  "
          f"({len(summ)} 链) -> {fp}")

if __name__ == '__main__':
    main()
