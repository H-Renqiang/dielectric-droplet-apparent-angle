# -*- coding: utf-8 -*-
"""n_structure_fit.py — N(θ,k) 结构修正：透镜等效 + 经验修正因子 S

模型：f = (εr−1)/(1 + N(θ,k)(εr−1))
  N(θ,k) = n_z(k_lens(θ,k)) · S(θ,k)
  k_lens = k√(1−ζ²)/(1+ζ)   （基板镜像 ⇒ 双冠透镜体的等效轴比）
  S = 1 + Σ a_ij u^i v^j,  u = max(0,−cosθ)（钝角凹陷角修正）, v = 1/k

产出：拟合系数、LOAO 交叉验证、修正前后 f 误差图、L1 全链对 M3 验证表
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from derive_hu import nz, EPSR, zeta_ellip, E0, geom, BETA, M3D, WS

SURF = os.path.join(WS, 'analysis/n_surface.csv')
FIG = os.path.join(WS, 'figs/fig_n_structure.png')
OUTCSV = os.path.join(WS, 'analysis/n_corr_chain_errors.csv')

# ---------------- 数据 ----------------
def load_surf():
    dat = {}
    for r in csv.DictReader(open(SURF)):
        dat.setdefault(float(r['theta']), {})[float(r['k'])] = float(r['fhat'])
    return dat

def N_bem_val(th, k, fhat):
    return ((EPSR-1)/fhat - 1)/(EPSR-1)

def k_lens(th_deg, k):
    z = zeta_ellip(th_deg, k)
    return k*np.sqrt(max(1-z*z, 0.0))/(1+z)

def f_of_N(N):
    return (EPSR-1)/(1+N*(EPSR-1))

# ---------------- S 拟合 ----------------
def basis(th_deg, k, terms):
    u = max(0.0, -np.cos(np.radians(th_deg)))
    v = 1.0/k
    return [u**i * v**j for i, j in terms]

def fit_S(dat, terms, skip_theta=None):
    X, Y = [], []
    for t in dat:
        if skip_theta is not None and abs(t-skip_theta) < 1e-9:
            continue
        for k, fh in dat[t].items():
            S = N_bem_val(t, k, fh)/nz(k_lens(t, k))
            X.append(basis(t, k, terms)); Y.append(S-1)
    a, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
    return a

def S_corr(th_deg, k, a, terms):
    return 1.0 + np.dot(a, basis(th_deg, k, terms))

def N_corr(th_deg, k, a, terms):
    return nz(k_lens(th_deg, k))*S_corr(th_deg, k, a, terms)

def f_err_map(dat, Nfunc):
    """返回整个曲面上 f 的相对误差（%）"""
    errs = []
    for t in dat:
        for k, fh in dat[t].items():
            errs.append((t, k, (f_of_N(Nfunc(t, k))/fh - 1)*100))
    return errs

# ---------------- L1 全链验证 ----------------
def chain_errs(th_deg, g, f_func):
    fp_ = os.path.join(M3D, f"chain_t{th_deg}_g{('0' if g==0 else '9.8')}.csv")
    if not os.path.exists(fp_):
        return None
    m3 = []
    for r in csv.DictReader(open(fp_)):
        if r.get('ok', '1') == '1':
            m3.append((float(r['U']), float(r['h_mm']), float(r['r_c_mm'])))
    k = 1.0
    eh = er = 0.0
    for U, h3, r3 in m3:
        res = minimize_scalar(lambda kk: E0(th_deg, kk, g)
                              - BETA*U*U*f_func(th_deg, kk),
                              bounds=(max(0.15, k-0.35), min(2.5, k+0.35)),
                              method='bounded', options=dict(xatol=1e-10))
        k = res.x
        h, rc, _ = geom(th_deg, k)
        h, rc = h*1e3, rc*1e3
        if h > 0.95*5.0:          # 近桥接，全局跳盆段不算
            break
        eh = max(eh, abs(h-h3)/h3*100)
        er = max(er, abs(rc-r3)/r3*100)
    return eh, er, len(m3)

def main():
    dat = load_surf()
    ths = sorted(dat)

    # ---- 候选项集对比（LOAO：留一角度交叉验证） ----
    cand = {
        'S=1（纯透镜）': [],
        'u2v2': [(2, 2)],
        'u2v2+u2v3': [(2, 2), (2, 3)],
        'u1v2+u2v2+u2v3': [(1, 2), (2, 2), (2, 3)],
        'u1v2+u1v3+u2v2+u2v3': [(1, 2), (1, 3), (2, 2), (2, 3)],
    }
    print('=== 候选 S 结构：全曲面 f 最大误差 / LOAO 最大误差（%）===')
    best = None
    for name, terms in cand.items():
        errs = [abs(e[2]) for e in f_err_map(dat, lambda t, k:
                nz(k_lens(t, k))*S_corr(t, k, fit_S(dat, terms), terms))]
        loao = []
        for t0 in ths:
            a = fit_S(dat, terms, skip_theta=t0)
            for k, fh in dat[t0].items():
                fv = f_of_N(nz(k_lens(t0, k))*S_corr(t0, k, a, terms))
                loao.append(abs(fv/fh-1)*100)
        print(f'{name:>22}: full={max(errs):6.2f}%  LOAO={max(loao):6.2f}%')
        if terms and (best is None or max(loao) < best[1]):
            best = (name, max(loao), terms)

    name, _, terms = best
    a = fit_S(dat, terms)
    print(f'\n选定: {name}  系数 a = {np.round(a, 4)}  (terms={terms})')

    N_old = lambda t, k: nz(k)*(1+k**-1.5*np.cos(np.radians(t))) if t < 90 else nz(k)
    N_lens = lambda t, k: nz(k_lens(t, k))
    N_new = lambda t, k: N_corr(t, k, a, terms)

    # ---- 误差地图（三面板） ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    ksv = sorted(dat[ths[0]])
    for ax, (title, Nf) in zip(axes, [
            ('current piecewise N (old)', N_old),
            ('lens model: $n_z(k_{\\rm lens})$', N_lens),
            ('lens + empirical S (new)', N_new)]):
        E = np.array([[abs(f_of_N(Nf(t, k))/dat[t][k]-1)*100 for k in ksv] for t in ths])
        im = ax.pcolormesh(ksv, ths, E, cmap='viridis', shading='auto',
                           vmin=0, vmax=max(50, np.percentile(E, 99)))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('$k$')
        fig.colorbar(im, ax=ax, label='$f$ error (%)')
    axes[0].set_ylabel('$\\theta_Y$ (deg)')
    fig.suptitle('Field-factor error vs BEM truth on the $(\\theta_Y, k)$ surface')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=200, bbox_inches='tight')
    print('->', FIG)

    # ---- L1 全链验证（新旧 f 对 M3） ----
    print('\n=== L1 全链最大高度误差（%，桥接前；old=分段N, new=透镜+S）===')
    rows = []
    for t in [30, 45, 60, 75, 90, 105, 120]:
        for g in (0.0, 9.8):
            eo = chain_errs(t, g, N_old and (lambda tt, kk: f_of_N(N_old(tt, kk))))
            en = chain_errs(t, g, lambda tt, kk: f_of_N(N_new(tt, kk)))
            if eo and en:
                rows.append(dict(theta=t, g=g, L1_old_h=round(eo[0], 2),
                                 L1new_h=round(en[0], 2),
                                 L1_old_rc=round(eo[1], 2), L1new_rc=round(en[1], 2)))
                print(f't{t:>3} g={g:4.1f}: h_err {eo[0]:6.2f} -> {en[0]:5.2f}   '
                      f'rc_err {eo[1]:6.2f} -> {en[1]:5.2f}')
    with open(OUTCSV, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('->', OUTCSV)
    # 把系数写成机器可读，供复用
    with open(os.path.join(WS, 'analysis/n_corr_params.csv'), 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['term_i', 'term_j', 'coef'])
        for (i, j), av in zip(terms, a):
            w.writerow([i, j, av])
    print('-> n_corr_params.csv')

if __name__ == '__main__':
    main()
