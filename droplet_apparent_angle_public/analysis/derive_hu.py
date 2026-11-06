# -*- coding: utf-8 -*-
"""derive_hu.py — L1 微扰展开导出 h(U)、r_c(U) 闭式公式并对 M3 链验证

框架：E(k;U) = E0(k) - G~ f(thetaY,k)，G~ = beta*U^2，beta = eps0*V0/(2 D^2)
  k* = k0 + a1*G~ + a2*G~^2
  a1 = f'/E0''；a2 = (f''*a1 - E0'''*a1^2/2)/E0''
  h(U)/h0 = 1 + C1h U^2 + C2h U^4 ；r_c 同理
导数用局部多项式拟合（被函数显式、光滑，精度 ~1e-8）。
f 用 L1 解析因子；另用 figs/fhat_cache.csv 的 BEM 采样做 f' 校准版对照。
"""
import os, csv
import numpy as np
from scipy.optimize import minimize_scalar

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3D = os.path.join(WS, 'efield_bem', 'efield_bem', 'results', 'hpc')
CACHE = os.path.join(WS, 'figs/fhat_cache.csv')
OUTCSV = os.path.join(WS, 'analysis/derive_hu_coeffs.csv')

SIGMA, DRHO, EPS0 = 0.073, 998.8, 8.8541878128e-12
V0, D, EPSR = 1.5e-8, 5e-3, 80.0
BETA = EPS0*V0/(2*D**2)          # J / V^2

THETAS = [30,35,40,45,50,55,60,65,70,75,80,82,84,86,88,90,95,100,105,110,115,120]

# ---------------- L1 几何与能量（与 scan_fhat.py 一致） ----------------
def quad(f, a, b, n=96):
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5*(b-a)*x + 0.5*(b+a)
    return 0.5*(b-a)*np.sum(w*f(t))

def zeta_ellip(th_deg, k):
    th = np.radians(th_deg)
    if abs(th_deg-90) < 1e-12:
        return 0.0
    return -np.sign(np.cos(th))/np.sqrt(1+k*k*np.tan(th)**2)

def geom(th_deg, k):
    """返回 h, r_c, b（SI，m）"""
    zeta = zeta_ellip(th_deg, k)
    F = 2/3 + zeta - zeta**3/3
    b = (V0/(np.pi*k*k*F))**(1/3)
    a = k*b
    return (1+zeta)*b, a*np.sqrt(1-zeta*zeta), b

def E0(th_deg, k, g):
    """毛细 + 重力能（J）"""
    zeta = zeta_ellip(th_deg, k)
    F = 2/3 + zeta - zeta**3/3
    b = (V0/(np.pi*k*k*F))**(1/3)
    a = k*b
    A_LG = 2*np.pi*a*b*quad(lambda u: np.sqrt(np.maximum(1-(1-k*k)*u*u, 0)), -zeta, 1)
    A_SL = np.pi*a*a*(1-zeta*zeta)
    zc = np.pi*a*a*b*b*quad(lambda u: (zeta+u)*(1-u*u), -zeta, 1)/V0
    return SIGMA*(A_LG - np.cos(np.radians(th_deg))*A_SL) + DRHO*g*V0*zc

# ---------------- L1 解析电场因子 ----------------
def nz(k):
    if k <= 1.0:
        e = np.sqrt(max(1-k*k, 0.0))
        return 1/3 if e < 1e-14 else k*k/e**3*(np.arctanh(e)-e)
    e = np.sqrt(k*k-1)
    return k*k/e**3*(e-np.arctan(e))

def f_analytic(th_deg, k):
    N = nz(k)
    if th_deg < 90:
        N = (1+k**(-1.5)*np.cos(np.radians(th_deg)))*N
    return (EPSR-1)/(1+N*(EPSR-1))

# ---------------- 局部多项式求导 ----------------
def poly_derivs(func, k0, hwin=0.03, deg=6, n=41):
    """在 k0 附近用 deg 次多项式拟合，返回 (f0, f', f'', f''')（对 k 的导数）"""
    ks = np.linspace(k0-hwin, k0+hwin, n)
    vs = np.array([func(k) for k in ks])
    c = np.polyfit(ks-k0, vs, deg)          # 降幂
    p = np.poly1d(c)
    return p(0), p.deriv(1)(0), p.deriv(2)(0), p.deriv(3)(0)

# ---------------- 单点系数计算 ----------------
def coeffs(th_deg, g, f_func):
    E0f = lambda k: E0(th_deg, k, g)
    if g == 0:
        k0 = 1.0                            # 毛细能量极小 = 球冠（严格）
    else:
        k0 = minimize_scalar(E0f, bounds=(0.5, 2.0), method='bounded',
                             options=dict(xatol=1e-12)).x
    _, E0p, E0pp, E0ppp = poly_derivs(E0f, k0)
    _, fp, fpp, _ = poly_derivs(lambda k: f_func(th_deg, k), k0)
    h0v, _, _ = geom(th_deg, k0)
    hh = lambda k: geom(th_deg, k)[0]
    rr = lambda k: geom(th_deg, k)[1]
    h0p, hp, hpp, _ = poly_derivs(hh, k0)
    r0p, rp, rpp, _ = poly_derivs(rr, k0)
    h0v, r0v = h0p, r0p
    Lh, H2 = hp/h0v, hpp/h0v
    Lr, R2 = rp/r0v, rpp/r0v
    a1 = fp/E0pp
    a2 = (fpp*a1 - 0.5*E0ppp*a1**2)/E0pp
    C1h = BETA*a1*Lh;            C2h = BETA**2*(a2*Lh + 0.5*a1**2*H2)
    C1r = BETA*a1*Lr;            C2r = BETA**2*(a2*Lr + 0.5*a1**2*R2)
    return dict(k0=k0, h0=h0v, r0=r0v, E0pp=E0pp, fp=fp,
                a1=a1, a2=a2, C1h=C1h, C2h=C2h, C1r=C1r, C2r=C2r)

# ---------------- L1 全解（数值极小化，对照用） ----------------
def l1_chain(th_deg, g, U_list):
    out = []
    k = 1.0
    for U in U_list:
        res = minimize_scalar(lambda kk: E0(th_deg, kk, g)
                              - BETA*U*U*f_analytic(th_deg, kk),
                              bounds=(max(0.2, k-0.4), min(2.5, k+0.4)),
                              method='bounded', options=dict(xatol=1e-10))
        k = res.x
        h, r, _ = geom(th_deg, k)
        out.append((U, h, r))
    return out

# ---------------- BEM 采样缓存（f' 校准对照） ----------------
def load_cache():
    dat = {}
    if not os.path.exists(CACHE):
        return dat
    for r in csv.DictReader(open(CACHE)):
        dat.setdefault(float(r['theta']), []).append((float(r['k']), float(r['fhat'])))
    for t in dat:
        dat[t].sort()
    return dat

def fhat_derivs(cache, th_deg, k0=1.0, win=0.35):
    if float(th_deg) not in cache:
        return None
    pts = [(k, f) for k, f in cache[float(th_deg)] if abs(k-k0) <= win]
    if len(pts) < 6:
        return None
    ks = np.array([p[0] for p in pts]); fs = np.array([p[1] for p in pts])
    c = np.polyfit(ks-k0, fs, 5)
    p = np.poly1d(c)
    return p.deriv(1)(0), p.deriv(2)(0)

# ---------------- M3 低电压斜率（验证） ----------------
def m3_slope(th_deg, g):
    fp_ = os.path.join(M3D, f"chain_t{th_deg}_g{('0' if g==0 else '9.8')}.csv")
    if not os.path.exists(fp_):
        return None
    U, H, R = [], [], []
    for r in csv.DictReader(open(fp_)):
        if r.get('ok', '1') != '1':
            continue
        U.append(float(r['U'])); H.append(float(r['h_mm'])); R.append(float(r['r_c_mm']))
    U, H, R = map(np.array, (U, H, R))
    m = (U > 0) & (U <= 1500)
    if m.sum() < 3:
        return None
    sh = np.sum((H[m]/H[0]-1)*U[m]**2)/np.sum(U[m]**4)   # 过原点最小二乘
    sr = np.sum((R[m]/R[0]-1)*U[m]**2)/np.sum(U[m]**4)
    return sh, sr, H[0], R[0]

# ---------------- 主流程 ----------------
def main():
    cache = load_cache()
    rows = []
    for th in THETAS:
        for g in (0.0, 9.8):
            c = coeffs(th, g, f_analytic)
            row = dict(theta=th, g=g, **c)
            fd = fhat_derivs(cache, th, c['k0'])
            if fd is not None:                     # BEM 校准版 C1
                fp_b, _ = fd
                a1b = fp_b/c['E0pp']
                hh = lambda k: geom(th, k)[0]
                rr = lambda k: geom(th, k)[1]
                _, hp, _, _ = poly_derivs(hh, c['k0'])
                _, rp, _, _ = poly_derivs(rr, c['k0'])
                row['C1h_bem'] = BETA*a1b*hp/c['h0']
                row['C1r_bem'] = BETA*a1b*rp/c['r0']
            ms = m3_slope(th, g)
            if ms is not None:
                row['m3_sh'], row['m3_sr'] = ms[0], ms[1]
                row['ratio_h'] = c['C1h']/ms[0] if ms[0] else np.nan
                row['ratio_r'] = c['C1r']/ms[1] if ms[1] else np.nan
            rows.append(row)
            print(f"t{th:>3} g={g:4.1f} k0={c['k0']:.5f} "
                  f"C1h={c['C1h']:.3e} C2h={c['C2h']:.3e} "
                  f"C1r={c['C1r']:.3e} "
                  + (f"M3sh={row['m3_sh']:.3e} ratio={row['ratio_h']:.3f}"
                     if 'm3_sh' in row else ""), flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUTCSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print('->', OUTCSV)

if __name__ == '__main__':
    main()
