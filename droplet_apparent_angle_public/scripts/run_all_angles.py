# -*- coding: utf-8 -*-
"""run_all_angles.py — 服务器版：对单个角度跑 L1 fold + 匹配变形电压 + M3 严格 BEM，拟合 c。"""
import os, sys, csv
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# 物理常量
SIGMA, EPS0 = 0.073, 8.8541878128e-12
V0, D_GAP, EPSR = 1.5e-8, 5e-3, 80.0
BETA = EPS0*V0/(2*D_GAP**2)

BASE = os.path.dirname(os.path.abspath(__file__))
NC = os.path.join(BASE, 'n_corr_params.csv')

# 读 N 结构修正系数
STERMS, SCOEF = [], []
for r in csv.DictReader(open(NC)):
    STERMS.append((int(r['term_i']), int(r['term_j'])))
    SCOEF.append(float(r['coef']))

# 匹配变形的目标轴比（与 kimi 一致）
KTARGETS = [0.985, 0.940, 0.850, 0.665]

def zeta_ellip(th, k):
    t = np.radians(th)
    if abs(th-90) < 1e-12: return 0.0
    return -np.sign(np.cos(t))/np.sqrt(1+k*k*np.tan(t)**2)

def nz(k):
    if k <= 1.0:
        e = np.sqrt(max(1-k*k, 0.0))
        return 1/3 if e < 1e-14 else k*k/e**3*(np.arctanh(e)-e)
    e = np.sqrt(k*k-1)
    return k*k/e**3*(e-np.arctan(e))

def N_lens(th, k):
    z = zeta_ellip(th, k)
    kl = k*np.sqrt(max(1-z*z, 0.0))/(1+z)
    u = max(0.0, -np.cos(np.radians(th)))
    v = 1.0/k
    S = 1.0 + sum(c*u**i*v**j for (i, j), c in zip(STERMS, SCOEF))
    return nz(kl)*S

def f_lens(th, k):
    return (EPSR-1)/(1+N_lens(th, k)*(EPSR-1))

def quad(f, a, b, n=96):
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5*(b-a)*x + 0.5*(b+a)
    return 0.5*(b-a)*np.sum(w*f(t))

def E0(th, k, g):
    z = zeta_ellip(th, k)
    F = 2/3 + z - z**3/3
    b = (V0/(np.pi*k*k*F))**(1/3)
    a = k*b
    A_LG = 2*np.pi*a*b*quad(lambda u: np.sqrt(np.maximum(1-(1-k*k)*u*u, 0)), -z, 1)
    A_SL = np.pi*a*a*(1-z*z)
    zc = np.pi*a*a*b*b*quad(lambda u: (z+u)*(1-u*u), -z, 1)/V0
    return SIGMA*(A_LG - np.cos(np.radians(th))*A_SL) + 998.8*g*V0*zc

def l1_k(th, U):
    r = minimize_scalar(lambda kk: E0(th, kk, 0.0) - BETA*U*U*f_lens(th, kk),
                        bounds=(0.3, 2.5), method='bounded', options=dict(xatol=1e-12))
    return r.x

def spheroid_theta(th, k, n=4000):
    z = zeta_ellip(th, k)
    F = 2/3 + z - z**3/3
    b = (V0/(np.pi*k*k*F))**(1/3)
    a = k*b
    u = np.linspace(-z, 1.0, n)
    du = u[1]-u[0]
    s = np.sqrt(np.maximum(1-u*u, 1e-16))
    drdu = -a*u/s
    dsdu = np.sqrt(drdu*drdu + b*b)
    ell = np.concatenate([[0.0], np.cumsum(dsdu[:-1]*du)])
    th_app = np.degrees(np.arctan2(1.0, k*u/s))
    th_app = np.where(u > 1-1e-9, 0.0, th_app)
    return ell, th_app

def R0(th):
    return (3*V0/(np.pi*(2-3*np.cos(np.radians(th))+np.cos(np.radians(th))**3)))**(1/3)

def find_fold(th):
    """L1 的 fold 电压（自由能极小点消失，k* 跑到下边界）。"""
    for U in range(300, 9000, 100):
        if l1_k(th, U) < 0.35:
            lo, hi = U-100, U
            for _ in range(30):
                mid = 0.5*(lo+hi)
                if l1_k(th, mid) < 0.35: hi = mid
                else: lo = mid
            return 0.5*(lo+hi)
    return 9000.0

def find_U(th, kt):
    """找使 l1_k(th,U) = kt 的电压。"""
    lo, hi = 50.0, 9000.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if l1_k(th, mid) > kt: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def main(idx):
    th = int(sys.argv[1]) if len(sys.argv) > 1 else idx
    out_dir = os.path.join(BASE, 'results_all_angles')
    os.makedirs(out_dir, exist_ok=True)
    outf = os.path.join(out_dir, f't{th}.csv')

    # 1. fold 电压 + 安全上限
    U_fold = find_fold(th)
    U_safe = 0.95 * U_fold

    # 2. 匹配变形电压（确保 < U_safe）
    Us = []
    for kt in KTARGETS:
        U = find_U(th, kt)
        if U > U_safe:
            U = U_safe  # 截断到安全上限
        Us.append(U)

    # 3. M3 严格跑（CheckedDrop）
    sys.path.insert(0, BASE)
    from freesurf_checked import CheckedDrop
    from freesurf import cap_guess
    d = CheckedDrop(th, g=0.0, N=160, mode='halfspace')
    R = cap_guess(np.radians(th))['R']
    X = None
    prev_U = 0.0
    m3_data = {}  # U -> (ell_over_R0, theta_app_deg)
    for U in [0.0] + Us:
        if U == 0.0:
            res = d.solve_voltage(0.0, X0=X, strict=True)
            X = [res['h'], res['dp0'], res['L']]
        else:
            base_X = X.copy(); base_fn = d.fn_hat.copy()
            result = None
            for nstep in [1, 2, 4, 8, 16]:
                Xtry = base_X.copy(); d.fn_hat = base_fn.copy(); ok = True
                for j in range(1, nstep+1):
                    Uj = prev_U + (U-prev_U)*j/nstep
                    cand = d.solve_voltage(float(Uj), X0=Xtry, strict=True)
                    if not cand.get('ok', 0): ok = False; result = cand; break
                    result = cand; Xtry = [cand['h'], cand['dp0'], cand['L']]
                if ok: break
            if not result or not result.get('ok', 0):
                print(f'[t{th}] U={U:.0f} REJECTED', flush=True)
                return
            X = [result['h'], result['dp0'], result['L']]
        prev_U = float(U)
        r, z, ph, L = (res if U == 0.0 else result)['r'], (res if U == 0.0 else result)['z'], \
                      (res if U == 0.0 else result)['phi'], (res if U == 0.0 else result)['L']
        s = d.x*L; ell = L - s; theta = -np.degrees(ph)
        m3_data[round(U, 1)] = (ell[::-1]/R, theta[::-1])

    # 4. 拟合 c（匹配变形，直接用 M3 数据 + U² 插值已在 Us 上）
    e0, t0 = m3_data[0.0]
    f0 = interp1d(e0*R, t0, bounds_error=False, fill_value='extrapolate')
    xs, ys = [], []
    for U in Us:
        key = min(m3_data.keys(), key=lambda x: abs(x-U))
        e, t = m3_data[key]
        G = EPS0*(EPSR-1)*U**2*R/(2*SIGMA*D_GAP**2)
        m = (e > 0.015) & (e < 0.12)
        x = np.sin(np.radians(th))*G*e[m]
        y = -(np.cos(np.radians(t[m])) - np.cos(np.radians(f0(e[m]*R))))
        xs.append(x); ys.append(y)
    x = np.concatenate(xs); y = np.concatenate(ys)
    c_m3 = np.sum(x*y)/np.sum(x*x)

    # 5. L1 的 c（完整最小化，同样电压）
    ell0, th0 = spheroid_theta(th, 1.0)
    f0_l1 = interp1d(ell0, th0, bounds_error=False, fill_value='extrapolate')
    xs, ys = [], []
    for U in Us:
        kU = l1_k(th, U)
        ell, th_app = spheroid_theta(th, kU)
        fi = interp1d(ell, th_app, bounds_error=False, fill_value='extrapolate')
        G = EPS0*(EPSR-1)*U**2*R/(2*SIGMA*D_GAP**2)
        eg = np.linspace(0.015, 0.12, 60)
        x = np.sin(np.radians(th))*G*eg
        y = -(np.cos(np.radians(fi(eg*R))) - np.cos(np.radians(f0_l1(eg*R))))
        xs.append(x); ys.append(y)
    x = np.concatenate(xs); y = np.concatenate(ys)
    c_l1 = np.sum(x*y)/np.sum(x*x)

    with open(outf, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['thetaY', 'U_fold', 'U_safe', 'U1', 'U2', 'U3', 'U4', 'c_M3', 'c_L1'])
        w.writerow([th, f'{U_fold:.1f}', f'{U_safe:.1f}'] + [f'{u:.1f}' for u in Us] +
                   [f'{c_m3:.6f}', f'{c_l1:.6f}'])
    print(f'[t{th}] fold={U_fold:.0f} c_M3={c_m3:.6f} c_L1={c_l1:.6f}', flush=True)

if __name__ == '__main__':
    main(0)
