# -*- coding: utf-8 -*-
"""gen_fig6_fixed.py — 重画 Fig.6：每角度标注真实电压，x 轴到拟合窗口 0.12。"""
import os, csv
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SIGMA, DRHO, EPS0 = 0.073, 998.8, 8.8541878128e-12
V0, D, EPSR = 1.5e-8, 5e-3, 80.0
BETA = EPS0*V0/(2*D**2)

# 严格收敛 BEM 数据（含锐角 30/45/60/75/90）
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, 'apparent_angle')
# L1 的 N 结构修正系数
NC = os.path.join(HERE, 'n_corr_params.csv')
OUT = os.path.join(HERE, '..', 'figures')

STERMS, SCOEF = [], []
for r in csv.DictReader(open(NC)):
    STERMS.append((int(r['term_i']), int(r['term_j'])))
    SCOEF.append(float(r['coef']))

def zeta_ellip(th_deg, k):
    th = np.radians(th_deg)
    if abs(th_deg-90) < 1e-12: return 0.0
    return -np.sign(np.cos(th))/np.sqrt(1+k*k*np.tan(th)**2)

def nz(k):
    if k <= 1.0:
        e = np.sqrt(max(1-k*k, 0.0))
        return 1/3 if e < 1e-14 else k*k/e**3*(np.arctanh(e)-e)
    e = np.sqrt(k*k-1)
    return k*k/e**3*(e-np.arctan(e))

def N_lens(th_deg, k):
    z = zeta_ellip(th_deg, k)
    kl = k*np.sqrt(max(1-z*z, 0.0))/(1+z)
    u = max(0.0, -np.cos(np.radians(th_deg)))
    v = 1.0/k
    S = 1.0 + sum(c*u**i*v**j for (i, j), c in zip(STERMS, SCOEF))
    return nz(kl)*S

def f_lens(th_deg, k): return (EPSR-1)/(1+N_lens(th_deg, k)*(EPSR-1))

def quad(f, a, b, n=96):
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5*(b-a)*x + 0.5*(b+a)
    return 0.5*(b-a)*np.sum(w*f(t))

def E0(th_deg, k, g):
    zeta = zeta_ellip(th_deg, k)
    F = 2/3 + zeta - zeta**3/3
    b = (V0/(np.pi*k*k*F))**(1/3)
    a = k*b
    A_LG = 2*np.pi*a*b*quad(lambda u: np.sqrt(np.maximum(1-(1-k*k)*u*u, 0)), -zeta, 1)
    A_SL = np.pi*a*a*(1-zeta*zeta)
    zc = np.pi*a*a*b*b*quad(lambda u: (zeta+u)*(1-u*u), -zeta, 1)/V0
    return SIGMA*(A_LG - np.cos(np.radians(th_deg))*A_SL) + DRHO*g*V0*zc

def l1_k(th, g, U):
    res = minimize_scalar(lambda kk: E0(th, kk, g) - BETA*U*U*f_lens(th, kk),
                          bounds=(0.3, 2.5), method='bounded', options=dict(xatol=1e-12))
    return res.x

def spheroid_theta(th_deg, k, n=4000):
    zeta = zeta_ellip(th_deg, k)
    F = 2/3 + zeta - zeta**3/3
    b = (V0/(np.pi*k*k*F))**(1/3)
    a = k*b
    u = np.linspace(-zeta, 1.0, n)
    du = u[1]-u[0]
    s = np.sqrt(np.maximum(1-u*u, 1e-16))
    drdu = -a*u/s
    dsdu = np.sqrt(drdu*drdu + b*b)
    ell = np.concatenate([[0.0], np.cumsum(dsdu[:-1]*du)])
    th_app = np.degrees(np.arctan2(1.0, k*u/s))
    th_app = np.where(u > 1-1e-9, 0.0, th_app)
    return ell, th_app

def load_m3(th, U):
    fp = os.path.join(AN, f't{th}_U{U}.csv')
    if not os.path.exists(fp): return None, None
    rows = list(csv.DictReader(open(fp)))
    e = np.array([float(r['ell_over_R0']) for r in rows])
    t = np.array([float(r['theta_app_deg']) for r in rows])
    return e, t

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7, 'axes.linewidth': 0.6,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'legend.frameon': False, 'legend.fontsize': 5.0, 'lines.linewidth': 1.0,
})

M3_CASES = {45:[1000,2000,3000,4000], 60:[1000,2000,3000,4000],
            75:[1000,2000,3000,4000], 90:[1000,2000,3000,4000]}
COLORS = {0:'#1f77b4', 1:'#2ca02c', 2:'#d62728', 3:'#9467bd'}
YLIM = {45:(0,1.5), 60:(0,2), 75:(0,2), 90:(0,2.5)}
LWIN = (0.015, 0.12)   # 拟合窗口

fig, axes = plt.subplots(2, 2, figsize=(3.5, 4.0))
for idx, th in enumerate([45, 60, 75, 90]):
    ax = axes[idx//2][idx%2]
    R0 = (3*V0/(np.pi*(2-3*np.cos(np.radians(th))+np.cos(np.radians(th))**3)))**(1/3)
    ell0, th0 = spheroid_theta(th, 1.0)
    f0 = interp1d(ell0, th0, bounds_error=False, fill_value='extrapolate')
    e0, t0 = load_m3(th, 0)
    f0_m3 = interp1d(e0*R0, t0, bounds_error=False, fill_value='extrapolate')

    volt_handles = []
    for j, U in enumerate(M3_CASES[th]):
        kU = l1_k(th, 0.0, U)
        ell, th_app = spheroid_theta(th, kU)
        e3, t3 = load_m3(th, U)
        if e3 is None: continue
        ell_m3 = e3*R0
        fi = interp1d(ell, th_app, bounds_error=False, fill_value='extrapolate')
        dth_L1 = fi(ell_m3) - f0(ell_m3)
        dth_M3 = t3 - f0_m3(ell_m3)
        ax.plot(e3, dth_M3, 'o', ms=2.5, alpha=0.7, color=COLORS[j], mfc='none')
        ax.plot(e3, dth_L1, '-', lw=1.0, color=COLORS[j])
        volt_handles.append(Line2D([], [], color=COLORS[j], lw=1.0,
                                   label=f'{U/1000:.2f} kV'))

    # 无阴影；x 轴到拟合窗口上限
    ax.set_xlim(0, 0.12)
    ax.set_ylim(*YLIM[th])
    ax.set_xlabel(r'$\ell/R_0$', fontsize=7)
    ax.set_ylabel(r'$\delta\theta_{\rm app}$ (deg)', fontsize=7)
    title = fr'$\theta_Y={th}^\circ$'
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    # 每个子图内放本角度的真实电压图例
    ax.legend(handles=volt_handles, loc='upper left', fontsize=4.8,
              handlelength=1.2, ncol=2, columnspacing=0.6)

# BEM vs theory 图例（放图下方，用 fig.legend）
fig.legend(handles=[Line2D([], [], marker='o', ls='', mfc='none', color='k', ms=3, label='BEM'),
                    Line2D([], [], ls='-', color='k', lw=1, label='theory')],
           loc='lower center', ncol=2, fontsize=6, frameon=False)
fig.tight_layout(pad=0.4, rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(OUT, 'fig6.png'), dpi=300, bbox_inches='tight', pad_inches=0.01)
print('saved fig6.png')
