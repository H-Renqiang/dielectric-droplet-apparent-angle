# -*- coding: utf-8 -*-
"""plot_fig5_apparent.py — Fig.5 apparent contact angle, PR style.

Field-induced apparent-angle change relative to the U=0 baseline profile:
  dth(l;U) = th_app(l;U) - th_app(l;0),  l = arc length from contact line.
Scaling: cos th_app(U) - cos th_app(0) ~ -c(thY) sin(thY) Gamma (l/R0).

(a) dth(l/R0) for theta_Y=105 at four voltages.
(b) collapse of all 16 datasets onto 1 after dividing by c(thY) sin thY Gamma l/R0.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AN = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(AN)
DAT = os.path.join(AN, 'apparent_angle')
OUT = os.path.join(WS, 'figures')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'legend.fontsize': 7,
    'lines.linewidth': 1.0,
})
CM = 1/2.54

ANGLES = [45, 60, 75, 90]
COL = {45: '#ff7f0e', 60: '#1f77b4', 75: '#2ca02c', 90: '#000000'}
EG = np.linspace(0.01, 0.30, 200)      # common ell/R0 grid
R0MM = {}

def read_prof(th, U):
    rows = list(csv.DictReader(open(os.path.join(DAT,
                f't{th}_U{int(U)}.csv'))))
    e = np.array([float(x['ell_over_R0']) for x in rows])
    t = np.array([float(x['theta_app_deg']) for x in rows])
    return e, t

def load(th):
    """list of (U, Gamma, dth on EG, thU on EG, th0 on EG)"""
    e0, t0 = read_prof(th, 0)
    th0 = np.interp(EG, e0, t0)
    out = []
    for r in csv.DictReader(open(os.path.join(DAT, f'summary_t{th}.csv'))):
        U = float(r['U']); G = float(r['Gamma'])
        if th == 45 and U > 2000:   # 45° 减小电压范围，去掉最大电压
            continue
        R0MM[th] = float(r['R0'])*1e3
        e, t = read_prof(th, U)
        thU = np.interp(EG, e, t)
        out.append((U, G, thU - th0, thU, th0))
    return out

DATA = {th: load(th) for th in ANGLES}

# fit c(thetaY): cos th_app(U) - cos th_app(0) = -c sin(thY) Gamma (ell/R0)
CC = {}
for th in ANGLES:
    xs, ys = [], []
    for U, G, dth, thU, th0 in DATA[th]:
        m = (EG > 0.015) & (EG < 0.12)
        x = np.sin(np.radians(th))*G*EG[m]
        y = -(np.cos(np.radians(thU[m])) - np.cos(np.radians(th0[m])))
        xs.append(x); ys.append(y)
    x = np.concatenate(xs); y = np.concatenate(ys)
    CC[th] = float(np.sum(x*y)/np.sum(x*x))
print('fitted c:', {k: round(v, 4) for k, v in CC.items()})

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.6*CM, 10.0*CM),
                              gridspec_kw={'hspace': 0.32})

# (a) t60 family
for U, G, dth, thU, th0 in DATA[60]:
    ax.plot(EG, dth, '-', lw=1.0, label=rf'$U={U/1000:.2f}\,$kV')
ax.set_xlabel(r'$\ell/R_0$')
ax.set_ylabel(r'$\delta\theta_{\rm app}$ (deg)')
ax.set_xlim(0, 0.3)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), title=r'$\theta_Y=60^\circ$', title_fontsize=7)
ax.text(0.04, 0.90, '(a)', transform=ax.transAxes, fontsize=9)
R0 = R0MM[60]
sec = ax.secondary_xaxis('top', functions=(lambda x: x*R0*1e3,
                                           lambda x: x/1e3/R0))
sec.set_xlabel(r'$\ell$ ($\mu$m)', fontsize=8)
sec.tick_params(axis='x', direction='in', labelsize=7)

# (b) collapse: y = -[cos th_app(U) - cos th_app(0)]  vs
#               x = c(thY) sin(thY) Gamma (ell/R0),  log-log, y=x reference
for th in ANGLES:
    for U, G, dth, thU, th0 in DATA[th]:
        m = EG > 0.012
        x = CC[th]*np.sin(np.radians(th))*G*EG[m]
        y = -(np.cos(np.radians(thU[m])) - np.cos(np.radians(th0[m])))
        ax2.plot(x, y, '-', lw=0.7, color=COL[th], alpha=0.85)
xx = np.array([5e-4, 0.5])
ax2.plot(xx, xx, ls=':', lw=0.7, color='0.4')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlim(5e-4, 0.5); ax2.set_ylim(5e-4, 0.5)
ax2.set_xlabel(r'$c(\theta_Y)\sin\theta_Y\,\Gamma\,\ell/R_0$')
ax2.set_ylabel(r'$-[\cos\theta_{\rm app}(U)-\cos\theta_{\rm app}(0)]$')
handles = [Line2D([], [], color=COL[th], lw=1.2,
                  label=rf'$\theta_Y={th}^\circ$') for th in ANGLES]
ax2.legend(handles=handles, loc='lower right', ncol=2, handlelength=1.4)
ax2.text(0.03, 0.92, '(b)', transform=ax2.transAxes, fontsize=9)

fig.savefig(os.path.join(OUT, 'fig5_apparent.pdf'),
            bbox_inches='tight', pad_inches=0.03)
fig.savefig(os.path.join(OUT, 'fig5_apparent.png'), dpi=300,
            bbox_inches='tight', pad_inches=0.03)
print('saved fig5')
