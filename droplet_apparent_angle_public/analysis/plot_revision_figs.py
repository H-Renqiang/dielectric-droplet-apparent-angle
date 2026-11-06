# -*- coding: utf-8 -*-
"""plot_revision_figs.py — 修订版论文图
fig2_combined.pdf : (a) N(theta,k) lens vs BEM; (b) f(theta,k) lens/piecewise/BEM
fig3_chains.pdf   : (a) h/D vs U; (b) r_c/D vs U  (L1 vs BEM)
fig4_phase.pdf    : (a) h(thetaY,U); (b) r_c(thetaY,U) 相图
                    底色 = BEM 全扫描, 虚线等值线 = L1 理论
figA1_errors.pdf  : 附录误差图 (a) prefactor 误差; (b) 沿链形状误差
"""
import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

AN = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(AN)
import sys
sys.path.insert(0, AN)
from derive_hu import nz, zeta_ellip, EPSR

OUT = os.path.join(WS, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'legend.fontsize': 7,
    'lines.linewidth': 1.0,
})
CM = 1/2.54
THS = [30, 45, 60, 75, 90]
COL = {30: '#9467bd', 45: '#ff7f0e', 60: '#1f77b4', 75: '#2ca02c',
       90: '#000000'}

# ---------------- 共享数据 ----------------
def load_surf():
    dat = {}
    for r in csv.DictReader(open(os.path.join(AN, 'n_surface.csv'))):
        dat.setdefault(float(r['theta']), {})[float(r['k'])] = float(r['fhat'])
    return dat

SURF = load_surf()
STERMS, SCOEF = [], []
for r in csv.DictReader(open(os.path.join(AN, 'n_corr_params.csv'))):
    STERMS.append((int(r['term_i']), int(r['term_j'])))
    SCOEF.append(float(r['coef']))

def N_bem(th, k, fhat):
    return ((EPSR-1)/fhat - 1)/(EPSR-1)

def N_old(th_deg, k):
    th = np.radians(th_deg)
    N = nz(k)
    if th_deg < 90:
        N = (1.0 + k**(-1.5)*np.cos(th))*N
    return N

def N_lens(th_deg, k):
    z = zeta_ellip(th_deg, k)
    kl = k*np.sqrt(max(1-z*z, 0.0))/(1+z)
    u = max(0.0, -np.cos(np.radians(th_deg)))
    v = 1.0/k
    S = 1.0 + sum(c*u**i*v**j for (i, j), c in zip(STERMS, SCOEF))
    return nz(kl)*S

def f_of_N(N):
    return (EPSR-1)/(1+N*(EPSR-1))

# ======================================================================
# fig2_combined: (a) N, (b) f
# ======================================================================
fig, (ax, axb) = plt.subplots(2, 1, figsize=(8.6*CM, 11.0*CM),
                              gridspec_kw={'hspace': 0.30})
kcont = np.linspace(0.4, 1.6, 200)
for th in THS:
    ks = sorted(SURF[float(th)])
    Nb = [N_bem(th, k, SURF[float(th)][k]) for k in ks]
    ax.plot(ks, Nb, 'o', ms=2.6, mfc='none', mec=COL[th], mew=0.7)
    ax.plot(kcont, [N_lens(th, k) for k in kcont], '-', color=COL[th])
ax.set_ylabel(r'$N(\theta_Y,k)$')
ax.set_ylim(0.04, 0.90)
ax.set_xlim(0.38, 1.86)
ax.text(0.03, 0.93, '(a)', transform=ax.transAxes, fontsize=9)
lab_y = {30: 0.710, 45: 0.630, 60: 0.585, 75: 0.520, 90: 0.470}
for th in THS:
    ax.text(1.64, lab_y[th], rf'${th}^\circ$', fontsize=7.5,
            color=COL[th], va='center')
ax.text(1.64, 0.79, r'$\theta_Y=$', fontsize=7.5, color='k', va='center')
handles = [Line2D([], [], color='k', lw=1.0, label='lens model'),
           Line2D([], [], color='k', marker='o', ls='none', ms=2.6,
                  mfc='none', mew=0.7, label='BEM')]
ax.legend(handles=handles, loc='lower right', handlelength=1.4,
          bbox_to_anchor=(0.99, 0.02))

for th in THS:
    ks = sorted(SURF[float(th)])
    fb = [SURF[float(th)][k] for k in ks]
    axb.plot(ks, fb, 'o', ms=2.8, mfc='none', mec=COL[th], mew=0.7)
    axb.plot(kcont, [f_of_N(N_lens(th, k)) for k in kcont], '-',
             color=COL[th])
# 端点标注：按 lens 曲线在 k=1.6 处的取值放置，并做最小间距排斥
lab_end = {th: f_of_N(N_lens(th, 1.6)) for th in THS}
order = sorted(THS, key=lambda t: lab_end[t])
MINSEP = 0.5
pos = {order[0]: lab_end[order[0]]}
for t in order[1:]:
    pos[t] = max(lab_end[t], pos[order[order.index(t)-1]] + MINSEP)
for th in THS:
    axb.text(1.67, pos[th], rf'${th}^\circ$', fontsize=7.5,
             color=COL[th], va='center')
axb.set_xlabel(r'$k$')
axb.set_ylabel(r'$f(\theta_Y,k)$')
axb.set_xlim(0.38, 1.95)
axb.set_ylim(0.7, 8.0)
axb.text(0.03, 0.93, '(b)', transform=axb.transAxes, fontsize=9)
handles = [Line2D([], [], color='k', lw=1.0, label='lens model'),
           Line2D([], [], color='k', marker='o', ls='none', ms=2.8,
                  mfc='none', mew=0.7, label='BEM')]
axb.legend(handles=handles, loc='upper right', handlelength=1.6)
fig.savefig(os.path.join(OUT, 'fig2_combined.pdf'), bbox_inches='tight')
plt.close(fig)
print('saved fig2_combined.pdf')

# ======================================================================
# fig3_chains: (a) h/D, (b) r_c/D
# ======================================================================
DAT = os.path.join(AN, 'l1_chains')
def read_csv(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

fig, (axa, axb) = plt.subplots(2, 1, figsize=(8.6*CM, 9.2*CM),
                               gridspec_kw={'hspace': 0.28})
angles = [45, 60, 75, 90]
colors = {45: '#9467bd', 60: '#1f77b4', 75: '#2ca02c', 90: '#d62728'}
Dd = 5.0
for th in angles:
    rows = read_csv(os.path.join(DAT, f'l1_t{th}_g9.8.csv'))
    U  = np.array([float(r['U']) for r in rows])
    h  = np.array([float(r['h_mm']) for r in rows])
    rc = np.array([float(r['r_c_mm']) for r in rows])
    h3 = np.array([float(r['M3_h']) for r in rows])
    r3 = np.array([float(r['M3_rc']) for r in rows])
    axa.plot(U/1000, h/Dd, '-', color=colors[th], lw=1.0)
    axa.plot(U/1000, h3/Dd, 'o', ms=2.2, mfc='none', mec=colors[th],
             mew=0.6, markevery=2)
    axb.plot(U/1000, rc/Dd, '-', color=colors[th], lw=1.0)
    axb.plot(U/1000, r3/Dd, 'o', ms=2.2, mfc='none', mec=colors[th],
             mew=0.6, markevery=2)
    axa.text(U[-1]/1000+0.10, h[-1]/Dd, rf'${th}^\circ$', fontsize=7.5,
             color=colors[th], va='center')
    axb.text(U[-1]/1000+0.10, rc[-1]/Dd, rf'${th}^\circ$', fontsize=7.5,
             color=colors[th], va='center')
axa.set_ylabel(r'$h/d$')
axa.set_xlim(0, 5.9); axa.set_ylim(0.15, 0.55)
axa.text(0.04, 0.90, '(a)', transform=axa.transAxes, fontsize=9)
handles = [Line2D([], [], color='k', lw=1.0, label='one-parameter theory'),
           Line2D([], [], color='k', marker='o', ls='none', ms=2.4,
                  mfc='none', mew=0.7, label='BEM')]
axa.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 0.90),
           handlelength=1.6)
axb.set_ylabel(r'$r_c/d$')
axb.set_xlim(0, 5.9); axb.set_ylim(0.3, 0.65)
axb.text(0.04, 0.90, '(b)', transform=axb.transAxes, fontsize=9)
axb.set_xlabel(r'$U$ (kV)')
fig.savefig(os.path.join(OUT, 'fig3_chains.pdf'), bbox_inches='tight')
plt.close(fig)
print('saved fig3_chains.pdf')

# ======================================================================
# fig4_phase: (a) h(thetaY,U), (b) r_c(thetaY,U)
# 底色 = BEM; 等值线 = L1
# ======================================================================
HPC = os.path.join(WS, 'efield_bem', 'efield_bem', 'results', 'hpc')
angs, Us = None, None
bem_h, bem_rc = {}, {}
for f in sorted(glob.glob(os.path.join(HPC, '*_g9.8.csv'))):
    for r in csv.DictReader(open(f)):
        if r['ok'] != '1':
            continue
        t = float(r['theta_Y']); u = float(r['U'])/1000.0
        if t > 90:
            continue
        bem_h[(t, u)] = float(r['h_mm'])
        bem_rc[(t, u)] = float(r['r_c_mm'])
angs = sorted({t for t, u in bem_h})
Us = sorted({u for t, u in bem_h})
THG, UG = np.meshgrid(angs, Us, indexing='ij')
HB = np.array([[bem_h[(t, u)] for u in Us] for t in angs])
RB = np.array([[bem_rc[(t, u)] for u in Us] for t in angs])

# L1 网格
l1_h, l1_rc = {}, {}
for t in angs:
    tag = int(t) if t == int(t) else t
    f = os.path.join(DAT, f'l1_t{tag}_g9.8.csv')
    if not os.path.exists(f):
        continue
    for r in read_csv(f):
        l1_h[(t, float(r['U'])/1000.0)] = float(r['h_mm'])
        l1_rc[(t, float(r['U'])/1000.0)] = float(r['r_c_mm'])
HL = np.full_like(HB, np.nan)
RL = np.full_like(RB, np.nan)
for i, t in enumerate(angs):
    for j, u in enumerate(Us):
        if (t, u) in l1_h:
            HL[i, j] = l1_h[(t, u)]
            RL[i, j] = l1_rc[(t, u)]
print('L1 coverage: %d/%d' % (np.isfinite(HL).sum(), HL.size))

# 细化网格（三次插值）用于平滑填色
tf = np.linspace(angs[0], angs[-1], 300)
uf = np.linspace(Us[0], Us[-1], 300)
TF, UF = np.meshgrid(tf, uf, indexing='ij')
Hi = RegularGridInterpolator((angs, Us), HB, method='cubic')(np.stack([TF.ravel(), UF.ravel()], 1)).reshape(TF.shape)
Ri = RegularGridInterpolator((angs, Us), RB, method='cubic')(np.stack([TF.ravel(), UF.ravel()], 1)).reshape(TF.shape)

fig, axes = plt.subplots(2, 1, figsize=(8.6*CM, 11.0*CM))
fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.10,
                    hspace=0.35)
for ax, Z, lab in [(axes[0], Hi, r'$h$ (mm)'),
                   (axes[1], Ri, r'$r_c$ (mm)')]:
    vmin = np.floor(Z.min()*10)/10
    vmax = np.ceil(Z.max()*10)/10
    lv = np.linspace(vmin, vmax, 19)
    cf = ax.contourf(TF, UF, Z, levels=lv, cmap='jet')
    # BEM 等值线（细实线）
    ax.contour(TF, UF, Z, levels=np.linspace(vmin, vmax, 10),
               colors='k', linewidths=0.3, alpha=0.4)
    ax.set_xlabel(r'$\theta_Y$ (deg)')
    ax.set_ylabel(r'$U$ (kV)')
    cb = fig.colorbar(cf, ax=ax, pad=0.02,
                      ticks=np.linspace(vmin, vmax, 6))
    cb.set_label(lab, fontsize=8)
    cb.ax.tick_params(labelsize=7, direction='in')
axes[0].text(0.04, 0.93, '(a)', transform=axes[0].transAxes, fontsize=10,
             color='w')
axes[1].text(0.04, 0.93, '(b)', transform=axes[1].transAxes, fontsize=10,
             color='w')
fig.savefig(os.path.join(OUT, 'fig4_phase.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig4_phase.png'), dpi=300,
            bbox_inches='tight')
plt.close(fig)
print('saved fig4_phase.pdf')

# ======================================================================
# figA1_errors: (a) prefactor 误差; (b) 沿链形状误差
# ======================================================================
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.6*CM, 10.5*CM),
                              gridspec_kw={'hspace': 0.30})
ths_all = [t for t in sorted(SURF) if t <= 90]
err_lens = []
for t in ths_all:
    el = max(abs(f_of_N(N_lens(t, k))/fh - 1) for k, fh in SURF[t].items())*100
    err_lens.append(el)
ax.plot(ths_all, err_lens, 'o-', ms=3, mfc='none', mew=0.8, lw=0.8,
        color='#d62728', label='lens model')
ax.set_yscale('log')
ax.set_ylim(0.1, 5)
ax.set_xlabel(r'$\theta_Y$ (deg)')
ax.set_ylabel(r'max $|\delta f/f|$ (\%)')
ax.legend(loc='upper center', handlelength=1.8)
ax.text(0.04, 0.90, '(a)', transform=ax.transAxes, fontsize=9)

summ = read_csv(os.path.join(DAT, 'l1_summary.csv'))
th = np.array([float(r['theta']) for r in summ])
g = np.array([float(r['g']) for r in summ])
eh = np.array([float(r['max_h_err']) for r in summ])
er = np.array([float(r['max_rc_err']) for r in summ])
mask = th <= 90
th = th[mask]; g = g[mask]; eh = eh[mask]; er = er[mask]
m0 = g == 0.0; m9 = g == 9.8
ax2.plot(th[m9], eh[m9], 'o-', ms=3.0, color='#d62728', lw=0.7,
         label=r'$h$, $g=9.8$ m/s$^2$')
ax2.plot(th[m0], eh[m0], 'o--', ms=3.0, mfc='none', color='#d62728', lw=0.7,
         label=r'$h$, $g=0$')
ax2.plot(th[m9], er[m9], 's-', ms=2.8, color='#1f77b4', lw=0.7,
         label=r'$r_c$, $g=9.8$ m/s$^2$')
ax2.plot(th[m0], er[m0], 's--', ms=2.8, mfc='none', color='#1f77b4', lw=0.7,
         label=r'$r_c$, $g=0$')
ax2.axhline(1.0, ls=':', lw=0.6, color='0.4')
ax2.set_yscale('log')
ax2.set_ylim(8e-3, 20)
ax2.set_xlim(25, 95)
ax2.set_xlabel(r'$\theta_Y$ (deg)')
ax2.set_ylabel(r'max error along branch (%)')
ax2.legend(loc='upper center', fontsize=6.2, ncol=2, handlelength=1.8,
           columnspacing=1.0, bbox_to_anchor=(0.5, 0.98))
ax2.text(0.03, 0.90, '(b)', transform=ax2.transAxes, fontsize=9)
fig.savefig(os.path.join(OUT, 'figA1_errors.pdf'), bbox_inches='tight')
plt.close(fig)
print('saved figA1_errors.pdf')
