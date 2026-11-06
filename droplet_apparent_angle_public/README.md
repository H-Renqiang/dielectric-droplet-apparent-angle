# Apparent contact angle equation for a dielectric droplet on a conductive hydrophilic surface in electric field

Code and data for the manuscript

> **"Apparent contact angle equation for a dielectric droplet on a conductive hydrophilic surface in electric field"**
> Ren-Qiang Huang, Gao-Xiao Jiang, Zhong-yi Li, Rui Ma, and Chen-Xu Wu (Xiamen University)

---

## Repository layout

```
├── analysis/
│   ├── derive_hu.py           # truncated-spheroid free energy, geometry, constants
│   ├── n_structure_fit.py     # lens depolarization factor N(theta,k)=n_z(k_lens)
│   ├── l1_chains_paper.py     # L1 equilibrium branches + error vs BEM
│   ├── plot_revision_figs.py  # Fig. 2, 3, 4, 7
│   ├── plot_fig5_apparent.py  # Fig. 5
│   ├── gen_fig6_fixed.py      # Fig. 6
│   ├── n_surface.csv          # BEM electrostatic prefactor f(theta,k)
│   ├── n_corr_params.csv      # lens-model coefficients
│   ├── l1_chains.zip          # theory branches + error (unzip: l1_t{theta}_g{g}.csv)
│   └── apparent_angle.zip     # tangent-angle profiles (unzip: t{theta}_U{U}.csv)
├── efield_bem/efield_bem/
│   ├── src/                   # axisymmetric BEM free-boundary solver
│   │   ├── bem_axisym.py / freesurf.py / freesurf_checked.py / wedge.py
│   └── results/hpc.zip        # BEM equilibrium branches (unzip: chain_t{theta}_g{g}.csv)
├── scripts/                   # generators (BEM solver copy + n_corr_params.csv)
│   ├── run_chain_smalltheta.py#   BEM equilibrium branches
│   ├── run_all_angles.py      #   apparent-angle profiles
│   └── bem_axisym.py / freesurf.py / freesurf_checked.py / wedge.py / n_corr_params.csv
├── data/c_theta_final_table.csv   # fitted prefactor c(theta_Y), 30-140 deg
├── figures/                   # fig1.png ... fig7.png
├── LICENSE                    # MIT (code) + CC-BY-4.0 (data)
└── README.md
```

---

## Physical system

A dielectric droplet (V0 = 15 uL, sigma = 0.073 N/m, eps_r = 80) sits on the
grounded lower plate of a parallel-plate capacitor of gap d = 5 mm. The
microscopic contact angle is pinned at the Young angle theta_Y; the interface
is approximated by a truncated spheroid with one shape parameter k = a/b. The
electrostatic prefactor is obtained in closed form by the lens construction
N(theta_Y, k) = n_z(k_lens). The benchmark is an independent boundary-element
(BEM) free-boundary solver.

The apparent contact angle obeys
  cos(theta_app;U) - cos(theta_app;0) = -c(theta_Y) sin(theta_Y) Gamma l/R0
with c(theta_Y) ~ (1/2) f_sph^2/(eps_r-1) (1 - cos(theta_Y)).

---

## Data files (unzip the .zip files first)

- analysis/n_surface.csv — BEM prefactor f, columns `theta, k, fhat`,
  grid theta_Y = 30-120 deg (step 2.5 deg), k = 0.4-1.6 (step 0.05).
- analysis/n_corr_params.csv — lens correction coefficients `term_i, term_j, coef`.
- efield_bem/efield_bem/results/hpc.zip — BEM equilibrium branches
  `chain_t{theta}_g{g}.csv` (columns `U, theta_c_deg, h_mm, r_c_mm, ok`), 30-120 deg.
- analysis/l1_chains.zip — theory branches `l1_t{theta}_g{g}.csv`
  (columns `U, k, h_mm, r_c_mm, M3_h, M3_rc, h_err, rc_err`) + l1_summary.csv.
- analysis/apparent_angle.zip — tangent-angle profiles `t{theta}_U{U}.csv`
  (columns `ell_over_R0, theta_app_deg`) + summary_t{theta}.csv.
- data/c_theta_final_table.csv — fitted c(theta_Y) for 30-140 deg.

---

## Reproducing the results

Dependencies: Python 3 with numpy, scipy, matplotlib.

    cd analysis
    python plot_revision_figs.py    # Fig. 2, 3, 4, 7 -> ../figures/
    python plot_fig5_apparent.py    # Fig. 5
    python gen_fig6_fixed.py        # Fig. 6
    python l1_chains_paper.py       # regenerate l1_chains/

BEM equilibrium branches (heavy; minutes per branch on one core):

    cd scripts
    python run_chain_smalltheta.py <theta>

---

## License

Code: MIT. Data: CC-BY-4.0. Please cite the paper above if you use this material.
