"""wedge2.py --- 接触线楔角奇性指数（修正版几何）：
   液体: alpha in (pi-theta, pi) ，气体: alpha in (0, pi-theta)，导体面 alpha=0 与 alpha=pi
   =>  eps_r cot(nu*theta) = -cot(nu*(pi-theta))
"""
import numpy as np
from scipy.optimize import brentq


def nu_of(theta_deg, epsr=80.0):
    th = np.radians(theta_deg)
    def f(nu):
        return epsr*np.cos(nu*th)*np.sin(nu*(np.pi-th)) + np.cos(nu*(np.pi-th))*np.sin(nu*th)
    xs = np.linspace(1e-6, 1.0, 200001)
    v = f(xs)
    roots = []
    for i in range(len(xs)-1):
        if v[i]*v[i+1] < 0:
            try:
                roots.append(brentq(f, xs[i], xs[i+1], xtol=1e-14))
            except Exception:
                pass
    return max(roots) if roots else 1.0


if __name__ == '__main__':
    print("修正后的楔角方程 eps_r cot(nu*theta) = -cot(nu*(pi-theta))")
    print("%8s %10s %10s %10s %10s" % ("theta", "epsr=2", "epsr=10", "epsr=80", "epsr=1000"))
    for t in (30, 45, 60, 75, 90, 95, 105, 120, 135, 150):
        print("%8d %10.4f %10.4f %10.4f %10.4f" % (t, nu_of(t, 2), nu_of(t, 10), nu_of(t, 80), nu_of(t, 1000)))
    print()
    print("极限检验：epsr->无穷 时 nu -> pi/(2*theta)（theta>90 时）")
    for t in (105, 120, 135):
        print("   theta=%3d: nu(epsr=1e6)=%.4f   pi/(2theta)=%.4f" % (t, nu_of(t, 1e6), np.pi/(2*np.radians(t))))
