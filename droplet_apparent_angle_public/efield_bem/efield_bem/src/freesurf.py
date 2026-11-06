"""
freesurf.py --- 自由边界求解器：Young–Laplace + 麦克斯韦应力（尖锐界面，强形式）
==============================================================================
液滴轴对称、坐落 z=0 接地极板，体积 V0 固定。界面按弧长 s 参数化（s=0 顶点 → s=L 接触线）：

    dr/ds   = cos phi
    dz/ds   = sin phi
    dphi/ds = -[ dp(z) + f_n(s) ] / sigma - sin(phi)/r
    dp(z)   = dp_0 + drho*g*(h_apex - z)
    f_n     = (U/D)^2 * fn_hat(s)      （BEM 给出单位外场的麦克斯韦法向应力，向外为正）

未知量 (h_apex, dp_0, L)，条件：z(L)=0、phi(L) = -theta_Y（Young 定律）、V=V0。
电场压强与形状迭代到自洽（欠松弛）。

接触线奇性：尖锐楔角下电场 ~ r^{nu-1}，nu≈0.66（epsr=80, theta=90°）> 1/2，
故电场能量在接触线附近 ∝ r^{2nu}，对接触线位置求导 ∝ δ^{2nu-1} → 0：
尖锐极限下"电场线力"消失，宏观接触角严格等于 Young 角。该结论用
(a) 带 Young 边界条件的强形式解 与 (b) 全能量极小化的形状导数 两条独立路线交叉验证。
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

import bem_axisym as B

EPS0 = B.EPS0
SIGMA = 0.073
DRHO = 998.8
V0 = 1.5e-8
EPSR = 80.0
D_GAP = 5.0e-3


def grad_x(N, grade=0.9):
    x = np.linspace(0.0, 1.0, N)
    if grade > 0:
        x = x - grade * np.sin(2 * np.pi * x) / (2 * np.pi)
    return x


def cap_guess(thetaY, V0=V0):
    """U=0、g=0 的球冠初值。"""
    c = np.cos(thetaY)
    R = (3 * V0 / (np.pi * (2 - 3 * c + c ** 3))) ** (1.0 / 3.0)
    return dict(R=R, h=R * (1 - c), rc=R * np.sin(thetaY), L=R * thetaY,
                dp0=2 * SIGMA / R)


class Drop:
    def __init__(self, thetaY_deg, g=0.0, N=160, grade=0.9, sigma=SIGMA, drho=DRHO,
                 V0=V0, epsr=EPSR, D=D_GAP, mode='halfspace', nimg=0):
        self.thetaY = float(np.radians(thetaY_deg))
        self.thetaY_deg = float(thetaY_deg)
        self.g = float(g)
        self.N = int(N)
        self.x = grad_x(N, grade)
        self.sigma = sigma
        self.drho = drho
        self.V0 = V0
        self.epsr = epsr
        self.D = D
        self.mode = mode
        self.nimg = nimg
        self.fn_hat = np.zeros(self.N - 1)      # 单位外场麦克斯韦应力（单元中点）
        self.sol_bem = None

    # ---------------- Young–Laplace 积分 ----------------
    def profile(self, h, dp0, L, scale):
        s = self.x * L
        fmid = self.fn_hat * scale
        smid = 0.5 * (s[:-1] + s[1:])
        fn0 = fmid[0]

        def fn_of_s(ss):
            return np.interp(ss, smid, fmid, left=fn0, right=fmid[-1])

        phip0 = -(dp0 + fn0) / (2.0 * self.sigma)
        s1 = s[1]
        y0 = [s1, h, phip0 * s1]

        def rhs(ss, y):
            r, z, ph = y
            dp = dp0 + self.drho * self.g * (h - z)
            return [np.cos(ph), np.sin(ph),
                    -(dp + fn_of_s(ss)) / self.sigma - np.sin(ph) / max(r, 1e-30)]

        sol = solve_ivp(rhs, (s1, L), y0, t_eval=s[1:], rtol=1e-11, atol=1e-15,
                        method='RK45', max_step=L / 8)
        if not sol.success or sol.y.shape[1] != self.N - 1:
            return None
        r = np.concatenate([[0.0], sol.y[0]])
        z = np.concatenate([[h], sol.y[1]])
        ph = np.concatenate([[0.0], sol.y[2]])
        return r, z, ph

    def volume(self, r, ph, L):
        s = self.x * L
        return np.pi * np.trapz(r ** 2 * (-np.sin(ph)), s)

    def residuals(self, X, scale):
        h, dp0, L = X
        if not (h > 0 and dp0 > 0 and L > 0):
            return np.array([1e3, 1e3, 1e3])
        pr = self.profile(h, dp0, L, scale)
        if pr is None:
            return np.array([1e3, 1e3, 1e3])
        r, z, ph = pr
        if np.any(~np.isfinite(r)) or np.any(~np.isfinite(z)):
            return np.array([1e3, 1e3, 1e3])
        V = self.volume(r, ph, L)
        return np.array([z[-1] / (0.05 * h), ph[-1] + self.thetaY, V / self.V0 - 1.0])

    def shoot(self, X0, scale):
        sol = root(self.residuals, np.asarray(X0, float), args=(scale,), method='hybr',
                   options=dict(xtol=1e-15, maxfev=600))
        return np.asarray(sol.x, float), sol

    # ---------------- 几何与能量 ----------------
    def geometry(self, r, z, ph, L):
        s = self.x * L
        V = self.volume(r, ph, L)
        A_LG = 2 * np.pi * np.trapz(r, s)
        r_c = float(r[-1])
        z_c = np.pi * np.trapz(r ** 2 * z * (-np.sin(ph)), s) / V
        return dict(V=V, A_LG=A_LG, A_SL=np.pi * r_c ** 2, r_c=r_c, h=float(z[0]), z_c=z_c)

    def energies(self, geo, sol, U, g):
        F_s = self.sigma * (geo['A_LG'] - np.cos(self.thetaY) * geo['A_SL'])
        F_g = self.drho * g * self.V0 * geo['z_c']
        dW = sol['dW_hat'] * (U / self.D) ** 2
        return dict(F_s=F_s, F_g=F_g, dW=dW, F_e=-dW, F_tot=F_s + F_g - dW)

    # ---------------- 单电压求解 ----------------
    def solve_voltage(self, U, X0=None, itmax=30, damp=0.6, tol=2e-3, verbose=False):
        scale = (U / self.D) ** 2
        if X0 is None:
            cg = cap_guess(self.thetaY, self.V0)
            X0 = [cg['h'], cg['dp0'], cg['L']]
        if U == 0.0:
            self.fn_hat = np.zeros(self.N - 1)
            X, solr = self.shoot(X0, 0.0)
            r, z, ph = self.profile(*X, 0.0)
            self.fn_hat = np.zeros(self.N - 1)
            self.sol_bem = None
            return self._pack(U, X, r, z, ph, None, 0.0)
        X = np.asarray(X0, float)
        for it in range(itmax):
            X, solr = self.shoot(X, scale)
            r, z, ph = self.profile(*X, scale)
            P = np.column_stack([r, z])
            sol = B.build(P, self.epsr, mode=self.mode, D=self.D, nimg=self.nimg)
            fn_new = B.surface_fields(sol, self.epsr)[2]
            fn_mix = (1 - damp) * self.fn_hat + damp * fn_new if it > 0 else fn_new
            change = np.max(np.abs(fn_mix - self.fn_hat)) / max(1e-30, np.max(np.abs(fn_mix)))
            self.fn_hat = fn_mix
            self.sol_bem = sol
            if verbose:
                print("    it=%2d h=%.6f rc=%.6f theta=%.4f dFn=%.2e"
                      % (it, z[0] * 1e3, r[-1] * 1e3, np.degrees(-ph[-1]), change))
            if it > 0 and change < tol:
                break
        r, z, ph = self.profile(*X, scale)
        P = np.column_stack([r, z])
        sol = B.build(P, self.epsr, mode=self.mode, D=self.D, nimg=self.nimg)
        self.sol_bem = sol
        return self._pack(U, X, r, z, ph, sol, scale)

    def _pack(self, U, X, r, z, ph, sol, scale):
        h, dp0, L = X
        geo = self.geometry(r, z, ph, L)
        out = dict(U=U, thetaY_deg=self.thetaY_deg, g=self.g, h=h, rc=geo['r_c'],
                   L=L, dp0=dp0, theta_c=float(np.degrees(-ph[-1])),
                   r=r, z=z, phi=ph, geo=geo)
        if sol is None:
            out.update(F_s=self.sigma * (geo['A_LG'] - np.cos(self.thetaY) * geo['A_SL']),
                       F_g=self.drho * self.g * self.V0 * geo['z_c'], dW=0.0,
                       F_e=0.0, F_tot=self.sigma * (geo['A_LG'] - np.cos(self.thetaY) * geo['A_SL'])
                       + self.drho * self.g * self.V0 * geo['z_c'])
        else:
            out.update(self.energies(geo, sol, U, self.g))
            out['dW_hat'] = sol['dW_hat']
        return out
