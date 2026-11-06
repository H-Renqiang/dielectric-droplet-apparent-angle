"""
bem_axisym.py --- 轴对称静电场边界积分求解器（尖锐界面 / 分片线性 sigma + 内部最小二乘配点）
=========================================================================================
物理：均匀外场中（含 z=0 导体平面镜像；可选 z=0 与 z=D 双板镜像链）一个轴对称介电体
      （相对介电常数 epsr）界面上的束缚面电荷。

表示：phi = phi0 + ∮_Gamma G(x,x') sigma(x') dS' ,  phi0 = z （单位外场 E0 = 1）
      G 含镜像使 z=0（与 z=D）上 phi = 0。
界面条件（phi 连续、eps1 dphi1/dn = eps2 dphi2/dn）-> 第二类 Fredholm 方程
      sigma(x) + 2 lam eps0 [ ∮ dG/dn (x,x') sigma(x') dS' + b(x) ] = 0
      lam = (eps1-eps2)/(eps1+eps2) , b = dphi0/dn
轴对称归约（方位角积掉）：
      G_ax = (r'/(pi eps0)) K(m)/d1 ,  d1^2=(r+r')^2+(z-z')^2 , m=4rr'/d1^2
      dG_ax/dn = (r'/(pi eps0 d1)) [ E(m)( -n.(x-x')/rho^2 + nr/(2r) ) - nr K(m)/(2r) ]
      rho^2=(r-r')^2+(z-z')^2；该式做了代数消去，避免 1/rho^2 大项相减的灾难性抵消。

离散：界面按弧长分段；sigma 分片线性（自由度 = 节点数）；配点取单元内部 Gauss 点，用最小
      二乘求解 —— 配点不落在折点上，避开折点法向造成的伪奇异。

线性性：对 sigma、b 都线性，故解一次得单位外场解；sigma=(U/D)sigma_hat，
      dW ∝ (U/D)^2，f_n ∝ (U/D)^2。
"""
import numpy as np
from scipy.special import ellipk, ellipe

EPS0 = 8.8541878128e-12


# ---------------------------------------------------------------- 核函数
def _md1(r, z, rp, zp):
    r, z, rp, zp = np.broadcast_arrays(r, z, rp, zp)
    d1sq = (r + rp) ** 2 + (z - zp) ** 2
    d1 = np.sqrt(d1sq)
    with np.errstate(divide='ignore', invalid='ignore'):
        m = np.where(d1sq > 0, 4.0 * r * rp / d1sq, 0.0)
    return np.clip(m, 0.0, 1.0 - 1e-15), d1


def kerns(r, z, nr, nz, rp, zp):
    r, z, nr, nz, rp, zp = np.broadcast_arrays(r, z, nr, nz, rp, zp)
    m, d1 = _md1(r, z, rp, zp)
    K, E = ellipk(m), ellipe(m)
    pre = rp / (np.pi * EPS0)
    dr, zeta = r - rp, z - zp
    Kg = pre * K / d1
    rho2 = np.maximum(dr ** 2 + zeta ** 2, 1e-300)
    ndot = nr * dr + nz * zeta
    rsafe = np.where(np.abs(r) < 1e-15, np.inf, r)
    term = E * (-ndot / rho2 + nr / (2.0 * rsafe)) - 0.5 * nr * K / rsafe
    return Kg, pre * term / d1


def kern_img(r, z, nr, nz, rp, zp, mode='halfspace', D=None, nimg=0):
    if mode == 'free':
        return kerns(r, z, nr, nz, rp, zp)
    if mode == 'halfspace':
        a, b = kerns(r, z, nr, nz, rp, zp)
        c, d = kerns(r, z, nr, nz, rp, -zp)
        return a - c, b - d
    if mode != 'slab':
        raise ValueError(mode)
    Kg, Kn = None, None
    for n in range(-int(nimg), int(nimg) + 1):
        off = 2.0 * n * D
        a1, b1 = kerns(r, z, nr, nz, rp, zp + off)
        a2, b2 = kerns(r, z, nr, nz, rp, -zp + off)
        Kg = (a1 - a2) if Kg is None else Kg + (a1 - a2)
        Kn = (b1 - b2) if Kn is None else Kn + (b1 - b2)
    return Kg, Kn


# ---------------------------------------------------------------- 网格
def resample_polyline(pts, N, grade=0.9, closed=False):
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    x = np.linspace(0.0, 1.0, N)
    if grade > 0:
        x = x - grade * np.sin(2 * np.pi * x) / (2 * np.pi)
    r = np.interp(x * L, s, pts[:, 0])
    z = np.interp(x * L, s, pts[:, 1])
    return np.column_stack([r, z]), L


def elements_of(nodes, closed=False):
    P = np.asarray(nodes, float)
    N = len(P)
    i0 = np.arange(N) if closed else np.arange(N - 1)
    i1 = (i0 + 1) % N if closed else i0 + 1
    A, B = P[i0], P[i1]
    d = B - A
    L = np.hypot(d[:, 0], d[:, 1])
    t = d / np.maximum(L, 1e-300)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return dict(A=A, B=B, mid=0.5 * (A + B), t=t, n=n, elen=L, i0=i0, i1=i1)


def _clustered(A, B, xc, qn=12, nsub=14, ratio=0.5):
    """单元 [0,1] 上聚类到最近点 u* 的求积（处理 K(m) 的弱对数奇异）。"""
    AB = B - A
    L2 = max(float(np.dot(AB, AB)), 1e-300)
    us = float(np.clip(np.dot(np.asarray(xc, float) - A, AB) / L2, 0.0, 1.0))
    xg, wg = np.polynomial.legendre.leggauss(qn)
    xg, wg = (xg + 1) / 2.0, wg / 2.0
    ou, ow = [], []
    for a, b, clus_a in ((0.0, us, False), (us, 1.0, True)):
        if b - a < 1e-15:
            continue
        t = np.arange(nsub + 1)
        frac = (1.0 - ratio ** t) / (1.0 - ratio ** nsub)
        br = a + (b - a) * frac if clus_a else b - (b - a) * frac[::-1]
        for k in range(nsub):
            lo, hi = br[k], br[k + 1]
            ou.append(lo + (hi - lo) * xg)
            ow.append((hi - lo) * wg)
    return np.concatenate(ou), np.concatenate(ow)


# ---------------------------------------------------------------- 组装 / 求解
def build(nodes, epsr, closed=False, mode='halfspace', D=None, nimg=0,
          nq=2, qfar=10, qself=18):
    P = np.asarray(nodes, float)
    el = elements_of(P, closed=closed)
    A, B, nrm, L, mid = el['A'], el['B'], el['n'], el['elen'], el['mid']
    M = len(L)
    Nu = M if closed else M + 1
    lam = (epsr - 1.0) / (epsr + 1.0)

    xg_c, _ = np.polynomial.legendre.leggauss(nq)
    uc = (xg_c + 1) / 2.0
    ncol = M * nq
    col_r = (A[:, 0:1] + (B - A)[:, 0:1] * uc[None, :]).reshape(-1)
    col_z = (A[:, 1:2] + (B - A)[:, 1:2] * uc[None, :]).reshape(-1)
    col_n = np.repeat(nrm, nq, axis=0)
    col_elem = np.repeat(np.arange(M), nq)
    col_u = np.tile(uc, M)

    xg, wg = np.polynomial.legendre.leggauss(qfar)
    uq, wq = (xg + 1) / 2.0, wg / 2.0
    QR = A[:, 0:1] + (B - A)[:, 0:1] * uq[None, :]
    QZ = A[:, 1:2] + (B - A)[:, 1:2] * uq[None, :]
    QW = L[:, None] * wq[None, :]

    Kg, Kn = kern_img(col_r[:, None, None], col_z[:, None, None],
                      col_n[:, 0][:, None, None], col_n[:, 1][:, None, None],
                      QR[None, :, :], QZ[None, :, :], mode=mode, D=D, nimg=nimg)
    W0 = QW * (1 - uq)[None, :]      # 基函数 (1-u) 的权（含 Jacobi L 与 Gauss 权）
    W1 = QW * uq[None, :]
    Gc = [np.einsum('kjq,jq->kj', Kg, W0), np.einsum('kjq,jq->kj', Kg, W1)]
    Kc = [np.einsum('kjq,jq->kj', Kn, W0), np.einsum('kjq,jq->kj', Kn, W1)]

    for k in range(ncol):                      # 自身单元用聚类细求积
        j = col_elem[k]
        uu, ww = _clustered(A[j], B[j], (col_r[k], col_z[k]), qn=qself, nsub=14, ratio=0.5)
        qr = A[j, 0] + (B[j] - A[j])[0] * uu
        qz = A[j, 1] + (B[j] - A[j])[1] * uu
        kg, kn = kern_img(np.full_like(uu, col_r[k]), np.full_like(uu, col_z[k]),
                          np.full_like(uu, col_n[k, 0]), np.full_like(uu, col_n[k, 1]),
                          qr, qz, mode=mode, D=D, nimg=nimg)
        wj = ww * L[j]
        Gc[0][k, j] = np.sum(kg * wj * (1 - uu))
        Gc[1][k, j] = np.sum(kg * wj * uu)
        Kc[0][k, j] = np.sum(kn * wj * (1 - uu))
        Kc[1][k, j] = np.sum(kn * wj * uu)

    rows = np.arange(ncol)
    Gmat = np.zeros((ncol, Nu))
    Kmat = np.zeros((ncol, Nu))
    Bmat = np.zeros((ncol, Nu))
    jj = np.arange(M)
    for s in (0, 1):
        cidx = (jj + s) % M if closed else jj + s
        Gmat[rows[:, None], cidx[None, :]] += Gc[s]
        Kmat[rows[:, None], cidx[None, :]] += Kc[s]
        # 配点处的插值权重（配点属于自己的单元）
        own = (col_elem[:, None] == jj[None, :])
        Bmat[rows[:, None], cidx[None, :]] += own * ((1 - col_u) if s == 0 else col_u)[:, None]

    Amat = Bmat + 2.0 * lam * EPS0 * Kmat
    rhs = -2.0 * lam * EPS0 * col_n[:, 1]
    sigma_nodes, *_ = np.linalg.lstsq(Amat, rhs, rcond=None)

    # ---- 输出点：单元中点 ----
    Kg_m, _ = kern_img(mid[:, 0][:, None, None], mid[:, 1][:, None, None],
                       nrm[:, 0][:, None, None], nrm[:, 1][:, None, None],
                       QR[None, :, :], QZ[None, :, :], mode=mode, D=D, nimg=nimg)
    Gm = [np.einsum('kjq,jq->kj', Kg_m, W0), np.einsum('kjq,jq->kj', Kg_m, W1)]
    for j in range(M):
        uu, ww = _clustered(A[j], B[j], mid[j], qn=qself, nsub=14, ratio=0.5)
        qr = A[j, 0] + (B[j] - A[j])[0] * uu
        qz = A[j, 1] + (B[j] - A[j])[1] * uu
        kg, _ = kern_img(np.full_like(uu, mid[j, 0]), np.full_like(uu, mid[j, 1]),
                         np.full_like(uu, nrm[j, 0]), np.full_like(uu, nrm[j, 1]),
                         qr, qz, mode=mode, D=D, nimg=nimg)
        wj = ww * L[j]
        Gm[0][j, j] = np.sum(kg * wj * (1 - uu))
        Gm[1][j, j] = np.sum(kg * wj * uu)
    Gmat_mid = np.zeros((M, Nu))
    for s in (0, 1):
        cidx = (np.arange(M) + s) % M if closed else np.arange(M) + s
        Gmat_mid[np.arange(M)[:, None], cidx] += Gm[s]

    if closed:
        sigma_mid = 0.5 * (sigma_nodes + np.roll(sigma_nodes, -1))
    else:
        sigma_mid = 0.5 * (sigma_nodes[:-1] + sigma_nodes[1:])
    phi_mid = Gmat_mid @ sigma_nodes + mid[:, 1]
    S = np.concatenate([[0.0], np.cumsum(L)])
    s_mid = S[:-1] + 0.5 * L
    dW_hat = -0.5 * np.sum(sigma_mid * mid[:, 1] * (2.0 * np.pi * mid[:, 0] * L))
    return dict(nodes=P, el=el, mid=mid, nrm=nrm, elen=L, s=s_mid, dW_hat=dW_hat,
                sigma=sigma_mid, sigma_nodes=sigma_nodes, phi=phi_mid,
                phi_s=phi_mid - mid[:, 1],
                lam=lam, epsr=epsr, mode=mode, D=D, nimg=nimg, closed=closed,
                Amat=Amat, rhs=rhs, Gmat_mid=Gmat_mid)


# ---------------------------------------------------------------- 诊断量
def energy_hat(sol, zref=None):
    """单位外场 (E0=1) 的场能增量 dW = -1/2 ∮ sigma (phi0-phi0_ref) dS（面元 2 pi r ds）。
    精确解电中性，phi0 常数参考无关；离散解取界面高度中位数作参考抑制放大。"""
    if zref is None:
        zref = float(np.median(sol['mid'][:, 1]))
    dS = 2.0 * np.pi * sol['mid'][:, 0] * sol['elen']
    return -0.5 * np.sum(sol['sigma'] * (sol['mid'][:, 1] - zref) * dS)


def surface_fields(sol, epsr):
    """E_n(内侧)、E_t、麦克斯韦法向应力 f_n（沿外法向向外，单位外场下的值）。"""
    s = sol['s']
    Et = -np.gradient(sol['phi'], s, edge_order=2)
    En = sol['sigma'] / (EPS0 * (epsr - 1.0))
    fn = 0.5 * EPS0 * (epsr - 1.0) * (epsr * En ** 2 + Et ** 2)
    return En, Et, fn, s


def potential_at(sol, r, z):
    """任意 (r,z) 处散射电位 phi_s（单位外场）。"""
    el = sol['el']
    A, B, L = el['A'], el['B'], el['elen']
    xg, wg = np.polynomial.legendre.leggauss(10)
    uq, wq = (xg + 1) / 2.0, wg / 2.0
    QR = A[:, 0:1] + (B - A)[:, 0:1] * uq[None, :]
    QZ = A[:, 1:2] + (B - A)[:, 1:2] * uq[None, :]
    QW = L[:, None] * wq[None, :]
    rr, zz = np.asarray(r, float), np.asarray(z, float)
    Kg, _ = kern_img(rr[..., None, None], zz[..., None, None],
                     np.zeros(rr.shape)[..., None, None], np.ones(rr.shape)[..., None, None],
                     QR, QZ, mode=sol['mode'], D=sol.get('D'), nimg=sol.get('nimg', 0))
    G0 = np.einsum('...jq,jq->...j', Kg, QW * (1 - uq)[None, :])
    G1 = np.einsum('...jq,jq->...j', Kg, QW * uq[None, :])
    sn = sol['sigma_nodes']
    if sol['closed']:
        return G0 @ sn + G1 @ np.roll(sn, -1)
    return G0 @ sn[:-1] + G1 @ sn[1:]
