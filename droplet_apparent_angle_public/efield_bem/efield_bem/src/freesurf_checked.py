# -*- coding: utf-8 -*-
"""Strict free-boundary iteration: no returned point is accepted without residuals."""
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import bem_axisym as B
from freesurf import Drop


class CheckedDrop(Drop):
    def geometric_metrics(self, X, r, z, ph):
        V = self.volume(r, ph, X[2])
        return dict(theta_error_deg=float(np.degrees(-ph[-1])-self.thetaY_deg),
                    volume_ratio=float(V/self.V0),
                    z_contact_over_h=float(abs(z[-1]/X[0])),
                    h_mm=float(z[0]*1e3), rc_mm=float(r[-1]*1e3))

    def geometry_ok(self, m):
        return (abs(m['theta_error_deg']) < 0.01
                and abs(m['volume_ratio']-1) < 2e-4
                and m['z_contact_over_h'] < 2e-4
                and 0 < m['h_mm'] < 0.98*self.D*1e3)

    def solve_voltage(self, U, X0=None, itmax=80, damp=0.6, tol=2e-3,
                      verbose=False, strict=True):
        """Try progressively smaller relaxation; return ok=0 instead of a false equilibrium."""
        if X0 is None:
            from freesurf import cap_guess
            cg = cap_guess(self.thetaY, self.V0)
            X0 = [cg['h'], cg['dp0'], cg['L']]
        X0 = np.asarray(X0, float)
        fn0 = self.fn_hat.copy()
        if U == 0:
            self.fn_hat[:] = 0
            X, solr = self.shoot(X0, 0.0)
            pr = self.profile(*X, 0.0)
            if pr is None:
                return dict(ok=0, reason='zero_field_shoot_failed')
            r, z, ph = pr
            m = self.geometric_metrics(X, r, z, ph)
            out = self._pack(U, X, r, z, ph, None, 0.0)
            out.update(ok=int(self.geometry_ok(m)), reason='converged', **m)
            return out
        scale = (U/self.D)**2
        last = dict(ok=0, reason='not_attempted')
        for relaxation in ([damp] if not strict else [damp, 0.3, 0.15]):
            X = X0.copy()
            self.fn_hat = fn0.copy()
            reason = 'iteration_limit'
            for it in range(itmax):
                X, solr = self.shoot(X, scale)
                pr = self.profile(*X, scale)
                if pr is None:
                    reason = 'profile_failed'
                    break
                r, z, ph = pr
                gm = self.geometric_metrics(X, r, z, ph)
                if not self.geometry_ok(gm):
                    reason = 'shoot_residual:' + str(gm)
                    break
                P = np.column_stack([r, z])
                sol = B.build(P, self.epsr, mode=self.mode, D=self.D, nimg=self.nimg)
                fn_new = B.surface_fields(sol, self.epsr)[2]
                fn_mix = ((1-relaxation)*self.fn_hat + relaxation*fn_new
                          if it > 0 else fn_new)
                change = np.max(np.abs(fn_mix-self.fn_hat))/max(1e-30, np.max(np.abs(fn_mix)))
                self.fn_hat = fn_mix
                self.sol_bem = sol
                if verbose:
                    print(f'    relax={relaxation:.2f} it={it:3d} h={gm["h_mm"]:.6f} '
                          f'rc={gm["rc_mm"]:.6f} dtheta={gm["theta_error_deg"]:.2e} '
                          f'dV={gm["volume_ratio"]-1:.2e} dFn={change:.2e}', flush=True)
                if it > 0 and change < tol:
                    # Make shape and field consistent at the same iterate before accepting.
                    X, solr = self.shoot(X, scale)
                    pr = self.profile(*X, scale)
                    if pr is None:
                        reason = 'final_profile_failed'
                        break
                    r, z, ph = pr
                    gm = self.geometric_metrics(X, r, z, ph)
                    if not self.geometry_ok(gm):
                        reason = 'final_geometry:' + str(gm)
                        break
                    sol = B.build(np.column_stack([r, z]), self.epsr,
                                  mode=self.mode, D=self.D, nimg=self.nimg)
                    fn_final = B.surface_fields(sol, self.epsr)[2]
                    residual = np.max(np.abs(fn_final-self.fn_hat))/max(1e-30, np.max(np.abs(fn_final)))
                    self.sol_bem = sol
                    out = self._pack(U, X, r, z, ph, sol, scale)
                    out.update(ok=int(residual < tol), reason='converged' if residual < tol else 'field_residual',
                               field_relative_residual=float(residual), iterations=it+1,
                               relaxation=relaxation, **gm)
                    if residual < tol:
                        return out
                    last = out
                    reason = 'field_residual'
                    break
                last = self._pack(U, X, r, z, ph, sol, scale)
                last.update(ok=0, reason=reason, field_relative_residual=float(change),
                            iterations=it+1, relaxation=relaxation, **gm)
            else:
                reason = 'iteration_limit'
            if isinstance(last, dict):
                last['ok'] = 0
                last['reason'] = reason
        self.fn_hat = fn0.copy()
        return last
