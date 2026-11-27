from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "2D")]

import numpy as np

from common._precision import ensure_double_precision
from src.nonparametric.lshape import integration as ig
from src.nonparametric.lshape.variational_energy import energy_and_load
from src.nonparametric.lshape.conformity import support_rule_free_mask
from src.nonparametric.solver_2d import galerkin_solve_lshape
from src.nonparametric.eta_estimator_2d import _eta_volume_lshape, _eta_jump_lshape
import jax.numpy as jnp


def axis_reference(knots, p):
    t = list(map(Fraction, knots))
    elements = []
    for s in range(p, len(t)-p-1):
        a, b = t[s:s+2]
        h = b-a
        if not h:
            continue
        previous = {s: np.array([Fraction(1)], dtype=object)}
        for degree in range(1, p+1):
            basis = {}
            for i in range(s-degree, s+1):
                c = np.full(degree+1, Fraction(0), dtype=object)
                if i in previous:
                    c += np.convolve(previous[i], [a-t[i], h])/(t[i+degree]-t[i])
                if i+1 in previous:
                    c += np.convolve(previous[i+1], [t[i+degree+1]-a, -h])/(t[i+degree+1]-t[i+1])
                basis[i] = c
            previous = basis
        B = np.array(list(basis.values()))
        D = derivative(B, h)
        moments = np.array([[h/(j+k+1) for j in range(p+1)] for k in range(p+1)])
        m = np.asarray(B @ moments @ B.T, float)
        k = np.asarray(D @ moments[:-1, :-1] @ D.T, float)
        f = np.asarray(B @ moments[:, 0], float)
        elements.append((s-p, a, h, B, m, k, f))
    return elements


def derivative(c, h):
    return c[:, 1:] * np.arange(1, c.shape[1]) / h


def integral(c):
    return sum(v/Fraction((i+1)*(j+1)) for (i, j), v in np.ndenumerate(c))


def squared(c):
    moments_x = np.array([[Fraction(1, i+j+1) for j in range(c.shape[0])] for i in range(c.shape[0])])
    moments_y = np.array([[Fraction(1, i+j+1) for j in range(c.shape[1])] for i in range(c.shape[1])])
    return np.sum(c * (moments_x @ c @ moments_y))


def check(p, mesh):
    ensure_double_precision()
    h = 2.0**-32 if mesh == "thin" else 0.0625
    def axis(left, right):
        return np.r_[np.zeros(p+1), left, np.full(p, 0.5), right, np.ones(p+1)]
    x = axis([0.25, 0.5-2*h, 0.5-h], [0.5+h, 0.5+3*h, 0.875])
    y = axis([0.125, 0.5-3*h, 0.5-h], [0.5+h, 0.5+2*h, 0.75])
    X, Y = axis_reference(x, p), axis_reference(y, p)
    nx, ny = len(x)-p-1, len(y)-p-1
    ne = len(x)-2*p-1
    s1, s2 = Fraction(1, 4), Fraction(7, 2)
    coeff = np.random.default_rng(192).integers(-16, 17, size=(nx, ny))/16
    coeff *= support_rule_free_mask(x, y, p)
    c = np.vectorize(Fraction, otypes=[object])(coeff)
    xj, yj = jnp.asarray(x), jnp.asarray(y)
    _, *fx = ig.element_forms(xj, p, p+1, 4)
    _, *fy = ig.element_forms(yj, p, p+1, 4)
    fx, fy = [np.asarray(a) for a in fx], [np.asarray(a) for a in fy]
    K = np.zeros((nx*ny, nx*ny))
    F = np.zeros(nx*ny)
    vol, strain, load, faces = Fraction(0), Fraction(0), Fraction(0), {}
    errK = errF = 0.
    for i, a, hx, bx, mx, kx, lx in X:
        for j, b, hy, by, my, ky, ly in Y:
            if a >= 0.5 and b < 0.5:
                continue
            sig = s1 if b < 0.5 else s2 if a >= 0.5 else Fraction(1)
            localK = float(sig)*(np.kron(kx,my) + np.kron(mx,ky))
            localF = np.outer(lx,ly).ravel()
            actualK = float(sig)*(np.kron(fx[1][i],fy[0][j]) + np.kron(fx[0][i],fy[1][j]))
            actualF = np.outer(fx[2][i],fy[2][j]).ravel()
            errK = max(errK, np.max(abs(actualK-localK))/np.max(abs(localK)))
            errF = max(errF, np.max(abs(actualF-localF))/np.max(abs(localF)))
            idx = (np.arange(i,i+p+1)[:,None]*ny + np.arange(j,j+p+1)).ravel()
            K[np.ix_(idx,idx)] += localK
            F[idx] += localF
            u = bx.T @ c[i:i+p+1, j:j+p+1] @ by
            ux, uy = derivative(u.T,hx).T, derivative(u,hy)
            strain += sig*hx*hy*(squared(ux)+squared(uy))
            load += hx*hy*integral(u)
            residual = np.full_like(u, Fraction(0))
            residual[0,0] = Fraction(1)
            residual[:-2,:] += sig*derivative(ux.T,hx).T
            residual[:,:-2] += sig*derivative(uy,hy)
            vol += hx*hy*(hx**2+hy**2)*squared(residual)/sig
            if b >= 0.5 and (a == 0.5 or a+hx == 0.5):
                key = ("x", j, hy)
                trace = sig*ux[0,:] if a == 0.5 else -sig*ux.sum(axis=0)
                faces[key] = faces.get(key, 0) + trace
            if a < 0.5 and (b == 0.5 or b+hy == 0.5):
                key = ("y", i, hx)
                trace = sig*uy[:,0] if b == 0.5 else -sig*uy.sum(axis=1)
                faces[key] = faces.get(key, 0) + trace
    exact_faces = {d: sum(h*h*squared(v[:,None]) / max(1, s2 if d == "x" else s1)
                          for (axis, _, h), v in faces.items() if axis == d) for d in ("x","y")}
    args = (jnp.asarray(coeff), xj, yj, p, ne, ne, float(s1), float(s2), 4)
    free = support_rule_free_mask(x, y, p).ravel()
    sol = galerkin_solve_lshape(xj,yj,p,ne,ne,float(s1),float(s2),q_K=p+1,q_F=4)
    if any(v.dtype != np.float64 for v in (sol.K,sol.F,sol.u_h,sol.ritz_energy)):
        raise AssertionError("float64 required")
    def rel(actual, reference):
        return np.max(abs(np.asarray(actual) - reference)) / np.max(abs(reference))
    def scalar(actual, expected):
        return abs(float(actual)-float(expected))/abs(float(expected))
    sol_strain, sol_load = energy_and_load(sol.u_h, x, y, p, float(s1), float(s2))
    matrix, force, u = np.asarray(sol.K), np.asarray(sol.F), np.asarray(sol.u_h).ravel()
    scale = np.linalg.norm(matrix, np.inf)*np.linalg.norm(u, np.inf) + np.linalg.norm(force, np.inf)
    backward = np.linalg.norm(matrix @ u - force, np.inf)/scale
    return dict(p=p, mesh=mesh, h_min=h, element_K=errK, element_F=errF,
                solve_backward=backward, solve_ritz=scalar(sol.ritz_energy, sol_strain/2-sol_load),
                free_K=rel(np.asarray(sol.K)[np.ix_(free,free)],K[np.ix_(free,free)]),
                free_F=rel(np.asarray(sol.F)[free],F[free]),
                volume=scalar(_eta_volume_lshape(*args), vol),
                face_x=scalar(ig.flux_face(args[0],xj,yj,p,1.,float(s2),True,4),exact_faces["x"]),
                face_y=scalar(ig.flux_face(args[0].T,yj,xj,p,float(s1),1.,False,4),exact_faces["y"]),
                eta2=scalar(_eta_volume_lshape(*args)+_eta_jump_lshape(*args), vol+sum(exact_faces.values())),
                ritz=scalar(ig.ritz(args[0],xj,yj,p,float(s1),float(s2),p+1,4),strain/2-load))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [check(p,m) for p in (2,3) for m in ("graded","thin")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if any(not np.isfinite(v) or v > 2e-14 for r in rows for k,v in r.items() if k not in ("p","mesh","h_min")):
        raise ValueError("integration error exceeds 2e-14")


if __name__ == "__main__":
    main()
