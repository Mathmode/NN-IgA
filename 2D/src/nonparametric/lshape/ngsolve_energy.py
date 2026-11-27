from __future__ import annotations

CORNER = 0.5
OUTER = "dir"
INTERFACE = "iface"


def _is_interface(x, y, corner=CORNER, tol=1e-9):
    return (abs(x - corner) < tol and y > corner) or (abs(y - corner) < tol and x < corner)


def build_lshape_mesh(levels=16, factor=0.15, maxh=0.2, origin=False, outer_hpref=None):
    from netgen.occ import Glue, MoveTo, OCCGeometry
    from ngsolve import Mesh

    c = 0.0 if origin else CORNER
    o = c - CORNER
    bl = MoveTo(o, o).Rectangle(CORNER, CORNER).Face()
    bl.faces.name = "bl"
    tl = MoveTo(o, c).Rectangle(CORNER, 1.0 - CORNER).Face()
    tl.faces.name = "tl"
    tr = MoveTo(c, c).Rectangle(1.0 - CORNER, 1.0 - CORNER).Face()
    tr.faces.name = "tr"
    shape = Glue([bl, tl, tr])
    for e in shape.edges:
        e.name = INTERFACE if _is_interface(e.center.x, e.center.y, corner=c) else OUTER
    for v in shape.vertices:
        if abs(v.p.x - c) < 1e-9 and abs(v.p.y - c) < 1e-9:
            v.hpref = 1
        elif outer_hpref is not None:
            v.hpref = float(outer_hpref)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
    mesh.RefineHP(int(levels), factor=float(factor))
    return mesh


def boundary_names(mesh):
    return sorted(set(mesh.GetBoundaries()))


def solve_energy(mesh, sigma1, sigma2, order):
    from ngsolve import H1, BilinearForm, GridFunction, InnerProduct, LinearForm, dx, grad

    sigma = mesh.MaterialCF({"bl": float(sigma1), "tl": 1.0, "tr": float(sigma2)})
    fes = H1(mesh, order=int(order), dirichlet=OUTER)
    u, v = fes.TnT()
    a = BilinearForm(sigma * grad(u) * grad(v) * dx).Assemble()
    f = LinearForm(1.0 * v * dx).Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    return float(InnerProduct(f.vec, gfu.vec)), int(fes.ndof)


__all__ = ["build_lshape_mesh", "boundary_names", "solve_energy", "OUTER", "INTERFACE"]
