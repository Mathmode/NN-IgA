from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from common.bspline_basis import basis_batch_for_degree
from scripts import alg1_lshape_Tscan as scan
from scripts import lshape_energy_errors as evaluator
from src.nonparametric.lshape.conformity import support_rule_free_mask
from src.nonparametric.lshape.energy_error import attach_energy_errors
from src.nonparametric.lshape.pde import sigma_field
from src.nonparametric.solver_2d import free_mask_lshape

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("p", [2, 3])
def test_graded_positive_span_keeps_partition_and_mask(p):
    h = 2.0 ** -35
    knots = np.r_[np.zeros(p + 1), 0.5 - 2*h, 0.5 - h, np.full(p, 0.5),
                  0.5 + h, 0.5 + 2*h, np.ones(p + 1)]
    i = np.flatnonzero(knots == 0.5)[-1] - p
    window = jnp.asarray(knots[None, i:i + 2*p + 2])
    values, derivatives, second = basis_batch_for_degree(p, jnp.array([[0.5 + h/4]]), window)
    assert np.sum(values) == pytest.approx(1, abs=1e-13)
    assert float(derivatives.sum()) * h == pytest.approx(0, abs=1e-12)
    assert float(second.sum()) * h**2 == pytest.approx(0, abs=1e-12)
    np.testing.assert_array_equal(free_mask_lshape(knots, knots, p), support_rule_free_mask(knots, knots, p))


def test_material_on_both_sides_of_thin_span():
    x = jnp.array([0.5 - 2e-13, 0.5 + 2e-13])
    np.testing.assert_array_equal(sigma_field(x, 0.75, 2, 3), [1, 3])


def test_selection_and_reference_lookup():
    val = scan.split_instances("val", 16)
    assert len(set(val)) == 16
    assert set(val).isdisjoint(scan.split_instances("test", 16))
    energies = pd.read_csv(REPO / "references_v3/pilot_L20_q14_16.csv")
    assert scan.reference_energy(energies, val[0], "val") == pytest.approx(energies.F_order16.iloc[1])
    with pytest.raises(ValueError):
        scan.reference_energy(energies, val[0], "test")
    with pytest.raises(ValueError):
        scan.reference_energy(energies, val[-1], "val")


def test_v3_reference_and_existing_energy_gaps():
    energies = pd.read_csv(REPO / "references_v3/lshape_energy_ngsolve.csv", float_precision="round_trip")
    assert len(energies.drop_duplicates(["sigma1", "sigma2"])) == len(energies) == 400
    assert energies.split.value_counts().to_dict() == {"train": 280, "val": 60, "test": 60}
    assert np.isfinite(energies.select_dtypes("number")).all().all()
    assert energies.rel_diff_orders.max() <= 1e-11
    old = pd.read_csv(REPO / "references_v2/lshape_energy_ngsolve.csv", float_precision="round_trip")
    common = evaluator.align_parameter_pairs(old, energies)
    np.testing.assert_allclose(common.F_order18, old.F_order14, rtol=4e-9, atol=0)
    rows = evaluator.load_rows("data_results/lshape/p*/eval/*.csv")
    checked = attach_energy_errors(rows, evaluator.align_parameter_pairs(rows, energies), "F_order18")
    assert len(checked) == len(rows) > 0
    assert np.isfinite(checked.gap).all()
    assert (checked.gap >= 0).all()


def test_validation_selection_handles_plateaus_and_unresolved_errors():
    from scripts.summarize_lshape_Tscan import validation_tables

    rows = []
    for N in (32, 64):
        for bound, relative_error in [(5, 0.20), (8, 0.199), (10, 0.10), (15, 0.099)]:
            for i in range(16):
                gap = (relative_error / N) ** 2 if i > 0 else -1e-13
                rows.append(dict(p=2, N=N, T=bound, gap=gap, h_min=(i+1)*1e-9,
                                 F_load=1, F_ref=1, ritz=-0.5, ritz_assembled=-0.5, gamma_loc=2, nit=3, stop="gradient"))
    data = pd.DataFrame(rows)
    medians, rates, schedule = validation_tables(data)
    assert (medians.n_unresolved == 1).all()
    assert np.allclose(rates.order_32_64, 1)
    assert schedule["T"].tolist() == [10, 10]
    assert np.allclose(schedule.h_min_floor, 1e-10, rtol=1e-12, atol=0)
    data.loc[data.N == 64, "gap"] = 0
    medians, rates, schedule = validation_tables(data)
    assert medians.loc[medians.N == 64, "median_err"].isna().all()
    assert rates.order_32_64.isna().all()
    assert schedule.loc[schedule.N == 64, "T"].isna().all()


@pytest.mark.parametrize("p", [2, 3])
def test_element_energy_and_load_on_thin_spans(p):
    from src.nonparametric.lshape.variational_energy import energy_and_load

    s1, s2 = 2, 7
    h = 2.0 ** -35
    knots = np.r_[np.zeros(p + 1), 0.5 - h, np.full(p, 0.5), 0.5 + h, np.ones(p + 1)]
    greville = np.array([np.mean(knots[i+1:i+p+1], dtype=np.longdouble) for i in range(len(knots)-p-1)])
    strain, load = energy_and_load(np.outer(greville, greville), knots, knots, p, s1, s2)
    assert load == pytest.approx((1 + 3 + 9)/64, abs=2e-15)
    assert strain == pytest.approx((2*s1 + 8 + 14*s2)/48, abs=3e-15)
