import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lshape_phase_e_eval import condition_numbers
from summarize_lshape_phase_e import LEVELS, energy_tables, paired_tables, precision_table, validate_rows


@pytest.mark.parametrize("n", [10, 87])
def test_condition_numbers_match_exact_dirichlet_spectrum(n):
    a = np.diag(np.full(n, 2.)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
    d = np.linspace(1., 4., n)
    a *= d[:,None]*d[None,:]
    free = np.arange(n) % 23 != 0
    restricted = a[np.ix_(free, free)]
    scales = np.sqrt(restricted.diagonal())
    result = condition_numbers(a, free)
    np.testing.assert_allclose(result, [np.linalg.cond(restricted),
        np.linalg.cond(restricted/scales[:,None]/scales[None,:])], rtol=5e-12)


def sample_rows():
    rows = []
    for n in LEVELS:
        for x in (0.25, 0.75):
            gap = (x/n)**2
            rows.append(dict(method="eta", p=2, N=n, sigma1=x, sigma2=1., split="test",
                             cpus=4, cpu="Platinum 8362 CPU", F_ref=1., strain=1-gap, load=1-gap,
                             gap=gap, err=np.sqrt(gap), eta=2*np.sqrt(gap), ieff=2., below_resolution=False,
                             gamma_loc=1., aspect=1., h_min=1/n, cond=2., cond_jacobi=2.,
                             free_dofs=(n+1)**2-(n//2+1)**2, online_s=0.1))
    return pd.DataFrame(rows)


def test_expected_cases_and_signed_energy_checked():
    d = sample_rows()
    refs = {(0.25,1.):1., (0.75,1.):1.}
    validate_rows(d, list(refs), refs, 2, "eta")
    with pytest.raises(ValueError, match="duplicated"):
        validate_rows(pd.concat([d,d.iloc[:1]]), list(refs), refs, 2, "eta")
    d.loc[0,"gap"] *= -1
    with pytest.raises(ValueError, match="gap"):
        validate_rows(d, list(refs), refs, 2, "eta")


def test_paired_pool_excludes_unpaired_cases_and_uses_all_seeds():
    d = sample_rows()
    uniform = d.assign(method="uniform")
    nets = pd.concat([d.assign(method=m, seed=s, err=d.err*(s+1)/10) for m in ("eta","ritz") for s in range(4)])
    whole = pd.concat([uniform, nets])
    alg = pd.concat([d[d.sigma1 == .25].assign(method=m, time_s=2, n_active=0, stop="gradient")
                     for m in ("eta","ritz")])
    training = pd.DataFrame([dict(method=m, p=2, seed=s, train_s=19, peak_rss_mib=50)
                             for m in ("eta","ritz") for s in range(4)])
    cmp, cost = paired_tables(whole, alg, training)
    assert (cost.n_network_rows == 4).all() and (cost.n_parameters == 1).all()
    np.testing.assert_allclose(cmp.net_over_alg1_eta, .25)
    np.testing.assert_allclose(cost.break_even, 10)
    np.testing.assert_allclose(cmp.uniform, .25/cmp.N)
    energy, orders, accuracy = energy_tables(whole)
    assert len(energy) == 15
    np.testing.assert_allclose(orders.order_32_64, 1)
    assert (accuracy.N_measured <= 64).all()


def test_reference_perturbation_brackets_measured_error():
    d = sample_rows().assign(method="uniform", ritz=lambda x: x.strain/2 - x.load)
    alg = d.assign(method="eta")
    df = precision_table(d, alg)
    unif = df[df.method == "uniform"]
    assert (unif.err_lo < unif.err_hi).all()
    assert (unif.improvement_lo <= 1).all() and (unif.improvement_hi >= 1).all()
    assert (df.ritz_discrepancy_max < 1e-15).all()
    assert df.loc[df.method == "alg1-eta", "improvement_lo"].isna().all()
