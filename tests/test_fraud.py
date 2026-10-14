import numpy as np

from drm import fraud as F


def test_simulator_is_reproducible_and_rates_reasonable():
    a, b = F.simulate_orders(5000, seed=3), F.simulate_orders(5000, seed=3)
    assert a.equals(b)
    df = F.simulate_orders(40000, seed=0)
    cod = df[df.payment_method == "cod"]
    pre = df[df.payment_method != "cod"]
    assert 0.08 < cod.cod_refusal.mean() < 0.25
    assert 0.005 < pre.chargeback.mean() < 0.05
    assert (cod.chargeback == 0).all() and (pre.cod_refusal == 0).all()


def test_sim_tasks_have_no_label_leakage_columns():
    for t in F.sim_tasks(8000).values():
        cols = set(t.train.X.columns)
        assert not cols & {"cod_refusal", "chargeback", "_latent_kind"}


def test_cost_threshold_beats_no_action_on_informative_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    s = np.clip(0.6 * y + rng.normal(0.2, 0.15, 3000), 0, 1)
    amt = rng.lognormal(3.5, 0.8, 3000)
    fn, fp = F.COSTS["cod_refusal"]
    thr, c = F.best_cost_threshold(y, s, amt, fn, fp)
    assert c < F.expected_cost(y, s, np.inf, amt, fn, fp)


def test_precision_at_budget():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .1])
    p, r = F.precision_at_budget(y, s, 0.2)
    assert p == 0.5 and r == 0.5
