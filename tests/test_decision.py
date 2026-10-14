import numpy as np
import pandas as pd

from drm import decision as D


def _cases(n=2000, seed=0, noise=0.15):
    rng = np.random.default_rng(seed)
    toxic = (rng.random(n) < 0.1).astype(int)
    spam = (rng.random(n) < 0.05).astype(int)
    risky = (rng.random(n) < 0.12).astype(int)

    def prob(y):
        return np.clip(np.where(y == 1, 0.7, 0.1) + rng.normal(0, noise, n), 0.001, 0.999)

    return pd.DataFrame({"toxic": toxic, "spam": spam, "risky": risky, "p_toxic": prob(toxic),
                         "p_spam": prob(spam), "p_risky": prob(risky), "amount": rng.lognormal(3.5, 0.8, n)})


def test_oracle_is_lower_bound():
    c, cost = _cases(), D.CostModel()
    o = D.evaluate(c, D.policy_oracle(c, 0, cost), cost)["cost_per_1k_cases"]
    for acts in [D.policy_rules(c, 100), D.policy_bayes(c, 100, cost),
                 D.policy_mcdm(c, 100, {"p_toxic": .4, "p_spam": .2, "p_risky": .3, "amount": .1})]:
        assert D.evaluate(c, acts, cost)["cost_per_1k_cases"] >= o - 1e-9


def test_bayes_with_true_probabilities_matches_oracle():
    c, cost = _cases(), D.CostModel()
    perfect = c.assign(p_toxic=c.toxic.astype(float), p_spam=c.spam.astype(float), p_risky=c.risky.astype(float))
    b = D.evaluate(perfect, D.policy_bayes(perfect, 0, cost), cost)["cost_per_1k_cases"]
    o = D.evaluate(perfect, D.policy_oracle(perfect, 0, cost), cost)["cost_per_1k_cases"]
    assert abs(b - o) < 1e-9


def test_capacity_respected():
    c, cost = _cases(), D.CostModel()
    for acts in [D.policy_rules(c, 50), D.policy_bayes(c, 50, cost), D.policy_threshold(c, 50, 0.3, 0.9)]:
        assert (acts == "review").sum() <= 50


def test_risk_coverage_monotone_end():
    rc = D.risk_coverage(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.4, 0.6]))
    assert rc["coverage"].iloc[-1] == 1.0 and abs(rc["error"].iloc[-1] - 0.5) < 1e-9
