"""Tests for the follow-up experiments: FI-2010 history features, BTC tradability, C++ codegen."""

import sys
from pathlib import Path

import numpy as np

from src.experiments.fi2010_benchmark import history_features, HISTORY_LAGS
from src.experiments.exp_e_tradability import cost_aware_labels, round_trips


def test_history_features_use_only_the_past():
    rng = np.random.default_rng(0)
    lob = rng.normal(size=(300, 40)).astype(np.float32)
    h = history_features(lob)
    assert h.shape == (300, 40 * len(HISTORY_LAGS))
    lob2 = lob.copy()
    lob2[200:] += 100.0  # change the future
    assert np.allclose(history_features(lob2)[:200], h[:200])
    # lag-1 block is the one-step change
    assert np.allclose(h[5, :40], lob[5] - lob[4])


def _book(mids, spread=0.02, n_levels=10):
    lob = np.zeros((len(mids), n_levels * 4))
    for lvl in range(n_levels):
        lob[:, lvl * 4] = np.asarray(mids) + spread / 2 + 0.01 * lvl
        lob[:, lvl * 4 + 1] = 10.0
        lob[:, lvl * 4 + 2] = np.asarray(mids) - spread / 2 - 0.01 * lvl
        lob[:, lvl * 4 + 3] = 10.0
    return lob


def test_cost_aware_labels_flag_only_moves_that_beat_fees():
    # mid jumps +1% after row 3: buying at the ask then selling at the bid 2 rows later beats 4 bp fees
    mids = [100.0] * 4 + [101.0] * 6
    lob, ts = _book(mids), np.arange(10) * 1000.0
    y = cost_aware_labels(lob, ts, k=2, fee_bps=4.0)
    assert y[1] == 1 and y[2] == 1         # entry at row 2/3 (next snapshot), exit at 4/5 after the jump
    assert (y[4:7] == 0).all()             # flat price afterwards: spread + fees make it unprofitable
    # a tiny move does not clear the fees
    lob_small = _book([100.0] * 4 + [100.005] * 6)
    assert (cost_aware_labels(lob_small, ts, k=2, fee_bps=4.0) == 0).all()


def test_round_trips_do_not_overlap():
    lob, ts = _book([100.0] * 50), np.arange(50) * 1000.0
    trips = round_trips(lob, ts, rows=np.arange(0, 40), sides=np.ones(40, dtype=int), k=5, lot=1.0)
    # each trip enters at t+1 and exits at t+5+1, so a new trip can start only every ~6 rows
    assert 5 <= len(trips) <= 7
    assert np.allclose(trips.gross_bps, -0.02 / 100.01 * 1e4)  # pays exactly the spread on a flat book


def test_cpp_codegen_matches_lightgbm():
    import lightgbm as lgb
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from benchmark_trained_model import tree_to_cpp

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) + (X[:, 2] > 1).astype(int)
    booster = lgb.LGBMClassifier(n_estimators=5, num_leaves=7, verbosity=-1).fit(X, y).booster_
    dump = booster.dump_model()

    # Evaluate the generated C++ source as Python to check the tree logic without a compiler
    def as_python(node):
        if "leaf_value" in node:
            return repr(node["leaf_value"])
        return (f"({as_python(node['left_child'])} if float(x[{node['split_feature']}]) <= {node['threshold']!r} "
                f"else {as_python(node['right_child'])})")

    n_class = dump["num_class"]
    raw = booster.predict(X[:50], raw_score=True)
    exprs = [as_python(t["tree_structure"]) for t in dump["tree_info"]]
    for i, row in enumerate(X[:50]):
        env = {"x": row}
        scores = [sum(eval(e, env) for e in exprs[c::n_class]) for c in range(n_class)]
        assert np.allclose(scores, raw[i], atol=1e-9)
    # and the C++ text contains one comparison per internal node
    code = tree_to_cpp(dump["tree_info"][0]["tree_structure"])
    assert code.count("if (") == dump["tree_info"][0]["num_leaves"] - 1
