import numpy as np
import pandas as pd
import pytest

from boas_unc import abstain as ab
from boas_unc import learned_abstain as la

A = np.array
W, N1, N2, N3, R, X = 0, 1, 2, 3, 4, -2


def test_lagged_and_edges():
    hb = A([N2, N3, R, W])
    assert la.lagged(hb, 0).tolist() == [N2, N3, R, W]
    assert la.lagged(hb, -1).tolist() == [la.EDGE, N2, N3, R]
    assert la.lagged(hb, 2).tolist() == [R, W, la.EDGE, la.EDGE]


def test_one_hot_has_one_column_per_category_and_edge():
    out = la.one_hot(A([W, X, la.EDGE, R]))
    assert out.shape == (4, 7) and out.sum(axis=1).tolist() == [1, 1, 1, 1]
    assert out[1, 5] == 1 and out[2, 6] == 1


def test_change_epochs_marks_where_a_new_stage_begins():
    assert la.change_epochs(A([N2, N2, N3, N3, X, N3])).tolist() == [False, False, True, False, False, False]


def test_capped_replaces_infinity_and_limits():
    assert la.capped(A([0.0, 3.0, 50.0, np.inf]), 10).tolist() == [0.0, 3.0, 10.0, 10.0]


def test_window_share_and_count_with_clipped_edges():
    x = A([False, True, False, False, True])
    assert la.window_share(x, 1, 1, as_count=True).tolist() == [1, 1, 1, 1, 1]
    assert la.window_share(x, 1, 0).tolist() == [0.0, 0.5, 0.5, 0.0, 0.5]


def test_feature_counts():
    hb, onset = A([N2] * 12), np.arange(12) * 30
    assert la.night_features(hb, onset, True).shape == (12, 33)
    assert la.night_features(hb, onset, False).shape == (12, 56)


def test_causal_features_never_look_at_future_labels():
    rng = np.random.default_rng(0)
    hb = rng.choice([W, N1, N2, N3, R, X], 40)
    onset = np.arange(40) * 30
    changed = hb.copy()
    changed[20:] = rng.choice([W, N1, N2, N3, R, X], 20)
    before, after = la.night_features(hb, onset, True), la.night_features(changed, onset, True)
    assert np.array_equal(before[:20], after[:20])  # rows up to the change are identical
    nc_before, nc_after = la.night_features(hb, onset, False), la.night_features(changed, onset, False)
    assert not np.array_equal(nc_before[:20], nc_after[:20])  # the non-causal set does use the future


def test_features_use_only_the_headband_labels_and_the_clock():
    import inspect
    assert list(inspect.signature(la.night_features).parameters) == ["hb", "onset", "causal"]


def test_oof_scores_never_score_a_person_the_model_trained_on():
    class Stub:
        def fit(self, X, y):
            self.seen = set(X[:, 0])
            return self

        def predict_proba(self, X):
            assert not (set(X[:, 0]) & self.seen)  # column 0 is the person's fingerprint
            return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])

    groups = np.repeat(np.arange(10), 6)
    X = np.column_stack([groups, np.arange(60)]).astype(float)
    scores = la.oof_scores(X, np.zeros(60, dtype=int), groups, Stub, n_splits=5)
    assert np.allclose(scores, 0.6)


def test_oof_scores_cover_every_row_with_probabilities():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 20)
    X, y = rng.normal(size=(200, 3)), rng.integers(0, 2, 200)
    scores = la.oof_scores(X, y, groups, la.MODELS["logistic"], n_splits=5)
    assert not np.isnan(scores).any() and scores.min() >= 0 and scores.max() <= 1


def test_kept_mask_keeps_the_lowest_scores_and_breaks_ties_stably():
    scores = A([0.9, 0.1, 0.5, 0.5, 0.2])
    assert la.kept_mask(scores, 0.6).tolist() == [False, True, True, False, True]
    assert la.kept_mask(scores, 1.0).all() and la.kept_mask(scores, 0.2).sum() == 1


def test_risk_coverage_with_informative_and_with_perfect_scores():
    error = A([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    perfect = error.astype(float)
    curve = la.risk_coverage(perfect, error, [1.0, 0.8, 0.5])
    assert curve[0]["accuracy"] == pytest.approx(0.8)  # coverage 1.0 is the baseline accuracy
    assert curve[1]["accuracy"] == 1.0 and curve[2]["accuracy"] == 1.0
    assert la.accuracy_at_coverage(perfect, error, 1.0) == pytest.approx(0.8)
    assert la.accuracy_at_coverage(np.zeros(10), error, 0.5) == pytest.approx(1 - error[:5].mean())  # stable ties


def test_per_night_at_coverage_summarises_both_quantities():
    meta = pd.DataFrame({"sub": ["a"] * 20 + ["b"] * 20, "error": [False] * 38 + [True, True]})
    scores = np.concatenate([np.zeros(20), np.r_[np.zeros(18), [1.0, 1.0]]])
    out = la.per_night_at_coverage(meta, scores, 0.95)
    assert out["coverage"]["n_defined"] == 2 and out["accuracy_kept"]["median"] == 1.0


def test_evaluate_model_compares_with_each_hand_made_flag():
    meta = pd.DataFrame({"sub": ["a"] * 12 + ["b"] * 12, "error": [False] * 20 + [True] * 4})
    scores = meta.error.to_numpy().astype(float)
    flags = {n: {"coverage": 0.8, "accuracy_kept": 0.9, "on_device": True} for n in
             la.CONTEXT_FLAGS + ["combined_1", "change_2", "change_5"]}
    out = la.evaluate_model(scores, meta, flags)
    assert out["auroc"] == 1.0
    assert out["versus_hand_made_flags"]["change_1"]["learned_accuracy"] == 1.0
    assert out["versus_hand_made_flags"]["change_1"]["difference"] == pytest.approx(0.1)


def make_table(n_nights=12, n_epochs=70, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_nights):
        hum = rng.choice([W, N1, N2, N3, R], n_epochs, p=[0.15, 0.1, 0.5, 0.1, 0.15])
        hb = np.where(rng.random(n_epochs) < 0.15, rng.choice([W, N1, N2, N3, R, X], n_epochs), hum)
        parts.append(pd.DataFrame({"sub": f"sub-{i}", "pid": str(i // 2), "epoch": range(n_epochs),
                                   "onset": np.arange(n_epochs) * 30, "psg_hum": hum, "psg_ai": hum, "hb_ai": hb,
                                   "hum_disconnected": False}))
    return pd.concat(parts, ignore_index=True)


def test_build_dataset_targets_and_shapes():
    table = make_table(3, 30)
    data = la.build_dataset(table)
    assert data["causal"].shape == (90, 33) and data["non_causal"].shape == (90, 56)
    meta = data["meta"]
    assert (meta.error <= meta.base).all() and meta.base.sum() == both_stage_count(table)


def both_stage_count(table):
    return int((table.psg_hum.isin(range(5)) & table.hb_ai.isin(range(5))).sum())


def test_run_on_a_small_table_returns_every_section():
    table = make_table()
    flags = ab.evaluate_flags(ab.add_flags(table))
    out = la.run(table, flags)
    assert set(out["feature_sets"]) == {"causal", "non_causal"}
    for fs in out["feature_sets"].values():
        assert set(fs["models"]) == set(la.MODELS)
        for model in fs["models"].values():
            assert 0 <= model["auroc"] <= 1 and model["curve"][0]["coverage"] == 1.0
            assert model["curve"][0]["accuracy"] == pytest.approx(out["baseline_accuracy"])
    assert set(out["flag_auroc_hand_made"]) == set(la.CONTEXT_FLAGS)
