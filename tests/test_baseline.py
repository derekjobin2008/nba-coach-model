import numpy as np
import pytest

from coachmodel.baseline import TASKS, build_model, calibration_table, evaluate, load_features


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_label_is_not_a_feature(task_name):
    task = TASKS[task_name]
    assert task.label not in task.features


def test_no_same_stoppage_bench_decision_is_a_feature():
    # Whether the coach called a timeout or made a sub at this stoppage is decided at
    # the same moment as the label, so it must never be an input.
    for task in TASKS.values():
        assert not {"called_timeout", "subbed_out"} & set(task.features)


def test_calibration_table_matches_a_calibrated_predictor():
    # Labels drawn with exactly the predicted probability: each bin's observed rate
    # should land close to its average prediction.
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0, 0.3, 200_000)
    labels = rng.random(200_000) < predicted
    table = calibration_table(labels.astype(int), predicted)
    assert len(table) == 10
    assert np.allclose(table["observed_rate"], table["mean_predicted"], atol=0.01)


def test_constant_prediction_has_no_advantage():
    labels = np.array([0, 0, 0, 1] * 25)
    results = evaluate(labels, labels, np.full(len(labels), 0.25))
    assert np.isclose(results["model_log_loss"], results["constant_log_loss"])


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_model_trains_on_fixture_games(database, task_name):
    task = TASKS[task_name]
    frame = load_features(database, task)
    assert len(frame) > 0 and frame[task.features].notna().all().all()
    assert set(frame[task.label].unique()) == {0, 1}
    predicted = build_model(task).fit(frame, frame[task.label]).predict_proba(frame)[:, 1]
    assert ((predicted > 0) & (predicted < 1)).all()
