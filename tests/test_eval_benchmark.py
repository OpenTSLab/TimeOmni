import csv
import json

import pytest
from sklearn.metrics import f1_score

from eval_benchmark import classification_eval, evaluate_all_files


def classification_rows(reference, predictions):
    return [
        {"gt_result": {"gt_class": {"default": label}}, "generated_text": prediction}
        for label, prediction in zip(reference, predictions)
    ]


@pytest.mark.parametrize(
    ("reference", "predictions", "accuracy", "uar", "macro_f1"),
    [
        pytest.param(
            ["ALPHA", "BETA"], ["ALPHA", "ALPHA"], 0.5, 0.5, 1 / 3,
            id="missed-reference-class",
        ),
        pytest.param(
            ["ALPHA", "BETA"], ["ALPHA", "BETA"], 1.0, 1.0, 1.0,
            id="both-correct",
        ),
        pytest.param(
            ["ALPHA", "BETA"], ["BETA", "ALPHA"], 0.0, 0.0, 0.0,
            id="both-incorrect",
        ),
        pytest.param(
            ["ALPHA", "ALPHA"], ["ALPHA", "GAMMA"], 0.5, 0.5, 2 / 3,
            id="one-reference-class",
        ),
        # ALPHA yields (TP, FP, FN) = (2, 1, 1). BETA yields (0, 1, 1).
        pytest.param(
            ["ALPHA", "ALPHA", "ALPHA", "BETA"], ["ALPHA", "BETA", "ALPHA", "ALPHA"],
            0.5, 1 / 3, 1 / 3, id="unequal-support-with-zero-true-positives",
        ),
        pytest.param(
            ["ALPHA", "BETA", "ALPHA"], ["ALPHA", "GAMMA", "ALPHA"],
            2 / 3, 0.5, 0.5, id="prediction-only-class",
        ),
    ],
)
def test_single_label_scores(reference, predictions, accuracy, uar, macro_f1):
    reference_f1 = f1_score(
        reference, predictions, average="macro", labels=sorted(set(reference))
    )
    assert reference_f1 == pytest.approx(macro_f1)
    assert classification_eval(classification_rows(reference, predictions)) == pytest.approx(
        {"accuracy": accuracy, "uar": uar, "f1_score": macro_f1}
    )


def test_multilabel_score_includes_missed_label():
    rows = classification_rows([["ALPHA", "BETA"], ["ALPHA"]], ["ALPHA", "ALPHA"])
    assert f1_score([[1, 1], [1, 0]], [[1, 0], [1, 0]], average="macro") == 0.5
    assert classification_eval(rows) == pytest.approx(
        {"accuracy": 0.75, "uar": 0.5, "f1_score": 0.5}
    )


def test_empty_input_returns_no_scores():
    assert classification_eval([]) == {}


@pytest.mark.parametrize("task", ["classification", "qa"])
def test_evaluation_reports_macro_f1_in_csv(tmp_path, task):
    input_path = tmp_path / "predictions.jsonl"
    with input_path.open("w", encoding="utf-8") as stream:
        for label in ("ALPHA", "BETA"):
            truth = {"answer": label} if task == "qa" else {"gt_class": {"default": label}}
            row = {"task": task, "gt_result": truth, "generated_text": "ALPHA"}
            stream.write(json.dumps(row) + "\n")

    output = tmp_path / "metrics.csv"
    evaluate_all_files(str(tmp_path), str(output))

    with output.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [
            {"filename": input_path.name, "accuracy": "50.00", "uar": "50.00", "f1_score": "33.33"}
        ]
