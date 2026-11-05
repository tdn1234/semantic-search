import pytest

from evaluate import precision_at_k, recall_at_k


def test_recall_at_1():
    retrieved = ["A", "X", "B"]
    relevant = ["A", "B"]

    assert recall_at_k(retrieved, relevant, 1) == 0.5


def test_recall_at_3():
    retrieved = ["A", "X", "B"]
    relevant = ["A", "B"]

    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_precision_at_1():
    retrieved = ["A", "X", "B"]
    relevant = ["A", "B"]

    assert precision_at_k(retrieved, relevant, 1) == 1.0


def test_precision_at_3():
    retrieved = ["A", "X", "B"]
    relevant = ["A", "B"]

    assert precision_at_k(
        retrieved,
        relevant,
        3,
    ) == pytest.approx(2 / 3)


def test_no_relevant_products_gives_zero_recall():
    assert recall_at_k(["A"], [], 1) == 0.0


def test_no_matches_gives_zero_precision():
    assert precision_at_k(
        ["X", "Y", "Z"],
        ["A"],
        3,
    ) == 0.0