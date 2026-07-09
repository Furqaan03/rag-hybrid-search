from src.eval.metrics import hit_rate, mrr, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=3) == 0.5


def test_mrr_first_hit_rank():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5   # first relevant at rank 2
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_hit_rate():
    assert hit_rate(["a", "b"], {"b"}, k=2) == 1.0
    assert hit_rate(["a", "b"], {"z"}, k=2) == 0.0
