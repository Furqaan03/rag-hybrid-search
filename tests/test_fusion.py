from src.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_ranks_highest():
    dense = ["a", "b", "c"]
    sparse = ["a", "d", "e"]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert fused[0][0] == "a"  # appears top of both -> highest fused score


def test_dense_weight_dominates_when_higher():
    dense = ["x", "y"]
    sparse = ["y", "x"]
    fused = reciprocal_rank_fusion(dense, sparse, dense_weight=0.9, sparse_weight=0.1)
    assert fused[0][0] == "x"  # x is #1 in the heavily-weighted dense list


def test_all_ids_preserved():
    fused = reciprocal_rank_fusion(["a", "b"], ["c", "d"])
    assert {cid for cid, _ in fused} == {"a", "b", "c", "d"}
