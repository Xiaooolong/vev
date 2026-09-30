import torch

from train.data import micro_batches


def _row(n: int, i: int) -> dict:
    return {"input_ids": torch.full((n,), i, dtype=torch.long), "attention_mask": torch.ones(n, dtype=torch.long),
            "decide": n - 1, "target": torch.tensor([1.0]), "qtype": "noul", "id": str(i)}


def test_micro_batches_respect_token_budget_and_keep_every_row():
    rows = [_row(n, i) for i, n in enumerate([10, 50, 20, 40, 30, 60, 5, 45])]
    out = list(micro_batches(iter(rows), micro_tokens=100, pad_id=0, buffer=4))
    seen = []
    for batch, decide, items in out:
        L = batch["input_ids"].shape[1]
        assert L * len(items) <= 100 or len(items) == 1
        assert len(decide) == len(items) == batch["input_ids"].shape[0]
        assert batch["attention_mask"].sum(1).tolist() == [int(r["input_ids"].shape[0]) for r in items]
        seen += [r["id"] for r in items]
    assert sorted(seen) == sorted(r["id"] for r in rows)


def test_micro_batches_single_oversized_row_still_yields():
    rows = [_row(500, 0)]
    out = list(micro_batches(iter(rows), micro_tokens=100, pad_id=0))
    assert len(out) == 1 and len(out[0][2]) == 1
