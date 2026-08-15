import argparse

import pytest

from actserve.cuda_benchmark import _parse_batch_sizes


def test_batch_size_sweep_is_sorted_and_deduplicated() -> None:
    assert _parse_batch_sizes("16, 4,8,16", 2) == [4, 8, 16]


@pytest.mark.parametrize("value", ["", "0,8", "four,8"])
def test_batch_size_sweep_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_batch_sizes(value, 8)
