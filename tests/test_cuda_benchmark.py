import argparse

import pytest

from actserve.cuda_benchmark import _parse_batch_sizes, _parse_batch_waits


def test_batch_size_sweep_is_sorted_and_deduplicated() -> None:
    assert _parse_batch_sizes("16, 4,8,16", 2) == [4, 8, 16]


@pytest.mark.parametrize("value", ["", "0,8", "four,8"])
def test_batch_size_sweep_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_batch_sizes(value, 8)


def test_batch_wait_sweep_is_sorted_and_deduplicated() -> None:
    assert _parse_batch_waits("2,0,0.5,2", 4) == [0.0, 0.5, 2.0]


def test_batch_wait_sweep_uses_single_default() -> None:
    assert _parse_batch_waits(None, 2.0) == [2.0]


@pytest.mark.parametrize("value", ["", "-0.1,2", "fast,2"])
def test_batch_wait_sweep_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_batch_waits(value, 2.0)
