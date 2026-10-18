"""Real coverage readers retain gaps and ambiguity with local archive dispatch."""

from types import SimpleNamespace

import pytest

from quant_research_platform.sources.binance.replay import ObservationCoverage
from quant_research_platform.sources.tardis.replay import _ArchiveCoverage


def _reader(rows, *, invalid=None):
    class Index:
        _groups = {("futures", "agg_trades", "btcusdt"): rows}

        def trade_intervals(self, identity):
            for start, stop, key in rows:
                if identity == key:
                    if invalid is not None and start <= invalid < stop:
                        return (
                            (start, invalid, key),
                            (invalid + 1, stop, key + ":after"),
                        )
                    return ((start, stop, key),)
            return ()

    result = ObservationCoverage(Index(), "futures", "agg_trades", "btcusdt")
    original = result.covering
    result.calls = 0

    def counted(start, end):
        result.calls += 1
        return original(start, end)

    result.covering = counted
    return result


def _linear(readers, start, end):
    found = []
    for reader in readers:
        found.extend(reader.covering(start, end))
        if len(found) >= 2:
            break
    return tuple(found[:2])


@pytest.mark.parametrize("archives", [1, 3])
def test_archive_dispatch_is_local_for_monotone_queries(archives):
    rows = [[(i * 10, (i + 1) * 10, str(i))] for i in range(archives)]
    fast = [_reader(row) for row in rows]
    reference = [_reader(row) for row in rows]
    coverage = _ArchiveCoverage(tuple(fast))
    queries = [
        (index * 10 + offset, str(index))
        for index in range(archives)
        for offset in (0, 4, 8)
    ]
    for value, identity in queries:
        expected = (identity,)
        assert _linear(reference, value, value + 1) == expected
        assert coverage.covering(value, value + 1) == expected
    assert sum(reader.calls for reader in fast) == len(queries)


@pytest.mark.parametrize("invalid", [None, 12])
def test_archive_local_candidates_preserve_overlap_gap_midnight_and_invalid_trade(
    invalid,
):
    # Adjacent ranges can straddle midnight; internal holes and overlaps are real.
    rows = [[(0, 8, "a"), (10, 18, "b")], [(15, 22, "c")], [(22, 30, "d")], []]
    fast = _ArchiveCoverage(tuple(_reader(row, invalid=invalid) for row in rows))
    reference = [_reader(row, invalid=invalid) for row in rows]
    for stamp in range(32):
        assert fast.covering(stamp, stamp + 1) == _linear(reference, stamp, stamp + 1)
    with pytest.raises(ValueError, match="coverage_window_non_monotonic"):
        fast.covering(0, 1)


def test_archive_empty_coverage_does_not_dispatch():
    reader = SimpleNamespace(
        rows=(), covering=lambda *_: pytest.fail("empty source queried")
    )
    assert _ArchiveCoverage((reader,)).covering(0, 1) == ()
