from __future__ import annotations

import pytest

from quant_research_platform.policy.lifecycle import closing_iterator


def test_closing_iterator_propagates_close_failure_without_primary() -> None:
    class IteratorWithFailingClose:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self) -> None:
            raise OSError("close_failure")

    with pytest.raises(OSError, match="close_failure"):
        with closing_iterator(IteratorWithFailingClose()):
            pass


def test_closing_iterator_yields_exact_object_and_closes_once() -> None:
    class IteratorWithClose:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    iterator = IteratorWithClose()
    with closing_iterator(iterator) as active:
        assert active is iterator

    assert iterator.close_calls == 1


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, KeyboardInterrupt, GeneratorExit),
)
def test_closing_iterator_records_secondary_without_replacing_primary(
    primary_type: type[BaseException],
) -> None:
    class IteratorWithFailingClose:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("close_failure")

    iterator = IteratorWithFailingClose()
    with pytest.raises(primary_type) as raised:
        with closing_iterator(iterator) as active:
            assert active is iterator
            raise primary_type("primary_failure")

    assert iterator.close_calls == 1
    assert raised.value.__notes__ == ["iterator close also failed: OSError"]


def test_generator_close_surfaces_nested_resource_failure() -> None:
    class Resource:
        def close(self) -> None:
            raise OSError("close_failure")

    def inner():
        with closing_iterator(Resource()):
            try:
                yield 1
            except GeneratorExit:
                return

    def outer():
        with closing_iterator(inner()) as source:
            for value in source:
                try:
                    yield value
                except GeneratorExit:
                    return

    stream = outer()
    assert next(stream) == 1
    with pytest.raises(OSError, match="close_failure"):
        stream.close()
