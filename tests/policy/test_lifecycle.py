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
    primary = primary_type("primary_failure")
    with pytest.raises(primary_type) as raised:
        with closing_iterator(iterator) as active:
            assert active is iterator
            raise primary

    assert raised.value is primary
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


@pytest.mark.parametrize(
    "primary_type", (RuntimeError, KeyboardInterrupt, GeneratorExit)
)
def test_nested_closing_iterators_preserve_primary_and_each_secondary(primary_type):
    closed = []

    class Resource:
        def __init__(self, name, failure):
            self.name, self.failure = name, failure

        def close(self):
            closed.append(self.name)
            raise self.failure

    primary = primary_type("consumer_failure")
    with pytest.raises(primary_type) as caught:
        with closing_iterator(Resource("outer", ValueError("outer_close"))):
            with closing_iterator(Resource("inner", OSError("inner_close"))):
                raise primary
    assert caught.value is primary
    assert closed == ["inner", "outer"]
    assert primary.__notes__ == [
        "iterator close also failed: OSError",
        "iterator close also failed: ValueError",
    ]
