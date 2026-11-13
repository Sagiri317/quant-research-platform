"""Read consumed source columns without pandas' duplicate-header renaming."""

import pandas as pd

from quant_research_platform.policy.lifecycle import closing_iterator


def _read_table(reader, source, columns, optional_columns, error, options):
    strip_headers = options.pop("strip_headers", False)
    position = source.tell() if hasattr(source, "tell") else None
    header_row = options.get("header", 0)
    header_options = {
        name: options[name] for name in ("sep", "encoding") if name in options
    }
    header = (
        reader(
            source,
            header=None,
            nrows=header_row + 1,
            dtype="str",
            keep_default_na=False,
            **header_options,
        )
        .iloc[header_row]
        .tolist()
    )
    if position is not None:
        source.seek(position)
    raw_header = header
    if strip_headers:
        header = [name.strip() for name in header]
    selected = set(columns) | (set(optional_columns) & set(header))
    if any(header.count(column) != 1 for column in selected):
        raise ValueError(error)
    actual = [
        raw for raw, name in zip(raw_header, header, strict=True) if name in selected
    ]
    result = reader(source, usecols=actual, **options)
    return result.rename(columns=str.strip) if strip_headers else result


def read_csv(
    source, columns, *, optional_columns=(), error="source_columns_invalid", **options
):
    if "chunksize" in options:

        def frames():
            with closing_iterator(source):
                with _read_table(
                    pd.read_csv, source, columns, optional_columns, error, options
                ) as chunks:
                    yield from chunks

        return frames()
    with closing_iterator(source):
        return _read_table(
            pd.read_csv, source, columns, optional_columns, error, options
        )


def read_excel(
    source, columns, *, optional_columns=(), error="source_columns_invalid", **options
):
    with closing_iterator(source):
        return _read_table(
            pd.read_excel, source, columns, optional_columns, error, options
        )
