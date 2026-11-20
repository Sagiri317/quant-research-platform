"""Read consumed source columns without pandas' duplicate-header renaming."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas._libs.parsers import STR_NA_VALUES

from quant_research_platform.policy.lifecycle import closing_iterator


def select_source_columns(
    header, required, optional=(), *, error="source_columns_invalid"
):
    """Require exactly one field for every consumed name in the original header."""
    selected = set(required) | (set(optional) & set(header))
    if any(header.count(column) != 1 for column in selected):
        raise ValueError(error)
    return [column for column in header if column in selected]


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
    selected = select_source_columns(header, columns, optional_columns, error=error)
    actual = selected
    if strip_headers:
        by_name = dict(zip(header, raw_header, strict=True))
        actual = [by_name[column] for column in selected]
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


def read_parquet(
    path, metadata, *, columns, optional_columns=(), chunksize=None, na_values=()
):
    selected = select_source_columns(metadata["columns"], columns, optional_columns)

    def frame(table):
        # Preserve CSV/TSV missing markers on consumed string columns.
        return (
            table.to_pandas()
            .astype("str")
            .replace([*STR_NA_VALUES, *na_values], np.nan)
        )

    if chunksize is None:
        with pq.ParquetFile(path) as source:
            return frame(
                source.read_row_groups(metadata["row_groups"], columns=selected)
            )

    def frames():
        with pq.ParquetFile(path) as source:
            for batch in source.iter_batches(
                batch_size=chunksize,
                row_groups=metadata["row_groups"],
                columns=selected,
            ):
                yield frame(batch)

    return frames()
