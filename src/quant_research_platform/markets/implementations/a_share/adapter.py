"""A-share canonical field adapter."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.catalog.fields import (
    canonical_field_names,
    platform_interim_column_names,
)


class AShareMarketAdapter:
    market_id = "a_share"

    def with_market_bars_1d_canonical_fields(
        self,
        frame: pd.DataFrame,
        *,
        source_id: str,
        source_asset_column: str,
        session_date_column: str,
        asset_namespace: str = "stock",
    ) -> pd.DataFrame:
        result = frame.copy()
        if source_asset_column not in result.columns:
            raise ValueError("missing_source_asset_id")
        if session_date_column not in result.columns:
            raise ValueError("missing_session_date")
        source_asset_id = self.source_asset_id(
            result[source_asset_column],
            asset_namespace=asset_namespace,
        )
        session_date = pd.to_datetime(result[session_date_column], errors="raise")
        timestamp = session_date.dt.normalize()
        self._set_canonical_columns(
            result,
            source_id=source_id,
            source_asset_id=source_asset_id,
            timestamp=timestamp,
            session_date=session_date.dt.normalize(),
            available_at=timestamp,
            asset_namespace=asset_namespace,
        )
        return result

    def with_metadata_canonical_fields(
        self,
        frame: pd.DataFrame,
        *,
        source_id: str,
        source_asset_column: str,
        time_columns: tuple[str, ...],
        asset_namespace: str = "stock",
    ) -> pd.DataFrame:
        result = frame.copy()
        if source_asset_column not in result.columns:
            raise ValueError("missing_source_asset_id")
        source_asset_id = self.source_asset_id(
            result[source_asset_column],
            asset_namespace=asset_namespace,
        )
        timestamp = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
        for column in time_columns:
            if column not in result.columns:
                continue
            parsed = pd.to_datetime(result[column], errors="coerce")
            result[column] = parsed
            timestamp = timestamp.combine_first(parsed)
        timestamp = timestamp.dt.normalize()
        self._set_canonical_columns(
            result,
            source_id=source_id,
            source_asset_id=source_asset_id,
            timestamp=timestamp,
            session_date=timestamp,
            available_at=timestamp,
            asset_namespace=asset_namespace,
        )
        return result

    def source_asset_id(
        self,
        values: pd.Series,
        *,
        asset_namespace: str,
    ) -> pd.Series:
        result = values.astype(str).str.strip()
        if asset_namespace == "stock":
            return result.str.zfill(6)
        return result

    def asset_id(
        self,
        source_asset_id: pd.Series,
        *,
        asset_namespace: str,
    ) -> pd.Series:
        if asset_namespace == "stock":
            return source_asset_id
        return asset_namespace + ":" + source_asset_id

    def _set_canonical_columns(
        self,
        frame: pd.DataFrame,
        *,
        source_id: str,
        source_asset_id: pd.Series,
        timestamp: pd.Series,
        session_date: pd.Series,
        available_at: pd.Series,
        asset_namespace: str,
    ) -> None:
        names = canonical_field_names()
        values: dict[str, object] = {
            names.market_id: self.market_id,
            names.asset_id: self.asset_id(
                source_asset_id,
                asset_namespace=asset_namespace,
            ),
            names.timestamp: timestamp,
            names.session_date: session_date,
            names.available_at: available_at,
            names.source_id: source_id,
            names.source_asset_id: source_asset_id,
        }
        for offset, column in enumerate(platform_interim_column_names()):
            if column in frame.columns:
                frame[column] = values[column]
            else:
                frame.insert(offset, column, values[column])


__all__ = ["AShareMarketAdapter"]
