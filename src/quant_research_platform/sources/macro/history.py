"""Normalize saved HKMA raw pages and merge native macro observations."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
from quant_research_platform.sources.macro.interim import (
    DATA_ID,
    _frame,
    interim_catalog,
    resolve_source_conflicts,
    validate_observations,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset

_HK_FIELDS = {
    "hk_base_rate": ("Base Rate", "Today's Overnight HIBOR"),
    "hk_hibor_on": ("Today's Overnight HIBOR", "HKAB 1-Month HIBOR Fixing"),
    "hk_hibor_1m": ("HKAB 1-Month HIBOR Fixing", "TWI"),
    "hk_balance_open": ("Opening Aggregate Balance", "Closing Aggregate Balance"),
    "hk_balance_close": ("Closing Aggregate Balance", "Change Attributable to"),
}


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def _page_text(payload):
    parser = _PageText()
    parser.feed(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
    parser.close()
    return "\n".join(parser.parts)


def parse_hkma_daily_page(payload):
    """Read values and displayed section clocks; a missing number remains missing."""
    text = _page_text(payload)
    clocks = re.findall(
        r"Date and Time[^\d]*(\d{1,2}:\d{2}),\s*(\d{2}/\d{2}/\d{4})", text
    )
    times = [
        pd.to_datetime(f"{day} {clock}", format="%d/%m/%Y %H:%M")
        .tz_localize("Asia/Hong_Kong")
        .isoformat()
        for clock, day in clocks
    ]
    values = {}
    for name, (start, end) in _HK_FIELDS.items():
        if start not in text or end not in text.split(start, 1)[1]:
            continue
        section = text.split(start, 1)[1].split(end, 1)[0]
        # Some historical HIBOR cells omit the percent sign; balances retain mn.
        suffix = r"%?" if name in {"hk_hibor_on", "hk_hibor_1m"} else r"(?:%|\s*mn)"
        number = re.search(r"(?:^|\n)([\d,.]+)" + suffix + r"(?:\n|$)", section)
        if number:
            values[name] = float(number[1].replace(",", ""))
    return {"section_times": times, "values": values}


def prepare_hkma_history(config):
    """Normalize saved daily pages with the declared historical time assumption."""
    config = qrp_json_mapping_snapshot(config)
    if set(config) != {"raw_root", "output"}:
        raise ValueError("hkma_history_prepare_fields_invalid")
    raw_root = Path(config["raw_root"])
    manifest = json.loads((raw_root / "download.json").read_text(encoding="utf-8"))
    frames, statuses = [], {}
    for item in manifest:
        statuses[item["status"]] = statuses.get(item["status"], 0) + 1
        if item["status"] != "parsed":
            continue
        record = item["responses"][0]
        member = raw_root / record["file"]
        if not member.resolve().is_relative_to(raw_root.resolve()):
            raise ValueError("macro_source_member_outside_root")
        day = pd.Timestamp(item["observation_date"])
        parsed = parse_hkma_daily_page(member.read_bytes())
        if not parsed["values"] or not any(
            pd.Timestamp(clock).date() == day.date()
            for clock in parsed["section_times"]
        ):
            raise ValueError("hkma_daily_page_date_or_values_missing")
        for name in _HK_FIELDS:
            frame = _frame(
                name,
                [parsed["values"].get(name, float("nan"))],
                [day],
                record=record,
                timezone="Asia/Hong_Kong",
            )
            frame["source_id"] = "hkma_historical_page"
            frames.append(frame)
    if not frames:
        raise ValueError("macro_no_observations")
    output = Path(config["output"])
    write_dataset(
        output,
        validate_observations(pd.concat(frames, ignore_index=True)),
        spec=interim_catalog().get(DATA_ID),
        metadata={
            "inputs": {"raw_root": str(raw_root)},
            "parameters": {"availability": "inferred_historical_release"},
            "producer_diagnostics": {"page_statuses": statuses},
        },
    )
    return output


def merge_macro(config):
    """Merge explicit native snapshots using the existing source-priority rule."""
    config = qrp_json_mapping_snapshot(config)
    if (
        set(config) != {"inputs", "output"}
        or type(config["inputs"]) is not list
        or not config["inputs"]
        or any(type(path) is not str or not path for path in config["inputs"])
    ):
        raise ValueError("macro_merge_fields_invalid")
    frames = [read_dataset(path) for path in config["inputs"]]
    frame = resolve_source_conflicts(
        validate_observations(pd.concat(frames, ignore_index=True))
    )
    output = Path(config["output"])
    write_dataset(
        output,
        frame,
        spec=interim_catalog().get(DATA_ID),
        metadata={
            "inputs": {"source_datasets": config["inputs"]},
            "parameters": {"source_priority": "official_sources_then_latest_retrieval"},
            "producer_diagnostics": {
                "input_rows": sum(len(part) for part in frames),
                "output_rows": len(frame),
            },
        },
    )
    return output
