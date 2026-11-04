"""Download explicit public sources into new, independently published snapshots."""

import hashlib
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
from quant_research_platform.sources.macro.catalog import SERIES
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
    reject_if_exists,
)

HKMA_URL = (
    "https://api.hkma.gov.hk/public/market-data-and-statistics/"
    "daily-monetary-statistics/daily-figures-interbank-liquidity"
)
EASTMONEY_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _response(client, root, records, name, url, *, method="GET", **kwargs):
    response = client.request(method, url, **kwargs)
    # Preserve the received payload, including a publisher's error page.
    (root / name).write_bytes(response.content)
    records.append(
        {
            "file": name,
            "url": str(response.url),
            "method": method,
            "status_code": response.status_code,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "bytes": len(response.content),
            **({"form": kwargs["data"]} if "data" in kwargs else {}),
        }
    )
    response.raise_for_status()
    return response


def _alfred(client, root, records, series, start, end):
    url = f"https://alfred.stlouisfed.org/series/downloaddata?seid={series.code}"
    form = _response(client, root, records, "form.html", url)
    options = re.search(
        r'<select[^>]*id="form_selected_vintage_dates"[^>]*>(.*?)</select>',
        form.text,
        re.S,
    )
    if options is None:
        raise ValueError("alfred_vintage_form_missing")
    bounds = {}
    for field in ("start", "end"):
        match = re.search(
            rf'<input[^>]*id="form_obs_{field}_date"[^>]*value="(\d{{4}}-\d{{2}}-\d{{2}})"',
            form.text,
        )
        if match is None:
            raise ValueError("alfred_observation_range_missing")
        bounds[field] = match[1]
    observation_start, observation_end = (
        max(start, bounds["start"]),
        min(end, bounds["end"]),
    )
    if observation_start > observation_end:
        raise ValueError("alfred_no_observations_in_range")
    dates = [
        value
        for value in re.findall(r'value="(\d{4}-\d{2}-\d{2})"', options[1])
        if value >= start
    ]
    if not dates:
        raise ValueError("alfred_no_vintages_in_range")
    # These are the publisher's download limits, not local resource limits.
    batch_size = 450 if series.frequency == "D" else 1000
    for offset in range(0, len(dates), batch_size):
        payload = {
            "form[units]": "lin",
            "form[obs_start_date]": observation_start,
            "form[obs_end_date]": observation_end,
            "form[selected_vintage_dates][]": dates[offset : offset + batch_size],
            "form[entered_vintage_dates]": "",
            "form[file_type]": "1",
            "form[file_format]": "csv",
            "form[download_data]": "",
        }
        response = _response(
            client,
            root,
            records,
            f"vintages-{offset:05d}.zip",
            url,
            method="POST",
            data=payload,
        )
        if not zipfile.is_zipfile(io.BytesIO(response.content)):
            raise ValueError("alfred_download_not_zip")


def _hkma(client, root, records, start, end):
    for year in range(int(start[:4]), int(end[:4]) + 1):
        offset = 0
        while True:
            response = _response(
                client,
                root,
                records,
                f"page-{year}-{offset:05d}.json",
                HKMA_URL,
                params={
                    "offset": offset,
                    "pagesize": 1000,
                    "sortby": "end_of_date",
                    "sortorder": "asc",
                    "choose": "end_of_date",
                    "from": max(start, f"{year}-01-01"),
                    "to": min(end, f"{year}-12-31"),
                },
            )
            payload = response.json()
            if not payload["header"]["success"]:
                raise ValueError("hkma_request_failed")
            rows = payload["result"]["records"]
            offset += len(rows)
            if len(rows) < 1000:
                break


def _eastmoney(client, root, records):
    page = 1
    while True:
        response = _response(
            client,
            root,
            records,
            f"page-{page:04d}.json",
            EASTMONEY_URL,
            params={
                "reportName": "RPT_ECONOMY_CURRENCY_SUPPLY",
                "columns": "ALL",
                "pageSize": 500,
                "pageNumber": page,
                "sortColumns": "REPORT_DATE",
                "sortTypes": "1",
            },
        )
        payload = response.json()
        if not payload["success"]:
            raise ValueError("eastmoney_request_failed")
        if page >= payload["result"]["pages"]:
            break
        page += 1


def fetch_macro(config):
    """Fetch selected histories; independent source failures remain in the report."""
    config = qrp_json_mapping_snapshot(config)
    required = {"raw_root", "start_date", "end_date", "series_ids", "workers"}
    if not required <= config.keys() or config.keys() - required - {"timeout_seconds"}:
        raise ValueError("macro_fetch_fields_invalid")
    start, end = (date.fromisoformat(config[key]) for key in ("start_date", "end_date"))
    selected = config["series_ids"]
    if (
        start > end
        or type(selected) is not list
        or not selected
        or any(type(item) is not str or item not in SERIES for item in selected)
        or len(set(selected)) != len(selected)
        or type(config["workers"]) is not int
        or config["workers"] < 1
    ):
        raise ValueError("macro_fetch_scope_invalid")
    timeout = config.get("timeout_seconds", 60)
    if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
        raise ValueError("macro_fetch_timeout_invalid")
    root = Path(config["raw_root"])
    reject_if_exists(root)
    root.mkdir(parents=True)
    jobs = {}
    for name in selected:
        series = SERIES[name]
        key = name if series.source == "alfred" else series.source
        jobs.setdefault(key, []).append(name)

    def fetch(job):
        key, names = job
        series = SERIES[names[0]]
        report = {
            "series_ids": names,
            "source": series.source,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }

        def write(staging):
            records = []
            try:
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    if series.source == "alfred":
                        _alfred(client, staging, records, series, str(start), str(end))
                    elif series.source == "hkma":
                        _hkma(client, staging, records, str(start), str(end))
                    elif series.source == "eastmoney":
                        _eastmoney(client, staging, records)
                    elif series.source == "boj":
                        response = _response(
                            client,
                            staging,
                            records,
                            "observations.json",
                            "https://www.stat-search.boj.or.jp/api/v1/getDataCode",
                            params={
                                "db": "MD08",
                                "format": "json",
                                "lang": "en",
                                "code": series.code,
                                "startDate": start.strftime("%Y%m"),
                                "endDate": end.strftime("%Y%m"),
                            },
                        )
                        if response.json()["STATUS"] != 200:
                            raise ValueError("boj_request_failed")
                    elif series.source == "sf_fed":
                        _response(
                            client,
                            staging,
                            records,
                            "observations.csv",
                            "https://www.frbsf.org/wp-content/uploads/chart1-monetary-policy-surprises.csv",
                        )
                    elif series.source == "bis":
                        _response(
                            client,
                            staging,
                            records,
                            "observations.xml",
                            f"https://stats.bis.org/api/v1/data/WS_GLI/{series.code}",
                            params={
                                "startPeriod": str(start.year),
                                "endPeriod": str(end.year),
                            },
                        )
                    else:
                        raise ValueError("explicit_release_file_required")
                report["status"] = "downloaded"
            except (httpx.HTTPError, ValueError, KeyError) as error:
                report["status"] = "unavailable"
                report["reason"] = (
                    str(error) if type(error) is ValueError else type(error).__name__
                )
            report["responses"] = records
            atomic_write_json(staging / "source.json", report)

        atomic_write_directory(root / key, write)
        return key, report

    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        results = dict(pool.map(fetch, jobs.items()))
    atomic_write_json(root / "download.json", {"request": config, "sources": results})
    return root
