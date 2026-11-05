"""Read the official ChinaBond daily yield table without browser dependencies."""

from html.parser import HTMLParser

import pandas as pd


class _YieldTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and len(self.row) == 10:
            self.rows.append(self.row)


def yield_table(payload):
    parser = _YieldTable()
    parser.feed(payload.decode("utf-8"))
    header = [
        "曲线名称",
        "日期",
        "3月",
        "6月",
        "1年",
        "3年",
        "5年",
        "7年",
        "10年",
        "30年",
    ]
    if header not in parser.rows:
        raise ValueError("chinabond_yield_table_missing")
    rows = parser.rows[parser.rows.index(header) + 1 :]
    return pd.DataFrame(rows, columns=header).replace({"": None, "--": None})
