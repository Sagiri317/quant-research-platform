from __future__ import annotations


def unsafe_assignment_text() -> str:
    key = "".join(chr(code) for code in (116, 111, 107, 101, 110))
    return f'credential = "{key}=unsafe"\n'
