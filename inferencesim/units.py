"""Unit constants and human-readable formatting helpers.

All internal quantities are plain SI floats: bytes, bytes/s, FLOP/s,
seconds, watts, USD.  These constants exist so preset files read like
spec sheets.
"""

from __future__ import annotations

# Decimal multipliers (spec sheets use decimal units for bandwidth/FLOPs).
KILO = 1e3
MEGA = 1e6
GIGA = 1e9
TERA = 1e12
PETA = 1e15

KB = 1e3
MB = 1e6
GB = 1e9
TB = 1e12

# Binary capacities, occasionally useful for SRAM sizes.
KiB = 2**10
MiB = 2**20
GiB = 2**30

US = 1e-6
MS = 1e-3


def fmt_si(value: float, unit: str, digits: int = 3) -> str:
    """Format a value with an SI prefix, e.g. fmt_si(3.35e12, 'B/s') -> '3.35 TB/s'."""
    if value == 0:
        return f"0 {unit}"
    for scale, prefix in [(1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]:
        if abs(value) >= scale:
            return f"{value / scale:.{digits}g} {prefix}{unit}"
    return f"{value:.{digits}g} {unit}"


def fmt_bytes(value: float, digits: int = 3) -> str:
    return fmt_si(value, "B", digits)


def fmt_time(seconds: float, digits: int = 3) -> str:
    if seconds == 0:
        return "0 s"
    if seconds >= 1:
        return f"{seconds:.{digits}g} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.{digits}g} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.{digits}g} us"
    return f"{seconds * 1e9:.{digits}g} ns"


def fmt_count(value: float, digits: int = 3) -> str:
    return fmt_si(value, "", digits).rstrip()
