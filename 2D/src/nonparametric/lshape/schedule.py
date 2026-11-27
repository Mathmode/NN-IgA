import csv
import math


def read_schedule(path, p):
    with open(path, newline="") as f:
        rows = [(int(r["N"]), float(r["T"]), float(r["h_min_floor"]))
                for r in csv.DictReader(f) if int(r["p"]) == p]
    if not rows or len({n for n, _, _ in rows}) != len(rows):
        raise ValueError(f"missing or duplicated schedule for p={p} in {path}")
    for n, t, h in rows:
        if n < 2 or n % 2 or not (math.isfinite(t) and t > 0 and 0 <= h < 1/n):
            raise ValueError(f"invalid mesh policy for p={p}, N={n}")
    return tuple(sorted(rows))


def policy_at(schedule, n):
    for level, t, floor in schedule:
        if level == n:
            return t, floor
    raise ValueError(f"no mesh policy for N={n}")
