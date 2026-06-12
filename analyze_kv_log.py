#!/usr/bin/env python3
"""
analyze_kv_log.py
-----------------
Reads kv_lifecycle.jsonl (produced by kv_cache_logger.h) and prints a summary
of the KV-cache lifecycle: init parameters, fill pressure, and clear events.

Usage:
    python analyze_kv_log.py kv_lifecycle.jsonl
    python analyze_kv_log.py kv_lifecycle.jsonl --verbose
"""

import json
import sys
import argparse
from collections import defaultdict
from typing import Any

# ─── helpers ─────────────────────────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] line {lineno}: bad JSON ({exc})", file=sys.stderr)
    return records


def fmt_ts(us: int) -> str:
    ms = us / 1000
    return f"{ms:>12.3f} ms"


# ─── analysis ────────────────────────────────────────────────────────────────

def analyse(records: list[dict], verbose: bool) -> None:
    init_events:  list[dict] = []
    fill_events:  list[dict] = []
    clear_events: list[dict] = []

    for r in records:
        phase = r.get("phase", "")
        if phase == "INIT":
            init_events.append(r)
        elif phase == "FILL":
            fill_events.append(r)
        elif phase == "CLEAR":
            clear_events.append(r)

    t0 = records[0]["ts_us"] if records else 0

    # ── INIT ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"INIT events  ({len(init_events)} total)")
    print("=" * 60)
    for e in init_events:
        ts = fmt_ts(e["ts_us"] - t0)
        kv = e.get("kv_size", "?")
        nl = e.get("n_layer", "?")
        ns = e.get("n_stream", "?")
        tk = e.get("type_k", "?")
        tv = e.get("type_v", "?")
        off = "GPU" if e.get("offload") else "CPU"
        print(f"  [{ts}]  kv_size={kv}  n_layer={nl}  "
              f"n_stream={ns}  k={tk}  v={tv} ")

    # ── FILL ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"FILL events  ({len(fill_events)} total)")
    print("=" * 60)

    # Batch-level records
    batches = [e for e in fill_events if e.get("event") == "batch_applied"]
    slots   = [e for e in fill_events if e.get("event") == "slot_filled"]

    print(f"  batch_applied records : {len(batches)}")
    print(f"  slot_filled records   : {len(slots)}")

    if batches:
        total_tokens = sum(e.get("n_tokens", 0) for e in batches)
        print(f"  total tokens filled   : {total_tokens}")

        # head progression
        heads = [e.get("head_before", 0) for e in batches]
        print(f"  head min / max        : {min(heads)} / {max(heads)}")

    if slots:
        # per-sequence token counts
        seq_counts: dict[int, int] = defaultdict(int)
        for e in slots:
            for sid in e.get("seq_ids", []):
                seq_counts[sid] += 1
        print(f"  tokens per sequence:")
        for sid in sorted(seq_counts):
            print(f"    seq {sid:>3d} : {seq_counts[sid]} tokens")

        # cell utilisation heat-map (bucket into 10 bands)
        kv_size = init_events[0].get("kv_size", 1) if init_events else 1
        band = max(1, kv_size // 10)
        bucket: dict[int, int] = defaultdict(int)
        for e in slots:
            bucket[e.get("cell", 0) // band] += 1
        print(f"  cell utilisation (buckets of {band}):")
        for b in range(10):
            lo, hi = b * band, (b + 1) * band - 1
            bar = "█" * (bucket[b] // max(1, max(bucket.values()) // 20))
            print(f"    [{lo:>5d}-{hi:>5d}]  {bucket[b]:>6d}  {bar}")

    if verbose and batches:
        print()
        print("  Batch timeline (first 20 batches):")
        for e in batches[:20]:
            ts = fmt_ts(e["ts_us"] - t0)
            print(f"    [{ts}]  n_tokens={e.get('n_tokens'):>4d}  "
                  f"head_before={e.get('head_before'):>5d}")

    # ── CLEAR ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"CLEAR events  ({len(clear_events)} total)")
    print("=" * 60)

    full_clears  = [e for e in clear_events if e.get("event") == "cache_cleared"]
    seq_rms      = [e for e in clear_events if e.get("event") == "seq_rm"]
    seq_keeps    = [e for e in clear_events if e.get("event") == "seq_keep"]
    defrags      = [e for e in clear_events if e.get("event") == "defrag"]

    print(f"  cache_cleared (full)  : {len(full_clears)}")
    print(f"  seq_rm                : {len(seq_rms)}")
    print(f"  seq_keep              : {len(seq_keeps)}")
    print(f"  defrag                : {len(defrags)}")

    if seq_rms:
        total_freed_rm = sum(e.get("cells_freed", 0) for e in seq_rms)
        print(f"  total cells freed (seq_rm)   : {total_freed_rm}")

    if seq_keeps:
        total_freed_keep = sum(e.get("cells_freed", 0) for e in seq_keeps)
        print(f"  total cells freed (seq_keep) : {total_freed_keep}")

    if verbose and seq_rms:
        print()
        print("  seq_rm events (first 20):")
        for e in seq_rms[:20]:
            ts  = fmt_ts(e["ts_us"] - t0)
            sid = e.get("seq_id", -1)
            p0  = e.get("p0", 0)
            p1  = e.get("p1", 0)
            cf  = e.get("cells_freed", 0)
            print(f"    [{ts}]  seq={sid:>3d}  pos=[{p0},{p1})  freed={cf}")

    # ── TIMELINE SUMMARY ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("TIMELINE (all events, chronological)")
    print("=" * 60)
    all_events = sorted(records, key=lambda r: r.get("ts_us", 0))
    for e in all_events:
        ts    = fmt_ts(e["ts_us"] - t0)
        phase = e.get("phase", "?")
        event = e.get("event", "?")
        # build a short detail string from whichever keys are present
        detail_keys = {k: v for k, v in e.items()
                       if k not in ("ts_us", "phase", "event")}
        detail = "  ".join(f"{k}={v}" for k, v in detail_keys.items())
        print(f"  [{ts}]  {phase:<5s}  {event:<20s}  {detail}")


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse kv_lifecycle.jsonl")
    parser.add_argument("logfile", help="Path to the JSONL log file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-event tables for FILL and CLEAR phases")
    args = parser.parse_args()

    records = load_records(args.logfile)
    if not records:
        print("No records found.")
        return

    print(f"Loaded {len(records)} records from '{args.logfile}'\n")
    analyse(records, verbose=args.verbose)


if __name__ == "__main__":
    main()
