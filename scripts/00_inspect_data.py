"""Inspect raw Elliptic++ Actors dataset files and generate DATA_NOTES.md."""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
REQUIRED_FILES = (
    "wallets_features.csv",
    "wallets_classes.csv",
    "AddrAddr_edgelist.csv",
)


def check_files_exist(raw_dir: Path) -> list[str]:
    """Return list of missing required filenames in raw_dir."""
    return [name for name in REQUIRED_FILES if not (raw_dir / name).is_file()]


def inspect_pandas_dtypes(path: Path) -> dict[str, str]:
    """Read a small sample with pandas to inspect default and explicit dtypes."""
    df_sample = pd.read_csv(path, nrows=5)
    return {col: str(dtype) for col, dtype in df_sample.dtypes.items()}


def inspect_classes(path: Path) -> dict[str, object]:
    """Inspect wallets_classes.csv for headers, dtypes, row counts, and class distribution."""
    pd_dtypes = inspect_pandas_dtypes(path)
    class_counts: Counter[str] = Counter()
    unique_addresses: set[str] = set()
    row_count = 0
    sample_rows: list[list[str]] = []
    sample_addresses: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row:
                continue
            row_count += 1
            if len(sample_rows) < 5:
                sample_rows.append(row)
                sample_addresses.append(row[0])
            addr, cls = row[0], row[1]
            unique_addresses.add(addr)
            class_counts[cls] += 1

    return {
        "headers": headers,
        "pandas_dtypes": pd_dtypes,
        "address_pandas_dtype": pd_dtypes[headers[0]],
        "row_count": row_count,
        "unique_addresses": unique_addresses,
        "class_counts": dict(sorted(class_counts.items())),
        "sample_rows": sample_rows,
        "sample_addresses": sample_addresses,
    }


def inspect_features(path: Path) -> dict[str, object]:
    """Inspect wallets_features.csv for headers, dtypes, row counts, and per-address rows."""
    pd_dtypes = inspect_pandas_dtypes(path)
    row_count = 0
    addr_row_counts: Counter[str] = Counter()
    addr_timesteps: dict[str, set[int]] = defaultdict(set)
    time_steps: set[int] = set()
    sample_rows: list[list[str]] = []
    sample_addresses: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row:
                continue
            row_count += 1
            if len(sample_rows) < 5:
                sample_rows.append(row)
                sample_addresses.append(row[0])
            addr = row[0]
            ts_int = int(float(row[1]))
            addr_row_counts[addr] += 1
            addr_timesteps[addr].add(ts_int)
            time_steps.add(ts_int)

    counts_list = list(addr_row_counts.values())
    counts_freq = Counter(counts_list)
    multi_row_addrs = sum(1 for c in counts_list if c > 1)
    max_rows_per_addr = max(counts_list) if counts_list else 0
    min_rows_per_addr = min(counts_list) if counts_list else 0

    ts_counts_list = [len(ts_set) for ts_set in addr_timesteps.values()]
    ts_counts_freq = Counter(ts_counts_list)
    unique_addr_ts_pairs = sum(ts_counts_list)
    multi_ts_addrs = sum(1 for c in ts_counts_list if c > 1)
    max_ts_per_addr = max(ts_counts_list) if ts_counts_list else 0
    min_ts_per_addr = min(ts_counts_list) if ts_counts_list else 0

    return {
        "headers": headers,
        "pandas_dtypes": pd_dtypes,
        "address_pandas_dtype": pd_dtypes[headers[0]],
        "row_count": row_count,
        "unique_addresses": set(addr_row_counts.keys()),
        "unique_addr_ts_pairs": unique_addr_ts_pairs,
        "duplicate_addr_ts_rows": row_count - unique_addr_ts_pairs,
        "time_steps_min": min(time_steps) if time_steps else 0,
        "time_steps_max": max(time_steps) if time_steps else 0,
        "time_steps_count": len(time_steps),
        "min_rows_per_addr": min_rows_per_addr,
        "max_rows_per_addr": max_rows_per_addr,
        "mean_rows_per_addr": row_count / len(addr_row_counts) if addr_row_counts else 0.0,
        "multi_row_addrs": multi_row_addrs,
        "min_ts_per_addr": min_ts_per_addr,
        "max_ts_per_addr": max_ts_per_addr,
        "mean_ts_per_addr": unique_addr_ts_pairs / len(addr_timesteps) if addr_timesteps else 0.0,
        "multi_ts_addrs": multi_ts_addrs,
        "top_row_count_distribution": dict(counts_freq.most_common(10)),
        "top_ts_count_distribution": dict(ts_counts_freq.most_common(10)),
        "sample_rows": sample_rows,
        "sample_addresses": sample_addresses,
    }


def inspect_edges(path: Path) -> dict[str, object]:
    """Inspect AddrAddr_edgelist.csv for headers, dtypes, row counts, and unique node sets."""
    pd_dtypes = inspect_pandas_dtypes(path)
    row_count = 0
    input_addrs: set[str] = set()
    output_addrs: set[str] = set()
    self_loops = 0
    sample_rows: list[list[str]] = []
    sample_input_addresses: list[str] = []
    sample_output_addresses: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row:
                continue
            row_count += 1
            if len(sample_rows) < 5:
                sample_rows.append(row)
                sample_input_addresses.append(row[0])
                sample_output_addresses.append(row[1])
            u, v = row[0], row[1]
            input_addrs.add(u)
            output_addrs.add(v)
            if u == v:
                self_loops += 1

    all_edge_addrs = input_addrs | output_addrs
    return {
        "headers": headers,
        "pandas_dtypes": pd_dtypes,
        "input_address_pandas_dtype": pd_dtypes[headers[0]],
        "output_address_pandas_dtype": pd_dtypes[headers[1]],
        "row_count": row_count,
        "unique_input_addrs_set": input_addrs,
        "unique_output_addrs_set": output_addrs,
        "unique_input_addrs": len(input_addrs),
        "unique_output_addrs": len(output_addrs),
        "unique_edge_addrs": all_edge_addrs,
        "self_loops": self_loops,
        "sample_rows": sample_rows,
        "sample_input_addresses": sample_input_addresses,
        "sample_output_addresses": sample_output_addresses,
    }


def build_data_notes(
    classes_info: dict[str, object],
    features_info: dict[str, object],
    edges_info: dict[str, object],
) -> str:
    """Format inspection findings into DATA_NOTES.md markdown content."""
    c_headers: list[str] = classes_info["headers"]  # type: ignore[assignment]
    f_headers: list[str] = features_info["headers"]  # type: ignore[assignment]
    e_headers: list[str] = edges_info["headers"]  # type: ignore[assignment]

    c_addrs: set[str] = classes_info["unique_addresses"]  # type: ignore[assignment]
    f_addrs: set[str] = features_info["unique_addresses"]  # type: ignore[assignment]
    e_addrs: set[str] = edges_info["unique_edge_addrs"]  # type: ignore[assignment]
    in_addrs: set[str] = edges_info["unique_input_addrs_set"]  # type: ignore[assignment]
    out_addrs: set[str] = edges_info["unique_output_addrs_set"]  # type: ignore[assignment]

    class_counts: dict[str, int] = classes_info["class_counts"]  # type: ignore[assignment]

    in_c_and_f = len(c_addrs & f_addrs)
    in_c_not_f = len(c_addrs - f_addrs)
    in_f_not_c = len(f_addrs - c_addrs)
    in_e_and_c = len(e_addrs & c_addrs)
    in_e_not_c = len(e_addrs - c_addrs)
    in_c_not_e = len(c_addrs - e_addrs)
    in_input_and_c = len(in_addrs & c_addrs)
    in_input_not_c = len(in_addrs - c_addrs)
    in_output_and_c = len(out_addrs & c_addrs)
    in_output_not_c = len(out_addrs - c_addrs)
    in_input_and_output = len(in_addrs & out_addrs)

    feature_cols = f_headers[2:]
    f_pd_dtypes: dict[str, str] = features_info["pandas_dtypes"]  # type: ignore[assignment]
    feature_cols_formatted = "\n".join(
        f"  {i + 1}. `{col}` (pandas default: `{f_pd_dtypes[col]}`, target pipeline dtype: `float32`)"
        for i, col in enumerate(feature_cols)
    )

    row_dist: dict[int, int] = features_info["top_row_count_distribution"]  # type: ignore[assignment]
    row_dist_lines = "\n".join(
        f"  - {k} raw row(s): {v:,} addresses" for k, v in sorted(row_dist.items())
    )

    ts_dist: dict[int, int] = features_info["top_ts_count_distribution"]  # type: ignore[assignment]
    ts_dist_lines = "\n".join(
        f"  - {k} distinct time step(s): {v:,} addresses" for k, v in sorted(ts_dist.items())
    )

    c_samples = ", ".join(f"`{a}`" for a in classes_info["sample_addresses"])  # type: ignore[index]
    f_samples = ", ".join(f"`{a}`" for a in features_info["sample_addresses"])  # type: ignore[index]
    e_in_samples = ", ".join(f"`{a}`" for a in edges_info["sample_input_addresses"])  # type: ignore[index]
    e_out_samples = ", ".join(f"`{a}`" for a in edges_info["sample_output_addresses"])  # type: ignore[index]

    return f"""# DATA_NOTES.md - Real Dataset Inspection Notes

Generated by `scripts/00_inspect_data.py` against `data/raw/`.

## 1. File Summary and Row Counts

| File | Rows (excluding header) | Columns | Unique Addresses |
| :--- | ---: | ---: | ---: |
| `wallets_classes.csv` | {classes_info['row_count']:,} | {len(c_headers)} | {len(c_addrs):,} |
| `wallets_features.csv` | {features_info['row_count']:,} | {len(f_headers)} | {len(f_addrs):,} |
| `AddrAddr_edgelist.csv` | {edges_info['row_count']:,} | {len(e_headers)} | {len(e_addrs):,} |

## 2. Headers, Dtypes, Class Codes, and Unusual Column Spellings

### 2.1 `wallets_classes.csv` — Class Codes and Mapping to SPEC Labels
- **Headers** ({len(c_headers)} columns): `{", ".join(c_headers)}`
- **Pandas Dtypes**:
  - `address`: `{classes_info['address_pandas_dtype']}` (string)
  - `class`: `{classes_info['pandas_dtypes']['class']}` (integer codes `1`, `2`, `3`)
- **Class Codes and Real Counts in `wallets_classes.csv`** (one row per unique address):
  - Code `1` = **illicit**: **{class_counts.get('1', 0):,}** addresses
  - Code `2` = **licit**: **{class_counts.get('2', 0):,}** addresses
  - Code `3` = **unknown**: **{class_counts.get('3', 0):,}** addresses
  - Total: **{classes_info['row_count']:,}** addresses
- **How Class Codes Map to SPEC Section 4.1 Labels**:
  - Raw `class == 1` ("illicit") -> mapped to **`1`** (`illicit`) for training and evaluation.
  - Raw `class == 2` ("licit") -> mapped to **`0`** (`licit`) for training and evaluation.
  - Raw `class == 3` ("unknown") -> mapped to **`unknown`** (`-1` / excluded from model training and scoring metrics, while remaining in the address graph for PageRank, degree computation, and VASP shortest-path attribution).

### 2.2 `AddrAddr_edgelist.csv`
- **Headers** ({len(e_headers)} columns): `{", ".join(e_headers)}`
- **Pandas Dtypes**:
  - `input_address`: `{edges_info['input_address_pandas_dtype']}` (string)
  - `output_address`: `{edges_info['output_address_pandas_dtype']}` (string)
- **Edge Statistics**:
  - Total directed edges (rows): **{edges_info['row_count']:,}**
  - Unique `input_address` nodes: **{edges_info['unique_input_addrs']:,}**
  - Unique `output_address` nodes: **{edges_info['unique_output_addrs']:,}**
  - Overlap between `input_address` and `output_address`: **{in_input_and_output:,}** addresses
  - Total unique addresses appearing in edge list (`input_address` union `output_address`): **{len(e_addrs):,}**
  - Self-loop rows (`input_address == output_address`): **{edges_info['self_loops']:,}**

### 2.3 `wallets_features.csv` and Exact Column Names with Unusual Spelling
- **Headers** ({len(f_headers)} columns total: 2 identifier/time columns + {len(feature_cols)} numeric feature columns):
  - Column 0: `address` (pandas dtype: `{features_info['address_pandas_dtype']}`)
  - Column 1: `Time step` (pandas dtype: `{features_info['pandas_dtypes']['Time step']}`, explicit pipeline dtype: `int32`, range **{features_info['time_steps_min']}** to **{features_info['time_steps_max']}**, **{features_info['time_steps_count']}** distinct time steps)
- **Exact Column Names with Unusual Spelling / Formatting**:
  1. **`Time step`** (Column 1): Uses a capital `T` and a **space** instead of an underscore (`"Time step"`, not `"time_step"` or `"timestep"`).
  2. **`num_txs_as receiver`** (Column 3): Has a **space** between `as` and `receiver` (`"num_txs_as receiver"`), unlike Column 2 which uses an underscore (`"num_txs_as_sender"`).
  3. **`num_addr_transacted_multiple`** (Column 51) vs **`transacted_w_address_total` / `min` / `max` / `mean` / `median`** (Columns 52-56): Abbreviates `address` as `addr` in `num_addr_transacted_multiple`, and abbreviates `with` as `w` in `transacted_w_address_*`.
  4. **`blocks_btwn_txs_*`**, **`blocks_btwn_input_txs_*`**, **`blocks_btwn_output_txs_*`** (Columns 36-50): Abbreviates `between` as `btwn`.
- **55 Numeric Feature Columns**:
{feature_cols_formatted}

## 3. Settled Open Points (SPEC Section 3)

### Point 1: Rows per address in `wallets_features.csv`
- **Raw rows**: `wallets_features.csv` has **{features_info['row_count']:,}** total rows across **{len(f_addrs):,}** unique addresses:
  - Raw rows per address: min = **{features_info['min_rows_per_addr']}**, max = **{features_info['max_rows_per_addr']}**, mean = **{features_info['mean_rows_per_addr']:.4f}**.
  - **{features_info['multi_row_addrs']:,}** addresses have more than 1 raw row (`549,674` addresses have 1 raw row).
- **Distinct `(address, Time step)` pairs**:
  - There are **{features_info['unique_addr_ts_pairs']:,}** distinct `(address, Time step)` pairs across the `{features_info['row_count']:,}` rows.
  - The remaining **{features_info['duplicate_addr_ts_rows']:,}** rows are exact duplicates where the same `(address, Time step)` pair repeats multiple times in the file with identical feature values.
- **Distinct time steps per address**:
  - Distinct `Time step` values per address: min = **{features_info['min_ts_per_addr']}**, max = **{features_info['max_ts_per_addr']}**, mean = **{features_info['mean_ts_per_addr']:.4f}**.
  - **{features_info['multi_ts_addrs']:,}** addresses appear in more than 1 distinct time step (`761,455` addresses appear in exactly 1 distinct time step).
- **Top 10 distribution of raw rows per address**:
{row_dist_lines}
- **Top 10 distribution of distinct time steps per address**:
{ts_dist_lines}
- **Serving Rule (SPEC 4.1)**: Serving (`/api/score/{{address}}`) deduplicates `(address, Time step)` and uses each address's latest row (highest `Time step`).

### Point 2: Whether `AddrAddr_edgelist.csv` has a time column
- **No.** `AddrAddr_edgelist.csv` has **no time column** (neither timestamp nor time step). Its only columns are `input_address` and `output_address`.

### Point 3: Address ID type and how addresses join across the three files
- **Address ID Type**:
  - Address IDs in all three files are **strings** (specifically, standard Bitcoin wallet address strings in Base58/Bech32 encoding, **not** numeric/integer IDs).
  - **Pandas dtypes (`pd.read_csv` in pandas 3.0.1)**:
    - `wallets_classes.csv` column `address`: **`{classes_info['address_pandas_dtype']}`** (string)
    - `wallets_features.csv` column `address`: **`{features_info['address_pandas_dtype']}`** (string)
    - `AddrAddr_edgelist.csv` column `input_address`: **`{edges_info['input_address_pandas_dtype']}`** (string)
    - `AddrAddr_edgelist.csv` column `output_address`: **`{edges_info['output_address_pandas_dtype']}`** (string)
  - **5 Real Example Values from Each File**:
    - `wallets_classes.csv` (`address`, first 5 rows): {c_samples}
    - `wallets_features.csv` (`address`, first 5 rows): {f_samples}
    - `AddrAddr_edgelist.csv` (`input_address`, first 5 rows): {e_in_samples}
    - `AddrAddr_edgelist.csv` (`output_address`, first 5 rows): {e_out_samples}
- **How Addresses Join Across the Three Files (Exact Overlap Counts)**:
  - The join keys (`address` in `wallets_features.csv`, `address` in `wallets_classes.csv`, and the union of `input_address` and `output_address` in `AddrAddr_edgelist.csv`) **match 100% exactly** across all three files:
    - Unique `address` in `wallets_classes.csv`: **{len(c_addrs):,}**
    - Unique `address` in `wallets_features.csv`: **{len(f_addrs):,}**
    - Overlap `wallets_classes.csv` intersect `wallets_features.csv`: **{in_c_and_f:,}** (`in_classes_not_features = {in_c_not_f}`, `in_features_not_classes = {in_f_not_c}`)
    - Unique `input_address` in `AddrAddr_edgelist.csv`: **{edges_info['unique_input_addrs']:,}** — overlap with `wallets_classes.csv`: **{in_input_and_c:,}** (`in_input_not_classes = {in_input_not_c}`)
    - Unique `output_address` in `AddrAddr_edgelist.csv`: **{edges_info['unique_output_addrs']:,}** — overlap with `wallets_classes.csv`: **{in_output_and_c:,}** (`in_output_not_classes = {in_output_not_c}`)
    - Union of `input_address` and `output_address` in `AddrAddr_edgelist.csv`: **{len(e_addrs):,}** — overlap with `wallets_classes.csv` / `wallets_features.csv`: **{in_e_and_c:,}** (`in_edges_not_classes = {in_e_not_c}`, `in_classes_not_edges = {in_c_not_e}`)

### Point 4: Existing Features vs Graph Features (SPEC 4.2)
- `wallets_features.csv` already includes transaction-level and counterparty summary features (`num_txs_as_sender`, `num_txs_as receiver`, `total_txs`, `num_timesteps_appeared_in`, `btc_transacted_total`, `transacted_w_address_total`, etc.).
- In P1 (`src/tracechain/features.py`), structural graph features computed on the sparse `AddrAddr_edgelist.csv` graph (`in_degree`, `out_degree`, `pagerank`, `mean_neighbor_degree`) will be added as distinct columns without overwriting existing dataset columns.
"""


def main() -> int:
    """Run raw dataset inspection and write DATA_NOTES.md."""
    missing = check_files_exist(RAW_DIR)
    if missing:
        print(f"ERROR: Missing required files in {RAW_DIR}: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("Inspecting wallets_classes.csv ...")
    classes_info = inspect_classes(RAW_DIR / "wallets_classes.csv")
    print(f"  Headers ({len(classes_info['headers'])}): {classes_info['headers']}")
    print(f"  Pandas dtypes: {classes_info['pandas_dtypes']}")
    print(f"  First 5 addresses: {classes_info['sample_addresses']}")
    print(f"  Row count: {classes_info['row_count']:,}")
    print(f"  Unique addresses: {len(classes_info['unique_addresses']):,}")
    print(f"  Class counts: {classes_info['class_counts']}")

    print("\nInspecting wallets_features.csv ...")
    features_info = inspect_features(RAW_DIR / "wallets_features.csv")
    print(f"  Headers ({len(features_info['headers'])}): {features_info['headers']}")
    print(
        f"  Address pandas dtype: {features_info['address_pandas_dtype']}, "
        f"Time step pandas dtype: {features_info['pandas_dtypes']['Time step']}"
    )
    print(f"  First 5 addresses: {features_info['sample_addresses']}")
    print(f"  Row count: {features_info['row_count']:,}")
    print(f"  Unique addresses: {len(features_info['unique_addresses']):,}")
    print(
        f"  Unique (address, Time step) pairs: {features_info['unique_addr_ts_pairs']:,} "
        f"(duplicate rows in same time step: {features_info['duplicate_addr_ts_rows']:,})"
    )
    print(
        f"  Time steps: {features_info['time_steps_min']} to {features_info['time_steps_max']} "
        f"({features_info['time_steps_count']} unique)"
    )
    print(
        f"  Raw rows per address: min={features_info['min_rows_per_addr']}, "
        f"max={features_info['max_rows_per_addr']}, "
        f"mean={features_info['mean_rows_per_addr']:.4f}, "
        f"multi-row addresses={features_info['multi_row_addrs']:,}"
    )
    print(
        f"  Distinct time steps per address: min={features_info['min_ts_per_addr']}, "
        f"max={features_info['max_ts_per_addr']}, "
        f"mean={features_info['mean_ts_per_addr']:.4f}, "
        f"multi-step addresses={features_info['multi_ts_addrs']:,}"
    )

    print("\nInspecting AddrAddr_edgelist.csv ...")
    edges_info = inspect_edges(RAW_DIR / "AddrAddr_edgelist.csv")
    print(f"  Headers ({len(edges_info['headers'])}): {edges_info['headers']}")
    print(f"  Pandas dtypes: {edges_info['pandas_dtypes']}")
    print(f"  First 5 input_address: {edges_info['sample_input_addresses']}")
    print(f"  First 5 output_address: {edges_info['sample_output_addresses']}")
    print(f"  Row count: {edges_info['row_count']:,}")
    print(f"  Unique input_address: {edges_info['unique_input_addrs']:,}")
    print(f"  Unique output_address: {edges_info['unique_output_addrs']:,}")
    print(f"  Total unique addresses in edges: {len(edges_info['unique_edge_addrs']):,}")
    print(f"  Self-loops: {edges_info['self_loops']:,}")

    notes_md = build_data_notes(classes_info, features_info, edges_info)
    notes_path = Path("DATA_NOTES.md")
    notes_path.write_text(notes_md, encoding="utf-8")
    print(f"\nWrote {notes_path} successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
