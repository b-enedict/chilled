"""
Extracts chiller log data from pivoted xlsx files into a flat CSV.

Each xlsx file covers one month; each sheet is one day. For every
(date, chiller_unit, timestamp) combination one row is produced,
with all measured metrics as columns.
"""

import re
from pathlib import Path

import fastexcel
import polars as pl

MONTH_MAP = {
    'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
    'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
    'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12,
}


def _clean_value(val: object) -> object:
    """Fix malformed numeric strings with extra dots (e.g. '50154..7', '0.53.', '2.8.7')."""
    if not isinstance(val, str):
        return val
    stripped = val.strip()
    if stripped.count('.') > 1:
        # Collapse extra dots: keep first dot as the decimal separator
        parts = stripped.split('.')
        fixed = parts[0] + '.' + ''.join(parts[1:])
        try:
            return float(fixed)
        except ValueError:
            pass
    return val


def _sanitize(text: str) -> str:
    text = text.replace('%', 'pct').replace('°', 'deg')
    text = re.sub(r'[^\w]+', '_', text).strip('_').lower()
    return text


def _parse_month_year(filename: str) -> tuple[int, int]:
    name = Path(filename).stem.upper()
    month = next((v for k, v in MONTH_MAP.items() if k in name), None)
    m = re.search(r'\b(20\d{2})\b', name)
    if not month or not m:
        raise ValueError(f"Cannot parse month/year from {filename!r}")
    return month, int(m.group(1))


def _build_metric_names(col0: list, col1: list) -> list[str]:
    """
    Derive clean column names from the two name columns (A and B in Excel).

    Rules:
    - Both non-null → hierarchical group start: name = "parent_sub"
    - col0 only     → standalone metric: name = col0  (resets parent)
    - col1 only     → sub-metric of current parent: name = "parent_sub"
    - Both null     → placeholder
    """
    last_parent: str | None = None
    names: list[str] = []

    for p_raw, s_raw in zip(col0, col1):
        p = str(p_raw).strip() if p_raw is not None else None
        s = str(s_raw).strip() if s_raw is not None else None

        if p and s:
            last_parent = p
            raw = f"{p}_{s}"
        elif p:
            last_parent = None  # standalone; no sub-rows should inherit this
            raw = p
        elif s:
            raw = f"{last_parent}_{s}" if last_parent else s
        else:
            # Both effectively null (blank label in source file).
            # Preserve parent context in the fallback name.
            raw = f"{last_parent}_metric_{len(names)}" if last_parent else f"metric_{len(names)}"

        name = _sanitize(raw)

        # ensure uniqueness
        base, i = name, 1
        while name in names:
            name = f"{base}_{i}"
            i += 1
        names.append(name)

    return names


def _extract_sheet(
    df_raw: pl.DataFrame,
    day: int,
    month: int,
    year: int,
) -> pl.DataFrame | None:
    cols = df_raw.columns
    n_cols = len(cols)

    # Find the chiller header row (first row where col 4 starts with "EC")
    chiller_row_idx = None
    for i in range(df_raw.height):
        v = df_raw.row(i)[4]
        if v and str(v).strip().startswith('EC'):
            chiller_row_idx = i
            break
    if chiller_row_idx is None:
        return None

    time_row_idx = chiller_row_idx + 1
    data_start = time_row_idx + 1

    # Bound the data at the REMARK row
    data_end = df_raw.height
    for i in range(data_start, df_raw.height):
        v0 = df_raw.row(i)[0]
        if v0 and 'REMARK' in str(v0).upper():
            data_end = i
            break

    # Chiller IDs live in every even-indexed data column starting at 4
    chiller_row = df_raw.row(chiller_row_idx)
    chiller_ids: list[str] = []
    chiller_col_starts: list[int] = []
    for c in range(4, n_cols - 1, 2):
        cid = chiller_row[c]
        if cid and str(cid).strip().startswith('EC'):
            chiller_ids.append(str(cid).strip())
            chiller_col_starts.append(c)

    if not chiller_ids:
        return None

    # Date: the 2025 files embed the actual date in col 2 of the row above
    # the chiller header; fall back to filename + sheet day number.
    date_str: str
    op_row = df_raw.row(chiller_row_idx - 1)
    raw_date = op_row[2] if n_cols > 2 else None
    if raw_date and re.match(r'\d{4}-\d{2}-\d{2}', str(raw_date)):
        date_str = str(raw_date)[:10]
    else:
        date_str = f"{year}-{month:02d}-{day:02d}"

    # Data slice (metric rows)
    data = df_raw.slice(data_start, data_end - data_start)
    if data.height == 0:
        return None

    metric_names = _build_metric_names(
        data.get_column(cols[0]).to_list(),
        data.get_column(cols[1]).to_list(),
    )

    # One output row per (chiller × timestamp)
    records: list[dict] = []
    for cid, c_start in zip(chiller_ids, chiller_col_starts):
        for ts_offset, ts in enumerate(['09:00', '00:00']):
            c = c_start + ts_offset
            if c >= n_cols:
                continue
            values = data.get_column(cols[c]).to_list()
            rec: dict = {'date': date_str, 'chiller_id': cid, 'timestamp': ts}
            for metric, val in zip(metric_names, values):
                rec[metric] = _clean_value(val)
            records.append(rec)

    return pl.DataFrame(records, infer_schema_length=len(records)) if records else None


def extract_all(
    input_dir: str | Path = 'input',
    output_path: str | Path = 'output/chiller_data.csv',
) -> pl.DataFrame:
    """
    Process every xlsx file in *input_dir* and write a combined CSV to
    *output_path*.  Returns the combined DataFrame.
    """
    xlsx_files = sorted(Path(input_dir).glob('*.xlsx'))
    all_dfs: list[pl.DataFrame] = []

    for filepath in xlsx_files:
        print(f"  {filepath.name} … ", end='', flush=True)
        try:
            month, year = _parse_month_year(filepath.name)
        except ValueError as e:
            print(f"SKIP ({e})")
            continue

        reader = fastexcel.read_excel(str(filepath))
        n = 0
        for sheet_name in reader.sheet_names:
            try:
                day = int(sheet_name)
            except ValueError:
                continue
            raw = reader.load_sheet(sheet_name, header_row=None).to_polars()
            sheet_df = _extract_sheet(raw, day, month, year)
            if sheet_df is not None:
                all_dfs.append(sheet_df)
                n += 1
        print(f"{n} days")

    if not all_dfs:
        print("No data extracted.")
        return pl.DataFrame()

    combined = (
        pl.concat(all_dfs, how='diagonal_relaxed')
        .sort(['date', 'chiller_id', 'timestamp'])
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(out)
    print(f"\n{combined.height} rows × {combined.width} cols → {out}")
    return combined


if __name__ == '__main__':
    extract_all()
