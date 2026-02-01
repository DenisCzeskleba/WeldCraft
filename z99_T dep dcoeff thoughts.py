import re
import numpy as np
from datetime import datetime

# --- user flags ---
rebuild_diffusion = True
csv_path = "00_D_eff_table.csv"

R = 8.315
grid_step_C = 0.1
Tmin_C, Tmax_C = 0.0, 2000.0  # use the acutal logic in param_config for this later
materials = ["none", "base_metal", "weld_metal", "HAZ"]

DIFF_SPEC = """

material: none
] -inf, +inf ]: D = 0

material: base_metal
] 20, 200 ]:   D = 8.7615 * (10 ** - 9) * (T_C ** 2.2285)
] 200, 740 ]:  D = 8.9963 * (10 ** - 9) * (T_C ** 2.2480)
] 740, 1450 ]: D = 0.6736  * exp(-45086 / (R * T_K))
] 1450, 1540 ]:D = 28.7905  * exp(-93534 / (R * T_K))
] 1540, 2000 ]:D = 0.246  * exp(-15450 / (R * T_K))

material: weld_metal
] 20, 200 ]:   D = 0.07465 * exp(-11072 / (R * T_K))
] 200, 740 ]:  D = 0.1104  * exp(-12431 / (R * T_K))
] 740, 1450 ]: D = 0.8753  * exp(-46396 / (R * T_K))
] 1450, 1540 ]:D = 1.2104  * exp(-37785 / (R * T_K))
] 1540, 2000 ]:D = 1.1578  * exp(-37007 / (R * T_K))

material: HAZ
] 0, 2000 ]:   D = 8.7615 * (10 ** - 9) * (T_C ** 2.2285)

""".strip()


def _parse_spec(spec: str):
    blocks = {}
    current = None
    for line in spec.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"material:\s*(\w+)", line, re.I)
        if m:
            current = m.group(1)
            blocks[current] = []
            continue
        m = re.match(r"\]\s*([^,]+)\s*,\s*([^\]]+)\s*\]\s*:\s*D\s*=\s*(.+)", line)
        if m and current:
            a_str, b_str, expr = m.groups()
            def num(s):
                s = s.strip().lower()
                if s in ("-inf", "-infty"): return -np.inf
                if s in ("+inf", "inf", "infty"): return np.inf
                return float(s)
            a, b = num(a_str), num(b_str)
            blocks[current].append((a, b, expr.strip()))
    return blocks


def build_Deff_table_from_spec():
    parsed = _parse_spec(DIFF_SPEC)

    # grid
    T_C = np.arange(Tmin_C, Tmax_C + grid_step_C, grid_step_C, dtype=float)
    T_K = T_C + 273.15
    table = np.zeros((len(materials), T_C.size), dtype=float)

    safe_env = {
        "np": np, "exp": np.exp, "log": np.log, "pow": np.power,
        "R": R
    }

    for mid, name in enumerate(materials):
        D = np.zeros_like(T_C)
        for (a, b, expr) in parsed.get(name, []):
            mask = (T_C > a) & (T_C <= b)
            if not np.any(mask):
                continue
            # expose both temperature units; keep T as alias to T_C for legacy specs
            local_env = {
                "T": T_C[mask],       # alias to T_C
                "T_C": T_C[mask],
                "T_K": T_K[mask],
            }
            D[mask] = eval(expr, safe_env, local_env)
        table[mid, :] = D
    return T_C, table


def save_Deff_csv(path, T_grid, table, materials, grid_step_C, model_version="spec"):
    header = (
        f"# units: T_C (C), D (mm^2/s)\n"
        f"# model_version: {model_version}\n"
        f"# grid_step_C: {grid_step_C}\n"
        f"# generated: {datetime.utcnow().isoformat()}Z\n"
    )
    cols = ["T_C"] + materials
    data = np.column_stack([T_grid] + [table[i, :] for i in range(table.shape[0])])
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(",".join(cols) + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%.16g")


def load_Deff_csv(path, materials, expected_step):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # strip comments
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    if not data_lines:
        raise RuntimeError("CSV has no data lines.")

    header_line = data_lines[0]
    # delimiter detect
    delim = ";" if header_line.count(";") > header_line.count(",") else ","
    header_cols = [c.strip() for c in header_line.split(delim)]

    expected_cols = ["T_C"] + list(materials)
    if header_cols != expected_cols:
        raise ValueError(f"CSV columns {header_cols} != expected {expected_cols}")

    # detect decimal commas if delimiter is ';'
    decimal_commas = (delim == ";")

    # parse rows
    rows = []
    for ln in data_lines[1:]:
        parts = [p.strip() for p in ln.split(delim)]
        if len(parts) != len(header_cols):
            raise ValueError(f"Row has {len(parts)} cols, expected {len(header_cols)}: {ln[:120]}...")
        if decimal_commas:
            parts = [p.replace(",", ".") for p in parts]
        try:
            rows.append([float(x) for x in parts])
        except ValueError as e:
            raise ValueError(f"Failed to parse numbers in line: {ln[:120]}...") from e

    arr = np.array(rows, dtype=float)  # shape (N, 1+len(materials))
    T_grid = arr[:, 0]
    diffs = np.diff(T_grid)
    if diffs.size and (not np.allclose(diffs, diffs[0]) or not np.isclose(diffs[0], expected_step)):
        raise ValueError(f"T_C grid step {diffs[0]} != expected {expected_step}")

    # table shape: (n_materials, N)
    table = arr[:, 1:].T
    if np.any(table < 0):
        raise ValueError("Negative diffusivity found in CSV.")
    return T_grid, table


def maybe_make_and_load_Deff_table(
    rebuild: bool,
    path: str,
    materials,
    grid_step_C,
    build_fn,           # should return (T_grid, table)
    model_version="spec"
):
    if rebuild:
        T_grid, table = build_fn()
        save_Deff_csv(path, T_grid, table, materials, grid_step_C, model_version=model_version)
    # always load the artifact used by the run
    return load_Deff_csv(path, materials, grid_step_C)


# --- usage ---
T_grid, Deff_table = maybe_make_and_load_Deff_table(
    rebuild_diffusion,
    csv_path,
    materials,
    grid_step_C,
    build_Deff_table_from_spec,
    model_version="table2_mean_v1"
)
