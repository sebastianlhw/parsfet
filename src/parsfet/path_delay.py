"""Path delay estimation for multi-stage combinational timing paths.

Public API
----------
**Input types** (user-facing, Pydantic v2):

    PathSpec       — one path stage: cell name, fanout, optional load overrides,
                     per-stage derate overrides.
    AnalysisConfig — all analysis parameters (derates, wire model, arc mode, …).
    WireLoadModel  — fanout-to-wire-cap table; several built-in process presets.

**Resolution result**:

    ManualResolution — NamedTuple returned by resolve_manual().

**Shared computation type** (internal):

    _ResolvedStage — fully-resolved per-stage atom: cell ref, load, derates,
                     arc_mode, linear fallback coefficients, skip flag.
                     Produced by resolve_manual(); future resolve_netlist() will
                     produce the same type.  Users never construct this directly.

**Output types** (engine-controlled dataclasses):

    TimingPoint    — per-stage annotated result (arrival, slew, load, method).
    TimingPath     — full result: list of TimingPoints + total_delay + warnings.

**Functions**:

    resolve_manual(path, dataset, config)  → ManualResolution
    propagate(stages, initial_slew_ns)     → TimingPath          (pure)
    estimate_path_delay(dataset, path,     → TimingPath          (convenience)
                        config)

Example::

    import parsfet
    from parsfet.path_delay import estimate_path_delay, PathSpec, AnalysisConfig

    ds = parsfet.Dataset().load_files(["lib.lib"])
    path   = [PathSpec(cell_name="INVD1", fanout=4),
              PathSpec(cell_name="NAND2D1", fanout=2),
              PathSpec(cell_name="INVD4")]
    config = AnalysisConfig(delay_derate=1.08,
                            wire_load=parsfet.WireLoadModel.typical_7nm())
    result = estimate_path_delay(ds, path, config)
    print(f"Total: {result.total_delay_ns*1e3:.1f} ps")
    for pt in result.points:
        print(f"  {pt.name:20s}  {pt.delay_ns*1e3:5.1f} ps  [{pt.method}]")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, Field, field_validator

# Emit a warning when the linear model R² falls below this threshold.
# Indicates the cell's delay-versus-load relationship is non-linear enough
# that the D0+k×load fit is unreliable.
_LINEAR_R2_WARN_THRESHOLD: float = 0.9

if TYPE_CHECKING:
    from .data import Dataset


# ---------------------------------------------------------------------------
# WireLoadModel — Pydantic, frozen, process-node presets
# ---------------------------------------------------------------------------


class WireLoadModel(BaseModel):
    """Maps fanout count to wire capacitance in pF for a given process node.

    All values are ORDER-OF-MAGNITUDE estimates for early design exploration.
    Use SPEF for sign-off accuracy.

    Attributes:
        fanout_cap_pf: dict mapping fanout (≥1) to wire cap in pF (≥0).
            Intermediate values linearly interpolated.  Clamped at edges.
        base_cap_pf: Fixed capacitance added per stage regardless of fanout.

    Example::

        >>> wlm = WireLoadModel(fanout_cap_pf={1: 0.001, 4: 0.006, 16: 0.018})
        >>> wlm.wire_cap(fanout=2)
        0.002666...
        >>> wlm = WireLoadModel.model_validate({'fanout_cap_pf': {1: 0.001, 4: 0.010}})
    """

    fanout_cap_pf: dict[int, float] = Field(default_factory=dict)
    base_cap_pf: float = Field(default=0.0, ge=0.0)

    model_config = {"frozen": True}

    @field_validator("fanout_cap_pf", mode="before")
    @classmethod
    def _validate_fanout_dict(cls, v: dict) -> dict[int, float]:
        result: dict[int, float] = {}
        for k, val in v.items():
            k_int = int(k)
            v_float = float(val)
            if k_int < 1:
                raise ValueError(f"fanout key must be ≥ 1, got {k_int}")
            if v_float < 0.0:
                raise ValueError(f"wire cap must be ≥ 0.0 pF, got {v_float}")
            result[k_int] = v_float
        return result

    def wire_cap(self, fanout: int) -> float:
        """Return interpolated wire cap in pF for the given fanout (clamped)."""
        if not self.fanout_cap_pf:
            return self.base_cap_pf
        xs = sorted(self.fanout_cap_pf.keys())
        ys = [self.fanout_cap_pf[x] for x in xs]
        return float(np.interp(fanout, xs, ys)) + self.base_cap_pf

    @classmethod
    def typical_7nm(cls) -> "WireLoadModel":
        """7nm FinFET — approximate global routing parasitics."""
        return cls(fanout_cap_pf={1: 0.001, 4: 0.006, 16: 0.018}, base_cap_pf=0.002)

    @classmethod
    def typical_14nm(cls) -> "WireLoadModel":
        """14nm FinFET — approximate global routing parasitics."""
        return cls(fanout_cap_pf={1: 0.002, 4: 0.010, 16: 0.030}, base_cap_pf=0.003)

    @classmethod
    def typical_28nm(cls) -> "WireLoadModel":
        """28nm — approximate global routing parasitics."""
        return cls(fanout_cap_pf={1: 0.005, 4: 0.020, 16: 0.060}, base_cap_pf=0.005)

    @classmethod
    def typical_65nm(cls) -> "WireLoadModel":
        """65nm — approximate global routing parasitics."""
        return cls(fanout_cap_pf={1: 0.010, 4: 0.040, 16: 0.120}, base_cap_pf=0.010)

    @classmethod
    def typical_130nm(cls) -> "WireLoadModel":
        """130nm — approximate global routing parasitics."""
        return cls(fanout_cap_pf={1: 0.020, 4: 0.080, 16: 0.250}, base_cap_pf=0.020)

    @classmethod
    def zero(cls) -> "WireLoadModel":
        """No wire load — ideal wires (comparison / unit-test baseline)."""
        return cls(fanout_cap_pf={}, base_cap_pf=0.0)


# ---------------------------------------------------------------------------
# PathSpec — user input for ONE stage (Pydantic, frozen)
# ---------------------------------------------------------------------------


class PathSpec(BaseModel):
    """Specification for one gate stage on a timing path.

    The output net of this stage drives:

    1. Exactly one on-path sink (the next ``PathSpec``) — always counted once.
    2. Zero or more off-path sinks (``off_path_loads``) — capacitive load only.
    3. Fanout shorthand (``fanout``) — (fanout-1) extra copies of the on-path sink.
    4. Manual extra capacitance (``extra_cap_pf``) — pads, ESD, long wires.

    Load resolution order:

    * ``off_path_loads`` non-empty → ``Cin(next) + Σ Cin(sinks) + wire_cap(1+N) + extra``
    * otherwise → ``fanout × Cin(next) + wire_cap(fanout) + extra``

    Per-stage derate overrides (``delay_derate``, ``slew_derate``):
        ``None`` → inherit from :class:`AnalysisConfig` (most stages).
        Set to override for one cell (e.g. known aging or process variation).

    Examples::

        PathSpec(cell_name="INVD1", fanout=4)
        PathSpec(cell_name="NAND2D1",
                 off_path_loads=["DFFRPQ_D", "AOI22D1"],
                 extra_cap_pf=0.005)
        PathSpec(cell_name="BUFFD4", extra_cap_pf=0.050)   # pad driver
        PathSpec.model_validate({"cell_name": "INVD1", "fanout": 4})  # from YAML
    """

    cell_name: str
    fanout: int = Field(1, ge=1, description="Fanout count (≥ 1)")
    off_path_loads: list[str] = Field(default_factory=list)
    extra_cap_pf: float = Field(0.0, ge=0.0, description="Extra additive cap in pF")
    # Per-stage derate overrides — None → use AnalysisConfig global
    delay_derate: float | None = Field(
        None, ge=0.5, le=2.0,
        description="Delay derate override for this stage only. Range [0.5, 2.0].",
    )
    slew_derate: float | None = Field(
        None, ge=0.5, le=2.0,
        description="Slew derate override for this stage only. Range [0.5, 2.0].",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# AnalysisConfig — all analysis parameters (Pydantic, frozen)
# ---------------------------------------------------------------------------


class AnalysisConfig(BaseModel):
    """All analysis parameters for a timing run.

    Separates *how* to analyse from *what* to analyse (``list[PathSpec]``).
    Can be constructed from a TOML/YAML config file via ``model_validate``.

    Attributes:
        delay_derate: Multiplicative factor on each stage's raw delay.
            1.0 = nominal.  >1.0 = pessimistic (late corner / OCV).
            Range [0.5, 2.0].
        slew_derate: Multiplicative factor on each stage's output slew
            before propagating to the next stage.  Range [0.5, 2.0].
        arc_mode: ``'worst'`` — slowest arc at the (slew, load) point (default).
            ``'average'`` — ``Cell.delay_at()`` averages all arcs.
        wire_load: :class:`WireLoadModel` for wire parasitics.
            ``None`` = ideal wires.
        initial_slew_ns: Input slew for the first stage in ns.
            ``None`` = use FO4 slew from the first entry's normalizer.
        output_load_pf: Capacitive load on the last stage's output pin (pF).

    Example::

        cfg = AnalysisConfig(delay_derate=1.08, slew_derate=1.05,
                             wire_load=WireLoadModel.typical_7nm())
        cfg = AnalysisConfig.model_validate(toml.load("sta.toml")["analysis"])
    """

    delay_derate: float = Field(1.0, ge=0.5, le=2.0)
    slew_derate: float = Field(1.0, ge=0.5, le=2.0)
    arc_mode: Literal["worst", "average"] = "worst"
    wire_load: WireLoadModel | None = None
    initial_slew_ns: float | None = Field(None, ge=0.0)
    output_load_pf: float = Field(0.05, ge=0.0)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# _ResolvedStage — internal computation atom (frozen dataclass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedStage:
    """Internal.  Produced by resolve_manual() (and future resolve_netlist()).

    Contains everything propagate() needs for one stage — no dataset,
    no PathSpec, no netlist reference.  Users never construct this directly.

    All analysis choices (arc_mode, derates) and all lookup results (linear
    fallback coefficients, unit conversions) are baked in at resolution time,
    keeping propagate() a pure function.

    Attributes:
        cell_name:          For reporting and identifying the stage.
        cell:               Liberty Cell object with timing arc tables.
                            ``None`` → use linear fallback unconditionally.
        load_pf:            Fully resolved output load (wire + fanout×Cin + extras).
        t_div:              ns → library time units  (e.g. 1000.0 for ns→ps).
        c_div:              pF → library cap units   (e.g. 1000.0 for pF→fF).
        t_mul:              library time units → ns.
        arc_mode:           Baked from AnalysisConfig — 'worst' or 'average'.
        delay_derate:       Effective delay derate (stage override ∨ config global).
        slew_derate:        Effective slew derate.
        instance_name:      Instance identifier from netlist (empty for manual mode).
        linear_d0_ns:       Linear model intercept in ns (D0+k×load fallback).
        linear_k_ns_per_pf: Linear model slope in ns/pF.
        linear_r2:          R² of the linear fit (used for warning threshold).
        linear_slew_ref_pf: Reference load for slew approximation in linear mode
                            (typically FO4 load from the dataset normalizer).
        linear_missing_from_df: True when the cell was absent from the metrics
                            DataFrame at resolve time.  propagate() uses this to
                            emit a warning only when linear is actually invoked —
                            avoiding false-positive warnings for NLDM-capable cells
                            that happen to lack normalised metrics.
        is_skipped:         True for sequential cells — propagate() emits a
                            TimingPoint with method='skipped' and delay=0.
    """

    # Core (required)
    cell_name: str
    cell: object | None       # Liberty Cell; typed as object to avoid circular import
    load_pf: float
    t_div: float              # ns → lib_units
    c_div: float              # pF → lib_units
    t_mul: float              # lib_units → ns
    # Analysis choices (baked in — defaults match AnalysisConfig defaults)
    arc_mode: Literal["worst", "average"] = "worst"
    delay_derate: float = 1.0
    slew_derate: float = 1.0
    instance_name: str = ""
    # Linear fallback (baked in from dataset at resolve time)
    linear_d0_ns: float = 0.0
    linear_k_ns_per_pf: float = 0.0
    linear_r2: float = 1.0
    linear_slew_ref_pf: float = 0.0  # fo4_load_pf for slew approximation
    linear_missing_from_df: bool = False  # True → emit warning only if linear is used
    is_skipped: bool = False


# ---------------------------------------------------------------------------
# ManualResolution — structured return from resolve_manual()
# ---------------------------------------------------------------------------


class ManualResolution(NamedTuple):
    """Return value of :func:`resolve_manual`.

    Attributes:
        stages:          One :class:`_ResolvedStage` per input :class:`PathSpec`
                         (including skipped sequential stages — ``len(stages)``
                         always equals ``len(path)``).
        warnings:        Non-fatal issues detected during resolution (unknown cells,
                         saturation, sequential skips, multi-entry datasets, …).
        initial_slew_ns: Resolved input slew for the first stage (from
                         ``AnalysisConfig.initial_slew_ns`` or FO4 baseline).
    """

    stages: list[_ResolvedStage]
    warnings: list[str]
    initial_slew_ns: float


# ---------------------------------------------------------------------------
# TimingPoint / TimingPath — output types (plain dataclasses)
# ---------------------------------------------------------------------------


@dataclass
class TimingPoint:
    """Annotated result for one path stage.

    Attributes:
        name:          Cell name (manual mode) or instance/pin name (STA).
        arrival_ns:    Cumulative arrival time at the OUTPUT of this stage (ns).
        delay_ns:      Derated stage delay (ns).  0.0 for skipped stages.
        slew_ns:       Derated output transition time propagated to next stage (ns).
        load_pf:       Effective output load (pF).  0.0 for skipped stages.
        method:        ``'nldm'``, ``'linear'``, or ``'skipped'``.
        delay_derate:  Effective delay derate applied (1.0 for skipped).
        slew_derate:   Effective slew derate applied (1.0 for skipped).
        instance_name: Instance name from netlist; ``''`` for manual paths.
    """

    name: str
    arrival_ns: float
    delay_ns: float
    slew_ns: float
    load_pf: float
    method: str          # "nldm" | "linear" | "skipped"
    delay_derate: float
    slew_derate: float
    instance_name: str = ""


@dataclass
class TimingPath:
    """Result of a timing propagation run.

    This is a library-controlled output struct — do not construct directly.

    Attributes:
        points:         One :class:`TimingPoint` per input stage (including
                        sequential stages with ``method='skipped'``).
        total_delay_ns: Sum of derated stage delays (skipped stages contribute 0).
        warnings:       Always inspect — combines resolution warnings (unknown cells,
                        saturation, multi-entry) with propagation warnings (no arcs,
                        low R²).
    """

    points: list[TimingPoint]
    total_delay_ns: float
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# resolve_manual — factory: list[PathSpec] → ManualResolution
# ---------------------------------------------------------------------------


def resolve_manual(
    path: list[PathSpec],
    dataset: "Dataset",
    config: AnalysisConfig,
) -> ManualResolution:
    """Translate a manual path spec into resolved stages ready for propagation.

    Called once per timing run.  Performs all dataset lookups (cell Cin,
    linear-model coefficients, unit conversions) and stores results in each
    :class:`_ResolvedStage` so that :func:`propagate` is a pure function.

    The returned ``stages`` list has exactly ``len(path)`` entries; sequential
    cells are represented as ``_ResolvedStage(is_skipped=True)`` rather than
    being silently dropped.

    Args:
        path:    Ordered list of :class:`PathSpec` objects (input → output).
        dataset: Loaded :class:`~parsfet.data.Dataset`.
        config:  :class:`AnalysisConfig` controlling wire model, derates, arc mode.

    Returns:
        :class:`ManualResolution` — stages, warnings, resolved initial slew.

    Raises:
        ValueError: If no entries loaded, path is empty, or DataFrame is empty.
    """
    if not dataset.entries:
        raise ValueError("No entries loaded. Call load_files() first.")
    if not path:
        raise ValueError("path must contain at least one PathSpec.")

    # FO4 baseline from first entry (used for initial slew and slew approximation)
    entry = dataset.entries[0]
    fo4_slew_ns: float = 0.0
    fo4_load_pf: float = 0.0
    if entry.normalizer:
        fo4_slew_ns = entry.normalizer.baseline.fo4_slew
        fo4_load_pf = entry.normalizer.baseline.fo4_load

    initial_slew_ns = (
        config.initial_slew_ns if config.initial_slew_ns is not None else fo4_slew_ns
    )

    # Single to_dataframe() call — used for both Cin lookup and linear fallback.
    df = dataset.to_dataframe()
    if df.empty:
        raise ValueError("Dataset produced empty DataFrame — no cells with metrics.")
    cell_df = df.drop_duplicates(subset="cell").set_index("cell")
    dataset_mean_d0 = float(cell_df["raw_d0_ns"].mean())
    dataset_mean_k = float(cell_df["raw_k_ns_per_pf"].mean())

    # Build name-keyed Cell dict (first entry wins for duplicates after reversed loop).
    lib_cell_tuples: dict[str, tuple] = {}  # name → (Cell, t_div, c_div, t_mul)
    warnings: list[str] = []
    if len(dataset.entries) > 1:
        warnings.append(
            "Multiple library entries detected. NLDM arc walk uses first-entry-wins "
            "for duplicate cell names. Call combine() for consistent behaviour."
        )
    for e in reversed(dataset.entries):
        if e.library:
            t_mul = e.library.time_unit_ns   # lib_unit → ns
            c_mul = e.library.cap_unit_pf    # lib_unit → pF
            t_div = 1.0 / t_mul if t_mul else 1.0
            c_div = 1.0 / c_mul if c_mul else 1.0
            for c in e.library.cells.values():
                lib_cell_tuples[c.name] = (c, t_div, c_div, t_mul)

    wlm = config.wire_load
    stages: list[_ResolvedStage] = []

    for i, spec in enumerate(path):
        cell_name = spec.cell_name

        # --- Sequential cell guard ---
        # Emit a skipped stage so len(stages) == len(path) always.
        if cell_name in cell_df.index and bool(cell_df.loc[cell_name, "is_sequential"]):
            warnings.append(
                f"Stage {i} '{cell_name}': sequential cell (DFF/latch). "
                f"raw_d0_ns is clock-to-Q delay, not combinational. Stage skipped."
            )
            stages.append(_ResolvedStage(
                cell_name=cell_name,
                cell=None,
                load_pf=0.0,
                t_div=1.0, c_div=1.0, t_mul=1.0,
                is_skipped=True,
            ))
            continue

        # --- Resolve output load ---
        if i + 1 < len(path):
            next_name = path[i + 1].cell_name
            if next_name in cell_df.index:
                cin_next = float(cell_df.loc[next_name, "raw_input_cap_pf"])
            elif cell_name in cell_df.index:
                cin_next = float(cell_df.loc[cell_name, "raw_input_cap_pf"])
                warnings.append(
                    f"Stage {i} '{cell_name}': next cell '{next_name}' not found — "
                    f"using current cell Cin={cin_next:.4f} pF as estimate."
                )
            else:
                cin_next = 0.005
                warnings.append(
                    f"Stage {i} '{cell_name}': neither this nor next cell found — "
                    f"using 0.005 pF Cin fallback."
                )

            if spec.off_path_loads:
                off_cap = 0.0
                for sink in spec.off_path_loads:
                    if sink in cell_df.index:
                        off_cap += float(cell_df.loc[sink, "raw_input_cap_pf"])
                    else:
                        warnings.append(
                            f"Stage {i} '{cell_name}': off-path sink '{sink}' not "
                            f"found — Cin contribution ignored."
                        )
                wire_cap = wlm.wire_cap(1 + len(spec.off_path_loads)) if wlm else 0.0
                load_pf = cin_next + off_cap + spec.extra_cap_pf + wire_cap
            else:
                wire_cap = wlm.wire_cap(spec.fanout) if wlm else 0.0
                load_pf = spec.fanout * cin_next + spec.extra_cap_pf + wire_cap
        else:
            # Last stage — output pin load only (no wire cap; models external sink)
            load_pf = config.output_load_pf + spec.extra_cap_pf

        if fo4_load_pf > 0 and load_pf > 2.0 * fo4_load_pf:
            warnings.append(
                f"Stage {i} '{cell_name}': load {load_pf:.4f} pF > 2× FO4 load "
                f"({fo4_load_pf:.4f} pF). NLDM table clamped at edge — "
                f"delay may be underestimated at high fanout."
            )

        # --- Resolve effective derates ---
        eff_delay_derate = (
            spec.delay_derate if spec.delay_derate is not None else config.delay_derate
        )
        eff_slew_derate = (
            spec.slew_derate if spec.slew_derate is not None else config.slew_derate
        )

        # --- Bake linear fallback data (single dataset access, already in cell_df) ---
        # No warning here: if NLDM succeeds, these values are never used.
        # The warning fires lazily in propagate() only when linear is invoked.
        if cell_name not in cell_df.index:
            lin_d0 = dataset_mean_d0
            lin_k = dataset_mean_k
            lin_r2 = 1.0
            lin_missing = True
        else:
            row = cell_df.loc[cell_name]
            lin_d0 = float(row["raw_d0_ns"])
            lin_k = float(row["raw_k_ns_per_pf"])
            lin_r2 = (
                float(cell_df.loc[cell_name, "fit_r_squared"])
                if "fit_r_squared" in cell_df.columns
                else 1.0
            )
            lin_missing = False

        # --- Lookup cell object and unit factors ---
        tup = lib_cell_tuples.get(cell_name)
        cell_obj = tup[0] if tup else None
        t_div, c_div, t_mul = (tup[1], tup[2], tup[3]) if tup else (1.0, 1.0, 1.0)

        stages.append(_ResolvedStage(
            cell_name=cell_name,
            cell=cell_obj,
            load_pf=load_pf,
            t_div=t_div,
            c_div=c_div,
            t_mul=t_mul,
            arc_mode=config.arc_mode,
            delay_derate=eff_delay_derate,
            slew_derate=eff_slew_derate,
            linear_d0_ns=lin_d0,
            linear_k_ns_per_pf=lin_k,
            linear_r2=lin_r2,
            linear_slew_ref_pf=fo4_load_pf,
            linear_missing_from_df=lin_missing,
        ))

    return ManualResolution(
        stages=stages,
        warnings=warnings,
        initial_slew_ns=initial_slew_ns,
    )


# ---------------------------------------------------------------------------
# propagate — pure engine: list[_ResolvedStage] → TimingPath
# ---------------------------------------------------------------------------


def propagate(
    stages: list[_ResolvedStage],
    initial_slew_ns: float,
) -> TimingPath:
    """Pure timing propagation engine.

    Iterates over :class:`_ResolvedStage` objects, computing delay and slew at
    each stage using the NLDM arc tables (preferred) or the pre-baked linear
    model (fallback).  Emits one :class:`TimingPoint` per stage, including
    skipped sequential stages (``method='skipped'``, ``delay_ns=0``).

    This function has **no side effects** and makes **no I/O calls** — all cell
    data, unit conversions, analysis choices (arc_mode, derates), and linear
    fallback coefficients are baked into each :class:`_ResolvedStage` by
    :func:`resolve_manual` (or a future ``resolve_netlist``).

    Args:
        stages:          Resolved stages from :func:`resolve_manual`.
        initial_slew_ns: Input slew at the first stage in ns.

    Returns:
        :class:`TimingPath`.  Combine ``.warnings`` with
        :attr:`ManualResolution.warnings` for the full warning picture.
    """
    prop_warnings: list[str] = []
    current_slew_ns = initial_slew_ns
    arrival_ns = 0.0
    points: list[TimingPoint] = []

    for i, rs in enumerate(stages):
        cell_name = rs.cell_name

        # --- Sequential / skipped stage ---
        if rs.is_skipped:
            points.append(TimingPoint(
                name=cell_name,
                arrival_ns=arrival_ns,
                delay_ns=0.0,
                slew_ns=current_slew_ns,
                load_pf=0.0,
                method="skipped",
                delay_derate=1.0,
                slew_derate=1.0,
                instance_name=rs.instance_name,
            ))
            continue

        method = "linear"
        stage_delay = 0.0
        next_slew = current_slew_ns

        # --- NLDM arc walk ---
        if rs.cell is not None and getattr(rs.cell, "timing_arcs", None):
            method = "nldm"
            slew_lib = current_slew_ns * rs.t_div   # ns → lib units
            load_lib = rs.load_pf * rs.c_div        # pF → lib units

            if rs.arc_mode == "worst":
                # Scan all arcs; pick the one with highest delay.
                # Accept that arc's output slew — don't double-pessimise by
                # independently maximising delay and slew from different arcs.
                best_arc = None
                best_delay_lib = 0.0
                for arc in rs.cell.timing_arcs:
                    d = arc.delay_at(slew_lib, load_lib)
                    if d > best_delay_lib:
                        best_delay_lib = d
                        best_arc = arc
                if best_arc is not None:
                    stage_delay = best_delay_lib * rs.t_mul
                    next_slew = best_arc.output_transition_at(slew_lib, load_lib) * rs.t_mul
                else:
                    method = "linear"
                    prop_warnings.append(
                        f"Stage {i} '{cell_name}': arc_mode='worst' found no "
                        f"positive-delay arcs at slew={current_slew_ns:.4f} ns, "
                        f"load={rs.load_pf:.4f} pF. Falling back to linear model."
                    )
            else:  # 'average'
                stage_delay = rs.cell.delay_at(slew_lib, load_lib) * rs.t_mul
                next_slew = rs.cell.output_transition_at(slew_lib, load_lib) * rs.t_mul

        # --- Linear fallback (uses pre-baked coefficients — no dataset access) ---
        if method == "linear":
            if rs.linear_r2 < _LINEAR_R2_WARN_THRESHOLD:
                prop_warnings.append(
                    f"Stage {i} '{cell_name}': linear fallback R²={rs.linear_r2:.3f} "
                    f"(threshold {_LINEAR_R2_WARN_THRESHOLD}) — NLDM table nonlinear; "
                    f"estimate less reliable."
                )
            if rs.linear_missing_from_df:
                # Deferred from resolve_manual: only fire when linear is actually used.
                prop_warnings.append(
                    f"Stage {i} '{cell_name}': not found in metrics dataset — "
                    f"using dataset mean d0 and k for linear fallback."
                )
            elif rs.cell is None:
                prop_warnings.append(
                    f"Stage {i} '{cell_name}': no timing arcs (JSON import?) — "
                    f"using linear model D0+k×load."
                )
            stage_delay = rs.linear_d0_ns + rs.linear_k_ns_per_pf * rs.load_pf
            # Slew approximation: scale with √(load / fo4_load)
            if rs.linear_slew_ref_pf > 0:
                next_slew = max(current_slew_ns, 1e-3) * (
                    rs.load_pf / rs.linear_slew_ref_pf
                ) ** 0.5

        # --- Apply derates ---
        stage_delay *= rs.delay_derate
        next_slew *= rs.slew_derate

        arrival_ns += stage_delay
        points.append(TimingPoint(
            name=cell_name,
            arrival_ns=arrival_ns,
            delay_ns=stage_delay,
            slew_ns=next_slew,
            load_pf=rs.load_pf,
            method=method,
            delay_derate=rs.delay_derate,
            slew_derate=rs.slew_derate,
            instance_name=rs.instance_name,
        ))
        current_slew_ns = next_slew

    return TimingPath(
        points=points,
        total_delay_ns=sum(pt.delay_ns for pt in points),
        warnings=prop_warnings,
    )


# ---------------------------------------------------------------------------
# estimate_path_delay — convenience wrapper (resolve_manual + propagate)
# ---------------------------------------------------------------------------


def estimate_path_delay(
    dataset: "Dataset",
    path: list[PathSpec],
    config: AnalysisConfig | None = None,
) -> TimingPath:
    """Estimate total delay for a manual multi-stage combinational timing path.

    Convenience wrapper: calls :func:`resolve_manual` once (single
    ``to_dataframe()`` call), then :func:`propagate` (pure, no I/O).
    For advanced usage, call those two directly.

    Args:
        dataset: Loaded :class:`~parsfet.data.Dataset`.
        path:    Ordered list of :class:`PathSpec` objects (input → output).
        config:  :class:`AnalysisConfig` — uses defaults if ``None``.

    Returns:
        :class:`TimingPath`.  ``points`` has exactly ``len(path)`` entries;
        sequential stages appear with ``method='skipped'``.

    Example::

        import parsfet
        from parsfet.path_delay import estimate_path_delay, PathSpec, AnalysisConfig

        ds = parsfet.Dataset().load_files(["lib.lib"])
        result = estimate_path_delay(
            ds,
            [PathSpec(cell_name="INVD1", fanout=4),
             PathSpec(cell_name="BUFFD4")],
            AnalysisConfig(delay_derate=1.08,
                           wire_load=parsfet.WireLoadModel.typical_7nm()),
        )
        print(f"Total: {result.total_delay_ns*1e3:.1f} ps")
        for pt in result.points:
            print(f"  {pt.name:20s}  {pt.delay_ns*1e3:5.1f} ps  [{pt.method}]")
        for w in result.warnings:
            print("WARN:", w)
    """
    if config is None:
        config = AnalysisConfig()

    res = resolve_manual(path, dataset, config)
    internal = propagate(res.stages, res.initial_slew_ns)

    # Merge resolution warnings (cell lookups, saturation) with
    # propagation warnings (arc failures, low R²) into the final result.
    return TimingPath(
        points=internal.points,
        total_delay_ns=internal.total_delay_ns,
        warnings=res.warnings + internal.warnings,
    )
