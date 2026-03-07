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

    _ResolvedStage — fully-resolved per-stage state: cell ref, load, derates,
                     arc_mode, skip flag.
                     Produced by resolve_manual(); future resolve_netlist() will
                     produce the same type. Users never construct this directly.

**Output types** (engine-controlled dataclasses):

    TimingPoint    — per-stage annotated result (arrival, slew, load, method).
    TimingPath     — full result: list of TimingPoints + total_delay + warnings.

**Functions**:

    resolve_manual(path, dataset, config)  → ManualResolution
    propagate(stages, initial_slew_ns)     → TimingPath          (stateless)
    estimate_path_delay(dataset, path,     → TimingPath          (convenience)
                        config)

**Data Flow Architecture**:

.. mermaid::

    flowchart TD
       A[Dataset] -->|Cell Arcs, Cin| RM
       B[list: PathSpec] -->|Target Path| RM
       C[AnalysisConfig] -->|Derates, Arc Mode| RM
       RM[resolve_manual] -->|$O(N)$ Validation\nUnit Conversion\nCell Lookup| RS
       RS[list: _ResolvedStage] -->|Stateless Evaluation| P[propagate]
       C -.->|initial_slew_ns| P
       P --> TP[TimingPath]
       C -.->|period_ns| CS[compute_slack]
       TP -.->|Base Arrival/Slew| CS
       RS -.->|Setup/Hold/Clk2Q Arcs| CS
       CS --> Final[Final TimingPath with Slack]

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

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .data import Dataset


# ---------------------------------------------------------------------------
# WireLoadModel — Pydantic, frozen, process-node presets
# ---------------------------------------------------------------------------


class WireLoadModel(BaseModel):
    """Maps fanout count to wire capacitance in pF for a given process node.

    All values are approximations intended for early design exploration.
    Use SPEF for sign-off accuracy.

    **Mathematical Provenance**:
    The default process templates (e.g. `typical_7nm`, etc., when implemented)
    are derived from rule-of-thumb FO4 logic scaling factors rather than extracted
    SPEF or empirical foundry data. Future process node configurations should
    document whether they are based on standard linear scaling ($C_{wire} \\propto N_{fanout}$)
    or physical interconnect resistance-capacitance (RC) polynomial fitting.

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
    def wlm_template(cls) -> "WireLoadModel":
        """Template for wire load model."""
        return cls(fanout_cap_pf={1: 0.001, 4: 0.006, 16: 0.018}, base_cap_pf=0.002)

    @classmethod
    def zero(cls) -> "WireLoadModel":
        """No wire load model."""
        return cls(fanout_cap_pf={}, base_cap_pf=0.0)


# ---------------------------------------------------------------------------
# PathSpec — user input for ONE stage (Pydantic, frozen)
# ---------------------------------------------------------------------------


class PathSpec(BaseModel):
    """Specification for one gate stage on a timing path.

    The output net of this stage drives:

    1. Exactly one on-path sink (the next ``PathSpec``) — always counted once.
    2. Zero or more off-path sinks (``off_path_loads``) — capacitive load only.
    3. Fanout multiplier (``fanout``) — (fanout-1) extra copies of the on-path sink.
    4. Manual extra capacitance (``extra_cap_pf``) — pads, ESD, long wires.

    Load resolution order:

    * ``off_path_loads`` non-empty → ``Cin(next) + Σ Cin(sinks) + wire_cap(1+N) + extra``
    * otherwise → ``fanout × Cin(next) + wire_cap(fanout) + extra``

    Per-stage derate overrides (``delay_derate``, ``slew_derate``):
        Default behavior inherits from :class:`AnalysisConfig`.
        Set these to define an override for one cell (e.g. known aging or process variation).

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
            1.0 = nominal.  >1.0 = pessimistic (e.g. process variation margin).
            Range [0.5, 2.0].
        slew_derate: Multiplicative factor on each stage's output slew
            before propagating to the next stage.  Range [0.5, 2.0].
        arc_mode: ``'worst'`` — slowest arc at the (slew, load) point (default).
            ``'average'`` — ``Cell.delay_at()`` averages all arcs.
        wire_load: :class:`WireLoadModel` for wire parasitics.
            Defaults to ideal wires.
        initial_slew_ns: Input slew for the first stage in ns.
            Defaults to the clock slew.
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
    # --- Slack calculation ---
    period_ns: float | None = Field(
        None, gt=0.0,
        description="Target clock period. When set, compute_slack() is called "
                    "automatically by estimate_path_delay().",
    )
    clock_slew_ns: float = Field(
        0.05, gt=0.0,
        description="Clock input slew used for Clk\u2192Q and setup/hold constraint "
                    "table lookups (in ns).",
    )
    clock_uncertainty_ns: float = Field(
        0.0, ge=0.0,
        description="Clock uncertainty (jitter + skew) deducted from setup slack "
                    "and added to hold slack (in ns).",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# _ResolvedStage — internal computation atom (frozen dataclass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedStage:
    """Internal. Produced by resolve_manual() (and future resolve_netlist()).

    Contains required data for one stage computation — independent of dataset,
    PathSpec, or netlist reference. Users never construct this directly.

    All analysis choices (arc_mode, derates) and lookup results (unit
    conversions) are computed at resolution time, maintaining propagate() as a stateless function.

    Attributes:
        cell_name:     For reporting and identifying the stage.
        cell:          Liberty Cell object with timing arc tables.
                       ``None`` → no arcs available; propagate() emits delay=0 + warning.
        load_pf:       Fully resolved output load (wire + fanout×Cin + extras).
        t_div:         ns → library time units  (e.g. 1000.0 for ns→ps).
        c_div:         pF → library cap units   (e.g. 1000.0 for pF→fF).
        t_mul:         library time units → ns.
        arc_mode:      Baked from AnalysisConfig — 'worst' or 'average'.
        delay_derate:  Effective delay derate (stage override ∨ config global).
        slew_derate:   Effective slew derate.
        instance_name: Instance identifier from netlist (empty for manual mode).
        is_skipped:    True for sequential cells — propagate() emits a
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
        initial_slew_ns: Resolved input slew for the first stage.
    """

    stages: list[_ResolvedStage]
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
    """Result of a full timing run.

    The ``points`` list always has exactly ``len(path_spec)`` entries — sequential
    stages appear with ``method='skipped'`` and ``delay_ns=0``.

    **Combinational vs Sequential Timing Constraints:**
    ``total_delay_ns`` is strictly the summed combinational delay.  Clock-to-Q and
    setup/hold constraints are evaluated automatically by :func:`compute_slack`
    when :attr:`AnalysisConfig.period_ns` is specified.

    Slack formula::

        setup_slack = period_ns - clk_to_q_ns - total_delay_ns
                      - setup_time_ns - clock_uncertainty_ns
        hold_slack  = clk_to_q_ns + total_delay_ns
                      - hold_time_ns - clock_uncertainty_ns

    Attributes:
        points:          One :class:`TimingPoint` per :class:`PathSpec`.
        total_delay_ns:  Sum of combinational delays (ns). Sequential stages
                         contribute 0. Does **not** include Clk→Q.
        clk_to_q_ns:     Launch element clock-to-Q delay (ns). Excluded from calculations if missing.
        setup_time_ns:   Capture element setup-time constraint (ns). Excluded if missing.
        hold_time_ns:    Capture element hold-time constraint (ns). Excluded if missing.
        setup_slack_ns:  Setup-timing slack (ns). Negative = violation. Missing if required terms are absent.
        hold_slack_ns:   Hold-timing slack (ns). Negative = violation. Missing if required terms are absent.
    """

    points: list[TimingPoint]
    total_delay_ns: float
    # Sequential element constraint fields
    clk_to_q_ns: float | None = None
    setup_time_ns: float | None = None
    hold_time_ns: float | None = None
    setup_slack_ns: float | None = None
    hold_slack_ns: float | None = None


# ---------------------------------------------------------------------------
# resolve_manual - factory: list[PathSpec] → ManualResolution
# ---------------------------------------------------------------------------


def resolve_manual(
    path: list[PathSpec],
    dataset: "Dataset",
    config: AnalysisConfig,
) -> ManualResolution:
    """Translate a manual path spec into resolved stages ready for propagation.

    Performs all dataset lookups (cell Cin, unit conversions) and computes
    parameter constraints for each :class:`_ResolvedStage` $O(N)$ so that
    :func:`propagate` operates state-free.

    The returned ``stages`` list has exactly ``len(path)`` entries; sequential
    cells are represented as ``_ResolvedStage(is_skipped=True)``.

    Args:
        path:    Ordered list of :class:`PathSpec` objects (input → output).
        dataset: Loaded :class:`~parsfet.data.Dataset`.
        config:  :class:`AnalysisConfig` controlling wire model, derates, arc mode.

    Returns:
        :class:`ManualResolution` — computed stages and resolved initial slew.

    Raises:
        ValueError: If no entries loaded or if the path is empty.
    """
    dataset.resolve()   # trigger lazy combine before accessing library
    if not dataset.entries:
        raise ValueError("No entries loaded. Call load_files() first.")
    if not path:
        raise ValueError("path must contain at least one PathSpec.")

    # Build name-keyed Liberty Cell dict from the single combined entry.
    lib_cell_tuples: dict[str, tuple] = {}  # name → (Cell, t_div, c_div, t_mul, c_mul)
    lib = dataset.library
    if lib:
        t_mul = lib.time_unit_ns   # lib_unit → ns
        c_mul = lib.cap_unit_pf    # lib_unit → pF
        t_div = 1.0 / t_mul if t_mul else 1.0
        c_div = 1.0 / c_mul if c_mul else 1.0
        for c in lib.cells.values():
            lib_cell_tuples[c.name] = (c, t_div, c_div, t_mul, c_mul)

    # Default initial slew: explicit config value  →  clock_slew_ns (a standard
    # process-agnostic timing parameter defined in AnalysisConfig).
    initial_slew_ns = (
        config.initial_slew_ns if config.initial_slew_ns is not None
        else config.clock_slew_ns
    )

    # Helper: Cin of a cell in pF from Liberty pin capacitances.
    def _cin_pf(cell_name: str) -> float | None:
        tup = lib_cell_tuples.get(cell_name)
        if tup is None:
            return None
        cell_obj, _, _, _, c_mul_local = tup
        return cell_obj.total_input_capacitance * c_mul_local  # lib_units → pF

    wlm = config.wire_load
    stages: list[_ResolvedStage] = []

    for i, spec in enumerate(path):
        cell_name = spec.cell_name
        tup = lib_cell_tuples.get(cell_name)
        cell_obj = tup[0] if tup else None
        t_div, c_div, t_mul = (tup[1], tup[2], tup[3]) if tup else (1.0, 1.0, 1.0)

        if cell_obj is None:
            logger.warning("Stage %d (%s): not found in any loaded Liberty library.", i, cell_name)

        # --- Sequential cell detection (flag only — do NOT continue early) ---
        # Sequential element stages still need load_pf (for Clk→Q lookup) and cell_obj
        # (for clk_to_q_arcs / setup_arcs in compute_slack).
        is_seq = cell_obj is not None and cell_obj.is_sequential
        if is_seq:
            logger.info(
                "Stage %d (%s): Sequential cell. Clk→Q handled by compute_slack(). "
                "Skipping delay calculation in propagate().",
                i, cell_name
            )

        # --- Resolve output load (Cin from Liberty pin capacitances) ---
        if i + 1 < len(path):
            next_name = path[i + 1].cell_name
            cin_next = _cin_pf(next_name)
            if cin_next is None:
                cin_next = _cin_pf(cell_name)
                if cin_next is not None:
                    logger.warning(
                        "Stage %d (%s): Next cell (%s) not found in library. "
                        "Using current cell Cin=%.4f pF as load estimate.",
                        i, cell_name, next_name, cin_next
                    )
                else:
                    cin_next = 0.005
                    logger.warning(
                        "Stage %d (%s): Neither this nor next cell found. "
                        "Forcing 0.005 pF Cin fallback.",
                        i, cell_name
                    )

            if spec.off_path_loads:
                off_cap = 0.0
                for sink in spec.off_path_loads:
                    sink_cin = _cin_pf(sink)
                    if sink_cin is not None:
                        off_cap += sink_cin
                    else:
                        logger.warning(
                            "Stage %d (%s): off-path sink (%s) not found - Cin disregarded.",
                            i, cell_name, sink
                        )
                wire_cap = wlm.wire_cap(1 + len(spec.off_path_loads)) if wlm else 0.0
                load_pf = cin_next + off_cap + spec.extra_cap_pf + wire_cap
            else:
                wire_cap = wlm.wire_cap(spec.fanout) if wlm else 0.0
                load_pf = spec.fanout * cin_next + spec.extra_cap_pf + wire_cap
        else:
            # Last stage — output pin load only (no wire cap; models external sink)
            load_pf = config.output_load_pf + spec.extra_cap_pf

        # --- Resolve effective derates ---
        eff_delay_derate = (
            spec.delay_derate if spec.delay_derate is not None else config.delay_derate
        )
        eff_slew_derate = (
            spec.slew_derate if spec.slew_derate is not None else config.slew_derate
        )

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
            is_skipped=is_seq,
        ))

    return ManualResolution(
        stages=stages,
        initial_slew_ns=initial_slew_ns,
    )


# ---------------------------------------------------------------------------
# propagate - stateless evaluation: list[_ResolvedStage] → TimingPath
# ---------------------------------------------------------------------------


def propagate(
    stages: list[_ResolvedStage],
    initial_slew_ns: float,
) -> TimingPath:
    """Stateless timing propagation function.

    Iterates over :class:`_ResolvedStage` objects, computing delay and slew at
    each stage using the NLDM timing arcs. Emits one :class:`TimingPoint` per
    stage, including skipped sequential stages (``method='skipped'``, ``delay_ns=0``).
    Stages without a Liberty cell emit ``delay_ns=0`` and a warning.

    This function isolates computation from data retrieval. All cell
    data, unit conversions, and analysis choices are provided within each
    :class:`_ResolvedStage` by :func:`resolve_manual` (or a future
    ``resolve_netlist``).

    Args:
        stages:          Resolved stages from :func:`resolve_manual`.
        initial_slew_ns: Input slew at the first stage in ns.

    Returns:
        :class:`TimingPath`.
    """
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

        method = "nldm"
        stage_delay = 0.0
        next_slew = current_slew_ns

        # --- NLDM arc evaluation ---
        slew_lib = current_slew_ns * rs.t_div   # ns → lib units
        load_lib = rs.load_pf * rs.c_div        # pF → lib units

        if rs.cell is None or not getattr(rs.cell, "timing_arcs", None):
            logger.warning(
                "Stage %d (%s): no Liberty cell or timing arcs — delay set to 0. "
                "Verify %s exists in the loaded library.",
                i, cell_name, cell_name
            )
        elif rs.arc_mode == "worst":
            # Scan all arcs; pick the one with highest delay.
            # Use the output slew from the selected arc to avoid combining maximum delay
            # and maximum slew from different arcs, which overestimates resulting delay.
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
                logger.warning(
                    "Stage %d (%s): arc_mode='worst' found no positive-delay arcs "
                    "at slew=%.4f ns, load=%.4f pF — delay set to 0.",
                    i, cell_name, current_slew_ns, rs.load_pf
                )
        else:  # 'average'
            stage_delay = rs.cell.delay_at(slew_lib, load_lib) * rs.t_mul
            next_slew = rs.cell.output_transition_at(slew_lib, load_lib) * rs.t_mul

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
    )


# ---------------------------------------------------------------------------
# compute_slack - stateless post-processing: adds setup/hold slack to TimingPath
# ---------------------------------------------------------------------------


def compute_slack(
    path_result: TimingPath,
    stages: list[_ResolvedStage],
    config: AnalysisConfig,
) -> TimingPath:
    """Add setup/hold slack fields to an existing :class:`TimingPath`.

    This is a stateless post-processing step that reads timing arc
    data stored in :class:`_ResolvedStage` objects by :func:`resolve_manual`.
    It does **not** modify ``path_result`` — it returns a new :class:`TimingPath`.

    Called automatically by :func:`estimate_path_delay` when
    :attr:`AnalysisConfig.period_ns` is set.  Can be called manually on the
    output of :func:`propagate` for advanced workflows.

    Slack formulae::

        setup_slack = period_ns - clk_to_q_ns - total_delay_ns
                      - setup_time_ns - clock_uncertainty_ns
        hold_slack  = clk_to_q_ns + total_delay_ns
                      - hold_time_ns - clock_uncertainty_ns

    Args:
        path_result: Output of :func:`propagate` (or :func:`estimate_path_delay`).
        stages:      Resolved stages from :func:`resolve_manual` — same list
                     that was passed to :func:`propagate`.
        config:      :class:`AnalysisConfig` with ``period_ns`` set.

    Returns:
        New :class:`TimingPath` with slack fields populated. Log messages
        are emitted for issues like missing sequential arcs.

    Raises:
        ValueError: If ``config.period_ns`` is ``None``.
    """
    if config.period_ns is None:
        raise ValueError("compute_slack() requires config.period_ns to be set.")

    clk_to_q_ns: float | None = None
    setup_time_ns: float | None = None
    hold_time_ns: float | None = None

    # --- Find launch FF (first skipped stage with Clk→Q arcs) ---
    launch_stage: _ResolvedStage | None = None
    for rs in stages:
        if rs.is_skipped and rs.cell is not None and getattr(rs.cell, "clk_to_q_arcs", []):
            launch_stage = rs
            break

    # --- Find capture FF (last skipped stage with setup arcs) ---
    capture_stage: _ResolvedStage | None = None
    capture_point_idx: int | None = None
    for j, rs in enumerate(reversed(stages)):
        if rs.is_skipped and rs.cell is not None and getattr(rs.cell, "setup_arcs", []):
            capture_stage = rs
            capture_point_idx = len(stages) - 1 - j
            break

    # --- Compute Clk→Q ---
    if launch_stage is not None:
        clk_slew_lib = config.clock_slew_ns * launch_stage.t_div
        q_load_lib = launch_stage.load_pf * launch_stage.c_div
        arcs = launch_stage.cell.clk_to_q_arcs  # type: ignore[union-attr]
        if config.arc_mode == "worst":
            best_d = 0.0
            for arc in arcs:
                d = arc.delay_at(clk_slew_lib, q_load_lib)
                if d > best_d:
                    best_d = d
            clk_to_q_ns = best_d * launch_stage.t_mul if best_d > 0 else None
        else:  # 'average'
            arc_delays = [arc.delay_at(clk_slew_lib, q_load_lib) for arc in arcs]
            arc_delays = [d for d in arc_delays if d > 0]
            clk_to_q_ns = (sum(arc_delays) / len(arc_delays) * launch_stage.t_mul) if arc_delays else None
        if clk_to_q_ns is None:
            logger.warning(
                "compute_slack: launch sequential element Clk→Q arc returned 0 at the given "
                "clock_slew=%.4f ns and load=%.4f pF. Tclk2q omitted from slack.",
                config.clock_slew_ns, launch_stage.load_pf
            )
    else:
        logger.info(
            "compute_slack: no launch sequential element with Clk→Q arcs found. "
            "Tclk2q set to 0 (combinational-only path or cell has no clk_to_q_arcs)."
        )
        clk_to_q_ns = 0.0

    # --- Find data slew at capture FF's D pin ---
    data_slew_ns = 0.0
    if capture_point_idx is not None and capture_point_idx > 0:
        for pt in reversed(path_result.points[:capture_point_idx]):
            if pt.method != "skipped":
                data_slew_ns = pt.slew_ns
                break

    # --- Compute setup / hold ---
    if capture_stage is not None:
        data_slew_lib = data_slew_ns * capture_stage.t_div
        clk_slew_lib = config.clock_slew_ns * capture_stage.t_div
        # Setup: worst case = max of rise/fall constraint across all setup arcs
        # Filter at arc level (arc has at least one table) — not at value level,
        # since a valid setup or hold time of exactly 0.0 must not be silently dropped.
        setup_vals = [
            arc.constraint_at(data_slew_lib, clk_slew_lib)
            for arc in capture_stage.cell.setup_arcs  # type: ignore[union-attr]
            if arc.rise_constraint is not None or arc.fall_constraint is not None
        ]
        setup_time_ns = (max(setup_vals) * capture_stage.t_mul) if setup_vals else None
        # Hold: most constraining = max of rise/fall constraint across all hold arcs.
        # Hold times may be negative (FF tolerates late data) — never filter by value.
        hold_vals = [
            arc.constraint_at(data_slew_lib, clk_slew_lib)
            for arc in capture_stage.cell.hold_arcs  # type: ignore[union-attr]
            if arc.rise_constraint is not None or arc.fall_constraint is not None
        ]
        hold_time_ns = (max(hold_vals) * capture_stage.t_mul) if hold_vals else None
    else:
        logger.info(
            "compute_slack: no capture sequential element with setup arcs found. "
            "Setup and hold slack omitted."
        )

    # --- Compute final slack values ---
    setup_slack: float | None = None
    hold_slack: float | None = None
    if setup_time_ns is not None and clk_to_q_ns is not None:
        setup_slack = (
            config.period_ns
            - clk_to_q_ns
            - path_result.total_delay_ns
            - setup_time_ns
            - config.clock_uncertainty_ns
        )
    if hold_time_ns is not None and clk_to_q_ns is not None:
        hold_slack = (
            clk_to_q_ns
            + path_result.total_delay_ns
            - hold_time_ns
            - config.clock_uncertainty_ns
        )

    return TimingPath(
        points=path_result.points,
        total_delay_ns=path_result.total_delay_ns,
        clk_to_q_ns=clk_to_q_ns,
        setup_time_ns=setup_time_ns,
        hold_time_ns=hold_time_ns,
        setup_slack_ns=setup_slack,
        hold_slack_ns=hold_slack,
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

    Convenience wrapper: calls :func:`resolve_manual` once,
    then :func:`propagate` (stateless execution).
    For advanced usage, call those two functions independently.

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
    """
    if config is None:
        config = AnalysisConfig()

    res = resolve_manual(path, dataset, config)
    result = propagate(res.stages, res.initial_slew_ns)

    # Run slack computation automatically when a clock period is specified.
    if config.period_ns is not None:
        result = compute_slack(result, res.stages, config)

    return result
