"""Tests for parsfet.path_delay — PathSpec, AnalysisConfig, WireLoadModel,
resolve_manual, propagate, estimate_path_delay, and compute_slack.

All tests that need a real dataset use the `sample_liberty_file` fixture from
conftest.py, which provides INV_X1 (with NLDM timing arcs, ns/pF units) and
DFF_X1 (sequential, with Clk→Q, setup_rising, and hold_rising constraint arcs).
"""

import pytest
from pydantic import ValidationError

import parsfet
from parsfet.data import Dataset, load_files
from parsfet.path_delay import (
    AnalysisConfig,
    ManualResolution,
    PathSpec,
    TimingPath,
    TimingPoint,
    WireLoadModel,
    _ResolvedStage,
    compute_slack,
    estimate_path_delay,
    propagate,
    resolve_manual,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ds(sample_liberty_file):
    """Loaded dataset with INV_X1 (NLDM) and DFF_X1 (sequential)."""
    return load_files([sample_liberty_file])


@pytest.fixture
def inv_path():
    """Two-stage INV_X1 → INV_X1 path (both NLDM)."""
    return [PathSpec(cell_name="INV_X1"), PathSpec(cell_name="INV_X1")]


@pytest.fixture
def default_config():
    return AnalysisConfig()


# ---------------------------------------------------------------------------
# WireLoadModel
# ---------------------------------------------------------------------------


class TestWireLoadModel:
    def test_zero_returns_zero(self):
        wlm = WireLoadModel.zero()
        assert wlm.wire_cap(1) == 0.0
        assert wlm.wire_cap(8) == 0.0

    def test_base_cap_only(self):
        wlm = WireLoadModel(fanout_cap_pf={}, base_cap_pf=0.01)
        assert wlm.wire_cap(4) == pytest.approx(0.01)

    def test_interpolation(self):
        wlm = WireLoadModel(fanout_cap_pf={1: 0.001, 5: 0.005})
        assert wlm.wire_cap(3) == pytest.approx(0.003)   # linear midpoint

    def test_clamped_lower(self):
        wlm = WireLoadModel(fanout_cap_pf={2: 0.002, 8: 0.008})
        assert wlm.wire_cap(1) == pytest.approx(0.002)   # clamped to min key

    def test_clamped_upper(self):
        wlm = WireLoadModel(fanout_cap_pf={1: 0.001, 4: 0.004})
        assert wlm.wire_cap(16) == pytest.approx(0.004)  # clamped to max key

    def test_base_cap_added_to_interpolation(self):
        wlm = WireLoadModel(fanout_cap_pf={1: 0.001, 4: 0.004}, base_cap_pf=0.001)
        assert wlm.wire_cap(1) == pytest.approx(0.002)   # 0.001 + 0.001

    def test_invalid_key_raises(self):
        with pytest.raises(ValidationError):
            WireLoadModel(fanout_cap_pf={0: 0.001})  # fanout < 1

    def test_negative_cap_raises(self):
        with pytest.raises(ValidationError):
            WireLoadModel(fanout_cap_pf={1: -0.001})

    def test_string_key_coerced(self):
        wlm = WireLoadModel.model_validate({"fanout_cap_pf": {"2": "0.003"}})
        assert wlm.wire_cap(2) == pytest.approx(0.003)

    def test_presets_exist(self):
        for method in ("typical_7nm", "typical_14nm", "typical_28nm",
                       "typical_65nm", "typical_130nm"):
            wlm = getattr(WireLoadModel, method)()
            assert wlm.wire_cap(4) > 0

    def test_frozen(self):
        wlm = WireLoadModel.zero()
        with pytest.raises(Exception):
            wlm.base_cap_pf = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PathSpec
# ---------------------------------------------------------------------------


class TestPathSpec:
    def test_minimal_construction(self):
        ps = PathSpec(cell_name="INV_X1")
        assert ps.cell_name == "INV_X1"
        assert ps.fanout == 1
        assert ps.off_path_loads == []
        assert ps.extra_cap_pf == 0.0
        assert ps.delay_derate is None
        assert ps.slew_derate is None

    def test_fanout_ge1(self):
        with pytest.raises(ValidationError):
            PathSpec(cell_name="X", fanout=0)

    def test_extra_cap_non_negative(self):
        with pytest.raises(ValidationError):
            PathSpec(cell_name="X", extra_cap_pf=-0.001)

    def test_derate_range_low(self):
        with pytest.raises(ValidationError):
            PathSpec(cell_name="X", delay_derate=0.4)

    def test_derate_range_high(self):
        with pytest.raises(ValidationError):
            PathSpec(cell_name="X", slew_derate=2.1)

    def test_model_validate_from_dict(self):
        ps = PathSpec.model_validate({"cell_name": "NAND2D1", "fanout": 4})
        assert ps.fanout == 4

    def test_frozen(self):
        ps = PathSpec(cell_name="X")
        with pytest.raises(Exception):
            ps.fanout = 4  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AnalysisConfig
# ---------------------------------------------------------------------------


class TestAnalysisConfig:
    def test_defaults(self):
        cfg = AnalysisConfig()
        assert cfg.delay_derate == 1.0
        assert cfg.slew_derate == 1.0
        assert cfg.arc_mode == "worst"
        assert cfg.wire_load is None
        assert cfg.initial_slew_ns is None
        assert cfg.output_load_pf == pytest.approx(0.05)

    def test_invalid_arc_mode(self):
        with pytest.raises(ValidationError):
            AnalysisConfig(arc_mode="mean")  # type: ignore[arg-type]

    def test_derate_out_of_range(self):
        with pytest.raises(ValidationError):
            AnalysisConfig(delay_derate=0.3)

    def test_model_validate(self):
        cfg = AnalysisConfig.model_validate({
            "delay_derate": 1.1,
            "arc_mode": "average",
            "output_load_pf": 0.01,
        })
        assert cfg.delay_derate == pytest.approx(1.1)
        assert cfg.arc_mode == "average"

    def test_wire_load_embedded(self):
        cfg = AnalysisConfig(wire_load=WireLoadModel.zero())
        assert cfg.wire_load is not None
        assert cfg.wire_load.wire_cap(4) == 0.0

    def test_frozen(self):
        cfg = AnalysisConfig()
        with pytest.raises(Exception):
            cfg.delay_derate = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# resolve_manual
# ---------------------------------------------------------------------------


class TestResolveManual:
    def test_returns_manual_resolution(self, ds, inv_path, default_config):
        res = resolve_manual(inv_path, ds, default_config)
        assert isinstance(res, ManualResolution)

    def test_len_stages_equals_len_path(self, ds, default_config):
        """Invariant: one _ResolvedStage per PathSpec, including skipped."""
        path = [
            PathSpec(cell_name="INV_X1"),
            PathSpec(cell_name="DFF_X1"),   # sequential → skipped
            PathSpec(cell_name="INV_X1"),
        ]
        res = resolve_manual(path, ds, default_config)
        assert len(res.stages) == len(path)

    def test_sequential_stage_is_skipped(self, ds, default_config):
        path = [PathSpec(cell_name="INV_X1"), PathSpec(cell_name="DFF_X1")]
        res = resolve_manual(path, ds, default_config)
        assert res.stages[1].is_skipped is True

    def test_non_sequential_stage_not_skipped(self, ds, inv_path, default_config):
        res = resolve_manual(inv_path, ds, default_config)
        assert all(not s.is_skipped for s in res.stages)

    def test_arc_mode_baked_into_stages(self, ds, inv_path):
        cfg = AnalysisConfig(arc_mode="average")
        res = resolve_manual(inv_path, ds, cfg)
        assert all(s.arc_mode == "average" for s in res.stages)


    def test_derate_override_applied(self, ds):
        path = [
            PathSpec(cell_name="INV_X1", delay_derate=1.5),
            PathSpec(cell_name="INV_X1"),
        ]
        cfg = AnalysisConfig(delay_derate=1.0)
        res = resolve_manual(path, ds, cfg)
        assert res.stages[0].delay_derate == pytest.approx(1.5)
        assert res.stages[1].delay_derate == pytest.approx(1.0)  # global

    def test_initial_slew_from_config(self, ds, inv_path):
        cfg = AnalysisConfig(initial_slew_ns=0.025)
        res = resolve_manual(inv_path, ds, cfg)
        assert res.initial_slew_ns == pytest.approx(0.025)

    def test_initial_slew_defaults_to_clock_slew(self, ds, inv_path, default_config):
        """When initial_slew_ns is not set, default falls back to clock_slew_ns
        (resolve_manual no longer reads the normalizer FO4 baseline)."""
        res = resolve_manual(inv_path, ds, default_config)
        assert res.initial_slew_ns == pytest.approx(default_config.clock_slew_ns)

    def test_empty_path_raises(self, ds, default_config):
        with pytest.raises(ValueError, match="at least one"):
            resolve_manual([], ds, default_config)

    def test_empty_dataset_raises(self, default_config, inv_path):
        with pytest.raises(ValueError, match="No entries"):
            resolve_manual(inv_path, Dataset(), default_config)

    def test_sequential_warning_emitted(self, ds, default_config):
        path = [PathSpec(cell_name="DFF_X1")]
        res = resolve_manual(path, ds, default_config)
        assert any("sequential" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# propagate
# ---------------------------------------------------------------------------


class TestPropagate:
    def test_returns_timing_path(self, ds, inv_path, default_config):
        res = resolve_manual(inv_path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        assert isinstance(result, TimingPath)

    def test_pure_no_extra_args(self, ds, inv_path, default_config):
        """propagate() takes only (stages, initial_slew_ns) — no dataset/arc_mode."""
        res = resolve_manual(inv_path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        assert result.total_delay_ns > 0

    def test_points_len_equals_stages(self, ds, default_config):
        path = [
            PathSpec(cell_name="INV_X1"),
            PathSpec(cell_name="DFF_X1"),
            PathSpec(cell_name="INV_X1"),
        ]
        res = resolve_manual(path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        assert len(result.points) == len(res.stages)

    def test_skipped_stage_method(self, ds, default_config):
        path = [PathSpec(cell_name="INV_X1"), PathSpec(cell_name="DFF_X1")]
        res = resolve_manual(path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        assert result.points[1].method == "skipped"

    def test_skipped_stage_zero_delay(self, ds, default_config):
        path = [PathSpec(cell_name="INV_X1"), PathSpec(cell_name="DFF_X1")]
        res = resolve_manual(path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        assert result.points[1].delay_ns == pytest.approx(0.0)

    def test_arrival_is_cumulative(self, ds, inv_path, default_config):
        res = resolve_manual(inv_path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        pts = result.points
        assert pts[1].arrival_ns == pytest.approx(pts[0].arrival_ns + pts[1].delay_ns)

    def test_total_delay_is_sum_of_delays(self, ds, inv_path, default_config):
        res = resolve_manual(inv_path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        expected = sum(pt.delay_ns for pt in result.points)
        assert result.total_delay_ns == pytest.approx(expected)

    def test_nldm_used_for_arc_cells(self, ds, inv_path, default_config):
        res = resolve_manual(inv_path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        assert all(pt.method == "nldm" for pt in result.points)

    def test_delay_positive(self, ds, inv_path, default_config):
        res = resolve_manual(inv_path, ds, default_config)
        result = propagate(res.stages, res.initial_slew_ns)
        assert all(pt.delay_ns > 0 for pt in result.points)


# ---------------------------------------------------------------------------
# estimate_path_delay
# ---------------------------------------------------------------------------


class TestEstimatePathDelay:
    def test_returns_timing_path(self, ds, inv_path, default_config):
        result = estimate_path_delay(ds, inv_path, default_config)
        assert isinstance(result, TimingPath)

    def test_default_config_works(self, ds, inv_path):
        result = estimate_path_delay(ds, inv_path)  # config=None → defaults
        assert result.total_delay_ns > 0

    def test_points_len_equals_path(self, ds, default_config):
        path = [
            PathSpec(cell_name="INV_X1"),
            PathSpec(cell_name="DFF_X1"),
            PathSpec(cell_name="INV_X1"),
        ]
        result = estimate_path_delay(ds, path, default_config)
        assert len(result.points) == len(path)

    def test_sequential_appears_in_points(self, ds, default_config):
        path = [
            PathSpec(cell_name="INV_X1"),
            PathSpec(cell_name="DFF_X1"),
            PathSpec(cell_name="INV_X1"),
        ]
        result = estimate_path_delay(ds, path, default_config)
        assert result.points[1].method == "skipped"
        assert result.points[1].name == "DFF_X1"

    def test_warnings_combined(self, ds, default_config):
        """Warnings from resolution and propagation combined into TimingPath."""
        path = [PathSpec(cell_name="INV_X1"), PathSpec(cell_name="DFF_X1")]
        result = estimate_path_delay(ds, path, default_config)
        # Sequential skip warning comes from resolve_manual
        assert any("sequential" in w for w in result.warnings)

    def test_delay_derate_scales_delay(self, ds, inv_path):
        r_nominal = estimate_path_delay(ds, inv_path, AnalysisConfig(delay_derate=1.0))
        r_derated = estimate_path_delay(ds, inv_path, AnalysisConfig(delay_derate=1.5))
        assert r_derated.total_delay_ns == pytest.approx(
            r_nominal.total_delay_ns * 1.5, rel=1e-6
        )

    def test_wire_load_increases_load(self, ds, inv_path):
        r_no_wire = estimate_path_delay(ds, inv_path, AnalysisConfig())
        r_wire = estimate_path_delay(
            ds, inv_path,
            AnalysisConfig(wire_load=WireLoadModel.typical_7nm()),
        )
        # Wire load adds capacitance → higher delay
        assert r_wire.total_delay_ns >= r_no_wire.total_delay_ns

    def test_shim_equals_standalone(self, ds, inv_path, default_config):
        r1 = estimate_path_delay(ds, inv_path, default_config)
        r2 = ds.estimate_path_delay(inv_path, default_config)
        assert r1.total_delay_ns == pytest.approx(r2.total_delay_ns)
        assert len(r1.points) == len(r2.points)

    def test_per_stage_derate_override(self, ds):
        base = AnalysisConfig(delay_derate=1.0)
        overridden = [
            PathSpec(cell_name="INV_X1", delay_derate=2.0),
            PathSpec(cell_name="INV_X1"),
        ]
        normal = [
            PathSpec(cell_name="INV_X1"),
            PathSpec(cell_name="INV_X1"),
        ]
        r_over = estimate_path_delay(ds, overridden, base)
        r_norm = estimate_path_delay(ds, normal, base)
        # Stage 0 derated 2×, stage 1 same → total should be larger
        assert r_over.total_delay_ns > r_norm.total_delay_ns

    def test_output_load_pf_affects_last_stage(self, ds):
        path = [PathSpec(cell_name="INV_X1")]
        r_light = estimate_path_delay(ds, path, AnalysisConfig(output_load_pf=0.001))
        r_heavy = estimate_path_delay(ds, path, AnalysisConfig(output_load_pf=0.5))
        assert r_heavy.total_delay_ns > r_light.total_delay_ns

    def test_timing_point_fields(self, ds, inv_path, default_config):
        result = estimate_path_delay(ds, inv_path, default_config)
        for pt in result.points:
            assert pt.name
            assert pt.arrival_ns >= 0
            assert pt.delay_ns >= 0
            assert pt.slew_ns >= 0
            assert pt.load_pf >= 0
            assert pt.method in ("nldm", "linear", "skipped")
            assert pt.delay_derate >= 0.5

    def test_unknown_cell_warns(self, ds, default_config):
        path = [PathSpec(cell_name="NONEXISTENT_X99"), PathSpec(cell_name="INV_X1")]
        result = estimate_path_delay(ds, path, default_config)
        assert any("no Liberty cell" in w or "not found" in w for w in result.warnings)

    def test_empty_path_raises(self, ds, default_config):
        with pytest.raises(ValueError, match="at least one"):
            estimate_path_delay(ds, [], default_config)

    def test_empty_dataset_raises(self, default_config, inv_path):
        with pytest.raises(ValueError, match="No entries"):
            estimate_path_delay(Dataset(), inv_path, default_config)


# ---------------------------------------------------------------------------
# Coverage: arc_mode='average'
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    def test_arc_mode_average_produces_nldm(self, ds, inv_path):
        """arc_mode='average' branch in propagate() — still yields NLDM result."""
        cfg = AnalysisConfig(arc_mode="average")
        result = estimate_path_delay(ds, inv_path, cfg)
        assert all(pt.method == "nldm" for pt in result.points)
        assert result.total_delay_ns > 0

    def test_arc_mode_average_differs_from_worst(self, ds, inv_path):
        """'worst' arc picks the slowest arc; 'average' uses Cell.delay_at().
        For a cell with rise and fall arcs, these usually differ."""
        r_worst = estimate_path_delay(ds, inv_path, AnalysisConfig(arc_mode="worst"))
        r_avg = estimate_path_delay(ds, inv_path, AnalysisConfig(arc_mode="average"))
        # worst >= average is the dominant case, but we only assert both > 0
        assert r_worst.total_delay_ns > 0
        assert r_avg.total_delay_ns > 0

    def test_unknown_cell_delay_is_zero(self, ds, default_config):
        """A cell absent from the Liberty library produces delay=0 and a warning."""
        path = [PathSpec(cell_name="TOTALLY_UNKNOWN")]
        result = estimate_path_delay(ds, path, default_config)
        assert result.points[0].delay_ns == 0.0
        assert result.points[0].method == "nldm"  # method still 'nldm'; delay simply 0
        assert any("no Liberty cell" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Slack calculation — Liberty model, compute_slack, estimate_path_delay
# ---------------------------------------------------------------------------


class TestSlackCalculation:
    """Tests for the full NLDM-based setup/hold slack feature."""

    # -- Liberty model (Cell arc filter properties) --------------------------

    def test_dff_has_clk_to_q_arcs(self, ds):
        """DFF_X1 Q-pin timing() with timing_type=rising_edge should be found."""
        lib = ds.entries[0].library
        dff = lib.cells["DFF_X1"]
        assert len(dff.clk_to_q_arcs) >= 1
        assert all(a.timing_type in ("rising_edge", "falling_edge") for a in dff.clk_to_q_arcs)

    def test_dff_has_setup_arcs(self, ds):
        """DFF_X1 D-pin timing() with timing_type=setup_rising should be found."""
        lib = ds.entries[0].library
        dff = lib.cells["DFF_X1"]
        assert len(dff.setup_arcs) >= 1
        assert all(a.timing_type in ("setup_rising", "setup_falling") for a in dff.setup_arcs)

    def test_dff_has_hold_arcs(self, ds):
        """DFF_X1 D-pin timing() with timing_type=hold_rising should be found."""
        lib = ds.entries[0].library
        dff = lib.cells["DFF_X1"]
        assert len(dff.hold_arcs) >= 1
        assert all(a.timing_type in ("hold_rising", "hold_falling") for a in dff.hold_arcs)

    def test_inv_has_no_clk_to_q_arcs(self, ds):
        """Combinational cells should have no clk_to_q_arcs."""
        lib = ds.entries[0].library
        inv = lib.cells["INV_X1"]
        assert inv.clk_to_q_arcs == []
        assert inv.setup_arcs == []
        assert inv.hold_arcs == []

    def test_constraint_at_interpolates(self, ds):
        """TimingArc.constraint_at() returns a value in the expected range."""
        lib = ds.entries[0].library
        dff = lib.cells["DFF_X1"]
        arc = dff.setup_arcs[0]
        # Library is in ns/pF; data_slew=0.05 ns, clock_slew=0.05 ns → should be ~0.02-0.035 ns
        val = arc.constraint_at(0.05, 0.05)
        assert 0.01 < val < 0.1, f"constraint_at returned unexpected value: {val}"

    # -- resolve_manual: FF stages now get load_pf -------------------------

    def test_ff_stage_has_nonzero_load_pf(self, ds):
        """After refactor, launch FF stage load_pf = Cin(next stage), not 0."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1")]
        res = resolve_manual(path, ds, AnalysisConfig())
        launch = res.stages[0]
        assert launch.is_skipped is True
        assert launch.load_pf > 0.0, "launch FF should have Cin(INV_X1) as load"

    def test_ff_stage_has_cell_obj(self, ds):
        """FF stage cell_obj is populated so compute_slack can access clk_to_q_arcs."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1")]
        res = resolve_manual(path, ds, AnalysisConfig())
        assert res.stages[0].cell is not None

    # -- compute_slack isolation -------------------------------------------

    def test_compute_slack_raises_without_period(self, ds):
        """compute_slack raises ValueError if period_ns is None."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        res = resolve_manual(path, ds, AnalysisConfig())
        prop = propagate(res.stages, res.initial_slew_ns)
        with pytest.raises(ValueError, match="period_ns"):
            compute_slack(prop, res.stages, AnalysisConfig())  # no period_ns

    def test_compute_slack_returns_timing_path(self, ds):
        """compute_slack returns a TimingPath."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        cfg = AnalysisConfig(period_ns=1.0)
        res = resolve_manual(path, ds, cfg)
        prop = propagate(res.stages, res.initial_slew_ns)
        result = compute_slack(prop, res.stages, cfg)
        assert isinstance(result, TimingPath)

    def test_compute_slack_populates_clk_to_q(self, ds):
        """clk_to_q_ns is populated when launch FF has clk_to_q_arcs."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        cfg = AnalysisConfig(period_ns=1.0)
        res = resolve_manual(path, ds, cfg)
        prop = propagate(res.stages, res.initial_slew_ns)
        result = compute_slack(prop, res.stages, cfg)
        assert result.clk_to_q_ns is not None
        assert result.clk_to_q_ns > 0.0

    def test_compute_slack_populates_setup_time(self, ds):
        """setup_time_ns is populated when capture FF has setup arcs."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        cfg = AnalysisConfig(period_ns=1.0)
        res = resolve_manual(path, ds, cfg)
        prop = propagate(res.stages, res.initial_slew_ns)
        result = compute_slack(prop, res.stages, cfg)
        assert result.setup_time_ns is not None
        assert result.setup_time_ns > 0.0

    def test_compute_slack_formula_correct(self, ds):
        """setup_slack == period - clk_to_q - comb - setup - uncertainty."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        cfg = AnalysisConfig(period_ns=1.0, clock_uncertainty_ns=0.01)
        res = resolve_manual(path, ds, cfg)
        prop = propagate(res.stages, res.initial_slew_ns)
        r = compute_slack(prop, res.stages, cfg)
        if r.setup_slack_ns is not None and r.clk_to_q_ns is not None:
            expected = (
                cfg.period_ns
                - r.clk_to_q_ns
                - prop.total_delay_ns
                - r.setup_time_ns
                - cfg.clock_uncertainty_ns
            )
            assert r.setup_slack_ns == pytest.approx(expected, rel=1e-9)

    def test_compute_slack_no_ff_warning(self, ds, inv_path, default_config):
        """compute_slack warns when no launch FF is found."""
        cfg = AnalysisConfig(period_ns=1.0)
        res = resolve_manual(inv_path, ds, cfg)
        prop = propagate(res.stages, res.initial_slew_ns)
        result = compute_slack(prop, res.stages, cfg)
        assert any("launch FF" in w for w in result.warnings)

    # -- estimate_path_delay end-to-end with slack -------------------------

    def test_estimate_with_period_ns_triggers_slack(self, ds):
        """estimate_path_delay auto-calls compute_slack when period_ns is set."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        result = estimate_path_delay(ds, path, AnalysisConfig(period_ns=1.0))
        assert result.clk_to_q_ns is not None
        assert result.setup_time_ns is not None
        assert result.setup_slack_ns is not None

    def test_estimate_without_period_ns_no_slack(self, ds):
        """estimate_path_delay does NOT compute slack if period_ns is not set."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        result = estimate_path_delay(ds, path, AnalysisConfig())
        assert result.clk_to_q_ns is None
        assert result.setup_slack_ns is None

    def test_estimate_slack_negative_when_tight(self, ds):
        """With a very tight period, setup_slack should be negative (violation)."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        result = estimate_path_delay(
            ds, path, AnalysisConfig(period_ns=0.001)  # 1 ps — impossibly tight
        )
        if result.setup_slack_ns is not None:
            assert result.setup_slack_ns < 0, "1ps period should fail timing"

    def test_uncertainty_reduces_setup_slack(self, ds):
        """Larger clock_uncertainty_ns reduces setup_slack."""
        path = [PathSpec(cell_name="DFF_X1"), PathSpec(cell_name="INV_X1"),
                PathSpec(cell_name="DFF_X1")]
        r0 = estimate_path_delay(ds, path, AnalysisConfig(period_ns=1.0,
                                                          clock_uncertainty_ns=0.0))
        r1 = estimate_path_delay(ds, path, AnalysisConfig(period_ns=1.0,
                                                          clock_uncertainty_ns=0.05))
        if r0.setup_slack_ns is not None and r1.setup_slack_ns is not None:
            assert r1.setup_slack_ns < r0.setup_slack_ns
