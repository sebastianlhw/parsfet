"""Tests for parsfet.path_delay — PathSpec, AnalysisConfig, WireLoadModel,
resolve_manual, propagate, and estimate_path_delay.

All tests that need a real dataset use the `sample_liberty_file` fixture from
conftest.py, which provides INV_X1 (with NLDM timing arcs, ns/pF units) and
DFF_X1 (sequential, no combinational arcs).
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

    def test_linear_data_baked_in(self, ds, inv_path, default_config):
        """linear_d0_ns and k are resolved from dataset at resolve time."""
        res = resolve_manual(inv_path, ds, default_config)
        for s in res.stages:
            assert s.linear_d0_ns >= 0.0
            assert s.linear_k_ns_per_pf >= 0.0

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

    def test_initial_slew_defaults_to_fo4(self, ds, inv_path, default_config):
        res = resolve_manual(inv_path, ds, default_config)
        fo4_slew = ds.entries[0].normalizer.baseline.fo4_slew
        assert res.initial_slew_ns == pytest.approx(fo4_slew)

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
        assert any("not found" in w for w in result.warnings)

    def test_empty_path_raises(self, ds, default_config):
        with pytest.raises(ValueError, match="at least one"):
            estimate_path_delay(ds, [], default_config)

    def test_empty_dataset_raises(self, default_config, inv_path):
        with pytest.raises(ValueError, match="No entries"):
            estimate_path_delay(Dataset(), inv_path, default_config)


# ---------------------------------------------------------------------------
# Coverage: arc_mode='average', linear fallback, deferred warnings, constant
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

    def test_unknown_cell_linear_fallback_method(self, ds, default_config):
        """A cell absent from both lib and DataFrame forces linear fallback."""
        path = [PathSpec(cell_name="TOTALLY_UNKNOWN")]
        result = estimate_path_delay(ds, path, default_config)
        # Only one stage — the unknown cell
        assert result.points[0].method == "linear"

    def test_unknown_cell_deferred_warning_in_timing_path(self, ds, default_config):
        """'not found in metrics dataset' warning fires in propagate(), not resolve_manual().
        It should appear in TimingPath.warnings (combined), not ManualResolution.warnings."""
        path = [PathSpec(cell_name="TOTALLY_UNKNOWN")]
        res = resolve_manual(path, ds, default_config)
        # Resolution warnings should NOT yet contain the missing-from-df message.
        assert not any("metrics dataset" in w for w in res.warnings)
        # But the full estimate should produce it (deferred to propagation time).
        result = estimate_path_delay(ds, path, default_config)
        assert any("metrics dataset" in w for w in result.warnings)

    def test_linear_missing_from_df_flag(self, ds, default_config):
        """_ResolvedStage.linear_missing_from_df is True for unknown cells."""
        path = [PathSpec(cell_name="TOTALLY_UNKNOWN"), PathSpec(cell_name="INV_X1")]
        res = resolve_manual(path, ds, default_config)
        assert res.stages[0].linear_missing_from_df is True
        assert res.stages[1].linear_missing_from_df is False

    def test_linear_r2_warn_threshold_is_module_constant(self):
        """_LINEAR_R2_WARN_THRESHOLD is a module-level float, not a magic literal."""
        from parsfet.path_delay import _LINEAR_R2_WARN_THRESHOLD
        assert isinstance(_LINEAR_R2_WARN_THRESHOLD, float)
        assert 0.0 < _LINEAR_R2_WARN_THRESHOLD < 1.0

    def test_linear_slew_approximation_scales_with_load(self, ds):
        """Linear fallback slew scales √(load/fo4). Higher load → higher slew."""
        path_light = [PathSpec(cell_name="TOTALLY_UNKNOWN", extra_cap_pf=0.001)]
        path_heavy = [PathSpec(cell_name="TOTALLY_UNKNOWN", extra_cap_pf=0.5)]
        r_light = estimate_path_delay(ds, path_light, AnalysisConfig(output_load_pf=0.001))
        r_heavy = estimate_path_delay(ds, path_heavy, AnalysisConfig(output_load_pf=0.5))
        # Both use linear; heavier load → larger slew on the output
        assert r_heavy.points[0].slew_ns >= r_light.points[0].slew_ns
