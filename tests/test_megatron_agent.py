from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TITAN_ROOT = ROOT / "torchtitan"
if str(TITAN_ROOT) not in sys.path:
    sys.path.insert(0, str(TITAN_ROOT))
MEGATRON_ROOT = ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))
if "tyro" not in sys.modules:
    tyro_stub = types.ModuleType("tyro")

    class _ConstructorRegistry:
        def primitive_rule(self, _target):
            def decorator(fn):
                return fn

            return decorator

    class _PrimitiveTypeInfo:
        type = None

    class _PrimitiveConstructorSpec:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    tyro_stub.cli = lambda config_cls, args=None, default=None, registry=None: default if default is not None else config_cls()
    tyro_stub.conf = types.SimpleNamespace(Suppress=object)
    tyro_stub.constructors = types.SimpleNamespace(
        ConstructorRegistry=_ConstructorRegistry,
        PrimitiveTypeInfo=_PrimitiveTypeInfo,
        PrimitiveConstructorSpec=_PrimitiveConstructorSpec,
    )
    sys.modules["tyro"] = tyro_stub

from megatron_agent import agent_loop, trial_runner  # noqa: E402
from megatron_agent.feedback_optimizer import (  # noqa: E402
    build_feedback_search_plan,
    build_trial_outcome,
    infer_runtime_schedule_family,
    record_trial_feedback,
    reflect_on_trial,
)
from megatron_agent.metrics_parser import parse_megatron_logs  # noqa: E402
from megatron_agent.policy_memory import PolicyMemoryBank  # noqa: E402
from megatron_agent.torchtitan_hybrid import (  # noqa: E402
    TorchTitanHybridController,
    TorchTitanHybridEvidence,
    TorchTitanHybridPlanIR,
    export_plan_to_hybrid_policy,
    verify_torchtitan_hybrid_plan,
)
from megatron_agent.config import (  # noqa: E402
    AgentProposal,
    AgentObservation,
    ExperimentSpec,
    BatchPlanSpec,
    LengthBucketPolicy,
    ProgramBank,
    ProgramPatchSpec,
    ProgramTemplate,
    ReplanDecision,
    VerifierReport,
    default_backend_caps,
    default_dense_program,
    default_moe_smoke_program,
)
from megatron_agent.programs import check_program, classify_program_family, compile_program, verify_program  # noqa: E402
from megatron_agent import trace_reducer  # noqa: E402
from megatron_agent.trace_reducer import (  # noqa: E402
    build_agent_observation,
    build_context_record,
    build_trial_artifact,
    classify_bottleneck,
    recommend_optimization_methods,
    reduce_trial_trace,
)
from torchtitan.experiments.hybrid_policy import apply_hybrid_policy  # noqa: E402
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _safe_module_isinstance  # noqa: E402


class _FakeActivationCheckpoint:
    def __init__(self) -> None:
        self.mode = "none"
        self.selective_ac_option = "op"
        self.preserve_rng_state = False
        self.early_stop = False


class _FakeParallelism:
    def __init__(self) -> None:
        self.pipeline_parallel_degree = 1
        self.pipeline_parallel_schedule = "1F1B"
        self.pipeline_parallel_microbatch_size = 1
        self.module_fqns_per_model_part = None
        self.pipeline_parallel_stage_to_node = None
        self.pipeline_parallel_vpp_per_rank = None
        self.tensor_parallel_degree = 1
        self.context_parallel_degree = 1
        self.expert_parallel_degree = 1
        self.expert_tensor_parallel_degree = 1
        self.disable_loss_parallel = False
        self.data_parallel_replicate_degree = 1
        self.data_parallel_shard_degree = 1
        self.fsdp_reshard_after_forward = "never"
        self.fsdp_parallelism_conditioned_policy = "module_groups"
        self.fsdp_attention_scope = "keep"
        self.fsdp_mlp_scope = "keep"
        self.fsdp_mlp_output_scope = "keep"
        self.fsdp_embhead_scope = "keep"
        self.fsdp_mlp_unit_mode = "block"
        self.fsdp_node_local_reshard_size = 0
        self.fsdp_policy_trace = False
        self.fsdp_prefetch_window = 1
        self.fsdp_recompute_forward_prefetch = "inherit"
        self.fsdp_recompute_backward_prefetch = "inherit"
        self.fsdp_materialization_watermark_gib = 0.0
        self.fsdp_stage_hbm_budget_gib = None
        self.rank_order = None


class _FakeDebug:
    def __init__(self) -> None:
        self.pipeline_trace = False
        self.pipeline_trace_collectives = False


class _FakeTrainerConfig:
    def __init__(self) -> None:
        self.parallelism = _FakeParallelism()
        self.activation_checkpoint = _FakeActivationCheckpoint()
        self.debug = _FakeDebug()


class TestMegatronAgentProgramFlow(unittest.TestCase):
    def _mock_runtime_stack(self) -> mock._patch:
        return mock.patch(
            "megatron_agent.trial_runner._validate_runtime_stack",
            return_value={
                "transformer_impl": "transformer_engine",
                "transformer_engine_version": "test-te",
                "apex_path": "/tmp/apex/__init__.py",
            },
        )

    def test_single_node_dense_default_exports_family_outside_candidate(self) -> None:
        baseline = default_dense_program("single_g5")
        rewrite = agent_loop._rewrite_space(baseline, {})
        candidates, rejected = agent_loop._synthesize_programs(baseline, rewrite, candidate_limit=8)

        self.assertEqual(baseline.machine_profile.name, "consumer_single_node_5090d")
        self.assertEqual(baseline.backend_caps.transformer_impl, "local")
        self.assertFalse(rewrite.allow_single_node_pp_split)
        self.assertTrue(rewrite.allow_stage_aware_schedule)
        self.assertEqual(rejected, [])
        candidate_kinds = [candidate.metadata.get("program_kind") for candidate in candidates]
        self.assertIn("candidate_stage_aware_schedule", candidate_kinds)
        self.assertIn("candidate_pp_scaleout", candidate_kinds)
        self.assertNotIn("candidate_sequence_parallel_toggle", candidate_kinds)

        target = next(candidate for candidate in candidates if candidate.metadata.get("program_kind") == "candidate_stage_aware_schedule")
        family = classify_program_family(target).to_dict()
        self.assertTrue(family["is_family_outside"])

    def test_sequence_parallel_toggle_requires_backend_caps_support(self) -> None:
        baseline = default_dense_program("single_g5")
        baseline.backend_caps = default_backend_caps("transformer_engine")

        rewrite = agent_loop._rewrite_space(baseline, {})
        candidates, rejected = agent_loop._synthesize_programs(baseline, rewrite, candidate_limit=8)

        self.assertTrue(rewrite.allow_sequence_parallel_toggle)
        self.assertEqual(rejected, [])
        sp_toggle = next(candidate for candidate in candidates if candidate.metadata.get("program_kind") == "candidate_sequence_parallel_toggle")
        self.assertEqual(sp_toggle.parallel.tp_degree, baseline.parallel.tp_degree)
        self.assertTrue(sp_toggle.parallel.sp_enabled)

    def test_single_g4_dense_baseline_is_legal_and_exports_candidate(self) -> None:
        baseline = default_dense_program("single_g4")
        rewrite = agent_loop._rewrite_space(baseline, {})
        candidates, rejected = agent_loop._synthesize_programs(baseline, rewrite, candidate_limit=8)

        self.assertFalse(rewrite.allow_single_node_pp_split)
        self.assertTrue(rewrite.allow_nonuniform_partition)
        self.assertFalse(rewrite.allow_sequence_parallel_toggle)
        self.assertEqual(rejected, [])
        self.assertEqual(baseline.layout.stage_to_node, ["g4", "g4"])
        self.assertEqual(baseline.machine_profile.name, "consumer_single_node_4090d")
        candidate_kinds = [candidate.metadata.get("program_kind") for candidate in candidates]
        self.assertIn("candidate_nonuniform_partition", candidate_kinds)
        self.assertIn("candidate_pp_scaleout", candidate_kinds)
        self.assertNotIn("candidate_sequence_parallel_toggle", candidate_kinds)

    def test_dual_target_candidate_synthesis_still_produces_expected_families(self) -> None:
        dense_baseline = default_dense_program("dual_g4_g5")
        dense_runtime = {"bubble_ratio": 0.07, "stage_spread_ratio": 0.12, "cross_node_exposed_ratio": 0.08}
        dense_rewrite = agent_loop._rewrite_space(dense_baseline, dense_runtime)
        dense_candidates, _ = agent_loop._synthesize_programs(
            dense_baseline,
            dense_rewrite,
            runtime_summary=dense_runtime,
            candidate_limit=8,
        )
        dense_kinds = {candidate.metadata.get("program_kind") for candidate in dense_candidates}

        self.assertEqual(dense_baseline.parallel.tp_degree, 4)
        self.assertEqual(dense_baseline.parallel.pp_degree, 4)
        self.assertEqual(agent_loop._data_parallel_size(dense_baseline), 1)
        self.assertEqual(dense_baseline.layout.stage_to_node, ["g4", "g4", "g5", "g5"])
        self.assertTrue(
            {
                "candidate_nonuniform_partition",
                "candidate_stage_aware_schedule",
                "candidate_dual_node_pp8_scaleout",
            }.issubset(dense_kinds)
        )
        self.assertNotIn("candidate_sequence_parallel_toggle", dense_kinds)

        moe_baseline = default_moe_smoke_program("dual_g4_g5")
        moe_runtime = {"bubble_ratio": 0.06, "stage_spread_ratio": 0.10, "cross_node_exposed_ratio": 0.09}
        moe_rewrite = agent_loop._rewrite_space(moe_baseline, moe_runtime)
        moe_candidates, _ = agent_loop._synthesize_programs(
            moe_baseline,
            moe_rewrite,
            runtime_summary=moe_runtime,
            candidate_limit=8,
        )
        moe_kinds = {candidate.metadata.get("program_kind") for candidate in moe_candidates}

        self.assertFalse(moe_rewrite.allow_sequence_parallel_toggle)
        self.assertFalse(moe_rewrite.allow_dual_plane)
        self.assertTrue(
            {
                "candidate_nonuniform_partition",
                "candidate_stage_aware_schedule",
                "candidate_runtime_guided_schedule",
            }.issubset(moe_kinds)
        )
        self.assertNotIn("candidate_dual_plane", moe_kinds)
        self.assertNotIn("candidate_sequence_parallel_toggle", moe_kinds)

    def test_dual_g5_g5_dense_baseline_and_candidates_are_pp_first(self) -> None:
        baseline = default_dense_program("dual_g5_g5")
        runtime = {"bubble_ratio": 0.11, "stage_spread_ratio": 0.09, "cross_node_exposed_ratio": 0.04}
        rewrite = agent_loop._rewrite_space(baseline, runtime)
        candidates, rejected = agent_loop._synthesize_programs(
            baseline,
            rewrite,
            runtime_summary=runtime,
            candidate_limit=8,
        )
        kinds = {candidate.metadata.get("program_kind") for candidate in candidates}

        self.assertEqual(rejected, [])
        self.assertEqual(baseline.parallel.tp_degree, 4)
        self.assertEqual(baseline.parallel.pp_degree, 4)
        self.assertEqual(agent_loop._data_parallel_size(baseline), 1)
        self.assertEqual(baseline.layout.stage_to_node, ["g5_0", "g5_0", "g5_1", "g5_1"])
        self.assertTrue(rewrite.allow_nonuniform_partition)
        self.assertGreaterEqual(int(rewrite.max_pp_size or 0), 8)
        self.assertIn("candidate_dual_node_pp8_scaleout", kinds)
        self.assertIn("candidate_stage_aware_schedule", kinds)

    def test_dual_plane_requires_backend_caps_support(self) -> None:
        moe_baseline = default_moe_smoke_program("dual_g4_g5")
        moe_baseline.backend_caps = default_backend_caps("transformer_engine")
        moe_runtime = {"bubble_ratio": 0.06, "stage_spread_ratio": 0.10, "cross_node_exposed_ratio": 0.09}

        moe_rewrite = agent_loop._rewrite_space(moe_baseline, moe_runtime)
        moe_candidates, _ = agent_loop._synthesize_programs(moe_baseline, moe_rewrite, candidate_limit=8)
        moe_kinds = {candidate.metadata.get("program_kind") for candidate in moe_candidates}

        self.assertTrue(moe_rewrite.allow_dual_plane)
        self.assertIn("candidate_dual_plane", moe_kinds)

    def test_export_only_writes_programs_and_summary_without_running_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "runs"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "agent_loop.py",
                    "--export-only",
                    "--workdir",
                    str(workdir),
                    "--run-target",
                    "single_g5",
                    "--model-track",
                    "dense",
                ],
            ):
                with mock.patch("megatron_agent.agent_loop.run_trial", side_effect=AssertionError("run_trial should not be called")):
                    agent_loop.main()

            summary_path = workdir / "summary_megatron.json"
            programs_dir = workdir / "programs"
            self.assertTrue(summary_path.exists())
            self.assertTrue(programs_dir.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "program_synthesis_export_only")
            self.assertEqual(summary["tested_trials"], [])
            self.assertGreaterEqual(summary["candidate_generation_count"], 3)
            self.assertEqual(summary["candidate_execution_count"], 0)
            self.assertEqual(summary["compile_success_rate"], 1.0)
            self.assertGreaterEqual(summary["family_outside_ratio"], 0.75)
            self.assertIn("candidate_manifest", summary)
            self.assertEqual(summary["recommended_execution_order"][0], "baseline")
            self.assertIn("program_bank", summary)
            self.assertIn("runtime_signature", summary)

            manifest_names = [entry["config_name"] for entry in summary["candidate_manifest"]]
            self.assertIn("baseline", manifest_names)
            self.assertIn("candidate_stage_aware_schedule", manifest_names)
            self.assertTrue(
                {
                    "candidate_morphable_pipeline",
                    "candidate_nonuniform_vpp_shape",
                    "candidate_apipe_pipe_heuristic_v1",
                }
                & set(manifest_names)
            )

    def test_trial_runner_dry_run_emits_launch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program_path = tmp / "baseline.json"
            output_path = tmp / "dry_run.json"
            program_path.write_text(json.dumps(default_dense_program("single_g5").to_dict(), indent=2), encoding="utf-8")

            with self._mock_runtime_stack():
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "trial_runner.py",
                        "--program-file",
                        str(program_path),
                        "--output",
                        str(output_path),
                        "--megatron-root",
                        str(megatron_root),
                        "--launcher-script",
                        "",
                        "--dry-run",
                    ],
                ):
                    trial_runner.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["trial_context"]["runner_mode"], "direct_entry")
            self.assertEqual(payload["trial_context"]["resolved_paths"]["megatron_entry"], str(megatron_root / "pretrain_gpt.py"))
            self.assertEqual(
                payload["trial_context"]["resolved_paths"]["torchrun_log_dir"],
                str((Path("./runs_megatron").resolve()) / "trial_000" / "torchrun_logs"),
            )
            self.assertIn("launcher_env", payload["launch_plan"])
            self.assertIn("megatron_command", payload["launch_plan"])
            self.assertGreater(len(payload["launch_plan"]["megatron_command"]), 0)
            self.assertEqual(payload["launch_plan"]["launcher_env"]["TRANSFORMER_IMPL"], "transformer_engine")
            self.assertEqual(payload["trial_context"]["runtime_stack"]["transformer_engine_version"], "test-te")
            self.assertNotIn("--no-rope-fusion", payload["launch_plan"]["megatron_command"])
            self.assertNotIn("--no-persist-layer-norm", payload["launch_plan"]["megatron_command"])
            self.assertIn("--log-dir", payload["launch_plan"]["megatron_command"])
            self.assertIn("--redirects", payload["launch_plan"]["megatron_command"])
            self.assertIn("transformer_engine", payload["launch_plan"]["megatron_command"])
            self.assertEqual(payload["launch_plan"]["launcher_env"]["USE_BF16"], "1")
            self.assertEqual(payload["launch_plan"]["launcher_env"]["USE_FP16"], "0")
            self.assertEqual(payload["trial_context"]["resolved_backend_caps"]["transformer_impl"], "transformer_engine")
            self.assertEqual(payload["launch_plan"]["resolved_backend_caps"]["transformer_impl"], "transformer_engine")
            self.assertEqual(payload["trial_context"]["resolved_profile"]["machine_profile"]["name"], "consumer_single_node_5090d")
            self.assertIn("PP split remains preferred", " ".join(payload["launch_plan"]["compile_notes"]))

    def test_trial_runner_dry_run_respects_single_g4_and_fp16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program = default_dense_program("single_g4")
            program.metadata.update({"use_bf16": False, "use_fp16": True})
            program_path = tmp / "single_g4_fp16.json"
            output_path = tmp / "single_g4_fp16_dry_run.json"
            program_path.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")

            with mock.patch.object(
                sys,
                "argv",
                [
                    "trial_runner.py",
                    "--program-file",
                    str(program_path),
                    "--output",
                    str(output_path),
                    "--megatron-root",
                    str(megatron_root),
                    "--launcher-script",
                    "",
                    "--run-target",
                    "single_g4",
                    "--model-track",
                    "dense",
                    "--dry-run",
                ],
            ):
                trial_runner.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["program"]["cluster"]["target"], "single_g4")
            self.assertEqual(payload["launch_plan"]["launcher_env"]["RUN_TARGET"], "single_g4")
            self.assertEqual(payload["launch_plan"]["launcher_env"]["TRANSFORMER_IMPL"], "local")
            self.assertEqual(payload["launch_plan"]["launcher_env"]["USE_BF16"], "0")
            self.assertEqual(payload["launch_plan"]["launcher_env"]["USE_FP16"], "1")
            self.assertEqual(payload["trial_context"]["resolved_backend_caps"]["transformer_impl"], "local")
            self.assertEqual(payload["trial_context"]["resolved_profile"]["machine_profile"]["name"], "consumer_single_node_4090d")

    def test_trial_runner_dry_run_single_g5_local_disables_sp_and_enables_local_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program_path = tmp / "single_g5_local.json"
            output_path = tmp / "single_g5_local_dry_run.json"
            program_path.write_text(json.dumps(default_dense_program("single_g5").to_dict(), indent=2), encoding="utf-8")

            with mock.patch.object(
                sys,
                "argv",
                [
                    "trial_runner.py",
                    "--program-file",
                    str(program_path),
                    "--output",
                    str(output_path),
                    "--megatron-root",
                    str(megatron_root),
                    "--launcher-script",
                    "",
                    "--transformer-impl",
                    "local",
                    "--dry-run",
                ],
            ):
                trial_runner.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            cmd = payload["launch_plan"]["megatron_command"]
            self.assertEqual(payload["launch_plan"]["launcher_env"]["TRANSFORMER_IMPL"], "local")
            self.assertEqual(payload["launch_plan"]["launcher_env"]["ENABLE_SP"], "0")
            self.assertEqual(payload["trial_context"]["runtime_stack"]["transformer_impl"], "local")
            self.assertIn("--no-rope-fusion", cmd)
            self.assertIn("--no-persist-layer-norm", cmd)
            self.assertIn("--no-gradient-accumulation-fusion", cmd)
            self.assertIn("--eval-iters", cmd)
            self.assertIn("--eval-interval", cmd)
            self.assertEqual(cmd[cmd.index("--eval-iters") + 1], "0")
            self.assertEqual(cmd[cmd.index("--eval-interval") + 1], "1")
            self.assertNotIn("--sequence-parallel", cmd)
            self.assertEqual(payload["launch_plan"]["resolved_backend_caps"]["transformer_impl"], "local")
            self.assertIn("SP candidate suppressed", " ".join(payload["launch_plan"]["compile_notes"]))

    def test_trial_runner_dry_run_sequence_parallel_toggle_disables_sp_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program = default_dense_program("single_g5")
            program.parallel.sp_enabled = False
            program.metadata["program_kind"] = "candidate_sequence_parallel_toggle"
            program_path = tmp / "sp_toggle_off.json"
            output_path = tmp / "sp_toggle_off_dry_run.json"
            program_path.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")

            with self._mock_runtime_stack():
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "trial_runner.py",
                        "--program-file",
                        str(program_path),
                        "--output",
                        str(output_path),
                        "--megatron-root",
                        str(megatron_root),
                        "--launcher-script",
                        "",
                        "--dry-run",
                    ],
                ):
                    trial_runner.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["launch_plan"]["launcher_env"]["ENABLE_SP"], "0")
            self.assertNotIn("--sequence-parallel", payload["launch_plan"]["megatron_command"])

    def test_trial_runner_dry_run_stateful_single_g5_emits_window_override_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program_path = tmp / "stateful_single_g5.json"
            output_path = tmp / "stateful_single_g5_dry_run.json"
            program_path.write_text(json.dumps(default_dense_program("single_g5").to_dict(), indent=2), encoding="utf-8")

            with self._mock_runtime_stack():
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "trial_runner.py",
                        "--program-file",
                        str(program_path),
                        "--output",
                        str(output_path),
                        "--megatron-root",
                        str(megatron_root),
                        "--launcher-script",
                        "",
                        "--run-root",
                        str(tmp / "runs"),
                        "--dry-run",
                        "--telemetry-budget",
                        "summary",
                        "--window-steps",
                        "4",
                        "--enable-stateful-schedule",
                    ],
                ):
                    trial_runner.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            env = payload["launch_plan"]["launcher_env"]
            self.assertEqual(env["ENABLE_STATEFUL_SCHEDULE"], "1")
            self.assertEqual(env["SCHEDULE_RUNTIME_TRACE_LEVEL"], "summary")
            self.assertIn("WINDOW_RECONFIG_PLAN", env)
            self.assertIn("SCHEDULE_WINDOW_OVERRIDE_HINTS", env)
            self.assertTrue(str(env["SCHEDULE_RUNTIME_TRACE_DIR"]).endswith("runtime_schedule_traces"))
            overrides = json.loads(env["SCHEDULE_WINDOW_OVERRIDE_HINTS"])
            self.assertTrue(any(str(item.get("stage_selector") or "") == "hotspot_stage" for item in overrides))
            self.assertTrue(any(str(item.get("stage_selector") or "") == "optimizer_sensitive_stage" for item in overrides))

    def test_trial_runner_dry_run_emits_optimizer_overlap_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program_path = tmp / "baseline.json"
            output_path = tmp / "deep_observability.json"
            program_path.write_text(json.dumps(default_dense_program("single_g5").to_dict(), indent=2), encoding="utf-8")

            with self._mock_runtime_stack():
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "trial_runner.py",
                        "--program-file",
                        str(program_path),
                        "--output",
                        str(output_path),
                        "--megatron-root",
                        str(megatron_root),
                        "--launcher-script",
                        "",
                        "--run-root",
                        str(tmp / "runs"),
                        "--dry-run",
                        "--observability-preset",
                        "deep",
                        "--profile-step-start",
                        "3",
                        "--profile-step-end",
                        "7",
                        "--wandb-project",
                        "megatron-tests",
                        "--wandb-exp-name",
                        "single-g5-qwen14b-deep",
                    ],
                ):
                    trial_runner.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            cmd = payload["launch_plan"]["megatron_command"]
            self.assertIn("--use-pytorch-profiler", cmd)
            self.assertIn("--log-timers-to-tensorboard", cmd)
            self.assertIn("--log-memory-to-tensorboard", cmd)
            self.assertIn("--record-memory-history", cmd)
            self.assertIn("--wandb-project", cmd)
            self.assertIn("--wandb-exp-name", cmd)
            self.assertEqual(payload["launch_plan"]["observability"]["profile_step_start"], 3)
            self.assertEqual(payload["launch_plan"]["observability"]["profile_step_end"], 7)
            self.assertTrue(payload["launch_plan"]["observability"]["enable_pytorch_profiler"])
            self.assertTrue(payload["launch_plan"]["observability"]["enable_memory_history"])
            self.assertEqual(
                payload["trial_context"]["resolved_paths"]["torch_profile_path"],
                str((tmp / "runs").resolve() / "trial_000" / "torch_profile"),
            )

    def test_trial_runner_dry_run_nsys_wraps_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program_path = tmp / "baseline.json"
            output_path = tmp / "nsys_dry_run.json"
            program_path.write_text(json.dumps(default_dense_program("single_g5").to_dict(), indent=2), encoding="utf-8")

            with self._mock_runtime_stack():
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "trial_runner.py",
                        "--program-file",
                        str(program_path),
                        "--output",
                        str(output_path),
                        "--megatron-root",
                        str(megatron_root),
                        "--launcher-script",
                        "",
                        "--run-root",
                        str(tmp / "runs"),
                        "--dry-run",
                        "--enable-nsys",
                        "--nsys-output",
                        "custom_nsys/profile",
                    ],
                ):
                    trial_runner.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            executed = payload["launch_plan"]["executed_command"]
            self.assertEqual(executed[0], "nsys")
            self.assertIn("custom_nsys", payload["launch_plan"]["observability"]["nsys_output"])
            self.assertTrue(payload["launch_plan"]["observability"]["enable_nsys"])

    def test_trial_runner_rejects_tp_comm_overlap_without_sequence_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program = default_dense_program("single_g5")
            program.parallel.sp_enabled = False
            program_path = tmp / "tp_overlap_invalid.json"
            output_path = tmp / "tp_overlap_invalid_dry_run.json"
            program_path.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")

            with self._mock_runtime_stack():
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "trial_runner.py",
                        "--program-file",
                        str(program_path),
                        "--output",
                        str(output_path),
                        "--megatron-root",
                        str(megatron_root),
                        "--launcher-script",
                        "",
                        "--enable-tp-comm-overlap",
                        "--dry-run",
                    ],
                ):
                    with self.assertRaises(SystemExit) as exc_info:
                        trial_runner.main()

            self.assertEqual(exc_info.exception.code, 1)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["returncode"], 1)
            self.assertIn("tp_comm_overlap requires tp_degree > 1 with sequence parallel enabled", payload["error_msg"])

    def test_trial_runner_failure_extracts_root_cause_from_torchrun_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program_path = tmp / "baseline.json"
            output_path = tmp / "failure.json"
            run_root = tmp / "runs"
            program_path.write_text(json.dumps(default_dense_program("single_g5").to_dict(), indent=2), encoding="utf-8")

            def _fake_run(cmd, capture_output=None, text=None, cwd=None, env=None, **_kwargs):
                torchrun_dir = run_root / "trial_000" / "torchrun_logs" / "test_run_id" / "attempt_0" / "0"
                torchrun_dir.mkdir(parents=True, exist_ok=True)
                (torchrun_dir / "stderr.log").write_text(
                    "Traceback (most recent call last):\n"
                    "  File \"pretrain_gpt.py\", line 1, in <module>\n"
                    "    raise ValueError('boom')\n"
                    "ValueError: boom\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="elastic wrapper stdout\n",
                    stderr="elastic wrapper stderr\n",
                )

            with self._mock_runtime_stack():
                with mock.patch("megatron_agent.trial_runner.subprocess.run", side_effect=_fake_run):
                    with mock.patch(
                        "megatron_agent.trial_runner._validate_cuda_toolchain",
                        return_value={"CUDA_HOME": "/usr/local/cuda", "CUDA_PATH": "/usr/local/cuda", "CUDACXX": "/usr/local/cuda/bin/nvcc"},
                    ):
                        with mock.patch.object(
                            sys,
                            "argv",
                            [
                                "trial_runner.py",
                                "--program-file",
                                str(program_path),
                                "--output",
                                str(output_path),
                                "--megatron-root",
                                str(megatron_root),
                                "--launcher-script",
                                "",
                                "--run-root",
                                str(run_root),
                                "--no-stream-trial-logs",
                            ],
                        ):
                            with self.assertRaises(SystemExit) as exc_info:
                                trial_runner.main()

            self.assertEqual(exc_info.exception.code, 1)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("ValueError: boom", payload["root_cause_excerpt"])
            self.assertTrue(payload["root_cause_source"].endswith("stderr.log"))
            self.assertEqual(
                payload["trial_context"]["resolved_paths"]["torchrun_log_dir"],
                str(run_root.resolve() / "trial_000" / "torchrun_logs"),
            )

    def test_runtime_env_defaults_discovers_cuda_home_from_nvcc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cuda_home = Path(tmpdir) / "cuda"
            bin_dir = cuda_home / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "nvcc").write_text("", encoding="utf-8")

            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("megatron_agent.trial_runner.shutil.which", return_value=str(bin_dir / "nvcc")):
                    env = trial_runner._runtime_env_defaults()

            self.assertEqual(env["CUDA_HOME"], str(cuda_home))
            self.assertEqual(env["CUDA_PATH"], str(cuda_home))
            self.assertEqual(env["CUDACXX"], str(bin_dir / "nvcc"))

    def test_single_g5_dense_dry_run_requires_perf_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            megatron_root = tmp / "Megatron-LM"
            megatron_root.mkdir()
            (megatron_root / "pretrain_gpt.py").write_text("print('stub')\n", encoding="utf-8")

            program_path = tmp / "baseline.json"
            output_path = tmp / "missing_perf_stack.json"
            program_path.write_text(json.dumps(default_dense_program("single_g5").to_dict(), indent=2), encoding="utf-8")

            with mock.patch(
                "megatron_agent.trial_runner._validate_runtime_stack",
                side_effect=RuntimeError("single_g5 dense high-performance path requires Apex with apex.optimizers.FusedAdam available in the active environment."),
            ):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "trial_runner.py",
                        "--program-file",
                        str(program_path),
                        "--output",
                        str(output_path),
                        "--megatron-root",
                        str(megatron_root),
                        "--launcher-script",
                        "",
                        "--dry-run",
                    ],
                ):
                    with self.assertRaises(SystemExit) as exc_info:
                        trial_runner.main()

            self.assertEqual(exc_info.exception.code, 1)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("single_g5 dense high-performance path requires Apex", payload["error_msg"])

    def test_prepare_trial_artifact_dirs_clears_stale_trial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trial_dir = Path(tmpdir) / "trial_000"
            stale_file = trial_dir / "torchrun_logs" / "old" / "stderr.log"
            stale_file.parent.mkdir(parents=True, exist_ok=True)
            stale_file.write_text("stale", encoding="utf-8")

            output_dirs = {
                "trial_dir": str(trial_dir),
                "checkpoint_path": str(trial_dir / "checkpoints"),
                "tensorboard_path": str(trial_dir / "tensorboard"),
                "torch_profile_path": str(trial_dir / "torch_profile"),
                "torchrun_log_dir": str(trial_dir / "torchrun_logs"),
                "chakra_path": str(trial_dir / "chakra"),
                "nsys_path": str(trial_dir / "nsys"),
                "data_cache_path": str(trial_dir / "cache"),
            }
            observability = {"memory_snapshot_path": None, "nsys_output": None}

            trial_runner._prepare_trial_artifact_dirs(output_dirs, observability)

            self.assertFalse(stale_file.exists())
            self.assertTrue((trial_dir / "torchrun_logs").exists())

    def test_summary_fields_stable_without_cross_node_signal(self) -> None:
        baseline = default_dense_program("single_g5")
        candidate = agent_loop._build_pp_scaleout_candidate(baseline)
        self.assertIsNotNone(candidate)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = agent_loop._export_programs(baseline, [candidate], Path(tmpdir) / "programs")

            for baseline_metrics, best_program, best_metrics in (
                (
                    {
                        "config_name": "baseline",
                        "throughput_tokens_per_s": 1000.0,
                        "step_time_ms_p50": 90.0,
                        "bubble_ratio": 0.20,
                        "comm_ratio_from_stages": 0.12,
                        "stage_window_summary": {
                            "0": {"window_ms": 90.0},
                            "1": {"window_ms": 110.0},
                        },
                    },
                    candidate,
                    {
                        "config_name": "candidate_single_node_pp_split",
                        "throughput_tokens_per_s": 1200.0,
                        "step_time_ms_p50": 80.0,
                        "bubble_ratio": 0.10,
                        "comm_ratio_from_stages": 0.08,
                        "stage_window_summary": {
                            "0": {"window_ms": 95.0},
                            "1": {"window_ms": 100.0},
                        },
                    },
                ),
                (
                    {
                        "config_name": "baseline",
                        "returncode": 1,
                        "error_msg": "compile failed",
                    },
                    baseline,
                    {
                        "config_name": "baseline",
                        "returncode": 1,
                        "error_msg": "compile failed",
                    },
                ),
                (
                    None,
                    None,
                    None,
                ),
            ):
                summary = agent_loop._build_summary_payload(
                    export_only=baseline_metrics is None,
                    programs_dir=Path(tmpdir) / "programs",
                    runtime_summary={"bubble_ratio": 0.15},
                    runtime_signature={"bubble_ratio": 0.15, "length_bucket": "short"},
                    context_record={
                        "hardware_context": {"target": "single_g5"},
                        "model_context": {"track": "dense"},
                        "workload_context": {"length_bucket": "short"},
                        "runtime_evidence": {"bubble_ratio": 0.15},
                        "evidence_record": {"stage_evidence": []},
                        "failure_modes": [{"label": "tp_overpartitioned"}],
                    },
                    replan_decision={"scope": "pipe", "trigger": "workload_drift"},
                    bottleneck_signature={"dominant_label": "tp_overpartitioned", "labels": ["tp_overpartitioned"]},
                    rewrite=agent_loop._rewrite_space(baseline, {}),
                    baseline=baseline,
                    baseline_metrics=baseline_metrics,
                    best_program=best_program,
                    best_metrics=best_metrics,
                    tested=[] if baseline_metrics is None else [baseline_metrics],
                    family_outside_trials=[],
                    rejected_candidates=[],
                    candidate_manifest=manifest,
                    program_bank=ProgramBank(
                        templates=[
                            ProgramTemplate(
                                name="baseline",
                                run_target="single_g5",
                                model_track="dense",
                                length_bucket="short",
                                bottleneck_tags=["tp_overpartitioned"],
                                program=baseline,
                            )
                        ]
                    ),
                    evidence_manifest=[{"config_name": "evidence_pp_fixed_pipe"}],
                )
                self.assertIn("compile_success_rate", summary)
                self.assertIn("family_outside_ratio", summary)
                self.assertIn("stage_load_variance", summary)
                self.assertIn("observed_comm_ratio", summary)
                self.assertIn("baseline_vs_best", summary)
                self.assertIn("program_bank", summary)
                self.assertIn("runtime_signature", summary)
                self.assertIn("bottleneck_signature", summary)
                self.assertIn("context_record", summary)
                self.assertIn("failure_modes", summary)
                self.assertIn("replan_decision", summary)
                self.assertIn("motivation_evidence_manifest", summary)
                self.assertIn("autotune_history", summary)
                self.assertIn("auto_tune_rounds_requested", summary)
                self.assertIn("auto_tune_rounds_completed", summary)

    def test_select_autotune_seed_promotes_improved_successful_candidate(self) -> None:
        baseline = default_dense_program("single_g5")
        candidate = agent_loop._build_optimizer_aware_pipeline_candidate(
            baseline,
            {
                "runtime_evidence": {
                    "optimizer_exposed_ratio": 0.24,
                    "peak_reserved_ratio": 0.80,
                    "stage_tail_ratio": 0.13,
                    "tail_step_jitter_ratio": 0.12,
                }
            },
        )
        self.assertIsNotNone(candidate)
        current_metrics = {
            "config_name": "baseline",
            "returncode": 0,
            "program_hash": baseline.semantic_hash(),
            "step_time_ms_p50": 1000.0,
            "throughput_tokens_per_s": 1000.0,
            "trace_summary": {"steady_state_step_time_ms_p50": 1000.0},
        }
        improved_metrics = {
            "config_name": "candidate_optimizer_aware_pipeline",
            "returncode": 0,
            "program_hash": candidate.semantic_hash(),
            "step_time_ms_p50": 940.0,
            "throughput_tokens_per_s": 1080.0,
            "trace_summary": {"steady_state_step_time_ms_p50": 940.0},
        }
        promotion = agent_loop._select_autotune_seed(
            baseline,
            current_metrics,
            [improved_metrics, current_metrics],
            {
                baseline.semantic_hash(): baseline,
                candidate.semantic_hash(): candidate,
            },
        )
        self.assertIsNotNone(promotion)
        self.assertEqual((promotion or {}).get("program").semantic_hash(), candidate.semantic_hash())

    def test_batch_plan_and_program_bank_roundtrip(self) -> None:
        program = default_dense_program("single_g5")
        program.batch_plan = BatchPlanSpec(
            micro_batch_size=1,
            global_batch_size=32,
            grad_accum_steps=16,
            target_tokens_per_step=32768,
        )
        template = ProgramTemplate(
            name="pp4_bank_entry",
            run_target="single_g5",
            model_track="dense",
            length_bucket="short",
            bottleneck_tags=["tp_overpartitioned"],
            selection_score=0.9,
            program=program,
        )
        bank = ProgramBank(templates=[template])

        payload = bank.to_dict()
        restored = ProgramBank.from_dict(payload)
        self.assertEqual(restored.templates[0].program.batch_plan.global_batch_size, 32)
        self.assertEqual(restored.templates[0].length_bucket, "short")

    def test_trace_reducer_classifies_tp_overpartitioned(self) -> None:
        program = default_dense_program("single_g5")
        program.parallel.tp_degree = 8
        program.parallel.pp_degree = 1
        program.batch_plan.global_batch_size = 16
        program.metadata["seq_len"] = 1024

        summary = reduce_trial_trace(
            program,
            runtime_summary={
                "bubble_ratio": 0.02,
                "stage_load_variance": 0.0,
                "peak_memory_ratio": 0.64,
                "cross_node_exposed_ratio": 0.0,
            },
        )
        bottleneck = classify_bottleneck(program, summary)
        self.assertEqual(bottleneck["dominant_label"], "tp_overpartitioned")

    def test_context_record_exposes_evidence_and_failure_modes(self) -> None:
        program = default_dense_program("single_g5")
        program.parallel.pp_degree = 2
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        runtime_summary = {
            "bubble_ratio": 0.16,
            "stage_load_variance": 0.07,
            "cross_node_exposed_ratio": 0.11,
            "peak_memory_ratio": 0.88,
            "stage_window_summary": {
                "0": {"compute_ms": 1400.0, "comm_ms": 200.0, "bubble_ms": 120.0, "window_ms": 1720.0, "peak_reserved_gib": 23.0, "peak_active_gib": 20.0},
                "1": {"compute_ms": 1100.0, "comm_ms": 90.0, "bubble_ms": 60.0, "window_ms": 1250.0, "peak_reserved_gib": 17.0, "peak_active_gib": 15.0},
            },
        }
        context = build_context_record(program, runtime_summary=runtime_summary)
        labels = {item["label"] for item in context["failure_modes"]}
        self.assertIn("compute_imbalance", labels)
        self.assertIn("communication_drag", labels)
        self.assertIn("memory_hotspot", labels)
        self.assertIn("schedule_coupling", labels)
        self.assertEqual(context["workload_context"]["length_bucket"], "short")
        self.assertEqual(len(context["evidence_record"]["stage_evidence"]), 2)

    def test_context_record_includes_apipe_problem_formulation_and_plan(self) -> None:
        program = default_dense_program("dual_g5_g5")
        runtime_summary = {
            "steady_state_step_time_ms_p50": 8063.5,
            "steady_state_step_time_ms_p95": 8420.0,
            "bubble_ratio": 0.11,
            "pipeline_wait_ratio": 0.14,
            "optimizer_exposed_ratio": 0.24,
            "stage_tail_ratio": 0.18,
            "stage_window_summary": {
                "0": {"compute_ms": 1600.0, "comm_ms": 220.0, "bubble_ms": 150.0, "window_ms": 1970.0, "peak_reserved_gib": 22.0, "peak_active_gib": 19.0},
                "1": {"compute_ms": 1300.0, "comm_ms": 120.0, "bubble_ms": 70.0, "window_ms": 1490.0, "peak_reserved_gib": 18.0, "peak_active_gib": 16.0},
                "2": {"compute_ms": 1320.0, "comm_ms": 125.0, "bubble_ms": 65.0, "window_ms": 1510.0, "peak_reserved_gib": 18.5, "peak_active_gib": 16.2},
                "3": {"compute_ms": 1700.0, "comm_ms": 260.0, "bubble_ms": 160.0, "window_ms": 2120.0, "peak_reserved_gib": 24.0, "peak_active_gib": 21.0},
            },
        }
        context = build_context_record(program, runtime_summary=runtime_summary)
        formulation = context["evidence_record"]["apipe_problem_formulation"]
        plan = context["evidence_record"]["apipe_heuristic_plan"]

        self.assertEqual(formulation["status"], "v1_partial_runtime_support")
        action_names = {item["name"] for item in formulation["action_space"]}
        self.assertIn("move_boundary", action_names)
        self.assertIn("set_local_vpp", action_names)
        self.assertIn("reorder_flush_microbatches", action_names)
        self.assertIn("place_optimizer_slice", action_names)
        self.assertEqual(plan["runtime_controls"]["flush_order_policy"], "reverse_last_group")
        planned_actions = {item["name"] for item in plan["actions"]}
        self.assertIn("move_boundary", planned_actions)
        self.assertIn("reorder_flush_microbatches", planned_actions)

    def test_context_record_includes_runtime_branch_plan(self) -> None:
        program = default_dense_program("single_g5")
        program.parallel.pp_degree = 4
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        runtime_summary = {
            "steady_state_step_time_ms_p50": 8063.5,
            "bubble_ratio": 0.12,
            "pipeline_wait_ratio": 0.15,
            "optimizer_exposed_ratio": 0.22,
            "peak_memory_ratio": 0.89,
            "stage_window_summary": {
                "0": {"compute_ms": 1700.0, "forward_ms": 900.0, "completion_ms": 2250.0, "window_ms": 2250.0, "peak_reserved_gib": 23.5},
                "1": {"compute_ms": 1300.0, "forward_ms": 760.0, "completion_ms": 1500.0, "window_ms": 1500.0, "peak_reserved_gib": 18.0},
                "2": {"compute_ms": 1320.0, "forward_ms": 770.0, "completion_ms": 1520.0, "window_ms": 1520.0, "peak_reserved_gib": 18.4},
                "3": {"compute_ms": 1680.0, "forward_ms": 880.0, "completion_ms": 2200.0, "window_ms": 2200.0, "peak_reserved_gib": 23.0},
            },
        }
        context = build_context_record(program, runtime_summary=runtime_summary)
        branch_plan = context["evidence_record"]["runtime_branch_plan"]
        activated_ids = {item["branch_id"] for item in branch_plan["activated_branches"]}

        self.assertEqual(branch_plan["status"], "v1_executable_branch_pack")
        self.assertIn("branch_hotspot_stage_local_vpp", activated_ids)
        self.assertIn("branch_peak_window_memory_relief", activated_ids)
        self.assertIn("branch_local_pipe_reorder", activated_ids)
        self.assertGreaterEqual(len(branch_plan["trigger_rules"]), 4)

    def test_context_record_includes_morphable_pipeline_problem_and_plan(self) -> None:
        program = default_dense_program("single_g5")
        program.parallel.pp_degree = 4
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        runtime_summary = {
            "steady_state_step_time_ms_p50": 8063.5,
            "bubble_ratio": 0.12,
            "pipeline_wait_ratio": 0.15,
            "optimizer_exposed_ratio": 0.22,
            "peak_memory_ratio": 0.89,
            "comm_exposure_ratio": 0.13,
            "stage_window_summary": {
                "0": {"compute_ms": 1700.0, "forward_ms": 900.0, "completion_ms": 2250.0, "window_ms": 2250.0, "peak_reserved_gib": 23.5},
                "1": {"compute_ms": 1300.0, "forward_ms": 760.0, "completion_ms": 1500.0, "window_ms": 1500.0, "peak_reserved_gib": 18.0},
                "2": {"compute_ms": 1320.0, "forward_ms": 770.0, "completion_ms": 1520.0, "window_ms": 1520.0, "peak_reserved_gib": 18.4},
                "3": {"compute_ms": 1680.0, "forward_ms": 880.0, "completion_ms": 2200.0, "window_ms": 2200.0, "peak_reserved_gib": 23.0},
            },
        }
        context = build_context_record(program, runtime_summary=runtime_summary)
        problem = dict(context["evidence_record"]["morphable_pipeline_problem"])
        plan = dict(context["evidence_record"]["morphable_pipeline_plan"])
        branch_plan = dict(context["evidence_record"]["runtime_branch_plan"])

        self.assertEqual(problem["status"], "executable_v1")
        self.assertEqual(plan["status"], "executable_v1")
        self.assertEqual(problem["shape_objective"], "memory_constrained_throughput_maximization")
        self.assertEqual(problem["objective"]["type"], "minimize_step_time_under_memory_budget")
        self.assertIn("memory_budget", problem)
        self.assertIn("runtime_memory_policy", plan)
        self.assertEqual(plan["objective"]["type"], "minimize_step_time_under_memory_budget")
        self.assertGreater(len(list((problem.get("three_semantic_execution_graph") or {}).get("units") or [])), 0)
        self.assertIn("structure_aware_partition_ir", problem)
        self.assertIn("selective_vpp_generator", problem)
        self.assertIn("critical_path_communication_model", problem)
        self.assertIn("liveness_aware_chunk_formation", problem)
        self.assertGreater(len(list(plan.get("stage_families") or [])), 0)
        self.assertGreater(len(list(plan.get("selective_vpp_decisions") or [])), 0)
        self.assertGreater(len(list(plan.get("local_family_assignment") or [])), 0)
        self.assertGreater(len(list(context["evidence_record"].get("critical_operator_clusters") or [])), 0)
        self.assertTrue(str(plan.get("shape_signature") or ""))
        self.assertIn(
            "branch_morphable_pipeline_shape",
            {item["branch_id"] for item in branch_plan["activated_branches"]},
        )

    def test_runtime_branch_candidates_follow_active_branches(self) -> None:
        program = default_dense_program("single_g5")
        program.parallel.pp_degree = 4
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        runtime_summary = {
            "steady_state_step_time_ms_p50": 8063.5,
            "bubble_ratio": 0.12,
            "pipeline_wait_ratio": 0.15,
            "optimizer_exposed_ratio": 0.22,
            "peak_memory_ratio": 0.89,
            "stage_window_summary": {
                "0": {"compute_ms": 1700.0, "forward_ms": 900.0, "completion_ms": 2250.0, "window_ms": 2250.0, "peak_reserved_gib": 23.5},
                "1": {"compute_ms": 1300.0, "forward_ms": 760.0, "completion_ms": 1500.0, "window_ms": 1500.0, "peak_reserved_gib": 18.0},
                "2": {"compute_ms": 1320.0, "forward_ms": 770.0, "completion_ms": 1520.0, "window_ms": 1520.0, "peak_reserved_gib": 18.4},
                "3": {"compute_ms": 1680.0, "forward_ms": 880.0, "completion_ms": 2200.0, "window_ms": 2200.0, "peak_reserved_gib": 23.0},
            },
        }
        context = build_context_record(program, runtime_summary=runtime_summary)
        candidates = agent_loop._build_runtime_branch_candidates(program, context)
        branch_ids = {str(candidate.metadata.get("runtime_branch_id") or "") for candidate in candidates}

        self.assertIn("branch_peak_window_memory_relief", branch_ids)
        self.assertIn("branch_local_pipe_reorder", branch_ids)
        self.assertTrue(any(str(candidate.metadata.get("program_kind") or "").startswith("candidate_branch_") for candidate in candidates))

    def test_runtime_branch_candidates_include_morphable_pipeline_candidate(self) -> None:
        program = default_dense_program("single_g5")
        program.parallel.pp_degree = 4
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        runtime_summary = {
            "steady_state_step_time_ms_p50": 8063.5,
            "bubble_ratio": 0.12,
            "pipeline_wait_ratio": 0.15,
            "optimizer_exposed_ratio": 0.22,
            "peak_memory_ratio": 0.89,
            "comm_exposure_ratio": 0.13,
            "stage_window_summary": {
                "0": {"compute_ms": 1700.0, "forward_ms": 900.0, "completion_ms": 2250.0, "window_ms": 2250.0, "peak_reserved_gib": 23.5},
                "1": {"compute_ms": 1300.0, "forward_ms": 760.0, "completion_ms": 1500.0, "window_ms": 1500.0, "peak_reserved_gib": 18.0},
                "2": {"compute_ms": 1320.0, "forward_ms": 770.0, "completion_ms": 1520.0, "window_ms": 1520.0, "peak_reserved_gib": 18.4},
                "3": {"compute_ms": 1680.0, "forward_ms": 880.0, "completion_ms": 2200.0, "window_ms": 2200.0, "peak_reserved_gib": 23.0},
            },
        }
        context = build_context_record(program, runtime_summary=runtime_summary)
        candidates = agent_loop._build_runtime_branch_candidates(program, context)
        branch_ids = {str(candidate.metadata.get("runtime_branch_id") or "") for candidate in candidates}
        kinds = {str(candidate.metadata.get("program_kind") or "") for candidate in candidates}
        morphable = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.metadata.get("program_kind") or "") == "candidate_branch_morphable_pipeline_shape"
            ),
            None,
        )

        self.assertIn("branch_morphable_pipeline_shape", branch_ids)
        self.assertIn("candidate_branch_morphable_pipeline_shape", kinds)
        self.assertIsNotNone(morphable)
        self.assertIn(
            str((morphable.metadata or {}).get("runtime_memory_policy_mode") or ""),
            {"budgeted_joint_runtime_policy", "selective_overlap_aware"},
        )
        self.assertTrue(
            bool((morphable.metadata or {}).get("runtime_recompute_modules"))
            or float((morphable.metadata or {}).get("morphable_estimated_step_time_ms") or 0.0) > 0.0
        )

    def test_apipe_plan_uses_structure_aware_dispatch_for_vpp_wait(self) -> None:
        program = default_dense_program("single_g5")
        program.parallel.pp_degree = 4
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        runtime_summary = {
            "steady_state_step_time_ms_p50": 8200.0,
            "bubble_ratio": 0.05,
            "pipeline_wait_ratio": 0.12,
            "optimizer_exposed_ratio": 0.10,
            "stage_tail_ratio": 0.07,
            "stage_window_summary": {
                "0": {"compute_ms": 1680.0, "forward_ms": 880.0, "completion_ms": 2200.0, "window_ms": 2200.0, "peak_reserved_gib": 23.0},
                "1": {"compute_ms": 1320.0, "forward_ms": 770.0, "completion_ms": 1520.0, "window_ms": 1520.0, "peak_reserved_gib": 18.4},
                "2": {"compute_ms": 1300.0, "forward_ms": 760.0, "completion_ms": 1500.0, "window_ms": 1500.0, "peak_reserved_gib": 18.0},
                "3": {"compute_ms": 1700.0, "forward_ms": 900.0, "completion_ms": 2250.0, "window_ms": 2250.0, "peak_reserved_gib": 23.5},
            },
        }

        plan = trace_reducer._build_apipe_heuristic_plan(program, runtime_summary, [])
        action_names = {item["name"] for item in plan["actions"]}

        self.assertEqual(
            plan["runtime_controls"]["dispatch_order"],
            "structure_aware_critical_first",
        )
        self.assertIn("reprioritize_chunks_by_structure", action_names)

    def test_compile_program_exports_pipe_runtime_envs(self) -> None:
        program = default_dense_program("dual_g5_g5").normalized()
        program.schedule.template = "interleaved_grouped_g4"
        program.schedule.skeleton = "stage_aware_grouped"
        program.schedule.dispatch_order = "tail_boundary_rewrite"
        program.schedule.microbatch_group_size_per_vp_stage = 8
        program.strategy_ir.pipe.warmup_policy = "balanced_fill"
        program.strategy_ir.pipe.cooldown_policy = "opt_prioritized"
        program.metadata["flush_order_policy"] = "reverse_last_group"
        program.metadata["flush_microbatches"] = [8, 9, 10, 11]

        compiled = compile_program(program)
        self.assertEqual(compiled.launcher_env["SCHEDULE_TEMPLATE"], "interleaved_grouped_g4")
        self.assertEqual(compiled.launcher_env["DISPATCH_ORDER"], "tail_boundary_rewrite")
        self.assertEqual(compiled.launcher_env["SCHEDULE_WARMUP_POLICY"], "balanced_fill")
        self.assertEqual(compiled.launcher_env["SCHEDULE_COOLDOWN_POLICY"], "opt_prioritized")
        self.assertEqual(compiled.launcher_env["SCHEDULE_FLUSH_ORDER_POLICY"], "reverse_last_group")
        self.assertEqual(compiled.launcher_env["SCHEDULE_FLUSH_MICROBATCHES"], "8,9,10,11")

    def test_compile_program_exports_runtime_memory_envs(self) -> None:
        program = default_dense_program("single_g5").normalized()
        program.metadata["runtime_recompute_granularity"] = "selective"
        program.metadata["runtime_enable_recompute_activations"] = True
        program.metadata["runtime_recompute_modules"] = ["core_attn", "mlp"]
        program.metadata["runtime_enable_fine_grained_activation_offloading"] = True
        program.metadata["runtime_offload_modules"] = ["core_attn", "attn_proj"]
        program.metadata["schedule_warmup_checkpoint_policy"] = "full"
        program.metadata["schedule_steady_checkpoint_policy"] = "default"
        program.metadata["schedule_cooldown_p2p_policy"] = "serial"
        program.metadata["schedule_warmup_combined_policy"] = "serial"
        program.metadata["schedule_cooldown_combined_policy"] = "serial"

        compiled = compile_program(program)
        self.assertEqual(compiled.launcher_env["RECOMPUTE_GRANULARITY"], "selective")
        self.assertEqual(compiled.launcher_env["ENABLE_RECOMPUTE_ACTIVATIONS"], "1")
        self.assertEqual(compiled.launcher_env["RECOMPUTE_MODULES"], "core_attn,mlp")
        self.assertEqual(
            compiled.launcher_env["ENABLE_FINE_GRAINED_ACTIVATION_OFFLOADING"],
            "1",
        )
        self.assertEqual(compiled.launcher_env["OFFLOAD_MODULES"], "core_attn,attn_proj")
        self.assertEqual(compiled.launcher_env["SCHEDULE_WARMUP_CHECKPOINT_POLICY"], "full")
        self.assertEqual(compiled.launcher_env["SCHEDULE_STEADY_CHECKPOINT_POLICY"], "default")
        self.assertEqual(compiled.launcher_env["SCHEDULE_COOLDOWN_P2P_POLICY"], "serial")
        self.assertEqual(compiled.launcher_env["SCHEDULE_WARMUP_COMBINED_POLICY"], "serial")
        self.assertEqual(compiled.launcher_env["SCHEDULE_COOLDOWN_COMBINED_POLICY"], "serial")

    def test_compile_program_exports_optimizer_runtime_envs(self) -> None:
        program = default_dense_program("single_g5").normalized()
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        program.schedule.template = "interleaved_grouped_g2"
        program.schedule.skeleton = "stage_aware_grouped"
        program.schedule.microbatch_group_size_per_vp_stage = 2
        program.metadata.update(
            {
                "runtime_optimizer_policy_mode": "tail_guarded_overlap",
                "runtime_optimizer_target_policy": "tail_stage_first",
                "runtime_optimizer_chunk_scope": "tail_and_hotspot",
                "runtime_optimizer_window_policy": "tail_flush_aligned",
                "morphable_stage_families": [
                    {
                        "stage_index": 1,
                        "family": "optimizer_guarded_tail",
                        "stage_tags": ["tail_sensitive", "optimizer_sensitive"],
                        "dispatch_order": "optimizer_tail_guarded",
                        "warmup_policy": "balanced_fill",
                        "cooldown_policy": "optimizer_tail_hide",
                        "optimizer_runtime_mode": "tail_guarded_overlap",
                        "optimizer_target_policy": "tail_stage_first",
                        "optimizer_chunk_scope": "tail_and_hotspot",
                        "optimizer_window_policy": "tail_flush_aligned",
                        "optimizer_target_chunk": "tail",
                    }
                ],
            }
        )

        compiled = compile_program(program)
        self.assertEqual(compiled.launcher_env["ENABLE_DISTRIBUTED_OPTIMIZER"], "1")
        self.assertEqual(compiled.launcher_env["ENABLE_OVERLAP_GRAD_REDUCE"], "1")
        self.assertEqual(compiled.launcher_env["ENABLE_OVERLAP_PARAM_GATHER"], "1")
        self.assertEqual(compiled.launcher_env["ENABLE_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP"], "1")
        self.assertEqual(compiled.launcher_env["SCHEDULE_OPTIMIZER_RUNTIME_MODE"], "tail_guarded_overlap")
        self.assertEqual(compiled.launcher_env["SCHEDULE_OPTIMIZER_TARGET_POLICY"], "tail_stage_first")
        self.assertEqual(compiled.launcher_env["SCHEDULE_OPTIMIZER_CHUNK_SCOPE"], "tail_and_hotspot")
        self.assertEqual(compiled.launcher_env["SCHEDULE_OPTIMIZER_WINDOW_POLICY"], "tail_flush_aligned")
        self.assertIn("optimizer_window_policy=tail_flush_aligned", compiled.launcher_env["SCHEDULE_STAGE_FAMILY_HINTS"])
        self.assertIn("stage_tags=tail_sensitive|optimizer_sensitive", compiled.launcher_env["SCHEDULE_STAGE_FAMILY_HINTS"])

    def test_check_program_rejects_optimizer_runtime_without_interleaving(self) -> None:
        program = default_dense_program("single_g5").normalized()
        program.metadata.update(
            {
                "runtime_optimizer_policy_mode": "tail_guarded_overlap",
                "runtime_optimizer_target_policy": "tail_stage_first",
                "runtime_optimizer_chunk_scope": "tail_only",
                "runtime_optimizer_window_policy": "tail_flush_aligned",
            }
        )

        legality = check_program(program)
        self.assertFalse(legality.is_valid)
        self.assertTrue(any("interleaved" in error for error in legality.errors))

    def test_check_program_rejects_optimizer_targeted_window_override_without_overlap_runtime(self) -> None:
        program = default_dense_program("single_g5").normalized()
        program.parallel.pp_degree = 2
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        program.schedule.template = "interleaved_grouped_g2"
        program.schedule.skeleton = "stage_aware_grouped"
        program.schedule.microbatch_group_size_per_vp_stage = 2
        program.metadata["runtime_window_overrides"] = [
            {
                "phase": "steady",
                "window": "last_2_groups",
                "stage_selector": "optimizer_sensitive_stage",
                "chunk_order_policy": "target_chunk_first",
                "optimizer_target_chunk": "tail",
            }
        ]

        legality = check_program(program)
        self.assertFalse(legality.is_valid)
        self.assertTrue(any("optimizer-targeted window overrides" in error for error in legality.errors))

    def test_compile_program_exports_morphable_pipeline_envs(self) -> None:
        program = default_dense_program("single_g5").normalized()
        program.parallel.pp_degree = 4
        program.parallel.vpp_degree = 2
        program.layout.vpp_degree = 2
        program.metadata["morphable_objective_type"] = "minimize_step_time_under_memory_budget"
        program.metadata["morphable_estimated_step_time_ms"] = 8021.25
        program.metadata["morphable_estimated_step_delta_ms"] = -144.5
        program.metadata["morphable_shape_signature"] = "shape:test"
        program.metadata["morphable_chunk_shape_vector"] = [2, 1, 1, 2]
        program.metadata["morphable_stage_families"] = [
            {
                "stage_index": 0,
                "family": "critical_path_first",
                "dispatch_order": "structure_aware_critical_first",
                "warmup_policy": "balanced_fill",
                "cooldown_policy": "opt_prioritized",
                "checkpoint_policy": "selective",
                "p2p_policy": "serial",
                "combined_policy": "serial",
                "chunk_priority_hints": [4, 2],
            },
            {
                "stage_index": 1,
                "family": "memory_guarded",
                "dispatch_order": "middle_stage_relief",
                "warmup_policy": "balanced_fill",
                "cooldown_policy": "tail_min",
                "checkpoint_policy": "selective",
                "p2p_policy": "serial",
                "combined_policy": "serial",
                "chunk_priority_hints": [3, 1],
            },
        ]

        compiled = compile_program(program)
        self.assertEqual(compiled.launcher_env["ENABLE_MORPHABLE_PIPELINE"], "1")
        self.assertEqual(compiled.launcher_env["MORPHABLE_PIPE_OBJECTIVE"], "minimize_step_time_under_memory_budget")
        self.assertEqual(compiled.launcher_env["MORPHABLE_PIPE_ESTIMATED_STEP_TIME_MS"], "8021.2500")
        self.assertEqual(compiled.launcher_env["MORPHABLE_PIPE_ESTIMATED_STEP_DELTA_MS"], "-144.5000")
        self.assertEqual(compiled.launcher_env["MORPHABLE_PIPE_SHAPE_SIGNATURE"], "shape:test")
        self.assertEqual(compiled.launcher_env["MORPHABLE_PIPE_CHUNK_SHAPE_VECTOR"], "2,1,1,2")
        self.assertIn("0,family=critical_path_first", compiled.launcher_env["SCHEDULE_STAGE_FAMILY_HINTS"])
        self.assertIn("0:4,2", compiled.launcher_env["SCHEDULE_STAGE_CHUNK_PRIORITY_HINTS"])

    def test_context_record_includes_runtime_memory_policy(self) -> None:
        program = default_dense_program("single_g5")
        program.constraints.memory_budget_gb = 24.0
        runtime_summary = {
            "stage_window_summary": {
                "0": {
                    "compute_ms": 1200.0,
                    "forward_ms": 700.0,
                    "completion_ms": 1800.0,
                    "window_ms": 1800.0,
                    "peak_reserved_gib": 23.1,
                },
                "1": {
                    "compute_ms": 1100.0,
                    "forward_ms": 760.0,
                    "completion_ms": 1500.0,
                    "window_ms": 1500.0,
                    "peak_reserved_gib": 18.4,
                },
            },
        }

        context = build_context_record(program, runtime_summary=runtime_summary)
        local_memory = context["evidence_record"]["local_memory_search_space"]
        runtime_policy = local_memory["runtime_policy"]

        self.assertEqual(
            runtime_policy["status"], "executable_now_with_module_level_approximation"
        )
        self.assertEqual(runtime_policy["recompute_modules"], ["core_attn", "mlp"])
        self.assertEqual(runtime_policy["offload_modules"], ["attn_proj", "core_attn"])
        self.assertEqual(runtime_policy["warmup_checkpoint_policy"], "full")
        self.assertEqual(runtime_policy["warmup_combined_policy"], "serial")
        self.assertEqual(runtime_policy["steady_checkpoint_policy"], "default")

    def test_schedule_phase_helpers_support_phase_local_policies(self) -> None:
        from megatron.core.pipeline_parallel.schedules import (
            _phase_uses_combined_overlap,
            _phase_uses_p2p_overlap,
            _resolve_execution_phase_for_virtual_microbatches,
            _resolve_phase_checkpoint_policy,
        )

        with mock.patch.dict(
            os.environ,
            {
                "SCHEDULE_WARMUP_CHECKPOINT_POLICY": "full",
                "SCHEDULE_STEADY_CHECKPOINT_POLICY": "off",
                "SCHEDULE_COOLDOWN_P2P_POLICY": "serial",
                "SCHEDULE_WARMUP_COMBINED_POLICY": "serial",
                "SCHEDULE_STEADY_COMBINED_POLICY": "combined",
            },
            clear=False,
        ):
            self.assertTrue(_resolve_phase_checkpoint_policy("warmup", None))
            self.assertFalse(_resolve_phase_checkpoint_policy("steady", True))
            self.assertFalse(_phase_uses_p2p_overlap("cooldown", True))
            self.assertTrue(_phase_uses_p2p_overlap("warmup", True))
            self.assertEqual(
                _resolve_execution_phase_for_virtual_microbatches(
                    f_virtual_microbatch_id=3, b_virtual_microbatch_id=None
                ),
                "warmup",
            )
            self.assertEqual(
                _resolve_execution_phase_for_virtual_microbatches(
                    f_virtual_microbatch_id=3, b_virtual_microbatch_id=1
                ),
                "steady",
            )
            self.assertEqual(
                _resolve_execution_phase_for_virtual_microbatches(
                    f_virtual_microbatch_id=None, b_virtual_microbatch_id=1
                ),
                "cooldown",
            )
            self.assertFalse(_phase_uses_combined_overlap("warmup", True))
            self.assertTrue(_phase_uses_combined_overlap("steady", True))

    def test_schedule_table_supports_flush_reordering_envs(self) -> None:
        from megatron.core.pipeline_parallel.schedules import get_schedule_table

        with mock.patch.dict(
            os.environ,
            {
                "SCHEDULE_TEMPLATE": "pp4_middle_relief",
                "DISPATCH_ORDER": "tail_boundary_rewrite",
                "SCHEDULE_WARMUP_POLICY": "balanced_fill",
                "SCHEDULE_COOLDOWN_POLICY": "opt_prioritized",
                "SCHEDULE_FLUSH_ORDER_POLICY": "reverse_last_group",
            },
            clear=False,
        ):
            schedule_table = get_schedule_table(16, 2, 8)

        self.assertEqual([microbatch_id for microbatch_id, _ in schedule_table[-16:-8]], list(range(15, 7, -1)))
        self.assertEqual([microbatch_id for microbatch_id, _ in schedule_table[-8:]], list(range(15, 7, -1)))

    def test_schedule_runtime_policy_parses_structured_env_hints(self) -> None:
        from megatron.core.pipeline_parallel import schedules as pp_schedules

        with mock.patch.dict(
            os.environ,
            {
                "SCHEDULE_FAMILY": "zero_bubble",
                "SCHEDULE_DISPATCH_ORDER": "zero_bubble_proxy",
                "SCHEDULE_LANE_POLICY": "2",
                "SCHEDULE_GROUP_SIZE_VECTOR": json.dumps([1, 2, 2, 1]),
                "SCHEDULE_STAGE_SEMANTIC_HINTS": json.dumps(
                    {"1": {"family": "tail_heavy", "local_dispatch_hint": "tail_boundary_rewrite"}}
                ),
                "SCHEDULE_OVERLAP_HINTS": json.dumps({"enable_p2p_overlap": True}),
                "SCHEDULE_MEMORY_HINTS": json.dumps({"offload_policy": "guarded"}),
                "SCHEDULE_PARTITION_HINTS": json.dumps({"partition_mode": "nonuniform"}),
            },
            clear=False,
        ):
            with mock.patch.object(pp_schedules.parallel_state, "model_parallel_is_initialized", return_value=True):
                with mock.patch.object(pp_schedules.parallel_state, "get_pipeline_model_parallel_rank", return_value=1):
                    with mock.patch.object(pp_schedules.parallel_state, "get_pipeline_model_parallel_world_size", return_value=4):
                        policy = pp_schedules.get_schedule_runtime_policy()
                        hook_payload = pp_schedules.invoke_schedule_runtime_hook(
                            "before_forward_hook", {"stage_id": 1}
                        )
                        local_group_size = pp_schedules._resolve_local_group_size(4)

        self.assertEqual(policy["family"], "zero_bubble")
        self.assertEqual(policy["dispatch_order"], "zero_bubble_proxy")
        self.assertEqual(policy["lane_policy"], "2")
        self.assertEqual(policy["group_size_vector"], [1, 2, 2, 1])
        self.assertEqual(policy["local_stage_hint"]["family"], "tail_heavy")
        self.assertEqual(local_group_size, 2)
        self.assertEqual(hook_payload["status"], "ready")
        self.assertEqual(hook_payload["schedule_family"], "zero_bubble")
        self.assertEqual(hook_payload["stage_semantics"]["family"], "tail_heavy")

    def test_compile_program_exports_schedule_grid_and_action_specs(self) -> None:
        program = default_dense_program("single_g5").normalized()
        program.schedule_ir.family = "zero_bubble"
        program.schedule_ir.dispatch_order = "zero_bubble_proxy"
        program.schedule_ir.microbatch_lanes = 2
        program.schedule_ir.microbatch_group_size_per_vp_stage = 2

        compiled = compile_program(program)
        schedule_grid = json.loads(compiled.launcher_env["SCHEDULE_GRID_SPEC"])
        derived_actions = json.loads(compiled.launcher_env["SCHEDULE_ACTION_SPECS"])

        self.assertEqual(schedule_grid["family"], "zero_bubble")
        self.assertGreaterEqual(schedule_grid["lanes"], 2)
        self.assertGreater(schedule_grid["time_slots"], 0)
        self.assertTrue(any(item.get("kind") == "FWD" for item in schedule_grid["cells"]))
        self.assertTrue(any(item.get("kind") == "BWD_ACT" for item in schedule_grid["cells"]))
        self.assertTrue(any(item.get("kind") == "WGRAD_OPT" for item in schedule_grid["cells"]))
        self.assertTrue(any(item.get("action_type") == "FWD" for item in derived_actions))
        self.assertTrue(any(item.get("action_type") == "BWD_ACT" for item in derived_actions))
        self.assertTrue(any(item.get("action_type") == "WAIT" for item in derived_actions))

    def test_compile_program_exports_stateful_plan_payloads(self) -> None:
        program = default_dense_program("single_g5").normalized()

        compiled = compile_program(program)

        self.assertEqual(compiled.launcher_env["ENABLE_STATEFUL_SCHEDULE"], "1")
        self.assertIn("SCHEDULE_NODE_SPECS", compiled.launcher_env)
        self.assertIn("SCHEDULE_EDGE_SPECS", compiled.launcher_env)
        self.assertIn("STATE_PLAN", compiled.launcher_env)
        self.assertIn("TELEMETRY_BUDGET", compiled.launcher_env)
        self.assertIn("WINDOW_RECONFIG_PLAN", compiled.launcher_env)
        self.assertIn("GLOBAL_STRATEGY_PLAN", compiled.launcher_env)
        self.assertIn("REWRITE_EXECUTION_PLAN", compiled.launcher_env)
        self.assertIn("WINDOW_FEEDBACK_PLAN", compiled.launcher_env)
        self.assertGreater(len(compiled.stateful_schedule_nodes), 0)
        self.assertGreater(len(compiled.stateful_schedule_edges), 0)
        self.assertGreater(len(list((compiled.state_plan or {}).get("objects") or [])), 0)
        self.assertEqual(str((compiled.telemetry_budget or {}).get("level") or ""), "summary")
        global_strategy = json.loads(compiled.launcher_env["GLOBAL_STRATEGY_PLAN"])
        rewrite_plan = json.loads(compiled.launcher_env["REWRITE_EXECUTION_PLAN"])
        window_feedback_plan = json.loads(compiled.launcher_env["WINDOW_FEEDBACK_PLAN"])
        self.assertEqual(str(global_strategy.get("primary_parallel_mode") or ""), "pp_vpp")
        self.assertEqual(str((rewrite_plan.get("global_strategy") or {}).get("primary_parallel_mode") or ""), "pp_vpp")
        self.assertIn("recommended_rewrites", window_feedback_plan)

    def test_megatron_program_normalized_derives_global_strategy_and_rewrite_plan(self) -> None:
        program = default_dense_program("single_g5")
        program.global_strategy_plan = None
        program.rewrite_plan = None

        restored = type(program).from_dict(program.to_dict()).normalized()

        self.assertIsNotNone(restored.global_strategy_plan)
        self.assertIsNotNone(restored.rewrite_plan)
        self.assertEqual(restored.global_strategy_plan.primary_parallel_mode, "pp_vpp")
        self.assertEqual(restored.global_strategy_plan.stage_count, restored.parallel.pp_degree)
        self.assertEqual(restored.rewrite_plan.global_strategy.primary_parallel_mode, "pp_vpp")
        self.assertEqual(restored.rewrite_plan.telemetry_budget.level, "summary")
        self.assertEqual(restored.rewrite_plan.window_reconfig.window_steps, 4)

    def test_stateful_default_is_activation_centric(self) -> None:
        program = default_dense_program("single_g5").normalized()

        compiled = compile_program(program)
        state_objects = list((compiled.state_plan or {}).get("objects") or [])
        state_types = {str(item.get("state_type") or "") for item in state_objects}

        self.assertIn("activation", state_types)
        self.assertNotIn("parameter", state_types)
        self.assertNotIn("optimizer", state_types)

    def test_runtime_action_view_parses_grid_and_action_specs(self) -> None:
        from megatron.core.pipeline_parallel import schedules as pp_schedules

        with mock.patch.dict(
            os.environ,
            {
                "SCHEDULE_FAMILY": "dualpipe_v",
                "SCHEDULE_GRID_SPEC": json.dumps(
                    {
                        "lanes": 2,
                        "time_slots": 6,
                        "cells": [{"kind": "FWD", "stage_id": 0, "lane_id": 0, "microbatch_id": 0, "vchunk_id": 0, "time_slot": 0}],
                        "family": "dualpipe_v",
                        "stage_count": 2,
                        "vstage_count": 2,
                        "microbatch_count": 2,
                        "weight_version_policy": "delayed_wgrad",
                        "constraints": {"dispatch_order": "balanced_round_robin"},
                        "notes": ["contract-test"],
                    }
                ),
                "SCHEDULE_ACTION_SPECS": json.dumps(
                    [
                        {
                            "action_type": "FWD",
                            "stage_id": 0,
                            "lane_id": 0,
                            "microbatch_id": 0,
                            "vchunk_id": 0,
                            "time_slot": 0,
                            "duration_hint": 1.0,
                            "dependency_ids": [],
                            "memory_delta": 0.0,
                        }
                    ]
                ),
            },
            clear=False,
        ):
            action_view = pp_schedules.get_schedule_action_view()
            registry = pp_schedules.get_schedule_family_registry()

        self.assertEqual(action_view["family"], "dualpipe_v")
        self.assertEqual(action_view["grid_spec"]["family"], "dualpipe_v")
        self.assertEqual(action_view["action_count"], 1)
        self.assertIn("dualpipe_v", registry)
        self.assertEqual(registry["dualpipe_v"]["steady_rule"], "dualpipe_overlap")

    def test_schedule_runtime_policy_parses_telemetry_budget(self) -> None:
        from megatron.core.pipeline_parallel import schedules as pp_schedules

        with mock.patch.dict(
            os.environ,
            {
                "TELEMETRY_BUDGET": json.dumps(
                    {
                        "level": "summary",
                        "max_trace_mb": 64,
                        "max_events_per_rank": 4096,
                        "sampled_windows": 1,
                        "emit_compare_svg": False,
                    }
                ),
            },
            clear=False,
        ):
            policy = pp_schedules.get_schedule_runtime_policy()

        self.assertEqual(policy["telemetry_budget"]["level"], "summary")
        self.assertEqual(policy["telemetry_budget"]["max_trace_mb"], 64)
        self.assertEqual(policy["telemetry_budget"]["max_events_per_rank"], 4096)

    def test_schedule_action_runner_writes_runtime_trace(self) -> None:
        from megatron.core.pipeline_parallel import schedules as pp_schedules

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {
                    "SCHEDULE_RUNTIME_TRACE_DIR": tmpdir,
                    "SCHEDULE_FAMILY": "fixed_1f1b",
                    "SCHEDULE_ACTION_SPECS": json.dumps(
                        [
                            {
                                "action_type": "FWD",
                                "stage_id": 1,
                                "lane_id": 0,
                                "microbatch_id": 2,
                                "vchunk_id": 0,
                                "time_slot": 3,
                            }
                        ]
                    ),
                },
                clear=False,
            ):
                with mock.patch.object(pp_schedules.parallel_state, "model_parallel_is_initialized", return_value=True):
                    with mock.patch.object(pp_schedules.parallel_state, "get_pipeline_model_parallel_rank", return_value=1):
                        with mock.patch.object(pp_schedules.parallel_state, "get_pipeline_model_parallel_world_size", return_value=4):
                            runner = pp_schedules.get_schedule_action_runner(force_reset=True)
                            runner.set_phase("steady")
                            runner.set_context(microbatch_id=2, vchunk_id=0, lane_id=0)
                            fwd_token = runner.begin_action("FWD", microbatch_id=2, vchunk_id=0, phase="steady")
                            runner.end_action(fwd_token)
                            comm_token = runner.begin_action(
                                "COMM",
                                microbatch_id=2,
                                vchunk_id=0,
                                phase="steady",
                                lane_id=1,
                                metadata={"op_name": "send_forward"},
                            )
                            runner.end_action(comm_token, metadata={"op_name": "send_forward"})
                            trace_path = runner.flush()

            self.assertIsNotNone(trace_path)
            payload = json.loads(Path(str(trace_path)).read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "schedule_runtime_event_trace")
            self.assertEqual(payload["family"], "fixed_1f1b")
            self.assertEqual(payload["stage_id"], 1)
            self.assertEqual(len(payload["events"]), 2)
            self.assertTrue(any(item.get("action_type") == "COMM" for item in payload["events"]))
            self.assertGreaterEqual(float((payload.get("metrics") or {}).get("comm_ms") or 0.0), 0.0)

    def test_schedule_action_runner_summary_mode_omits_full_events(self) -> None:
        from megatron.core.pipeline_parallel import schedules as pp_schedules

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {
                    "SCHEDULE_RUNTIME_TRACE_DIR": tmpdir,
                    "SCHEDULE_FAMILY": "fixed_1f1b",
                    "TELEMETRY_BUDGET": json.dumps(
                        {
                            "level": "summary",
                            "max_trace_mb": 16,
                            "max_events_per_rank": 8,
                            "sampled_windows": 1,
                            "emit_compare_svg": False,
                        }
                    ),
                },
                clear=False,
            ):
                with mock.patch.object(pp_schedules.parallel_state, "model_parallel_is_initialized", return_value=True):
                    with mock.patch.object(pp_schedules.parallel_state, "get_pipeline_model_parallel_rank", return_value=0):
                        with mock.patch.object(pp_schedules.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2):
                            runner = pp_schedules.get_schedule_action_runner(force_reset=True)
                            runner.set_context(microbatch_id=0, vchunk_id=0, lane_id=0)
                            token = runner.begin_action("FWD", microbatch_id=0, vchunk_id=0, phase="steady")
                            runner.end_action(token)
                            trace_path = runner.flush()
                            payload = json.loads(Path(str(trace_path)).read_text(encoding="utf-8"))

        self.assertEqual(payload["telemetry"]["effective_level"], "summary")
        self.assertEqual(payload["events"], [])
        self.assertGreaterEqual(float((payload.get("metrics") or {}).get("observed_action_count") or 0.0), 1.0)

    def test_trial_runner_loads_runtime_schedule_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "runtime_schedule_traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "format": "schedule_runtime_event_trace",
                "family": "zero_bubble",
                "stage_id": 0,
                "events": [{"action_type": "FWD", "stage_id": 0, "microbatch_id": 0, "vchunk_id": 0, "start_ms": 0.0, "end_ms": 1.0, "duration_ms": 1.0}],
            }
            (trace_dir / "schedule_runtime_trace_rank000_stage000.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            traces = trial_runner._load_runtime_schedule_traces({"runtime_trace_dir": str(trace_dir)})

        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["family"], "zero_bubble")
        self.assertTrue(str(traces[0]["artifact_path"]).endswith(".json"))

    def test_trial_runner_skips_oversized_runtime_schedule_trace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "runtime_schedule_traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            huge_path = trace_dir / "schedule_runtime_trace_rank000_stage000.json"
            with huge_path.open("wb") as handle:
                handle.truncate(129 * 1024 * 1024)

            traces = trial_runner._load_runtime_schedule_traces({"runtime_trace_dir": str(trace_dir)})

        self.assertEqual(len(traces), 1)
        self.assertTrue(bool(traces[0].get("skipped_due_to_size")))
        self.assertEqual(str((traces[0].get("telemetry") or {}).get("effective_level") or ""), "summary")

    def test_write_analysis_artifacts_emits_window_feedback_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dirs = {
                "trial_dir": str(root / "trial_000"),
                "analysis_dir": str(root / "trial_000" / "analysis"),
            }
            metrics = {
                "trial_artifact": {},
                "context_record": {},
                "trace_summary": {},
                "returncode": 0,
                "window_feedback": {
                    "window_index": 1,
                    "policy_signature": "sig-window",
                    "critical_component_type": "reload",
                    "rollback_triggered": True,
                },
                "window_feedback_history": [
                    {"window_index": 0, "policy_signature": "sig-window"},
                    {"window_index": 1, "policy_signature": "sig-window", "rollback_triggered": True},
                ],
            }

            outputs = trial_runner._write_analysis_artifacts(output_dirs, metrics)

            window_feedback = json.loads(Path(outputs["window_feedback_json"]).read_text(encoding="utf-8"))
            history = json.loads(Path(outputs["window_feedback_history_json"]).read_text(encoding="utf-8"))
            self.assertEqual(window_feedback["policy_signature"], "sig-window")
            self.assertTrue(window_feedback["rollback_triggered"])
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1]["window_index"], 1)

    def test_runtime_schedule_traces_feed_visualization_artifacts(self) -> None:
        program = default_dense_program("single_g5").normalized()
        metrics = {
            "step_time_ms_p50": 1000.0,
            "steady_state_step_time_ms_p50": 1000.0,
            "bubble_ratio": 0.08,
            "runtime_schedule_traces": [
                {
                    "format": "schedule_runtime_event_trace",
                    "family": "zero_bubble",
                    "stage_id": 0,
                    "stage_semantics": {"family": "tail_heavy"},
                    "metrics": {"comm_ms": 22.0, "reload_stall_ms": 11.0},
                    "events": [
                        {
                            "action_type": "FWD",
                            "stage_id": 0,
                            "lane_id": 0,
                            "microbatch_id": 0,
                            "vchunk_id": 0,
                            "phase": "warmup",
                            "start_ms": 0.0,
                            "end_ms": 8.0,
                            "duration_ms": 8.0,
                        },
                        {
                            "action_type": "COMM",
                            "stage_id": 0,
                            "lane_id": 1,
                            "microbatch_id": 0,
                            "vchunk_id": 0,
                            "phase": "warmup",
                            "start_ms": 8.0,
                            "end_ms": 10.0,
                            "duration_ms": 2.0,
                            "metadata": {"op_name": "send_forward"},
                        },
                        {
                            "action_type": "WGRAD_OPT",
                            "stage_id": 0,
                            "lane_id": 0,
                            "microbatch_id": 0,
                            "vchunk_id": 0,
                            "phase": "cooldown",
                            "start_ms": 12.0,
                            "end_ms": 18.0,
                            "duration_ms": 6.0,
                        },
                    ],
                }
            ],
        }

        observation = build_agent_observation(program, metrics=metrics)
        artifact = build_trial_artifact(program, observation)
        visual = dict((artifact.get("visualization_artifacts") or {}))
        event_trace = dict(visual.get("pipeline_event_trace") or {})
        grid_trace = dict(visual.get("pipeline_grid_trace") or {})
        next_step = dict(artifact.get("next_step_hypotheses") or {})
        critical_path = dict((((observation.to_dict().get("evidence_record") or {}).get("critical_path_breakdown")) or {}))

        self.assertEqual(event_trace.get("timing_basis"), "runtime_observed")
        self.assertEqual((event_trace.get("summary") or {}).get("schedule_template"), "zero_bubble")
        self.assertTrue(any(item.get("op_kind") == "comm" for item in event_trace.get("events", [])))
        self.assertTrue(any(item.get("kind") == "WGRAD_OPT" for item in grid_trace.get("cells", [])))
        self.assertIn("next_schedule_family", next_step)
        self.assertIn("stop_signal", next_step)
        self.assertIn("local_verticalization_targets", next_step)
        self.assertIn("critical_stage_id", critical_path)
        self.assertIn("critical_layer_group_id", critical_path)
        self.assertIn("critical_component_type", critical_path)

    def test_megatron_program_roundtrip_preserves_new_schedule_ir(self) -> None:
        from megatron_agent.config import (
            MemoryIntentSpec,
            OverlapIntentSpec,
            PartitionOptimizationSpec,
            ProgramPatchSpec,
            ScheduleIRSpec,
            StageSemanticSpec,
        )

        program = default_dense_program("single_g5")
        program.schedule_ir = ScheduleIRSpec(
            family="zero_bubble",
            skeleton="zero_bubble",
            microbatch_lanes=2,
            microbatch_group_size_per_vp_stage=2,
            dispatch_order="zero_bubble_proxy",
            warmup_policy="balanced_fill",
            steady_state_policy="zero_bubble",
            cooldown_policy="optimizer_tail_hide",
            weight_version_policy="delayed_wgrad",
            virtual_stage_grouping=[1, 1],
            stage_semantics=[StageSemanticSpec(stage_id=0, family="tail_heavy", prefer_delayed_wgrad=True)],
            overlap_intents=OverlapIntentSpec(enable_p2p_overlap=True, enable_optimizer_tail_overlap=True),
            memory_intents=MemoryIntentSpec(checkpoint_policy="selective", offload_policy="guarded"),
        )
        program.partition_optimization = PartitionOptimizationSpec(
            partition_mode="nonuniform",
            allow_nonuniform_partition=True,
            stage_layer_counts=[18, 22],
            stage_local_vpp_vector=[1, 2],
            preferred_boundary_modules=["attention"],
        )
        program.applied_patch = ProgramPatchSpec(
            patch_id="patch-zero-bubble",
            patch_family="change_schedule_family",
            target_scope="schedule",
            changes={"family": "zero_bubble"},
        )
        restored = type(program).from_dict(program.to_dict()).normalized()

        self.assertEqual(restored.schedule_ir.family, "zero_bubble")
        self.assertEqual(restored.schedule_ir.dispatch_order, "zero_bubble_proxy")
        self.assertEqual(restored.partition_optimization.partition_mode, "nonuniform")
        self.assertEqual(restored.partition_optimization.stage_layer_counts, [18, 22])
        self.assertEqual(restored.applied_patch.patch_family, "change_schedule_family")

    def test_policy_memory_recommend_patch_families_prefers_useful_entries(self) -> None:
        from megatron_agent.policy_memory import summarize_state_for_memory

        bank = PolicyMemoryBank()
        target_state = {
            "model": {"model_track": "dense", "size_bucket": "qwen14b"},
            "hardware": {"run_target": "single_g5", "backend_family": "megatron"},
            "policy": {"runtime_schedule_family": "fixed_1f1b", "pp_degree": 4, "vpp_degree": 2},
            "runtime": {"bubble_ratio": 0.18, "peak_reserved_ratio": 0.88},
            "bottleneck_signature": "memory_bound",
        }
        key = json.dumps(
            {
                "model_family": "dense",
                "size_bucket": "qwen14b",
                "hardware_profile": "single_g5",
                "backend_family": "megatron",
                "bottleneck_signature": "memory_bound",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        bank.patch_memory[key] = {
            "schedule_family": "zero_bubble",
            "useful_patch_families": ["change_schedule_family", "add_offload_policy"],
            "harmful_patch_families": ["aggressive_vpp_expansion"],
            "uncertain_patch_families": ["change_partition_boundary"],
            "pareto_stats": {"attempts": 4, "successes": 3, "best_step_improvement_ms": 120.0, "best_throughput_gain": 0.12},
            "recent_failure_signatures": ["oom,tail_drag"],
        }

        guidance = bank.recommend_patch_families(target_state, top_k=3)

        self.assertEqual(guidance["state_summary"], summarize_state_for_memory(target_state))
        self.assertIn("change_schedule_family", guidance["useful_patch_families"])
        self.assertIn("aggressive_vpp_expansion", guidance["harmful_patch_families"])
        self.assertIn("zero_bubble", guidance["schedule_families"])
        self.assertTrue(bank.should_avoid_patch_family(target_state, "aggressive_vpp_expansion"))

    def test_policy_memory_records_rewrite_targets_and_rollback_history(self) -> None:
        search_state = {
            "model": {"model_track": "dense", "size_bucket": "qwen14b"},
            "hardware": {"run_target": "single_g5", "backend_family": "megatron"},
            "policy": {"runtime_schedule_family": "fixed_1f1b", "pp_degree": 2, "vpp_degree": 2},
            "runtime": {"bubble_ratio": 0.06, "peak_reserved_ratio": 0.86},
            "bottleneck_signature": "reload_bound",
        }
        candidate = default_dense_program("single_g5").normalized()
        candidate.metadata["runtime_branch_target_stage_ids"] = [1]
        candidate.metadata["target_layer_groups"] = ["group_reload_hot"]
        candidate.metadata["target_state_objects"] = ["activation:group_reload_hot"]
        candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-reload-shift",
            patch_family="reload_shift_patch",
            target_scope="memory",
            changes={"direction": "earlier"},
        )
        proposal = AgentProposal(
            proposal_id="candidate_reload_shift",
            scope="memory",
            program=candidate,
            priority_rank=8,
        ).normalized()
        outcome = build_trial_outcome(
            candidate,
            {
                "config_name": "candidate_reload_shift",
                "returncode": 1,
                "error_msg": "runtime regression",
                "window_feedback": {
                    "window_index": 1,
                    "policy_signature": "sig-reload-shift",
                    "critical_stage_id": 1,
                    "critical_layer_group_id": "group_reload_hot",
                    "critical_component_type": "reload",
                    "step_time_ms_p50": 1110.0,
                    "throughput_tokens_per_s": 1420.0,
                    "reload_stall_ms": 82.0,
                    "rollback_triggered": True,
                    "critical_path_breakdown": {"target_state_objects": ["activation:group_reload_hot"]},
                },
                "trace_summary": {
                    "critical_path_breakdown": {"critical_component_type": "reload"},
                    "reload_interference_summary": {"reload_stall_ms": 82.0},
                },
            },
            baseline_metrics={"step_time_ms_p50": 1000.0, "throughput_tokens_per_s": 1500.0},
        )
        reflection = reflect_on_trial(search_state, proposal, outcome, baseline_metrics={"step_time_ms_p50": 1000.0})
        bank = PolicyMemoryBank()

        case = record_trial_feedback(bank, search_state, proposal, outcome, reflection)
        guidance = bank.recommend_patch_families(search_state, top_k=3)

        self.assertEqual(case.local_policy["target_stage_ids"], [1])
        self.assertEqual(case.local_policy["target_layer_group_ids"], ["group_reload_hot"])
        self.assertEqual(case.local_policy["target_state_ids"], ["activation:group_reload_hot"])
        self.assertIn("reload_shift_patch", guidance["harmful_patch_families"])
        self.assertIn("reload_shift_patch", guidance["harmful_rewrite_families"])
        self.assertEqual(list((guidance.get("rewrite_targets") or {}).get("target_stage_ids") or []), [1])
        self.assertIn("performance_regression", list(guidance.get("recent_rollback_reasons") or []))
        self.assertTrue(bank.should_avoid_patch_family(search_state, "reload_shift_patch"))

    def test_build_beam_snapshot_prefers_high_value_patch(self) -> None:
        baseline = default_dense_program("single_g5").normalized()
        stronger = baseline.normalized()
        stronger.metadata["program_kind"] = "candidate_stronger"
        stronger.applied_patch = ProgramPatchSpec(
            patch_id="p-strong",
            patch_family="change_schedule_family",
            target_scope="schedule",
            changes={"family": "zero_bubble"},
        )
        weaker = baseline.normalized()
        weaker.metadata["program_kind"] = "candidate_weaker"
        weaker.applied_patch = ProgramPatchSpec(
            patch_id="p-weak",
            patch_family="add_offload_policy",
            target_scope="memory",
            changes={"offload_policy": "guarded"},
        )
        stronger_proposal = AgentProposal(
            proposal_id="stronger",
            scope="schedule",
            program=stronger,
            priority_rank=5,
        ).normalized()
        weaker_proposal = AgentProposal(
            proposal_id="weaker",
            scope="memory",
            program=weaker,
            priority_rank=9,
        ).normalized()
        proposal_nodes = {
            "stronger": agent_loop.SearchTreeNode(
                node_id="stronger",
                depth=1,
                patch_family="change_schedule_family",
                visits=3,
                total_value=1.8,
                last_result={"score": 0.8, "config_name": "candidate_stronger"},
            ),
            "weaker": agent_loop.SearchTreeNode(
                node_id="weaker",
                depth=1,
                patch_family="add_offload_policy",
                visits=3,
                total_value=0.2,
                last_result={"score": 0.1, "config_name": "candidate_weaker"},
            ),
        }

        snapshot = agent_loop._build_beam_snapshot(
            [stronger_proposal, weaker_proposal],
            proposal_nodes,
            beam_width=1,
        )

        self.assertEqual(len(snapshot), 1)
        self.assertEqual(str(snapshot[0].get("proposal_id") or ""), "stronger")
        self.assertEqual(str(snapshot[0].get("patch_family") or ""), "change_schedule_family")

    def test_build_search_tree_root_whole_config_ignores_patch_family_bias(self) -> None:
        baseline = default_dense_program("single_g5").normalized()
        schedule_candidate = baseline.normalized()
        schedule_candidate.metadata["program_kind"] = "candidate_schedule"
        schedule_candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-schedule",
            patch_family="change_schedule_family",
            target_scope="schedule",
            changes={"family": "zero_bubble"},
        )
        schedule_candidate.schedule_ir.family = "zero_bubble"
        memory_candidate = baseline.normalized()
        memory_candidate.metadata["program_kind"] = "candidate_memory"
        memory_candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-memory",
            patch_family="add_offload_policy",
            target_scope="memory",
            changes={"offload_policy": "guarded"},
        )
        memory_candidate.metadata["priority_rank"] = 0
        schedule_proposal = AgentProposal(
            proposal_id="candidate_schedule",
            scope="schedule",
            program=schedule_candidate,
            priority_rank=10,
        ).normalized()
        memory_proposal = AgentProposal(
            proposal_id="candidate_memory",
            scope="memory",
            program=memory_candidate,
            priority_rank=10,
        ).normalized()
        bank = PolicyMemoryBank()
        with mock.patch.object(
            bank,
            "recommend_patch_families",
            return_value={
                "state_summary": {"bottleneck_signature": "bubble_bound"},
                "useful_patch_families": ["change_schedule_family"],
                "harmful_patch_families": ["add_offload_policy"],
            },
        ):
            root, _ = agent_loop._build_search_tree_root(
                search_state={"bottleneck_signature": "bubble_bound"},
                proposal_pool=[schedule_proposal, memory_proposal],
                policy_memory_bank=bank,
                search_unit="whole_config",
                patch_memory_enabled=False,
                next_step_hypotheses={"next_schedule_family": "zero_bubble"},
            )

        self.assertEqual(len(root.children), 2)
        self.assertAlmostEqual(float(root.children[0].priority), float(root.children[1].priority), places=6)
        self.assertEqual(list(root.children[0].priority_reason or []), [])
        self.assertEqual(list(root.children[1].priority_reason or []), [])

    def test_build_search_tree_root_skips_patch_memory_when_disabled(self) -> None:
        baseline = default_dense_program("single_g5").normalized()
        proposal = AgentProposal(
            proposal_id="candidate_schedule",
            scope="schedule",
            program=baseline,
            priority_rank=5,
        ).normalized()
        bank = PolicyMemoryBank()
        with mock.patch.object(bank, "recommend_patch_families", side_effect=AssertionError("should not be called")):
            root, guidance = agent_loop._build_search_tree_root(
                search_state={"bottleneck_signature": "bubble_bound"},
                proposal_pool=[proposal],
                policy_memory_bank=bank,
                search_unit="patch",
                patch_memory_enabled=False,
                next_step_hypotheses={},
            )

        self.assertEqual(len(root.children), 1)
        self.assertEqual(list(guidance.get("useful_patch_families") or []), [])

    def test_build_search_tree_root_applies_next_step_hypothesis_bonus(self) -> None:
        baseline = default_dense_program("single_g5").normalized()
        schedule_candidate = baseline.normalized()
        schedule_candidate.metadata["program_kind"] = "candidate_schedule"
        schedule_candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-schedule",
            patch_family="change_schedule_family",
            target_scope="schedule",
            changes={"family": "zero_bubble"},
        )
        schedule_candidate.schedule_ir.family = "zero_bubble"
        overlap_candidate = baseline.normalized()
        overlap_candidate.metadata["program_kind"] = "candidate_overlap"
        overlap_candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-overlap",
            patch_family="enable_p2p_overlap",
            target_scope="overlap",
            changes={"enable_p2p_overlap": True},
        )
        schedule_proposal = AgentProposal(
            proposal_id="candidate_schedule",
            scope="schedule",
            program=schedule_candidate,
            priority_rank=10,
        ).normalized()
        overlap_proposal = AgentProposal(
            proposal_id="candidate_overlap",
            scope="overlap",
            program=overlap_candidate,
            priority_rank=10,
        ).normalized()
        root, _ = agent_loop._build_search_tree_root(
            search_state={"bottleneck_signature": "bubble_bound"},
            proposal_pool=[schedule_proposal, overlap_proposal],
            policy_memory_bank=PolicyMemoryBank(),
            search_unit="patch",
            patch_memory_enabled=False,
            next_step_hypotheses={"next_schedule_family": "zero_bubble"},
        )

        self.assertGreater(float(root.children[0].priority), float(root.children[1].priority))
        self.assertTrue(any("hypothesis_" in str(item) for item in (root.children[0].priority_reason or [])))

    def test_build_search_tree_root_applies_target_hit_bonus_for_rewrite_patch(self) -> None:
        baseline = default_dense_program("single_g5").normalized()
        targeted = baseline.normalized()
        targeted.metadata["program_kind"] = "candidate_targeted_reload"
        targeted.metadata["runtime_branch_target_stage_ids"] = [1]
        targeted.metadata["target_layer_groups"] = ["lg_reload_hot"]
        targeted.metadata["target_state_objects"] = ["activation:lg_reload_hot"]
        targeted.applied_patch = ProgramPatchSpec(
            patch_id="patch-targeted-reload",
            patch_family="reload_shift_patch",
            target_scope="memory",
            changes={"direction": "earlier"},
        )
        untargeted = baseline.normalized()
        untargeted.metadata["program_kind"] = "candidate_untargeted_reload"
        untargeted.applied_patch = ProgramPatchSpec(
            patch_id="patch-untargeted-reload",
            patch_family="reload_shift_patch",
            target_scope="memory",
            changes={"direction": "later"},
        )
        targeted_proposal = AgentProposal(
            proposal_id="candidate_targeted_reload",
            scope="memory",
            program=targeted,
            priority_rank=10,
        ).normalized()
        untargeted_proposal = AgentProposal(
            proposal_id="candidate_untargeted_reload",
            scope="memory",
            program=untargeted,
            priority_rank=10,
        ).normalized()

        root, _ = agent_loop._build_search_tree_root(
            search_state={"bottleneck_signature": "reload_bound"},
            proposal_pool=[untargeted_proposal, targeted_proposal],
            policy_memory_bank=PolicyMemoryBank(),
            search_unit="patch",
            patch_memory_enabled=False,
            next_step_hypotheses={
                "next_memory_patch_family": "reload_shift_patch",
                "target_stages": [1],
                "target_layer_groups": ["lg_reload_hot"],
                "target_state_objects": ["activation:lg_reload_hot"],
            },
        )

        best_child = max(root.children, key=lambda item: float(item.priority))
        worst_child = min(root.children, key=lambda item: float(item.priority))

        self.assertEqual(str(best_child.patch_family or ""), "reload_shift_patch")
        self.assertTrue(any("hypothesis_target:stage" in str(item) for item in (best_child.priority_reason or [])))
        self.assertTrue(any("hypothesis_target:layer_group" in str(item) for item in (best_child.priority_reason or [])))
        self.assertTrue(any("hypothesis_target:state_object" in str(item) for item in (best_child.priority_reason or [])))
        self.assertGreater(float(best_child.priority), float(worst_child.priority))

    def test_build_search_tree_root_critical_path_reload_biases_memory_patch(self) -> None:
        baseline = default_dense_program("single_g5").normalized()
        reload_candidate = baseline.normalized()
        reload_candidate.metadata["program_kind"] = "candidate_reload"
        reload_candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-reload",
            patch_family="tune_reload_prefetch",
            target_scope="memory",
            changes={"reload_policy": "prefetch"},
        )
        overlap_candidate = baseline.normalized()
        overlap_candidate.metadata["program_kind"] = "candidate_overlap"
        overlap_candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-overlap",
            patch_family="enable_p2p_overlap",
            target_scope="overlap",
            changes={"enable_p2p_overlap": True},
        )
        reload_proposal = AgentProposal(
            proposal_id="candidate_reload",
            scope="memory",
            program=reload_candidate,
            priority_rank=10,
        ).normalized()
        overlap_proposal = AgentProposal(
            proposal_id="candidate_overlap",
            scope="overlap",
            program=overlap_candidate,
            priority_rank=10,
        ).normalized()
        root, _ = agent_loop._build_search_tree_root(
            search_state={"bottleneck_signature": "memory_bound"},
            proposal_pool=[reload_proposal, overlap_proposal],
            policy_memory_bank=PolicyMemoryBank(),
            search_unit="patch",
            patch_memory_enabled=False,
            next_step_hypotheses={
                "critical_path_breakdown": {
                    "critical_component_type": "reload",
                    "critical_shift_candidates": ["activation_reload_shift_patch"],
                },
                "reload_interference_summary": {"reload_stall_ratio": 0.18},
                "policy_outcome_summary": {"promotion_action": "promote_memory_patch"},
            },
        )

        self.assertGreater(float(root.children[0].priority), float(root.children[1].priority))
        self.assertTrue(any("critical_component:reload" in str(item) or "reload_pressure" in str(item) for item in (root.children[0].priority_reason or [])))

    def test_build_search_tree_root_demotes_overlap_when_policy_outcome_says_so(self) -> None:
        baseline = default_dense_program("single_g5").normalized()
        overlap_candidate = baseline.normalized()
        overlap_candidate.metadata["program_kind"] = "candidate_overlap"
        overlap_candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-overlap",
            patch_family="enable_p2p_overlap",
            target_scope="overlap",
            changes={"enable_p2p_overlap": True},
        )
        schedule_candidate = baseline.normalized()
        schedule_candidate.metadata["program_kind"] = "candidate_schedule"
        schedule_candidate.applied_patch = ProgramPatchSpec(
            patch_id="patch-schedule",
            patch_family="change_schedule_family",
            target_scope="schedule",
            changes={"family": "zero_bubble"},
        )
        schedule_candidate.schedule_ir.family = "zero_bubble"
        overlap_proposal = AgentProposal(
            proposal_id="candidate_overlap",
            scope="overlap",
            program=overlap_candidate,
            priority_rank=10,
        ).normalized()
        schedule_proposal = AgentProposal(
            proposal_id="candidate_schedule",
            scope="schedule",
            program=schedule_candidate,
            priority_rank=10,
        ).normalized()
        root, _ = agent_loop._build_search_tree_root(
            search_state={"bottleneck_signature": "comm_bound"},
            proposal_pool=[overlap_proposal, schedule_proposal],
            policy_memory_bank=PolicyMemoryBank(),
            search_unit="patch",
            patch_memory_enabled=False,
            next_step_hypotheses={
                "critical_path_breakdown": {
                    "critical_component_type": "comm",
                    "critical_shift_candidates": ["comm_chunk_patch"],
                },
                "comm_chunk_exposure_summary": {"comm_exposure_ratio": 0.12},
                "policy_outcome_summary": {"demotion_action": "rollback_overlap"},
                "next_schedule_family": "zero_bubble",
            },
        )

        self.assertGreater(float(root.children[1].priority), float(root.children[0].priority))
        self.assertTrue(any("demote_overlap" in str(item) or "rollback_overlap" in str(item) for item in (root.children[0].priority_reason or [])))

    def test_stage_memory_gate_rejects_hotspot_candidate(self) -> None:
        program = default_dense_program("single_g5")
        program.constraints.memory_budget_gb = 4.0
        program.metadata["seq_len"] = 4096
        for local in program.strategy_ir.local_parallel:
            local.vpp_degree = 4
            local.cp_degree = 1
        program.parallel.vpp_degree = 4
        program.layout.vpp_degree = 4
        report = agent_loop.check_program(program)
        self.assertFalse(report.is_valid)
        self.assertIn("stage_memory_hotspot", report.diagnosis)
        self.assertGreaterEqual(len(report.stage_memory), 1)

    def test_replanner_prefers_pipe_for_bucket_drift(self) -> None:
        program = default_dense_program("single_g5")
        previous_context = {
            "workload_context": {"length_bucket": "short"},
            "runtime_evidence": {"bubble_ratio": 0.03, "cross_node_exposed_ratio": 0.0, "peak_reserved_ratio": 0.60},
        }
        current_context = {
            "workload_context": {"length_bucket": "mid"},
            "runtime_evidence": {"bubble_ratio": 0.12, "cross_node_exposed_ratio": 0.0, "peak_reserved_ratio": 0.62},
            "failure_modes": [{"label": "schedule_coupling"}],
        }
        decision = agent_loop._build_replan_decision(program, current_context, previous_context)
        self.assertEqual(decision["scope"], "pipe")
        self.assertEqual(decision["trigger"], "workload_drift")

    def test_agent_observation_and_trial_artifact_roundtrip(self) -> None:
        program = default_dense_program("single_g5")
        program.applied_patch = ProgramPatchSpec(
            patch_id="patch-memory",
            patch_family="add_offload_policy",
            target_scope="memory",
            changes={"offload_policy": "guarded", "reload_policy": "prefetch"},
        )
        trace_summary = reduce_trial_trace(
            program,
            runtime_summary={
                "bubble_ratio": 0.11,
                "stage_load_variance": 0.04,
                "cross_node_exposed_ratio": 0.03,
                "peak_memory_ratio": 0.72,
                "steady_state_step_time_ms_p50": 1400.0,
                "optimizer_exposed_ms": 320.0,
                "stage_window_summary": {
                    "0": {"compute_ms": 1200.0, "comm_ms": 100.0, "bubble_ms": 90.0, "window_ms": 1390.0},
                    "1": {"compute_ms": 1180.0, "comm_ms": 80.0, "bubble_ms": 70.0, "window_ms": 1330.0},
                },
            },
        )
        observation = build_agent_observation(
            program,
            trace_summary=trace_summary,
            motivation_evidence_manifest=[{"config_name": "evidence_pp_fixed_pipe"}],
        )
        restored = AgentObservation.from_dict(observation.to_dict())
        artifact = build_trial_artifact(
            program,
            restored,
            bottleneck_signature={"dominant_label": "stage_imbalanced"},
            experiment=ExperimentSpec(
                experiment_id="A_problem_existence",
                category="A",
                label="problem_existence",
                objective="study",
                program_kinds=["baseline"],
            ),
            search_unit="patch",
            patch_memory_enabled=True,
        )
        self.assertEqual(restored.motivation_evidence_manifest[0]["config_name"], "evidence_pp_fixed_pipe")
        self.assertEqual(artifact["experiment"]["experiment_id"], "A_problem_existence")
        self.assertEqual(str(artifact.get("patch_family") or ""), "add_offload_policy")
        self.assertEqual(str(artifact.get("patch_category") or ""), "memory")
        self.assertEqual(int(artifact.get("patch_count") or 0), 2)
        self.assertEqual(str(artifact.get("search_unit") or ""), "patch")
        self.assertTrue(bool(artifact.get("patch_memory_enabled")))
        self.assertGreaterEqual(len(artifact["stage_time_distribution"]), 2)
        self.assertIn("bottleneck_breakdown", artifact)
        self.assertIn("search_space_blueprint", artifact)
        self.assertIn("visualization_artifacts", artifact)
        self.assertIn("stage_cost_model", artifact)
        self.assertIn("boundary_semantics", artifact)
        self.assertIn("nonuniform_vpp_shape", artifact)
        self.assertIn("morphable_pipeline_problem", artifact)
        self.assertIn("morphable_pipeline_plan", artifact)
        self.assertIn("critical_operator_clusters", artifact)
        self.assertIn("pipe_search_space", artifact)
        self.assertIn("local_memory_search_space", artifact)
        perfetto = dict((artifact.get("visualization_artifacts") or {}).get("perfetto_trace") or {})
        self.assertEqual(perfetto.get("format"), "perfetto_trace")
        self.assertGreater(len(list(perfetto.get("traceEvents") or [])), 4)
        projection = dict((artifact.get("visualization_artifacts") or {}).get("pipeline_schedule_projection") or {})
        self.assertEqual(projection.get("format"), "pipeline_schedule_projection")
        self.assertGreater(len(list(projection.get("stage_tracks") or [])), 0)
        self.assertIn("local_window_observability", projection)
        self.assertIn("tail_window_ms", dict(projection.get("summary") or {}))
        self.assertTrue(any(item.get("name") == "optimizer_tail_guarded" for item in (projection.get("strategy_hypotheses") or [])))
        event_trace = dict((artifact.get("visualization_artifacts") or {}).get("pipeline_event_trace") or {})
        self.assertEqual(event_trace.get("format"), "pipeline_event_trace")
        self.assertTrue(len(list(event_trace.get("events") or [])) > 0)
        svg = dict((artifact.get("visualization_artifacts") or {}).get("pipeline_projection_svg") or {})
        self.assertEqual(svg.get("format"), "svg_inline")
        self.assertIn("<svg", str(svg.get("content") or ""))
        blueprint = dict(artifact.get("search_space_blueprint") or {})
        self.assertIn("executable_now", blueprint)
        self.assertTrue(any(item.get("name") == "parallel.vpp_degree" for item in (blueprint.get("executable_now") or [])))
        self.assertTrue(any(item.get("name") == "parallel.vpp_vector" for item in (blueprint.get("executable_now") or [])))
        self.assertGreater(len(list(artifact.get("stage_cost_model") or [])), 0)
        self.assertGreater(len(list(artifact.get("boundary_semantics") or [])), 0)
        self.assertEqual(dict(artifact.get("nonuniform_vpp_shape") or {}).get("vector_form"), "v = (v1, v2, ..., vS)")
        self.assertTrue(str(dict(artifact.get("morphable_pipeline_plan") or {}).get("shape_signature") or ""))
        self.assertIn("structure_aware_partition_ir", dict(artifact.get("morphable_pipeline_problem") or {}))
        self.assertIn("selective_vpp_generator", dict(artifact.get("morphable_pipeline_problem") or {}))

    def _write_patch_observation_fixture(self, root: Path) -> Path:
        run_dir = root / "run_patch_fixture"
        analysis_root = run_dir / "trial_analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)

        def _artifact_paths(trial_name: str) -> Dict[str, str]:
            trial_dir = analysis_root / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            compare_path = trial_dir / "compare_pipeline.svg"
            projection_path = trial_dir / "pipeline_projection.svg"
            svg_payload = "<svg xmlns='http://www.w3.org/2000/svg' width='120' height='60'><rect width='120' height='60' fill='#f4f4f4'/><text x='8' y='30'>trace</text></svg>"
            compare_path.write_text(svg_payload, encoding="utf-8")
            projection_path.write_text(svg_payload, encoding="utf-8")
            return {
                "compare_pipeline": str(compare_path),
                "pipeline_projection": str(projection_path),
            }

        baseline_metrics = {
            "trial_id": 0,
            "config_name": "baseline",
            "returncode": 0,
            "oom": False,
            "throughput_tokens_per_s": 100.0,
            "step_time_ms_p50": 1000.0,
            "search_unit": "patch",
            "patch_memory_enabled": True,
            "patch_family": "baseline",
            "patch_category": "schedule",
            "patch_count": 0,
            "bottleneck_signature": {"canonical_dominant_label": "bubble_bound"},
            "trace_summary": {
                "steady_state_step_time_ms_p50": 1000.0,
                "bubble_ratio": 0.14,
                "stage_load_variance": 0.10,
                "mem_skew_ratio": 0.08,
                "stage_tail_ratio": 0.07,
                "optimizer_exposed_ratio": 0.06,
            },
            "trial_artifact": {
                "schedule_template": "fixed_1f1b",
                "patch_family": "baseline",
                "patch_category": "schedule",
                "patch_count": 0,
                "runtime_trace_summary": {"wait_ms": 12.0, "comm_ms": 20.0},
            },
            "analysis_artifact_paths": _artifact_paths("baseline"),
        }
        candidate_bubble = {
            "trial_id": 1,
            "config_name": "candidate_zero_bubble",
            "returncode": 0,
            "oom": False,
            "throughput_tokens_per_s": 118.0,
            "step_time_ms_p50": 920.0,
            "search_unit": "patch",
            "patch_memory_enabled": True,
            "patch_family": "change_schedule_family",
            "patch_category": "schedule",
            "patch_count": 1,
            "bottleneck_signature": {"canonical_dominant_label": "bubble_bound"},
            "trace_summary": {
                "steady_state_step_time_ms_p50": 920.0,
                "bubble_ratio": 0.05,
                "stage_load_variance": 0.04,
                "mem_skew_ratio": 0.07,
                "stage_tail_ratio": 0.03,
                "optimizer_exposed_ratio": 0.04,
            },
            "trial_artifact": {
                "schedule_template": "zero_bubble",
                "patch_family": "change_schedule_family",
                "patch_category": "schedule",
                "patch_count": 1,
                "runtime_trace_summary": {"wait_ms": 6.0, "comm_ms": 15.0},
            },
            "analysis_artifact_paths": _artifact_paths("candidate_zero_bubble"),
        }
        candidate_memory = {
            "trial_id": 2,
            "config_name": "candidate_offload",
            "returncode": 0,
            "oom": False,
            "throughput_tokens_per_s": 109.0,
            "step_time_ms_p50": 970.0,
            "search_unit": "patch",
            "patch_memory_enabled": True,
            "patch_family": "add_offload_policy",
            "patch_category": "memory",
            "patch_count": 2,
            "bottleneck_signature": {"canonical_dominant_label": "memory_bound"},
            "trace_summary": {
                "steady_state_step_time_ms_p50": 970.0,
                "bubble_ratio": 0.10,
                "stage_load_variance": 0.06,
                "mem_skew_ratio": 0.03,
                "stage_tail_ratio": 0.05,
                "optimizer_exposed_ratio": 0.05,
            },
            "trial_artifact": {
                "schedule_template": "interleaved",
                "patch_family": "add_offload_policy",
                "patch_category": "memory",
                "patch_count": 2,
                "runtime_trace_summary": {"wait_ms": 8.0, "comm_ms": 18.0},
            },
            "analysis_artifact_paths": _artifact_paths("candidate_offload"),
        }
        summary = {
            "search_unit": "patch",
            "patch_memory_enabled": True,
            "baseline_family": {"runtime_schedule_family": "fixed_1f1b"},
            "baseline_metrics": baseline_metrics,
            "bottleneck_signature": {"canonical_dominant_label": "bubble_bound"},
            "tested_trials": [baseline_metrics, candidate_bubble, candidate_memory],
        }
        summary_path = run_dir / "summary_megatron.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary_path

    def test_analyze_patch_observations_outputs_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = self._write_patch_observation_fixture(Path(tmpdir))
            out_dir = Path(tmpdir) / "analysis"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_patch_observations.py"),
                    "--runs",
                    str(summary_path.parent),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("patch_observations", result.stdout)
            self.assertTrue((out_dir / "patch_observations.csv").exists())
            self.assertTrue((out_dir / "bottleneck_patch_success.csv").exists())
            self.assertTrue((out_dir / "bottleneck_patch_gain.csv").exists())
            self.assertTrue((out_dir / "search_ablation.csv").exists())
            header = (out_dir / "patch_observations.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("primary_parallel_mode", header)
            self.assertIn("rewrite_family", header)
            self.assertIn("critical_component_type", header)
            self.assertIn("rollback_triggered", header)
            manifest = json.loads((out_dir / "case_study_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(str(manifest.get("format") or ""), "patch_case_study_manifest")
            self.assertGreaterEqual(len(list(manifest.get("cases") or [])), 2)

    def test_plot_patch_paper_figures_outputs_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = self._write_patch_observation_fixture(Path(tmpdir))
            analysis_dir = Path(tmpdir) / "analysis"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_patch_observations.py"),
                    "--runs",
                    str(summary_path.parent),
                    "--out-dir",
                    str(analysis_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            out_dir = Path(tmpdir) / "figures"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "plot_patch_paper_figures.py"),
                    "--analysis-dir",
                    str(analysis_dir),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("fig_patch_sparsity", result.stdout)
            for name in (
                "fig_patch_sparsity.png",
                "fig_patch_count_hist.png",
                "fig_bottleneck_patch_success_heatmap.png",
                "fig_bottleneck_patch_gain_heatmap.png",
                "fig_search_ablation_curve.png",
                "fig_case_study_compare.png",
                "fig_stateful_vs_coarse.png",
                "fig_reload_shift_gain.png",
                "fig_adaptive_chunking_gain.png",
                "fig_local_verticalization_gain.png",
                "fig_budgeted_telemetry_cost.png",
            ):
                self.assertTrue((out_dir / name).exists(), name)

    def test_run_patch_paper_ablation_analysis_only_outputs_manifest_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_root = Path(tmpdir) / "paper_runs"
            for variant in ("patch", "whole_config", "patch_memory_off"):
                self._write_patch_observation_fixture(work_root / variant)
            analysis_dir = work_root / "analysis"
            figures_dir = analysis_dir / "figures"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_patch_paper_ablation.py"),
                    "--work-root",
                    str(work_root),
                    "--analysis-only",
                    "--analysis-dir",
                    str(analysis_dir),
                    "--figures-dir",
                    str(figures_dir),
                    "--enable-hierarchical-orchestrator",
                    "--enable-reload-shift",
                    "--enable-adaptive-chunking",
                    "--enable-local-verticalization",
                    "--telemetry-budget",
                    "summary",
                    "--window-steps",
                    "4",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("manifest:", result.stdout)
            self.assertTrue((work_root / "paper_ablation_manifest.json").exists())
            manifest = json.loads((work_root / "paper_ablation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(str(manifest.get("format") or ""), "patch_paper_ablation_manifest")
            variants = list(manifest.get("variants") or [])
            self.assertEqual(len(variants), 3)
            whole_config = next(item for item in variants if str(item.get("name") or "") == "whole_config")
            patch_memory_off = next(item for item in variants if str(item.get("name") or "") == "patch_memory_off")
            patch_variant = next(item for item in variants if str(item.get("name") or "") == "patch")
            self.assertIn("--search-unit", list(whole_config.get("command") or []))
            self.assertIn("whole_config", list(whole_config.get("command") or []))
            self.assertIn("--disable-patch-memory", list(patch_memory_off.get("command") or []))
            self.assertIn("--enable-hierarchical-orchestrator", list(patch_variant.get("command") or []))
            self.assertIn("--enable-reload-shift", list(patch_variant.get("command") or []))
            self.assertIn("--enable-adaptive-chunking", list(patch_variant.get("command") or []))
            self.assertIn("--enable-local-verticalization", list(patch_variant.get("command") or []))
            self.assertIn("--telemetry-budget", list(patch_variant.get("command") or []))
            self.assertIn("--window-steps", list(patch_variant.get("command") or []))
            self.assertTrue((analysis_dir / "patch_observations.csv").exists())
            self.assertTrue((analysis_dir / "search_ablation.csv").exists())
            self.assertTrue((analysis_dir / "case_study_manifest.json").exists())
            self.assertTrue((figures_dir / "fig_search_ablation_curve.png").exists())
            self.assertTrue((figures_dir / "fig_case_study_compare.png").exists())

    def test_context_record_includes_perfetto_trace_and_search_space_blueprint(self) -> None:
        program = default_dense_program("single_g5")
        context = build_context_record(
            program,
            runtime_summary={
                "bubble_ratio": 0.14,
                "stage_load_variance": 0.05,
                "peak_memory_ratio": 0.81,
                "steady_state_step_time_ms_p50": 1400.0,
                "stage_window_summary": {
                    "0": {"compute_ms": 760.0, "comm_ms": 120.0, "bubble_ms": 110.0, "window_ms": 990.0},
                    "1": {"compute_ms": 700.0, "comm_ms": 100.0, "bubble_ms": 70.0, "window_ms": 870.0},
                },
            },
        )
        evidence = dict(context.get("evidence_record") or {})
        runtime = dict(context.get("runtime_evidence") or {})
        self.assertIn("bottleneck_breakdown", evidence)
        self.assertIn("search_space_blueprint", evidence)
        self.assertIn("visualization_artifacts", evidence)
        self.assertIn("stage_cost_model", evidence)
        self.assertIn("boundary_semantics", evidence)
        self.assertIn("nonuniform_vpp_shape", evidence)
        self.assertIn("pipe_search_space", evidence)
        self.assertIn("local_memory_search_space", evidence)
        self.assertIn("single_node_deep_stats", evidence)
        self.assertIn("critical_operator_clusters", evidence)
        perfetto = dict((evidence.get("visualization_artifacts") or {}).get("perfetto_trace") or {})
        self.assertEqual(perfetto.get("format"), "perfetto_trace")
        projection = dict((evidence.get("visualization_artifacts") or {}).get("pipeline_schedule_projection") or {})
        self.assertEqual(projection.get("format"), "pipeline_schedule_projection")
        self.assertGreater(len(list(projection.get("stage_tracks") or [])), 0)
        event_trace = dict((evidence.get("visualization_artifacts") or {}).get("pipeline_event_trace") or {})
        self.assertEqual(event_trace.get("format"), "pipeline_event_trace")
        self.assertTrue(len(list(event_trace.get("events") or [])) > 0)
        self.assertIn("tail_window_ms", runtime)
        self.assertIn("cooldown_idle_ms", runtime)
        self.assertIn("optimizer_exposed_window_ms", runtime)
        self.assertIn("last_groups_idle_by_stage", runtime)
        svg = dict((evidence.get("visualization_artifacts") or {}).get("pipeline_projection_svg") or {})
        self.assertEqual(svg.get("format"), "svg_inline")
        self.assertIn("<svg", str(svg.get("content") or ""))
        self.assertTrue(any(item.get("label") == "pipeline_idle" for item in (evidence.get("bottleneck_breakdown") or [])))
        self.assertTrue(any(item.get("semantic") in {"normal", "tail-aware", "comm-aware", "memory-aware"} for item in (evidence.get("boundary_semantics") or [])))
        self.assertTrue(any(item.get("stage_id") == 0 for item in (evidence.get("stage_cost_model") or [])))
        self.assertEqual(dict(evidence.get("single_node_deep_stats") or {}).get("mode"), "single_node_8gpu")

    def test_write_analysis_artifacts_persists_visualization_and_blueprint_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dirs = {"trial_dir": tmpdir}
            metrics = {
                "context_record": {
                    "runtime_evidence": {"bubble_ratio": 0.1},
                    "evidence_record": {
                        "bottleneck_breakdown": [{"label": "pipeline_idle", "time_ms": 123.0}],
                        "stage_cost_model": [{"stage_id": 0, "total_cost_ms": 456.0}],
                        "boundary_semantics": [{"boundary_id": "0->1", "semantic": "comm-aware"}],
                        "nonuniform_vpp_shape": {"vector_form": "v = (v1, v2, ..., vS)"},
                        "pipe_search_space": {"variants": [{"name": "fixed_1f1b"}]},
                        "local_memory_search_space": {"per_stage_policy": [{"stage_id": 0}]},
                        "single_node_deep_stats": {"mode": "single_node_8gpu"},
                        "critical_operator_clusters": [{"stage_index": 0, "cluster_role": "memory_hotspot"}],
                        "search_space_blueprint": {"executable_now": [{"name": "parallel.pp_degree"}]},
                        "visualization_artifacts": {
                            "perfetto_trace": {"format": "perfetto_trace", "traceEvents": [{"name": "forward"}]},
                            "pipeline_schedule_projection": {"format": "pipeline_schedule_projection", "stage_tracks": [{"stage_id": 0}]},
                            "pipeline_event_trace": {"format": "pipeline_event_trace", "events": [{"stage_id": 0}]},
                            "pipeline_projection_svg": {"format": "svg_inline", "content": "<svg></svg>"},
                        },
                    },
                },
                "trial_artifact": {
                    "bottleneck_breakdown": [{"label": "pipeline_idle", "time_ms": 123.0}],
                    "stage_cost_model": [{"stage_id": 0, "total_cost_ms": 456.0}],
                    "boundary_semantics": [{"boundary_id": "0->1", "semantic": "comm-aware"}],
                    "nonuniform_vpp_shape": {"vector_form": "v = (v1, v2, ..., vS)"},
                    "pipe_search_space": {"variants": [{"name": "fixed_1f1b"}]},
                    "local_memory_search_space": {"per_stage_policy": [{"stage_id": 0}]},
                    "single_node_deep_stats": {"mode": "single_node_8gpu"},
                    "critical_operator_clusters": [{"stage_index": 0, "cluster_role": "memory_hotspot"}],
                    "search_space_blueprint": {"executable_now": [{"name": "parallel.pp_degree"}]},
                    "visualization_artifacts": {
                        "perfetto_trace": {"format": "perfetto_trace", "traceEvents": [{"name": "forward"}]},
                        "pipeline_schedule_projection": {"format": "pipeline_schedule_projection", "stage_tracks": [{"stage_id": 0}]},
                        "pipeline_event_trace": {"format": "pipeline_event_trace", "events": [{"stage_id": 0}]},
                        "pipeline_projection_svg": {"format": "svg_inline", "content": "<svg></svg>"},
                    },
                },
            }
            paths = trial_runner._write_analysis_artifacts(output_dirs, metrics)
            self.assertTrue(Path(paths["trial_artifact_json"]).exists())
            self.assertTrue(Path(paths["context_record_json"]).exists())
            self.assertTrue(Path(paths["perfetto_trace_json"]).exists())
            self.assertTrue(Path(paths["pipeline_schedule_projection_json"]).exists())
            self.assertTrue(Path(paths["pipeline_event_trace_json"]).exists())
            self.assertTrue(Path(paths["pipeline_projection_svg"]).exists())
            self.assertTrue(Path(paths["search_space_blueprint_json"]).exists())
            self.assertTrue(Path(paths["bottleneck_breakdown_json"]).exists())
            self.assertTrue(Path(paths["critical_operator_clusters_json"]).exists())
            self.assertTrue(Path(paths["failure_diagnosis_json"]).exists())

    def test_write_analysis_artifacts_for_failed_trial_persists_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(run_root=tmpdir)
            metrics = {
                "config_name": "candidate_branch_local_pipe_reorder",
                "returncode": 1,
                "oom": False,
                "error_msg": "Runtime Error: rank 5 failed",
                "root_cause_excerpt": "rank 5 exited first",
                "stdout_tail": "stdout tail",
                "stderr_tail": "stderr tail",
                "bottleneck_signature": {"labels": ["tail_heavy", "optimizer_exposed"]},
                "context_record": {
                    "runtime_evidence": {
                        "bubble_ratio": 0.12,
                        "peak_reserved_ratio": 0.88,
                        "stage_tail_ratio": 0.24,
                        "tail_step_jitter_ratio": 0.19,
                        "optimizer_exposed_ratio": 0.33,
                        "comm_exposure_ratio": 0.09,
                    },
                    "failure_modes": [{"label": "tail_heavy", "severity": "medium"}],
                    "derived_bottlenecks": [{"label": "optimizer_exposed", "severity": "high"}],
                    "optimization_hints": [
                        {
                            "scope": "cooldown",
                            "action": "target_chunk_first",
                            "rationale": "shorten optimizer-sensitive tail window",
                        }
                    ],
                },
                "trial_artifact": {
                    "program_kind": "candidate_branch_local_pipe_reorder",
                    "failure_modes": [{"label": "tail_heavy", "severity": "medium"}],
                    "derived_bottlenecks": [{"label": "optimizer_exposed", "severity": "high"}],
                    "optimization_hints": [
                        {
                            "scope": "cooldown",
                            "action": "target_chunk_first",
                            "rationale": "shorten optimizer-sensitive tail window",
                        }
                    ],
                    "visualization_artifacts": {},
                },
                "trial_reflection": {
                    "family": "dual_overlap_optimizer_hide",
                    "config_name": "candidate_branch_local_pipe_reorder",
                    "failure_sources": ["trial_failed"],
                    "recommended_next_action": "retain_family_adjust_local_policy",
                    "summary": "dual_overlap_optimizer_hide failed due to trial_failed",
                },
                "trial_outcome": {
                    "config_name": "candidate_branch_local_pipe_reorder",
                    "success": False,
                    "launch_failure": True,
                    "oom": False,
                },
            }
            paths = trial_runner.write_analysis_artifacts_for_trial(args, 7, metrics)
            artifact_dir = Path(tmpdir) / "trial_007" / "analysis"
            self.assertEqual(Path(paths["failure_diagnosis_json"]), artifact_dir / "failure_diagnosis.json")
            self.assertTrue((artifact_dir / "trial_artifact.json").exists())
            self.assertTrue((artifact_dir / "context_record.json").exists())
            self.assertTrue((artifact_dir / "trial_reflection.json").exists())
            self.assertTrue((artifact_dir / "trial_outcome.json").exists())
            diagnosis = json.loads((artifact_dir / "failure_diagnosis.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnosis["returncode"], 1)
            self.assertEqual(diagnosis["error_msg"], "Runtime Error: rank 5 failed")
            self.assertEqual(diagnosis["trial_reflection"]["recommended_next_action"], "retain_family_adjust_local_policy")
            self.assertTrue(diagnosis["recommended_actions"])

    def test_pipeline_schedule_projection_highlights_runtime_strategy_options(self) -> None:
        program = default_dense_program("single_g5")
        context = build_context_record(
            program,
            runtime_summary={
                "bubble_ratio": 0.17,
                "stage_load_variance": 0.05,
                "peak_memory_ratio": 0.86,
                "steady_state_step_time_ms_p50": 1000.0,
                "optimizer_exposed_ms": 320.0,
                "steady_state_step_time_ms_p95": 1280.0,
                "stage_window_summary": {
                    "0": {"compute_ms": 860.0, "comm_ms": 90.0, "bubble_ms": 70.0, "window_ms": 1020.0, "peak_reserved_gib": 22.0},
                    "1": {"compute_ms": 720.0, "comm_ms": 80.0, "bubble_ms": 160.0, "window_ms": 960.0, "peak_reserved_gib": 28.0},
                },
            },
        )
        projection = dict((((context.get("evidence_record") or {}).get("visualization_artifacts") or {}).get("pipeline_schedule_projection") or {}))
        self.assertEqual(projection.get("format"), "pipeline_schedule_projection")
        self.assertTrue(
            any(
                {"memory_hotspot", "optimizer_sensitive"} & set(track.get("stage_tags") or [])
                or str(track.get("role") or "") in {"tail_hotspot", "optimizer_sensitive_tail", "memory_hotspot"}
                for track in (projection.get("stage_tracks") or [])
            )
        )
        hypothesis_names = {str(item.get("name")) for item in (projection.get("strategy_hypotheses") or [])}
        self.assertIn("optimizer_tail_guarded", hypothesis_names)
        self.assertIn("checkpoint_boundary_joint", hypothesis_names)
        event_trace = dict((((context.get("evidence_record") or {}).get("visualization_artifacts") or {}).get("pipeline_event_trace") or {}))
        self.assertEqual(event_trace.get("format"), "pipeline_event_trace")
        self.assertTrue(any(str(item.get("op_kind") or "") == "fwd" for item in (event_trace.get("events") or [])))
        self.assertTrue(any(str(item.get("op_kind") or "") == "bwd" for item in (event_trace.get("events") or [])))

    def test_pipeline_schedule_projection_falls_back_when_stage_windows_are_sparse(self) -> None:
        program = default_dense_program("single_g5")
        program.parallel.pp_degree = 2
        context = build_context_record(
            program,
            runtime_summary={
                "steady_state_step_time_ms_p50": 6838.75,
                "optimizer_exposed_ms": 5013.67,
                "peak_reserved_ratio": 0.8926,
                "bubble_ratio": 0.0,
                "stage_window_summary": {
                    "0": {"window_ms": 0.0, "compute_ms": 0.0, "comm_ms": 0.0, "bubble_ms": 0.0},
                    "1": {"window_ms": 940.3281, "compute_ms": 0.0, "comm_ms": 0.0, "bubble_ms": 0.0},
                },
            },
        )
        projection = dict((((context.get("evidence_record") or {}).get("visualization_artifacts") or {}).get("pipeline_schedule_projection") or {}))
        self.assertEqual(projection.get("projection_mode"), "fallback_estimated")
        self.assertEqual(dict(projection.get("summary") or {}).get("evidence_source"), "fallback_estimated")
        tracks = list(projection.get("stage_tracks") or [])
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(list(track.get("segments") or []) for track in tracks))
        self.assertTrue(all(str(track.get("evidence_source") or "") == "fallback_estimated" for track in tracks))
        event_trace = dict((((context.get("evidence_record") or {}).get("visualization_artifacts") or {}).get("pipeline_event_trace") or {}))
        self.assertEqual(event_trace.get("format"), "pipeline_event_trace")
        self.assertTrue(any(str(item.get("op_kind") or "") == "fwd" for item in (event_trace.get("events") or [])))
        self.assertTrue(any(str(item.get("op_kind") or "") == "bwd" for item in (event_trace.get("events") or [])))
        phase_windows = list(projection.get("phase_windows") or [])
        self.assertTrue(phase_windows)
        previous_end = 0.0
        for phase in phase_windows:
            self.assertGreaterEqual(float(phase.get("start_ms") or 0.0), previous_end)
            self.assertGreaterEqual(float(phase.get("end_ms") or 0.0), float(phase.get("start_ms") or 0.0))
            previous_end = float(phase.get("end_ms") or 0.0)

    def test_verify_program_returns_structured_verifier_report(self) -> None:
        program = default_dense_program("single_g5")
        program.constraints.memory_budget_gb = 4.0
        program.metadata["seq_len"] = 4096
        observation = build_agent_observation(
            program,
            runtime_summary={"peak_memory_ratio": 0.96, "bubble_ratio": 0.07},
        )
        report = verify_program(program, observation=observation)
        restored = VerifierReport.from_dict(report.to_dict())
        self.assertFalse(restored.is_legal)
        self.assertIsNotNone(restored.rejection_reason)
        self.assertIn(restored.next_scope_hint, {"local", "skeleton", "pipe"})

    def test_synthesize_proposals_embeds_verifier_report(self) -> None:
        baseline = default_dense_program("single_g5")
        rewrite = agent_loop._rewrite_space(baseline, {"bubble_ratio": 0.14, "stage_load_variance": 0.05})
        context = build_context_record(
            baseline,
            runtime_summary={
                "bubble_ratio": 0.14,
                "stage_load_variance": 0.05,
                "peak_memory_ratio": 0.70,
                "stage_window_summary": {
                    "0": {"window_ms": 1500.0},
                    "1": {"window_ms": 1100.0},
                },
            },
        )
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary={"bubble_ratio": 0.14, "stage_load_variance": 0.05},
            context_record=context,
            replan_decision=ReplanDecision(trigger="bubble_spike", scope="pipe").to_dict(),
            candidate_limit=4,
        )
        self.assertEqual(rejected, [])
        self.assertGreaterEqual(len(proposals), 1)
        self.assertTrue(all(proposal.verifier_report is not None for proposal in proposals))
        self.assertTrue(any(proposal.scope == "pipe" for proposal in proposals))

    def test_context_record_exposes_tail_comm_and_memory_skew(self) -> None:
        program = default_dense_program("single_g5")
        context = build_context_record(
            program,
            runtime_summary={
                "bubble_ratio": 0.15,
                "stage_load_variance": 0.06,
                "cross_node_exposed_ratio": 0.10,
                "peak_memory_ratio": 0.82,
                "steady_state_step_time_ms_p50": 1000.0,
                "steady_state_step_time_ms_p95": 1350.0,
                "stage_window_summary": {
                    "0": {"compute_ms": 900.0, "comm_ms": 220.0, "bubble_ms": 110.0, "window_ms": 1230.0, "peak_reserved_gib": 23.0, "peak_active_gib": 19.0},
                    "1": {"compute_ms": 650.0, "comm_ms": 60.0, "bubble_ms": 35.0, "window_ms": 745.0, "peak_reserved_gib": 14.0, "peak_active_gib": 11.0},
                },
            },
        )
        runtime = context["runtime_evidence"]
        labels = {item["label"] for item in context["derived_bottlenecks"]}
        self.assertGreater(runtime["stage_tail_ratio"], 0.1)
        self.assertGreater(runtime["comm_exposure_ratio"], 0.1)
        self.assertGreater(runtime["mem_skew_ratio"], 0.1)
        self.assertIn("tail_heavy", labels)
        self.assertIn("comm_exposed", labels)
        self.assertIn("memory_skew", labels)

    def test_recommend_methods_prefers_megatron_pp_vpp_actions(self) -> None:
        program = default_dense_program("single_g5")
        context = build_context_record(
            program,
            runtime_summary={
                "bubble_ratio": 0.16,
                "stage_load_variance": 0.05,
                "cross_node_exposed_ratio": 0.09,
                "peak_memory_ratio": 0.78,
                "stage_window_summary": {
                    "0": {"compute_ms": 980.0, "comm_ms": 180.0, "bubble_ms": 120.0, "window_ms": 1280.0, "peak_reserved_gib": 21.0, "peak_active_gib": 17.0},
                    "1": {"compute_ms": 760.0, "comm_ms": 70.0, "bubble_ms": 45.0, "window_ms": 875.0, "peak_reserved_gib": 15.0, "peak_active_gib": 12.0},
                },
            },
        )
        signature = classify_bottleneck(program, context)
        hints = recommend_optimization_methods(program, context, signature)
        methods = {item["method"] for item in hints}
        self.assertIn("tail_aware_stage_partition", methods)
        self.assertIn("bubble_driven_partition_placement_vpp_tuning", methods)
        self.assertIn("communication_exposure_aware_vpp_chunking", methods)

    def test_torchtitan_backend_gets_schedule_sandbox_hint(self) -> None:
        program = default_dense_program("single_g5")
        program.metadata["execution_backend"] = "torchtitan"
        context = build_context_record(
            program,
            runtime_summary={
                "bubble_ratio": 0.18,
                "stage_load_variance": 0.02,
                "peak_memory_ratio": 0.70,
                "stage_window_summary": {
                    "0": {"compute_ms": 820.0, "comm_ms": 90.0, "bubble_ms": 150.0, "window_ms": 1060.0},
                    "1": {"compute_ms": 800.0, "comm_ms": 85.0, "bubble_ms": 140.0, "window_ms": 1025.0},
                },
            },
        )
        self.assertEqual(context["backend_context"]["backend_family"], "torchtitan")
        methods = {item["method"] for item in context["optimization_hints"]}
        self.assertIn("torchtitan_schedule_sandbox", methods)
        self.assertIn("torchtitan_zero_bubble_schedule_probe", methods)

    def test_torchtitan_rewrite_space_enables_hybrid_shard_and_schedule_sandbox(self) -> None:
        program = default_dense_program("dual_g5_g5")
        program.metadata["execution_backend"] = "torchtitan"
        runtime_summary = {
            "bubble_ratio": 0.18,
            "stage_load_variance": 0.07,
            "cross_node_exposed_ratio": 0.04,
            "peak_memory_ratio": 0.92,
            "steady_state_step_time_ms_p50": 1000.0,
            "stage_window_summary": {
                "0": {"compute_ms": 900.0, "comm_ms": 120.0, "bubble_ms": 180.0, "window_ms": 1200.0, "peak_reserved_gib": 28.0},
                "1": {"compute_ms": 880.0, "comm_ms": 115.0, "bubble_ms": 165.0, "window_ms": 1160.0, "peak_reserved_gib": 27.0},
                "2": {"compute_ms": 760.0, "comm_ms": 95.0, "bubble_ms": 125.0, "window_ms": 980.0, "peak_reserved_gib": 21.0},
                "3": {"compute_ms": 740.0, "comm_ms": 90.0, "bubble_ms": 130.0, "window_ms": 960.0, "peak_reserved_gib": 20.0},
            },
        }
        rewrite = agent_loop._rewrite_space(program, runtime_summary)
        self.assertTrue(rewrite.allow_hybrid_shard)
        self.assertTrue(rewrite.allow_torchtitan_schedule_sandbox)
        self.assertIn("torchtitan_zero_bubble", rewrite.allowed_schedule_templates)
        self.assertIn("torchtitan_dualpipev", rewrite.allowed_schedule_templates)

    def test_torchtitan_candidates_include_hybrid_shard_and_zero_bubble(self) -> None:
        program = default_dense_program("dual_g5_g5")
        program.metadata["execution_backend"] = "torchtitan"
        runtime_summary = {
            "bubble_ratio": 0.18,
            "stage_load_variance": 0.07,
            "cross_node_exposed_ratio": 0.04,
            "peak_memory_ratio": 0.92,
            "steady_state_step_time_ms_p50": 1000.0,
            "stage_window_summary": {
                "0": {"compute_ms": 900.0, "comm_ms": 120.0, "bubble_ms": 180.0, "window_ms": 1200.0, "peak_reserved_gib": 28.0},
                "1": {"compute_ms": 880.0, "comm_ms": 115.0, "bubble_ms": 165.0, "window_ms": 1160.0, "peak_reserved_gib": 27.0},
                "2": {"compute_ms": 760.0, "comm_ms": 95.0, "bubble_ms": 125.0, "window_ms": 980.0, "peak_reserved_gib": 21.0},
                "3": {"compute_ms": 740.0, "comm_ms": 90.0, "bubble_ms": 130.0, "window_ms": 960.0, "peak_reserved_gib": 20.0},
            },
        }
        rewrite = agent_loop._rewrite_space(program, runtime_summary)
        context = build_context_record(program, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            program,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="bubble_spike", scope="pipe").to_dict(),
            candidate_limit=8,
        )
        kinds = {str(proposal.program.metadata.get("program_kind")) for proposal in proposals}
        self.assertIn("candidate_pp_hsdp_hybrid", kinds)
        self.assertIn("candidate_torchtitan_zero_bubble_schedule", kinds)
        hybrid = next(proposal.program for proposal in proposals if str(proposal.program.metadata.get("program_kind")) == "candidate_pp_hsdp_hybrid")
        self.assertGreater(float(hybrid.metadata.get("hybrid_shard_score") or 0.0), 0.0)
        self.assertTrue(any("max_vpp_size=1" in str(item.get("reason") or "") for item in rejected))

    def test_compile_program_exports_torchtitan_shard_and_schedule_env(self) -> None:
        program = default_dense_program("dual_g5_g5")
        program.metadata["execution_backend"] = "torchtitan"
        context = build_context_record(
            program,
            runtime_summary={
                "bubble_ratio": 0.18,
                "stage_load_variance": 0.05,
                "cross_node_exposed_ratio": 0.03,
                "peak_memory_ratio": 0.90,
                "stage_window_summary": {
                    "0": {"window_ms": 1200.0, "peak_reserved_gib": 28.0},
                    "1": {"window_ms": 980.0, "peak_reserved_gib": 20.0},
                    "2": {"window_ms": 970.0, "peak_reserved_gib": 19.0},
                    "3": {"window_ms": 960.0, "peak_reserved_gib": 19.0},
                },
            },
        )
        candidate = agent_loop._build_pp_hsdp_hybrid_candidate(program, {"bubble_ratio": 0.18}, context)
        self.assertIsNotNone(candidate)
        compiled = compile_program(candidate)
        self.assertEqual(compiled.launcher_env["EXECUTION_BACKEND"], "torchtitan")
        self.assertIn("LOCAL_SHARD_POLICY", compiled.launcher_env)
        schedule_candidate = agent_loop._build_torchtitan_zero_bubble_schedule_candidate(program, {"bubble_ratio": 0.18}, context)
        self.assertIsNotNone(schedule_candidate)
        compiled_schedule = compile_program(schedule_candidate)
        self.assertEqual(compiled_schedule.launcher_env["SCHEDULE_TEMPLATE"], "torchtitan_zero_bubble")
        self.assertEqual(compiled_schedule.launcher_env["SCHEDULE_WARMUP_POLICY"], "max_forward_fill")
        self.assertEqual(compiled_schedule.launcher_env["SCHEDULE_COOLDOWN_POLICY"], "drain_with_w")

    def test_tail_aware_partition_scoring_prefers_runtime_guided_rebalance(self) -> None:
        baseline = default_dense_program("single_g5")
        runtime_summary = {
            "bubble_ratio": 0.06,
            "stage_load_variance": 0.05,
            "cross_node_exposed_ratio": 0.02,
            "peak_memory_ratio": 0.76,
            "steady_state_step_time_ms_p50": 1100.0,
            "stage_window_summary": {
                "0": {"compute_ms": 1080.0, "comm_ms": 80.0, "bubble_ms": 35.0, "window_ms": 1195.0, "peak_reserved_gib": 23.0},
                "1": {"compute_ms": 760.0, "comm_ms": 55.0, "bubble_ms": 20.0, "window_ms": 835.0, "peak_reserved_gib": 15.0},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = build_context_record(baseline, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="tail_drift", scope="skeleton").to_dict(),
            candidate_limit=6,
        )
        self.assertEqual(rejected, [])
        scores = {
            str(proposal.program.metadata.get("program_kind")): float(proposal.program.metadata.get("tail_partition_score") or 0.0)
            for proposal in proposals
            if str(proposal.program.metadata.get("program_kind")) in {"candidate_nonuniform_partition", "candidate_runtime_guided_partition"}
        }
        self.assertIn("candidate_nonuniform_partition", scores)
        self.assertIn("candidate_runtime_guided_partition", scores)
        self.assertGreater(scores["candidate_runtime_guided_partition"], scores["candidate_nonuniform_partition"])
        self.assertEqual(str(proposals[0].program.metadata.get("program_kind")), "candidate_runtime_guided_partition")

    @mock.patch(
        "megatron_agent.agent_loop._call_llm_chat_completion",
        return_value=json.dumps(
            {
                "selected_proposal_ids": ["candidate_runtime_guided_partition"],
                "rationales": {"candidate_runtime_guided_partition": "tail-heavy partition should be fixed before broader search"},
                "agent_topology": {"llm_planner_agents": 1, "verifier_agents": 1, "executor_agents": 1},
                "notes": ["prefer throughput under memory budget"],
            },
            ensure_ascii=False,
        ),
    )
    def test_llm_supervisor_reorders_and_tags_megatron_proposals(self, _mock_llm: mock.Mock) -> None:
        baseline = default_dense_program("single_g5")
        runtime_summary = {
            "bubble_ratio": 0.06,
            "stage_load_variance": 0.05,
            "cross_node_exposed_ratio": 0.02,
            "peak_memory_ratio": 0.76,
            "steady_state_step_time_ms_p50": 1100.0,
            "stage_window_summary": {
                "0": {"compute_ms": 1080.0, "comm_ms": 80.0, "bubble_ms": 35.0, "window_ms": 1195.0, "peak_reserved_gib": 23.0},
                "1": {"compute_ms": 760.0, "comm_ms": 55.0, "bubble_ms": 20.0, "window_ms": 835.0, "peak_reserved_gib": 15.0},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = build_context_record(baseline, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="tail_drift", scope="skeleton").to_dict(),
            candidate_limit=6,
            llm_config={
                "enabled": True,
                "endpoint": "http://10.100.1.93:12365/v1/chat/completions",
                "model": "/models/Qwen2.5-72B-Instruct",
                "temperature": 0.2,
                "log_llm": False,
            },
        )
        self.assertEqual(rejected, [])
        self.assertEqual(str(proposals[0].program.metadata.get("program_kind")), "candidate_runtime_guided_partition")
        self.assertEqual(str(proposals[0].source), "llm_supervisor")
        self.assertEqual(str(proposals[0].program.metadata.get("planner_backend")), "llm_http")
        self.assertEqual(str(proposals[0].program.metadata.get("planner_model")), "/models/Qwen2.5-72B-Instruct")
        topology = dict(proposals[0].program.metadata.get("agent_topology") or {})
        self.assertEqual(int(topology.get("llm_planner_agents") or 0), 1)
        self.assertEqual(int(topology.get("verifier_agents") or 0), 1)
        self.assertEqual(int(topology.get("executor_agents") or 0), 1)

    def test_external_inputs_are_merged_into_context_and_digested(self) -> None:
        baseline = default_dense_program("single_g5")
        context = build_context_record(baseline, runtime_summary={"bubble_ratio": 0.04, "peak_memory_ratio": 0.72})
        merged = agent_loop._augment_context_with_external_inputs(
            context,
            {
                "model_structure_summary": {
                    "num_layers": 40,
                    "layers": [{"type": "attention"}, {"type": "mlp"}, {"type": "embedding"}, {"type": "lm_head"}],
                },
                "hardware_topology_summary": {
                    "node_count": 1,
                    "gpu_count": 8,
                    "links": [{"type": "nvlink", "bandwidth_gbps": 900}, {"type": "pcie", "bandwidth_gbps": 64}],
                },
                "profile_summary": {
                    "layers": [{"forward_ms": 1.2, "backward_ms": 2.4, "activation_mb": 110.0, "communication_ms": 0.3}],
                    "peak_memory_gib": 28.4,
                },
                "baseline_catalog": {
                    "baselines": [
                        {"name": "vpp_neighbor_g8", "pp": 4, "vpp": 2, "family": "pp_vpp", "step_time_ms": 8040.0, "throughput": 43.4}
                    ]
                },
            },
        )
        self.assertIn("external_structure_summary", merged["model_context"])
        self.assertEqual(int((merged["model_context"]["structure_digest"] or {}).get("total_layers") or 0), 40)
        self.assertEqual(str((merged["hardware_context"]["topology_digest"] or {}).get("dominant_fabric") or ""), "nvlink")
        self.assertGreater(float((merged["evidence_record"]["profile_digest"] or {}).get("peak_activation_mb") or 0.0), 0.0)
        self.assertEqual(
            str((((merged["evidence_record"]["baseline_catalog_digest"] or {}).get("best_baseline") or {}).get("name")) or ""),
            "vpp_neighbor_g8",
        )

    def test_verified_template_library_and_selector_prefer_long_context_template(self) -> None:
        baseline = default_dense_program("single_g5")
        baseline.metadata["seq_len"] = 4096
        runtime_summary = {
            "bubble_ratio": 0.12,
            "pipeline_wait_ratio": 0.13,
            "optimizer_exposed_ratio": 0.10,
            "peak_memory_ratio": 0.91,
            "steady_state_step_time_ms_p50": 1600.0,
            "stage_window_summary": {
                "0": {"compute_ms": 1180.0, "comm_ms": 180.0, "bubble_ms": 120.0, "window_ms": 1480.0, "peak_reserved_gib": 28.0},
                "1": {"compute_ms": 980.0, "comm_ms": 110.0, "bubble_ms": 95.0, "window_ms": 1185.0, "peak_reserved_gib": 24.0},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = build_context_record(baseline, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="workload_drift", scope="local_parallel").to_dict(),
            candidate_limit=8,
        )
        self.assertTrue(len(proposals) > 0)
        self.assertTrue(any("estimated memory pressure" in str(item.get("reason") or "") for item in rejected))
        library = agent_loop._build_verified_strategy_template_library(baseline, proposals, context)
        template_ids = {str(item.get("template_id") or "") for item in library}
        self.assertIn("D_long_context_conservative", template_ids)
        decision = agent_loop._select_verified_strategy_template(library, context)
        self.assertEqual(str(decision.get("batch_profile") or ""), "long_context")
        self.assertEqual(str(decision.get("selector_mode") or ""), "verified_template_switch_only")
        self.assertEqual(str(decision.get("selected_template_id") or ""), "D_long_context_conservative")

    def test_runtime_guided_partition_adds_position_aware_metadata(self) -> None:
        baseline = default_dense_program("single_g5")
        runtime_summary = {
            "bubble_ratio": 0.09,
            "pipeline_wait_ratio": 0.18,
            "optimizer_exposed_ratio": 0.24,
            "stage_skew": 1.22,
            "stage_window_summary": {
                "0": {"compute_ms": 1180.0, "comm_ms": 260.0, "bubble_ms": 120.0, "window_ms": 1560.0, "peak_reserved_gib": 24.5},
                "1": {"compute_ms": 720.0, "comm_ms": 70.0, "bubble_ms": 22.0, "window_ms": 812.0, "peak_reserved_gib": 18.0},
            },
        }
        candidate = agent_loop._build_runtime_guided_partition(baseline, runtime_summary)
        self.assertIsNotNone(candidate)
        self.assertEqual(str(candidate.metadata.get("program_kind")), "candidate_runtime_guided_partition")
        self.assertEqual(str(candidate.metadata.get("runtime_partition_focus")), "tail-aware")
        self.assertGreaterEqual(int(candidate.metadata.get("runtime_partition_shift") or 0), 1)
        self.assertLess(int(candidate.partition.stages[0].decoder_layers), int(baseline.partition.stages[0].decoder_layers))
        self.assertGreater(int(candidate.partition.stages[1].decoder_layers), int(baseline.partition.stages[1].decoder_layers))
        self.assertIn("runtime_partition_stage_objectives", candidate.metadata)

    def test_verify_program_rejects_vpp_when_comm_pressure_outweighs_bubble_relief(self) -> None:
        baseline = default_dense_program("single_g5")
        candidate = agent_loop._build_stage_aware_schedule(baseline)
        self.assertIsNotNone(candidate)
        report = verify_program(
            candidate,
            observation={
                "runtime_evidence": {
                    "bubble_ratio": 0.04,
                    "stage_tail_ratio": 0.03,
                    "comm_exposure_ratio": 0.22,
                    "cross_node_exposed_ratio": 0.10,
                    "grouped_interleave_overhead": 0.08,
                }
            },
        )
        self.assertFalse(report.is_legal)
        self.assertIn("comm-exposure-aware VPP veto", str(report.rejection_reason))
        self.assertIn("comm_exposure_vpp_veto", list(report.diagnosis or []))
        self.assertEqual(report.next_scope_hint, "local")

    def test_synthesize_proposals_rejects_vpp_schedule_when_comm_exposed(self) -> None:
        baseline = default_dense_program("single_g5")
        runtime_summary = {
            "bubble_ratio": 0.05,
            "stage_load_variance": 0.04,
            "cross_node_exposed_ratio": 0.10,
            "peak_memory_ratio": 0.72,
            "steady_state_step_time_ms_p50": 1000.0,
            "stage_window_summary": {
                "0": {"compute_ms": 710.0, "comm_ms": 250.0, "bubble_ms": 30.0, "window_ms": 990.0},
                "1": {"compute_ms": 690.0, "comm_ms": 235.0, "bubble_ms": 25.0, "window_ms": 950.0},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = build_context_record(baseline, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="comm_exposure", scope="pipe").to_dict(),
            candidate_limit=6,
        )
        accepted_kinds = {str(proposal.program.metadata.get("program_kind")) for proposal in proposals}
        self.assertNotIn("candidate_stage_aware_schedule", accepted_kinds)
        self.assertNotIn("candidate_runtime_guided_schedule", accepted_kinds)
        self.assertTrue(
            any(
                "comm-exposure-aware VPP veto" in str(((item.get("reason") or {}).get("rejection_reason") or ""))
                for item in rejected
            )
        )

    def test_synthesize_proposals_emits_nonuniform_vpp_shape_candidate_and_stage_cost_score(self) -> None:
        baseline = default_dense_program("single_g5")
        runtime_summary = {
            "bubble_ratio": 0.14,
            "stage_load_variance": 0.06,
            "cross_node_exposed_ratio": 0.02,
            "peak_memory_ratio": 0.74,
            "steady_state_step_time_ms_p50": 980.0,
            "stage_window_summary": {
                "0": {"compute_ms": 760.0, "comm_ms": 70.0, "bubble_ms": 120.0, "window_ms": 950.0, "peak_reserved_gib": 21.0, "peak_active_gib": 17.0},
                "1": {"compute_ms": 620.0, "comm_ms": 55.0, "bubble_ms": 55.0, "window_ms": 730.0, "peak_reserved_gib": 15.0, "peak_active_gib": 12.0},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = build_context_record(baseline, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="bubble_spike", scope="local_parallel").to_dict(),
            candidate_limit=12,
        )
        self.assertTrue(len(rejected) >= 0)
        kinds = {str(proposal.program.metadata.get("program_kind")) for proposal in proposals}
        self.assertIn("candidate_nonuniform_vpp_shape", kinds)
        nonuniform = next(
            proposal.program
            for proposal in proposals
            if str(proposal.program.metadata.get("program_kind")) == "candidate_nonuniform_vpp_shape"
        )
        self.assertTrue(bool(nonuniform.metadata.get("preserve_stage_local_vpp")))
        self.assertGreater(len(list(nonuniform.metadata.get("stage_local_vpp_vector") or [])), 0)
        self.assertTrue(any("stage_cost_score" in proposal.program.metadata for proposal in proposals))

    def test_synthesize_proposals_emits_boundary_semantic_and_pipe_variant_candidates(self) -> None:
        baseline = default_dense_program("single_g5")
        runtime_summary = {
            "bubble_ratio": 0.17,
            "stage_load_variance": 0.05,
            "cross_node_exposed_ratio": 0.03,
            "peak_memory_ratio": 0.79,
            "steady_state_step_time_ms_p50": 1100.0,
            "stage_window_summary": {
                "0": {"compute_ms": 780.0, "comm_ms": 150.0, "bubble_ms": 130.0, "window_ms": 1060.0, "peak_reserved_gib": 22.0, "peak_active_gib": 18.0},
                "1": {"compute_ms": 700.0, "comm_ms": 140.0, "bubble_ms": 90.0, "window_ms": 930.0, "peak_reserved_gib": 18.0, "peak_active_gib": 14.0},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = build_context_record(baseline, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="bubble_spike", scope="pipe").to_dict(),
            candidate_limit=10,
        )
        proposal_kinds = {str(proposal.program.metadata.get("program_kind")) for proposal in proposals}
        rejected_kinds = {
            str((((item.get("proposal") or {}).get("program") or {}).get("metadata") or {}).get("program_kind"))
            for item in rejected
        }
        all_kinds = proposal_kinds | rejected_kinds
        self.assertTrue(any(kind.startswith("candidate_boundary_semantic_") for kind in all_kinds))
        pipe_variants = agent_loop._build_pipe_search_space_candidates(
            baseline,
            runtime_summary,
            context,
        )
        self.assertTrue(
            any(
                str(candidate.metadata.get("program_kind") or "").startswith("candidate_pipe_variant_")
                for candidate in pipe_variants
            )
        )

    def test_synthesize_proposals_emits_stage_local_memory_policy_candidate(self) -> None:
        baseline = default_dense_program("single_g5")
        runtime_summary = {
            "bubble_ratio": 0.09,
            "stage_load_variance": 0.04,
            "cross_node_exposed_ratio": 0.02,
            "peak_memory_ratio": 0.91,
            "steady_state_step_time_ms_p50": 1020.0,
            "stage_window_summary": {
                "0": {"compute_ms": 820.0, "comm_ms": 90.0, "bubble_ms": 70.0, "window_ms": 980.0, "peak_reserved_gib": 29.0, "peak_active_gib": 25.0},
                "1": {"compute_ms": 700.0, "comm_ms": 75.0, "bubble_ms": 45.0, "window_ms": 820.0, "peak_reserved_gib": 20.0, "peak_active_gib": 16.0},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = build_context_record(baseline, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="memory_pressure", scope="local_parallel").to_dict(),
            candidate_limit=16,
        )
        proposal_kinds = {str(proposal.program.metadata.get("program_kind")) for proposal in proposals}
        rejected_kinds = {
            str((((item.get("proposal") or {}).get("program") or {}).get("metadata") or {}).get("program_kind"))
            for item in rejected
        }
        all_kinds = proposal_kinds | rejected_kinds
        direct = agent_loop._build_stage_local_memory_policy_candidate(baseline, context)
        self.assertIsNotNone(direct)
        self.assertGreater(len(list((direct.metadata if direct is not None else {}).get("stage_local_memory_policy") or [])), 0)
        self.assertTrue(
            "candidate_stage_local_memory_policy" in all_kinds
            or "candidate_boundary_semantic_memory" in all_kinds
            or "candidate_local_fsdp_scope" in all_kinds
        )
        if "candidate_stage_local_memory_policy" in proposal_kinds:
            candidate = next(
                proposal.program
                for proposal in proposals
                if str(proposal.program.metadata.get("program_kind")) == "candidate_stage_local_memory_policy"
            )
            self.assertGreater(len(list(candidate.metadata.get("stage_local_memory_policy") or [])), 0)

    def test_feedback_search_plan_prioritizes_optimizer_family_and_tags_proposals(self) -> None:
        baseline = default_dense_program("single_g5")
        runtime_summary = {
            "bubble_ratio": 0.07,
            "optimizer_exposed_ratio": 0.26,
            "optimizer_ratio": 0.60,
            "peak_reserved_ratio": 0.85,
            "stage_tail_ratio": 0.14,
            "tail_step_jitter_ratio": 0.13,
            "stage_window_summary": {
                "0": {"window_ms": 980.0, "peak_reserved_gib": 21.0},
                "1": {"window_ms": 1120.0, "peak_reserved_gib": 27.5},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = {
            "runtime_evidence": dict(runtime_summary),
            "failure_modes": [{"label": "tail_heavy"}],
            "derived_bottlenecks": [{"label": "tail_heavy"}],
        }
        proposals, _ = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="steady", scope="pipe").to_dict(),
            candidate_limit=10,
            policy_memory_bank=PolicyMemoryBank(),
        )
        plan = build_feedback_search_plan(
            baseline,
            context,
            replan_decision=ReplanDecision(trigger="steady", scope="pipe").to_dict(),
            proposals=proposals,
            memory_bank=PolicyMemoryBank(),
        )
        self.assertEqual(str((plan.get("selected_families") or [""])[0]), "dual_overlap_optimizer_hide")
        optimizer_proposal = next(
            proposal
            for proposal in proposals
            if str(proposal.program.metadata.get("program_kind") or "") == "candidate_optimizer_aware_pipeline"
        )
        self.assertEqual(str(optimizer_proposal.program.metadata.get("runtime_schedule_family") or ""), "dual_overlap_optimizer_hide")
        self.assertIn("dual_overlap_optimizer_hide", list(optimizer_proposal.program.metadata.get("feedback_selected_families") or []))

    def test_policy_memory_bank_records_reflection_and_updates_scoreboard(self) -> None:
        baseline = default_dense_program("single_g5")
        context = {
            "runtime_evidence": {
                "optimizer_exposed_ratio": 0.25,
                "optimizer_ratio": 0.59,
                "peak_reserved_ratio": 0.85,
                "stage_tail_ratio": 0.15,
                "tail_step_jitter_ratio": 0.14,
            },
            "failure_modes": [{"label": "tail_heavy"}],
            "derived_bottlenecks": [{"label": "tail_heavy"}],
        }
        candidate = agent_loop._build_optimizer_aware_pipeline_candidate(baseline, context)
        self.assertIsNotNone(candidate)
        proposal = agent_loop._build_agent_proposal(
            candidate,
            scope="pipe",
            rationale="unit test",
            source="heuristic_supervisor",
        )
        search_plan = build_feedback_search_plan(
            baseline,
            context,
            replan_decision=ReplanDecision(trigger="steady", scope="pipe").to_dict(),
            proposals=[proposal],
            memory_bank=PolicyMemoryBank(),
        )
        baseline_metrics = {
            "step_time_ms_p50": 1000.0,
            "throughput_tokens_per_s": 1500.0,
            "trace_summary": {
                "optimizer_exposed_ratio": 0.25,
                "stage_tail_ratio": 0.15,
                "tail_step_jitter_ratio": 0.14,
                "peak_reserved_ratio": 0.85,
            },
        }
        metrics = {
            "config_name": "candidate_optimizer_aware_pipeline",
            "returncode": 0,
            "step_time_ms_p50": 910.0,
            "throughput_tokens_per_s": 1700.0,
            "trace_summary": {
                "optimizer_exposed_ratio": 0.16,
                "stage_tail_ratio": 0.08,
                "tail_step_jitter_ratio": 0.07,
                "peak_reserved_ratio": 0.80,
            },
            "window_feedback": {
                "window_index": 1,
                "policy_signature": "sig-optimizer-window",
                "critical_stage_id": 1,
                "critical_layer_group_id": "group_1",
                "critical_component_type": "reload",
                "step_time_ms_p50": 910.0,
                "throughput_tokens_per_s": 1700.0,
                "rollback_triggered": False,
                "recommended_rewrites": [{"rewrite_type": "reload_shift"}],
            },
        }
        outcome = build_trial_outcome(candidate, metrics, baseline_metrics=baseline_metrics)
        reflection = reflect_on_trial(
            dict(search_plan.get("search_state") or {}),
            proposal,
            outcome,
            baseline_metrics=baseline_metrics,
        )
        memory_bank = PolicyMemoryBank()
        case = record_trial_feedback(
            memory_bank,
            dict(search_plan.get("search_state") or {}),
            proposal,
            outcome,
            reflection,
        )
        self.assertEqual(case.family, "dual_overlap_optimizer_hide")
        self.assertEqual(outcome.policy_signature, "sig-optimizer-window")
        self.assertFalse(outcome.rollback_triggered)
        self.assertEqual(outcome.critical_component_type, "reload")
        self.assertEqual(int((outcome.latest_window_outcome or {}).get("window_index") or -1), 1)
        self.assertEqual(str((reflection.window_feedback_digest or {}).get("policy_signature") or ""), "sig-optimizer-window")
        self.assertEqual(
            str((((reflection.rewrite_recommendation or {}).get("recommended_rewrites") or [{}])[0].get("rewrite_type") or "")),
            "reload_shift",
        )
        self.assertTrue(
            any(
                str(item.get("window") or "") == "last_2_groups"
                for item in list(case.local_policy.get("runtime_window_overrides") or [])
            )
        )
        self.assertTrue(
            any(
                str(item.get("cluster_role") or "") == "optimizer_sensitive"
                for item in list(case.local_policy.get("runtime_operator_cluster_overrides") or [])
            )
        )
        scoreboard = memory_bank.family_scoreboard()
        self.assertEqual(str(scoreboard[0].get("family") or ""), "dual_overlap_optimizer_hide")
        self.assertGreater(float(scoreboard[0].get("score") or 0.0), 0.0)
        retrieved = memory_bank.retrieve_cases(
            dict(search_plan.get("search_state") or {}),
            family="dual_overlap_optimizer_hide",
            top_k=1,
        )
        self.assertEqual(len(retrieved), 1)

    def test_compile_program_exports_feedback_runtime_family_env(self) -> None:
        baseline = default_dense_program("single_g5")
        context = {
            "runtime_evidence": {
                "bubble_ratio": 0.13,
                "peak_reserved_ratio": 0.84,
                "optimizer_exposed_ratio": 0.19,
                "stage_tail_ratio": 0.16,
                "tail_step_jitter_ratio": 0.18,
                "stage_window_summary": {
                    "0": {"peak_reserved_gib": 23.0},
                    "1": {"peak_reserved_gib": 25.5},
                },
            },
            "failure_modes": [{"label": "tail_heavy"}],
            "derived_bottlenecks": [{"label": "tail_heavy"}],
        }
        candidate = agent_loop._build_tail_aware_execution_candidate(baseline, context)
        self.assertIsNotNone(candidate)
        compiled = compile_program(candidate)
        self.assertEqual(str(compiled.launcher_env.get("SCHEDULE_POLICY_FAMILY") or ""), "dual_overlap_tail_guarded")
        self.assertIn("SCHEDULE_STAGE_FAMILY_HINTS", compiled.launcher_env)

    def test_compile_program_lowers_runtime_window_override_hints(self) -> None:
        baseline = default_dense_program("single_g5")
        context = {
            "runtime_evidence": {
                "optimizer_exposed_ratio": 0.26,
                "optimizer_ratio": 0.61,
                "peak_reserved_ratio": 0.84,
                "stage_tail_ratio": 0.14,
                "tail_step_jitter_ratio": 0.15,
                "stage_window_summary": {
                    "0": {"peak_reserved_gib": 22.0},
                    "1": {"peak_reserved_gib": 27.0},
                },
            },
            "failure_modes": [{"label": "tail_heavy"}],
            "derived_bottlenecks": [{"label": "tail_heavy"}],
        }
        candidate = agent_loop._build_optimizer_aware_pipeline_candidate(baseline, context)
        self.assertIsNotNone(candidate)
        compiled = compile_program(candidate)
        encoded = str(compiled.launcher_env.get("SCHEDULE_WINDOW_OVERRIDE_HINTS") or "")
        self.assertTrue(encoded)
        payload = json.loads(encoded)
        self.assertTrue(any(str(item.get("window") or "") == "last_2_groups" for item in payload))
        self.assertTrue(any(str(item.get("stage_selector") or "") == "optimizer_sensitive_stage" for item in payload))

    def test_compile_program_lowers_runtime_operator_cluster_hints(self) -> None:
        baseline = default_dense_program("single_g5")
        context = {
            "runtime_evidence": {
                "optimizer_exposed_ratio": 0.24,
                "optimizer_ratio": 0.58,
                "peak_reserved_ratio": 0.84,
                "stage_tail_ratio": 0.14,
                "tail_step_jitter_ratio": 0.15,
            },
            "failure_modes": [{"label": "tail_heavy"}],
            "derived_bottlenecks": [{"label": "tail_heavy"}],
        }
        candidate = agent_loop._build_optimizer_aware_pipeline_candidate(baseline, context)
        self.assertIsNotNone(candidate)
        compiled = compile_program(candidate)
        encoded = str(compiled.launcher_env.get("SCHEDULE_OPERATOR_CLUSTER_HINTS") or "")
        self.assertTrue(encoded)
        payload = json.loads(encoded)
        self.assertTrue(any(str(item.get("cluster_role") or "") == "optimizer_sensitive" for item in payload))
        self.assertTrue(any(str(item.get("cluster_role") or "") == "backward_critical" for item in payload))

    def test_check_program_rejects_fine_grained_offload_without_transformer_engine(self) -> None:
        baseline = default_dense_program("single_g5")
        baseline.metadata["runtime_enable_fine_grained_activation_offloading"] = True
        baseline.metadata["runtime_offload_modules"] = ["core_attn"]
        report = check_program(baseline)
        self.assertFalse(report.is_valid)
        self.assertIn("fine_grained_offload_requires_te", list(report.diagnosis or []))
        self.assertTrue(
            any(
                "fine-grained activation offloading requires transformer_engine" in str(item)
                for item in (report.errors or [])
            )
        )

    def test_build_summary_payload_includes_feedback_memory_fields(self) -> None:
        baseline = default_dense_program("single_g5")
        rewrite = agent_loop._rewrite_space(baseline, {})
        memory_bank = PolicyMemoryBank()
        summary = agent_loop._build_summary_payload(
            export_only=True,
            programs_dir=Path("e:/python/Agent/runs_megatron_programs"),
            runtime_summary={},
            runtime_signature={},
            context_record={},
            replan_decision={},
            bottleneck_signature={},
            rewrite=rewrite,
            baseline=baseline,
            baseline_metrics=None,
            best_program=None,
            best_metrics=None,
            tested=[],
            family_outside_trials=[],
            rejected_candidates=[],
            candidate_manifest=[],
            program_bank=ProgramBank(),
            evidence_manifest=[],
            feedback_search_plan={"selected_families": ["dual_overlap_tail_guarded"]},
            policy_memory=memory_bank.to_dict(),
            family_scoreboard=memory_bank.family_scoreboard(),
            trial_reflections=[{"family": "dual_overlap_tail_guarded"}],
            autotune_history=[{"round_index": 1}],
            auto_tune_rounds_requested=2,
            search_unit="whole_config",
            patch_memory_enabled=False,
        )
        self.assertIn("feedback_search_plan", summary)
        self.assertIn("policy_memory", summary)
        self.assertIn("family_scoreboard", summary)
        self.assertIn("trial_reflections", summary)
        self.assertEqual(int(summary.get("auto_tune_rounds_requested") or 0), 2)
        self.assertEqual(int(summary.get("auto_tune_rounds_completed") or 0), 1)
        self.assertEqual(str(summary.get("search_unit") or ""), "whole_config")
        self.assertFalse(bool(summary.get("patch_memory_enabled")))

    def test_build_optimizer_aware_pipeline_candidate_emits_tail_guarded_execution_semantics(self) -> None:
        baseline = default_dense_program("single_g5")
        context = {
            "runtime_evidence": {
                "optimizer_exposed_ratio": 0.26,
                "optimizer_ratio": 0.61,
                "peak_reserved_ratio": 0.85,
                "stage_tail_ratio": 0.14,
                "tail_step_jitter_ratio": 0.16,
                "stage_window_summary": {
                    "0": {"peak_reserved_gib": 22.0},
                    "1": {"peak_reserved_gib": 28.5},
                },
            },
            "failure_modes": [{"label": "tail_heavy"}],
            "derived_bottlenecks": [{"label": "tail_heavy"}],
        }
        candidate = agent_loop._build_optimizer_aware_pipeline_candidate(baseline, context)
        self.assertIsNotNone(candidate)
        self.assertEqual(str(candidate.metadata.get("program_kind") or ""), "candidate_optimizer_aware_pipeline")
        self.assertEqual(str(candidate.schedule.dispatch_order or ""), "optimizer_tail_guarded")
        self.assertEqual(str(candidate.metadata.get("flush_order_policy") or ""), "optimizer_tail_hide")
        self.assertEqual(int(candidate.parallel.vpp_degree), 2)
        self.assertEqual(int(candidate.layout.vpp_degree), 2)
        self.assertEqual(str(candidate.metadata.get("runtime_optimizer_policy_mode") or ""), "tail_guarded_overlap")
        self.assertEqual(str(candidate.metadata.get("runtime_optimizer_target_policy") or ""), "tail_stage_first")
        self.assertEqual(str(candidate.metadata.get("runtime_optimizer_window_policy") or ""), "tail_flush_aligned")
        self.assertTrue(bool(candidate.metadata.get("runtime_recompute_modules")))
        window_overrides = list(candidate.metadata.get("runtime_window_overrides") or [])
        cluster_overrides = list(candidate.metadata.get("runtime_operator_cluster_overrides") or [])
        self.assertTrue(any(str(item.get("window") or "") == "last_2_groups" for item in window_overrides))
        self.assertTrue(any(str(item.get("stage_selector") or "") == "optimizer_sensitive_stage" for item in window_overrides))
        self.assertTrue(any(str(item.get("cluster_role") or "") == "optimizer_sensitive" for item in cluster_overrides))
        self.assertTrue(any(str(item.get("cluster_role") or "") == "backward_critical" for item in cluster_overrides))
        stage_families = list(candidate.metadata.get("morphable_stage_families") or [])
        self.assertTrue(any(str(item.get("family") or "") == "optimizer_guarded_tail" for item in stage_families))

    def test_build_tail_aware_execution_candidate_emits_heterogeneous_tail_controls(self) -> None:
        baseline = default_dense_program("single_g5")
        context = {
            "runtime_evidence": {
                "bubble_ratio": 0.14,
                "peak_reserved_ratio": 0.83,
                "optimizer_exposed_ratio": 0.19,
                "stage_tail_ratio": 0.16,
                "tail_step_jitter_ratio": 0.18,
                "stage_window_summary": {
                    "0": {"peak_reserved_gib": 23.0},
                    "1": {"peak_reserved_gib": 25.0},
                },
            },
            "failure_modes": [{"label": "tail_heavy"}],
            "derived_bottlenecks": [{"label": "tail_heavy"}],
        }
        candidate = agent_loop._build_tail_aware_execution_candidate(baseline, context)
        self.assertIsNotNone(candidate)
        self.assertEqual(str(candidate.metadata.get("program_kind") or ""), "candidate_tail_aware_execution")
        self.assertEqual(str(candidate.metadata.get("runtime_checkpoint_boundary_mode") or ""), "tail_stage_guarded")
        self.assertEqual(str(candidate.metadata.get("runtime_optimizer_policy_mode") or ""), "tail_guarded_overlap")
        self.assertEqual(str(candidate.metadata.get("runtime_optimizer_window_policy") or ""), "tail_flush_aligned")
        self.assertTrue(bool(candidate.metadata.get("stage_local_vpp_vector")))
        window_overrides = list(candidate.metadata.get("runtime_window_overrides") or [])
        cluster_overrides = list(candidate.metadata.get("runtime_operator_cluster_overrides") or [])
        self.assertTrue(any(str(item.get("window") or "") == "last_1_group" for item in window_overrides))
        self.assertTrue(any(str(item.get("stage_selector") or "") == "tail_stage" for item in window_overrides))
        self.assertTrue(any(str(item.get("cluster_role") or "") == "backward_critical" for item in cluster_overrides))
        stage_families = list(candidate.metadata.get("morphable_stage_families") or [])
        self.assertTrue(any(str(item.get("family") or "") == "tail_guarded" for item in stage_families))

    def test_build_stage_local_vpp_shape_candidate_lowers_heterogeneous_layout_and_stage_tags(self) -> None:
        baseline = default_dense_program("single_g5")
        context = {
            "runtime_evidence": {
                "optimizer_exposed_ratio": 0.22,
                "peak_reserved_ratio": 0.86,
                "stage_tail_ratio": 0.13,
                "tail_step_jitter_ratio": 0.12,
                "stage_window_summary": {
                    "0": {"peak_reserved_gib": 19.0},
                    "1": {"peak_reserved_gib": 21.2},
                },
            },
            "evidence_record": {
                "nonuniform_vpp_shape": {
                    "per_stage_candidates": [
                        {
                            "stage_id": 0,
                            "recommended_v": 2,
                            "currently_executable_values": [1, 2],
                            "candidate_chunk_shapes": [[9, 11]],
                        },
                        {
                            "stage_id": 1,
                            "recommended_v": 1,
                            "currently_executable_values": [1],
                            "candidate_chunk_shapes": [[20]],
                        },
                    ]
                }
            },
        }
        candidate = agent_loop._build_stage_local_vpp_shape_candidate(baseline, context)
        self.assertIsNotNone(candidate)
        self.assertEqual(str(candidate.metadata.get("program_kind") or ""), "candidate_nonuniform_vpp_shape")
        self.assertEqual(int(candidate.parallel.vpp_degree), 2)
        self.assertEqual(int(candidate.layout.vpp_degree), 2)
        self.assertEqual(str(candidate.layout.pipeline_layout or ""), "Ettttttttt|tttttttttttttttttttt|ttttttttttt|L")
        self.assertEqual(str(candidate.metadata.get("runtime_optimizer_policy_mode") or ""), "tail_guarded_overlap")
        stage_families = list(candidate.metadata.get("morphable_stage_families") or [])
        tail_hint = next(item for item in stage_families if int(item.get("stage_index") or -1) == 1)
        self.assertIn("tail_sensitive", list(tail_hint.get("stage_tags") or []))
        self.assertIn("optimizer_sensitive", list(tail_hint.get("stage_tags") or []))

    def test_build_checkpoint_boundary_refinement_candidate_marks_joint_checkpoint_control(self) -> None:
        baseline = default_dense_program("single_g5")
        baseline.parallel.pp_degree = 4
        baseline.parallel.vpp_degree = 2
        baseline.layout.vpp_degree = 2
        baseline.schedule.template = "interleaved_grouped_g2"
        baseline.schedule.skeleton = "stage_aware_grouped"
        baseline.schedule.microbatch_group_size_per_vp_stage = 2
        context = {
            "runtime_evidence": {
                "peak_reserved_ratio": 0.91,
                "stage_tail_ratio": 0.12,
                "tail_step_jitter_ratio": 0.15,
                "stage_window_summary": {
                    "0": {"peak_reserved_gib": 26.0},
                    "1": {"peak_reserved_gib": 29.4},
                },
            },
            "failure_modes": [{"label": "memory_hotspot"}],
            "derived_bottlenecks": [{"label": "tail_heavy"}],
        }
        candidate = agent_loop._build_checkpoint_boundary_refinement_candidate(baseline, context)
        self.assertIsNotNone(candidate)
        self.assertEqual(str(candidate.metadata.get("program_kind") or ""), "candidate_checkpoint_boundary_refinement")
        self.assertEqual(str(candidate.metadata.get("runtime_checkpoint_boundary_mode") or ""), "hotspot_tail_staggered")
        self.assertEqual(str(candidate.metadata.get("schedule_steady_checkpoint_policy") or ""), "guarded_selective")
        self.assertTrue(bool(candidate.metadata.get("stage_local_memory_policy")))
        self.assertTrue(any(str(item.get("cluster_role") or "") == "memory_hotspot" for item in list(candidate.metadata.get("runtime_operator_cluster_overrides") or [])))
        self.assertFalse(
            any(
                str(item.get("stage_selector") or "") == "optimizer_sensitive_stage"
                for item in list(candidate.metadata.get("runtime_window_overrides") or [])
            )
        )
        self.assertFalse(
            any(
                str(item.get("cluster_role") or "") == "optimizer_sensitive"
                for item in list(candidate.metadata.get("runtime_operator_cluster_overrides") or [])
            )
        )
        stage_families = list(candidate.metadata.get("morphable_stage_families") or [])
        self.assertTrue(any(str(item.get("family") or "") == "checkpoint_guarded" for item in stage_families))

    def test_synthesize_proposals_emits_morphable_pipeline_candidate(self) -> None:
        baseline = default_dense_program("single_g5")
        baseline.parallel.pp_degree = 4
        baseline.parallel.vpp_degree = 2
        baseline.layout.vpp_degree = 2
        runtime_summary = {
            "steady_state_step_time_ms_p50": 8063.5,
            "bubble_ratio": 0.12,
            "pipeline_wait_ratio": 0.15,
            "optimizer_exposed_ratio": 0.22,
            "peak_memory_ratio": 0.89,
            "comm_exposure_ratio": 0.13,
            "stage_window_summary": {
                "0": {"compute_ms": 1700.0, "forward_ms": 900.0, "completion_ms": 2250.0, "window_ms": 2250.0, "peak_reserved_gib": 23.5},
                "1": {"compute_ms": 1300.0, "forward_ms": 760.0, "completion_ms": 1500.0, "window_ms": 1500.0, "peak_reserved_gib": 18.0},
                "2": {"compute_ms": 1320.0, "forward_ms": 770.0, "completion_ms": 1520.0, "window_ms": 1520.0, "peak_reserved_gib": 18.4},
                "3": {"compute_ms": 1680.0, "forward_ms": 880.0, "completion_ms": 2200.0, "window_ms": 2200.0, "peak_reserved_gib": 23.0},
            },
        }
        rewrite = agent_loop._rewrite_space(baseline, runtime_summary)
        context = build_context_record(baseline, runtime_summary=runtime_summary)
        proposals, rejected = agent_loop._synthesize_proposals(
            baseline,
            rewrite,
            runtime_summary=runtime_summary,
            context_record=context,
            replan_decision=ReplanDecision(trigger="joint_structure_memory_comm", scope="pipe").to_dict(),
            candidate_limit=16,
        )
        proposal_kinds = {str(proposal.program.metadata.get("program_kind")) for proposal in proposals}
        rejected_kinds = {
            str((((item.get("proposal") or {}).get("program") or {}).get("metadata") or {}).get("program_kind"))
            for item in rejected
        }
        self.assertIn("candidate_morphable_pipeline", proposal_kinds | rejected_kinds)

    def test_torchtitan_hybrid_plan_normalization_and_validation(self) -> None:
        plan = TorchTitanHybridPlanIR(
            shard_mode="HSDP",
            pp_degree=4,
            vp_degree=2,
            schedule_kind="interleaved_1f1b",
            tp_degree=4,
            reshard_policy="node_local",
            stage_to_node=["g5_0", "g5_0", "g5_0", "g5_0", "g5_1", "g5_1", "g5_1", "g5_1"],
            stage_ranges=[[0, 4], [5, 9], [10, 14], [15, 19], [20, 24], [25, 29], [30, 34], [35, 39]],
        ).normalized()
        self.assertEqual(plan.shard_mode, "hsdp")
        self.assertEqual(plan.reshard_policy, "node_local")
        self.assertEqual(plan.schedule_kind, "interleaved_1f1b")
        self.assertEqual(len(plan.stage_ranges), 8)
        self.assertTrue(plan.name.startswith("hsdp_pp4"))
        with self.assertRaises(ValueError):
            TorchTitanHybridPlanIR(
                shard_mode="none",
                pp_degree=1,
                vp_degree=2,
                schedule_kind="interleaved_1f1b",
                tp_degree=1,
                reshard_policy="default",
            ).normalized()

    def test_torchtitan_hybrid_regime_classifier_prefers_memory_bubble_and_allgather(self) -> None:
        memory = TorchTitanHybridEvidence(peak_reserved_ratio=0.93, bubble_ratio=0.05)
        bubble = TorchTitanHybridEvidence(bubble_ratio=0.16, all_gather_exposed_ratio=0.05)
        allgather = TorchTitanHybridEvidence(all_gather_exposed_ratio=0.09, reduce_scatter_exposed_ratio=0.06, bubble_ratio=0.05)
        mixed = TorchTitanHybridEvidence(bubble_ratio=0.08, all_gather_exposed_ratio=0.05, reduce_scatter_exposed_ratio=0.03)
        self.assertEqual(TorchTitanHybridController().classify_regime(memory), "memory_dominated")
        self.assertEqual(TorchTitanHybridController().classify_regime(bubble), "pipeline_bubble_dominated")
        self.assertEqual(TorchTitanHybridController().classify_regime(allgather), "all_gather_dominated")
        self.assertEqual(TorchTitanHybridController().classify_regime(mixed), "mixed")

    def test_torchtitan_hybrid_verifier_rejects_non_hsdp_node_local_and_penalizes_global_fsdp_in_allgather_regime(self) -> None:
        evidence = TorchTitanHybridEvidence(
            all_gather_exposed_ratio=0.10,
            reduce_scatter_exposed_ratio=0.05,
            bubble_ratio=0.04,
            peak_reserved_ratio=0.80,
            n_microbatches=4,
        )
        with self.assertRaises(ValueError):
            TorchTitanHybridPlanIR(
                shard_mode="fsdp2",
                pp_degree=1,
                vp_degree=1,
                schedule_kind="1f1b",
                tp_degree=1,
                reshard_policy="node_local",
            ).normalized()
        plan = TorchTitanHybridPlanIR(
            shard_mode="fsdp2",
            pp_degree=1,
            vp_degree=1,
            schedule_kind="1f1b",
            tp_degree=1,
            reshard_policy="default",
            n_microbatches=4,
        )
        report = verify_torchtitan_hybrid_plan(plan, evidence, "all_gather_dominated")
        self.assertTrue(report.is_legal)
        self.assertIn("global_fsdp_penalty_for_allgather", list(report.diagnosis))
        self.assertLess(float(report.score.get("total_score") or 0.0), 0.0)

    def test_torchtitan_hybrid_controller_ranks_bubble_and_allgather_regimes_in_expected_direction(self) -> None:
        controller = TorchTitanHybridController()
        bubble_result = controller.synthesize(
            runtime_summary={
                "bubble_ratio": 0.18,
                "p2p_exposed_ratio": 0.05,
                "peak_reserved_ratio": 0.78,
                "n_microbatches": 8,
                "stage_window_summary": {
                    "0": {"forward_ms": 520.0, "backward_ms": 700.0, "peak_reserved_gib": 24.0},
                    "1": {"forward_ms": 480.0, "backward_ms": 660.0, "peak_reserved_gib": 23.0},
                    "2": {"forward_ms": 410.0, "backward_ms": 600.0, "peak_reserved_gib": 21.0},
                    "3": {"forward_ms": 405.0, "backward_ms": 590.0, "peak_reserved_gib": 20.0},
                },
            },
            top_k=6,
        )
        self.assertEqual(bubble_result.regime, "pipeline_bubble_dominated")
        self.assertTrue(any(candidate.plan.schedule_kind == "zero_bubble" for candidate in bubble_result.candidates[:3]))
        self.assertEqual(bubble_result.canary_baseline_name, "none_pp4_vp1_1f1b_tp4_default")

        allgather_result = controller.synthesize(
            runtime_summary={
                "all_gather_exposed_ratio": 0.11,
                "reduce_scatter_exposed_ratio": 0.05,
                "bubble_ratio": 0.04,
                "peak_reserved_ratio": 0.82,
                "n_microbatches": 4,
                "stage_window_summary": {
                    "0": {"forward_ms": 500.0, "backward_ms": 680.0, "peak_reserved_gib": 25.0},
                    "1": {"forward_ms": 490.0, "backward_ms": 670.0, "peak_reserved_gib": 24.0},
                    "2": {"forward_ms": 480.0, "backward_ms": 660.0, "peak_reserved_gib": 24.0},
                    "3": {"forward_ms": 470.0, "backward_ms": 650.0, "peak_reserved_gib": 23.0},
                },
            },
            top_k=6,
        )
        self.assertEqual(allgather_result.regime, "all_gather_dominated")
        self.assertEqual(allgather_result.candidates[0].plan.shard_mode, "hsdp")
        self.assertEqual(allgather_result.canary_baseline_name, "fsdp2_pp1_vp1_1f1b_tp1_default")

    def test_torchtitan_hybrid_export_roundtrip_applies_to_trainer_config(self) -> None:
        plan = TorchTitanHybridPlanIR(
            shard_mode="hsdp",
            pp_degree=4,
            vp_degree=1,
            schedule_kind="zero_bubble",
            tp_degree=2,
            reshard_policy="node_local",
            stage_to_node=["g5_0", "g5_0", "g5_1", "g5_1"],
            stage_ranges=[[0, 8], [9, 19], [20, 30], [31, 39]],
            n_microbatches=4,
        ).normalized()
        policy = export_plan_to_hybrid_policy(plan)
        self.assertEqual(policy["pipeline"]["schedule"], "ZBVZeroBubble")
        self.assertEqual(policy["fsdp2"]["replicate_degree"], 2)
        self.assertEqual(policy["fsdp2"]["shard_degree"], 1)
        cfg = apply_hybrid_policy(_FakeTrainerConfig(), policy_path=self._write_policy(policy))
        self.assertEqual(cfg.parallelism.pipeline_parallel_degree, 4)
        self.assertEqual(cfg.parallelism.tensor_parallel_degree, 2)
        self.assertEqual(cfg.parallelism.data_parallel_replicate_degree, 2)
        self.assertEqual(cfg.parallelism.data_parallel_shard_degree, 1)
        self.assertEqual(cfg.parallelism.pipeline_parallel_schedule, "ZBVZeroBubble")
        self.assertEqual(cfg.parallelism.pipeline_parallel_stage_to_node, ["g5_0", "g5_0", "g5_1", "g5_1"])

    def test_hybrid_policy_applies_fine_grained_fsdp_controls(self) -> None:
        policy = {
            "pipeline": {
                "degree": 2,
                "stage_hbm_budget_gib": [28.0, 30.0],
            },
            "fsdp2": {
                "enabled": True,
                "policy_mode": "module_groups",
                "mlp_unit_mode": "split_gate_up_down",
                "mlp_scope": "node",
                "mlp_output_scope": "keep",
                "prefetch_window": 2,
                "recompute_forward_prefetch": "none",
                "recompute_backward_prefetch": "none",
                "materialization_watermark_gib": 29.0,
            },
        }
        cfg = apply_hybrid_policy(_FakeTrainerConfig(), policy_path=self._write_policy(policy))
        self.assertEqual(cfg.parallelism.fsdp_mlp_unit_mode, "split_gate_up_down")
        self.assertEqual(cfg.parallelism.fsdp_mlp_scope, "node")
        self.assertEqual(cfg.parallelism.fsdp_mlp_output_scope, "keep")
        self.assertEqual(cfg.parallelism.fsdp_prefetch_window, 2)
        self.assertEqual(cfg.parallelism.fsdp_recompute_forward_prefetch, "none")
        self.assertEqual(cfg.parallelism.fsdp_recompute_backward_prefetch, "none")
        self.assertEqual(cfg.parallelism.fsdp_materialization_watermark_gib, 29.0)
        self.assertEqual(cfg.parallelism.fsdp_stage_hbm_budget_gib, [28.0, 30.0])

    def test_safe_module_isinstance_tolerates_non_type_candidate(self) -> None:
        class _Dummy:
            pass

        instance = _Dummy()
        self.assertTrue(_safe_module_isinstance(instance, _Dummy))
        self.assertFalse(_safe_module_isinstance(instance, None))
        self.assertFalse(_safe_module_isinstance(instance, "not-a-type"))

    def _write_policy(self, policy: Dict[str, Any]) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        try:
            json.dump(policy, handle, indent=2, ensure_ascii=False)
            handle.flush()
            return handle.name
        finally:
            handle.close()


class TestMegatronMetricsParser(unittest.TestCase):
    def test_parse_megatron_logs_derives_optimizer_exposure_and_pipeline_wait(self) -> None:
        stdout = """
        stage_metrics/stage_0/fwd(ms): 100
        stage_metrics/stage_0/bwd(ms): 200
        stage_metrics/stage_0/ag_estimated(ms): 10
        stage_metrics/stage_0/rs_estimated(ms): 5
        stage_metrics/stage_0/bubble(ms): 20
        stage_metrics/stage_0/peak_reserved(GiB): 25
        stage_metrics/stage_1/fwd(ms): 90
        stage_metrics/stage_1/bwd(ms): 180
        stage_metrics/stage_1/ag_estimated(ms): 8
        stage_metrics/stage_1/rs_estimated(ms): 4
        stage_metrics/stage_1/bubble(ms): 15
        stage_metrics/stage_1/peak_reserved(GiB): 24
        INFO:megatron.core.timers:(min, max) time across ranks (ms):
            forward-backward ...............................: (4300.00, 4400.00)
            all-grads-sync .................................: (4.20, 6.00)
            optimizer-copy-to-main-grad ....................: (0.20, 0.30)
            optimizer-inner-step ...........................: (4100.00, 4200.00)
            optimizer-copy-main-to-model-params ............: (0.10, 0.20)
            optimizer ......................................: (5100.00, 5200.00)
            forward-recv ...................................: (30.00, 40.00)
            backward-recv ..................................: (50.00, 60.00)
            forward-send-backward-recv .....................: (950.00, 1000.00)
            backward-send-forward-recv .....................: (10.00, 20.00)
         [2026-03-31 07:27:16.160756] iteration        4/       6 | consumed samples:          128 | elapsed time per iteration (ms): 10000.0 | throughput per GPU (TFLOP/s/GPU): 34.8 | learning rate: 8.000000E-05 | global batch size:    32 | lm loss: 1.407003E+01 |
         [2026-03-31 07:27:26.160756] iteration        5/       6 | consumed samples:          160 | elapsed time per iteration (ms): 10200.0 | throughput per GPU (TFLOP/s/GPU): 34.0 | learning rate: 8.000000E-05 | global batch size:    32 | lm loss: 1.307003E+01 |
         [2026-03-31 07:27:36.160756] iteration        6/       6 | consumed samples:          192 | elapsed time per iteration (ms): 9800.0 | throughput per GPU (TFLOP/s/GPU): 35.1 | learning rate: 8.000000E-05 | global batch size:    32 | lm loss: 1.207003E+01 |
        """
        parsed = parse_megatron_logs(
            stdout=stdout,
            stderr="",
            global_batch_size=32,
            seq_len=1024,
        )

        self.assertAlmostEqual(float(parsed["optimizer_total_ms"] or 0.0), 5200.0, places=4)
        self.assertAlmostEqual(float(parsed["optimizer_exposed_ms"] or 0.0), 5200.0, places=4)
        self.assertAlmostEqual(float(parsed["pipeline_wait_ms"] or 0.0), 1120.0, places=4)
        self.assertAlmostEqual(float(parsed["pipeline_wait_ratio"] or 0.0), 0.112, places=3)
        self.assertAlmostEqual(float(parsed["optimizer_exposed_ratio"] or 0.0), 0.52, places=3)
        self.assertAlmostEqual(float(parsed["bubble_exposure_ratio"] or 0.0), 35.0 / 632.0, places=4)
        self.assertAlmostEqual(float(parsed["stage_skew"] or 0.0), 335.0 / ((335.0 + 297.0) / 2.0), places=4)
        self.assertIn("timer_summary", parsed)
        self.assertIn("stage_window_summary", parsed)


if __name__ == "__main__":
    unittest.main()
