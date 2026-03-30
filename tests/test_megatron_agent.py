from __future__ import annotations

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
from megatron_agent.torchtitan_hybrid import (  # noqa: E402
    TorchTitanHybridController,
    TorchTitanHybridEvidence,
    TorchTitanHybridPlanIR,
    export_plan_to_hybrid_policy,
    verify_torchtitan_hybrid_plan,
)
from megatron_agent.config import (  # noqa: E402
    AgentObservation,
    ExperimentSpec,
    BatchPlanSpec,
    LengthBucketPolicy,
    ProgramBank,
    ProgramTemplate,
    ReplanDecision,
    VerifierReport,
    default_backend_caps,
    default_dense_program,
    default_moe_smoke_program,
)
from megatron_agent.programs import classify_program_family, compile_program, verify_program  # noqa: E402
from megatron_agent.trace_reducer import (  # noqa: E402
    build_agent_observation,
    build_context_record,
    build_trial_artifact,
    classify_bottleneck,
    recommend_optimization_methods,
    reduce_trial_trace,
)
from torchtitan.experiments.hybrid_policy import apply_hybrid_policy  # noqa: E402


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
        self.fsdp_embhead_scope = "keep"
        self.fsdp_node_local_reshard_size = 0
        self.fsdp_policy_trace = False
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
            self.assertEqual(summary["family_outside_ratio"], 1.0)
            self.assertIn("candidate_manifest", summary)
            self.assertEqual(summary["recommended_execution_order"][0], "baseline")
            self.assertIn("program_bank", summary)
            self.assertIn("runtime_signature", summary)

            manifest_names = [entry["config_name"] for entry in summary["candidate_manifest"]]
            self.assertIn("baseline", manifest_names)
            self.assertIn("candidate_pp_scaleout", manifest_names)
            self.assertIn("candidate_stage_aware_schedule", manifest_names)

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

    def test_trial_runner_dry_run_deep_observability_emits_profiler_and_wandb_flags(self) -> None:
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

            def _fake_run(cmd, capture_output, text, cwd, env=None):
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
        trace_summary = reduce_trial_trace(
            program,
            runtime_summary={
                "bubble_ratio": 0.11,
                "stage_load_variance": 0.04,
                "cross_node_exposed_ratio": 0.03,
                "peak_memory_ratio": 0.72,
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
        )
        self.assertEqual(restored.motivation_evidence_manifest[0]["config_name"], "evidence_pp_fixed_pipe")
        self.assertEqual(artifact["experiment"]["experiment_id"], "A_problem_existence")
        self.assertGreaterEqual(len(artifact["stage_time_distribution"]), 2)
        self.assertIn("bottleneck_breakdown", artifact)
        self.assertIn("search_space_blueprint", artifact)
        self.assertIn("visualization_artifacts", artifact)
        perfetto = dict((artifact.get("visualization_artifacts") or {}).get("perfetto_trace") or {})
        self.assertEqual(perfetto.get("format"), "perfetto_trace")
        self.assertGreater(len(list(perfetto.get("traceEvents") or [])), 4)
        blueprint = dict(artifact.get("search_space_blueprint") or {})
        self.assertIn("executable_now", blueprint)
        self.assertTrue(any(item.get("name") == "parallel.vpp_degree" for item in (blueprint.get("executable_now") or [])))

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
        self.assertIn("bottleneck_breakdown", evidence)
        self.assertIn("search_space_blueprint", evidence)
        self.assertIn("visualization_artifacts", evidence)
        perfetto = dict((evidence.get("visualization_artifacts") or {}).get("perfetto_trace") or {})
        self.assertEqual(perfetto.get("format"), "perfetto_trace")
        self.assertTrue(any(item.get("label") == "pipeline_idle" for item in (evidence.get("bottleneck_breakdown") or [])))

    def test_write_analysis_artifacts_persists_visualization_and_blueprint_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dirs = {"trial_dir": tmpdir}
            metrics = {
                "context_record": {
                    "runtime_evidence": {"bubble_ratio": 0.1},
                    "evidence_record": {
                        "bottleneck_breakdown": [{"label": "pipeline_idle", "time_ms": 123.0}],
                        "search_space_blueprint": {"executable_now": [{"name": "parallel.pp_degree"}]},
                        "visualization_artifacts": {
                            "perfetto_trace": {"format": "perfetto_trace", "traceEvents": [{"name": "forward"}]}
                        },
                    },
                },
                "trial_artifact": {
                    "bottleneck_breakdown": [{"label": "pipeline_idle", "time_ms": 123.0}],
                    "search_space_blueprint": {"executable_now": [{"name": "parallel.pp_degree"}]},
                    "visualization_artifacts": {
                        "perfetto_trace": {"format": "perfetto_trace", "traceEvents": [{"name": "forward"}]}
                    },
                },
            }
            paths = trial_runner._write_analysis_artifacts(output_dirs, metrics)
            self.assertTrue(Path(paths["trial_artifact_json"]).exists())
            self.assertTrue(Path(paths["context_record_json"]).exists())
            self.assertTrue(Path(paths["perfetto_trace_json"]).exists())
            self.assertTrue(Path(paths["search_space_blueprint_json"]).exists())
            self.assertTrue(Path(paths["bottleneck_breakdown_json"]).exists())

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

    def _write_policy(self, policy: Dict[str, Any]) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        try:
            json.dump(policy, handle, indent=2, ensure_ascii=False)
            handle.flush()
            return handle.name
        finally:
            handle.close()


if __name__ == "__main__":
    unittest.main()
