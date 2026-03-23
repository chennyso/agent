from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megatron_agent import agent_loop, trial_runner  # noqa: E402
from megatron_agent.config import default_dense_program, default_moe_smoke_program  # noqa: E402
from megatron_agent.programs import classify_program_family  # noqa: E402


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

        self.assertTrue(rewrite.allow_single_node_pp_split)
        self.assertTrue(rewrite.allow_sequence_parallel_toggle)
        self.assertEqual(rejected, [])
        candidate_kinds = [candidate.metadata.get("program_kind") for candidate in candidates]
        self.assertIn("candidate_single_node_pp_split", candidate_kinds)
        self.assertIn("candidate_sequence_parallel_toggle", candidate_kinds)

        target = next(candidate for candidate in candidates if candidate.metadata.get("program_kind") == "candidate_single_node_pp_split")
        family = classify_program_family(target).to_dict()
        self.assertTrue(family["is_family_outside"])
        sp_toggle = next(candidate for candidate in candidates if candidate.metadata.get("program_kind") == "candidate_sequence_parallel_toggle")
        self.assertEqual(sp_toggle.parallel.tp_degree, baseline.parallel.tp_degree)
        self.assertFalse(sp_toggle.parallel.sp_enabled)

    def test_single_g4_dense_baseline_is_legal_and_exports_candidate(self) -> None:
        baseline = default_dense_program("single_g4")
        rewrite = agent_loop._rewrite_space(baseline, {})
        candidates, rejected = agent_loop._synthesize_programs(baseline, rewrite, candidate_limit=8)

        self.assertFalse(rewrite.allow_single_node_pp_split)
        self.assertTrue(rewrite.allow_nonuniform_partition)
        self.assertTrue(rewrite.allow_sequence_parallel_toggle)
        self.assertEqual(rejected, [])
        self.assertEqual(baseline.layout.stage_to_node, ["g4", "g4"])
        candidate_kinds = [candidate.metadata.get("program_kind") for candidate in candidates]
        self.assertIn("candidate_nonuniform_partition", candidate_kinds)
        self.assertIn("candidate_sequence_parallel_toggle", candidate_kinds)

    def test_dual_target_candidate_synthesis_still_produces_expected_families(self) -> None:
        dense_baseline = default_dense_program("dual_g4_g5")
        dense_runtime = {"bubble_ratio": 0.07, "stage_spread_ratio": 0.12, "cross_node_exposed_ratio": 0.08}
        dense_rewrite = agent_loop._rewrite_space(dense_baseline, dense_runtime)
        dense_candidates, _ = agent_loop._synthesize_programs(dense_baseline, dense_rewrite, candidate_limit=8)
        dense_kinds = {candidate.metadata.get("program_kind") for candidate in dense_candidates}

        self.assertTrue(
            {
                "candidate_nonuniform_partition",
                "candidate_stage_aware_schedule",
                "candidate_topology_layout",
                "candidate_sequence_parallel_toggle",
            }.issubset(dense_kinds)
        )

        moe_baseline = default_moe_smoke_program("dual_g4_g5")
        moe_runtime = {"bubble_ratio": 0.06, "stage_spread_ratio": 0.10, "cross_node_exposed_ratio": 0.09}
        moe_rewrite = agent_loop._rewrite_space(moe_baseline, moe_runtime)
        moe_candidates, _ = agent_loop._synthesize_programs(moe_baseline, moe_rewrite, candidate_limit=8)
        moe_kinds = {candidate.metadata.get("program_kind") for candidate in moe_candidates}

        self.assertFalse(moe_rewrite.allow_sequence_parallel_toggle)
        self.assertTrue(
            {
                "candidate_nonuniform_partition",
                "candidate_stage_aware_schedule",
                "candidate_dual_plane",
                "candidate_topology_layout",
            }.issubset(moe_kinds)
        )
        self.assertNotIn("candidate_sequence_parallel_toggle", moe_kinds)

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
            self.assertEqual(summary["candidate_generation_count"], 2)
            self.assertEqual(summary["candidate_execution_count"], 0)
            self.assertEqual(summary["compile_success_rate"], 1.0)
            self.assertEqual(summary["family_outside_ratio"], 0.5)
            self.assertIn("candidate_manifest", summary)
            self.assertEqual(summary["recommended_execution_order"][0], "baseline")

            exported_files = sorted(path.name for path in programs_dir.glob("*.json"))
            self.assertIn("00_baseline.json", exported_files)
            self.assertIn("01_candidate_single_node_pp_split.json", exported_files)
            self.assertIn("02_candidate_sequence_parallel_toggle.json", exported_files)

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
            self.assertNotIn("--sequence-parallel", cmd)

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
        candidate = agent_loop._build_single_node_pipeline_candidate(baseline)
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
                    rewrite=agent_loop._rewrite_space(baseline, {}),
                    baseline=baseline,
                    baseline_metrics=baseline_metrics,
                    best_program=best_program,
                    best_metrics=best_metrics,
                    tested=[] if baseline_metrics is None else [baseline_metrics],
                    family_outside_trials=[],
                    rejected_candidates=[],
                    candidate_manifest=manifest,
                )
                self.assertIn("compile_success_rate", summary)
                self.assertIn("family_outside_ratio", summary)
                self.assertIn("stage_load_variance", summary)
                self.assertIn("observed_comm_ratio", summary)
                self.assertIn("baseline_vs_best", summary)


if __name__ == "__main__":
    unittest.main()
