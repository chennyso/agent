# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

from typing import Any

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.dual_pipe_v import get_dual_pipe_v_flag
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten.linear.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch.ops.torch_attn._varlen_attn.default,
    torch._higher_order_ops.inductor_compiled_code,
}


def _default_scope_name(base_reshard: bool) -> str:
    return "global" if bool(base_reshard) else "keep"


def _scope_to_reshard_value(
    scope: str,
    *,
    base_reshard: bool,
    node_local_size: int,
) -> bool | int:
    match scope:
        case "global":
            return True
        case "keep":
            return False
        case "node":
            if node_local_size > 0:
                return int(node_local_size)
            logger.warning(
                "module-group FSDP scope resolved to 'node' but fsdp_node_local_reshard_size<=0; "
                "falling back to the base FSDP policy"
            )
            return base_reshard
        case _:
            raise ValueError(f"Unsupported FSDP scope {scope!r}")


def _resolve_qwen3_group_scope(
    *,
    group_name: str,
    configured_scope: str,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    stage_idx: int,
    num_stages: int,
    stage_to_node: tuple[str, ...],
    is_last_local_stage: bool,
    base_reshard: bool,
) -> str:
    if configured_scope != "auto":
        return configured_scope

    current_stage_node = stage_to_node[stage_idx] if 0 <= stage_idx < len(stage_to_node) else None
    has_multi_stage_pipeline = num_stages > 1 and stage_idx >= 0
    crosses_prev = (
        current_stage_node is not None
        and stage_idx > 0
        and stage_to_node[stage_idx - 1] != current_stage_node
    )
    crosses_next = (
        current_stage_node is not None
        and stage_idx + 1 < num_stages
        and stage_to_node[stage_idx + 1] != current_stage_node
    )
    near_cross_node_boundary = crosses_prev or crosses_next

    if group_name == "attention":
        if parallel_dims.tp_enabled or parallel_dims.cp_enabled:
            return "keep"
        if (
            has_multi_stage_pipeline
            and is_last_local_stage
            and parallelism.fsdp_node_local_reshard_size > 0
        ):
            return "node"
        return _default_scope_name(base_reshard)

    if group_name == "mlp":
        if near_cross_node_boundary and parallelism.fsdp_node_local_reshard_size > 0:
            return "node"
        if has_multi_stage_pipeline and is_last_local_stage:
            return "keep"
        return _default_scope_name(base_reshard)

    if group_name == "embhead":
        if has_multi_stage_pipeline and (is_last_local_stage or stage_idx in {0, num_stages - 1}):
            return "keep"
        return _default_scope_name(base_reshard)

    return _default_scope_name(base_reshard)


def apply_parallelism_conditioned_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
) -> None:
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    base_reshard = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )
    stage_idx = int(getattr(model, "_tt_stage_idx", -1))
    num_stages = int(getattr(model, "_tt_num_stages", 1))
    stage_to_node = tuple(getattr(model, "_tt_stage_to_node", tuple()))
    is_last_local_stage = bool(getattr(model, "_tt_is_last_local_stage", True))

    attention_scope = _resolve_qwen3_group_scope(
        group_name="attention",
        configured_scope=parallelism.fsdp_attention_scope,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        stage_idx=stage_idx,
        num_stages=num_stages,
        stage_to_node=stage_to_node,
        is_last_local_stage=is_last_local_stage,
        base_reshard=base_reshard,
    )
    mlp_scope = _resolve_qwen3_group_scope(
        group_name="mlp",
        configured_scope=parallelism.fsdp_mlp_scope,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        stage_idx=stage_idx,
        num_stages=num_stages,
        stage_to_node=stage_to_node,
        is_last_local_stage=is_last_local_stage,
        base_reshard=base_reshard,
    )
    embhead_scope = _resolve_qwen3_group_scope(
        group_name="embhead",
        configured_scope=parallelism.fsdp_embhead_scope,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        stage_idx=stage_idx,
        num_stages=num_stages,
        stage_to_node=stage_to_node,
        is_last_local_stage=is_last_local_stage,
        base_reshard=base_reshard,
    )

    attention_reshard = _scope_to_reshard_value(
        attention_scope,
        base_reshard=base_reshard,
        node_local_size=parallelism.fsdp_node_local_reshard_size,
    )
    mlp_reshard = _scope_to_reshard_value(
        mlp_scope,
        base_reshard=base_reshard,
        node_local_size=parallelism.fsdp_node_local_reshard_size,
    )
    embhead_reshard = _scope_to_reshard_value(
        embhead_scope,
        base_reshard=base_reshard,
        node_local_size=parallelism.fsdp_node_local_reshard_size,
    )

    if parallelism.fsdp_policy_trace:
        logger.info(
            "[fsdp-policy] "
            f"stage={stage_idx} local_last={is_last_local_stage} "
            f"stage_node={getattr(model, '_tt_stage_node', 'n/a')} "
            f"attention={attention_scope}->{attention_reshard} "
            f"mlp={mlp_scope}->{mlp_reshard} "
            f"embhead={embhead_scope}->{embhead_reshard}"
        )

    if getattr(model, "enable_weight_tying", False):
        logger.warning(
            "module-group FSDP policy currently treats tied embedding/output as a single embhead group"
        )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=embhead_reshard,
        )

    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=embhead_reshard,
        )

    for transformer_block in model.layers.values():
        fully_shard(
            [transformer_block.attention_norm, transformer_block.attention],
            **fsdp_config,
            reshard_after_forward=attention_reshard,
        )
        if getattr(transformer_block, "moe_enabled", False):
            fully_shard(
                [transformer_block.ffn_norm, transformer_block.moe],
                **fsdp_config,
                reshard_after_forward=mlp_reshard,
            )
        else:
            fully_shard(
                [transformer_block.ffn_norm, transformer_block.feed_forward],
                **fsdp_config,
                reshard_after_forward=mlp_reshard,
            )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=base_reshard,
        )

    fully_shard(model, **fsdp_config)


def parallelize_qwen3(
    model: Qwen3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
            cp_enabled=parallel_dims.cp_enabled,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        dual_pipe_v = get_dual_pipe_v_flag(
            parallelism=parallelism, ac_config=ac_config, parallel_dims=parallel_dims
        )

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            dual_pipe_v=dual_pipe_v,
        )

    if parallel_dims.cp_enabled:
        if parallel_dims.tp_enabled:
            raise NotImplementedError(
                "Context Parallel with Tensor Parallel is not yet supported for Qwen3. "
                "See https://github.com/pytorch/torchtitan/issues/2446"
            )
        attn_backend = getattr(model.config.layer.attention, "attn_backend", "sdpa")
        apply_cp_to_attention_module(
            # pyrefly: ignore [missing-attribute, not-callable]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
            attn_backend,
        )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            # pyrefly: ignore [bad-argument-type]
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config, parallel_dims.ep_enabled)

    if parallel_dims.fsdp_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        if parallelism.fsdp_parallelism_conditioned_policy == "module_groups":
            if parallel_dims.ep_enabled:
                logger.warning(
                    "module-group FSDP policy does not yet customize MoE expert meshes; "
                    "falling back to standard FSDP application"
                )
                apply_fsdp(
                    model,
                    dp_mesh,
                    param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
                    reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
                    pp_enabled=parallel_dims.pp_enabled,
                    cpu_offload=training.enable_cpu_offload,
                    reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
                    ep_degree=parallel_dims.ep,
                    edp_mesh=edp_mesh,
                    gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
                )
            else:
                apply_parallelism_conditioned_fsdp(
                    model,
                    dp_mesh,
                    parallel_dims=parallel_dims,
                    parallelism=parallelism,
                    param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
                    reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
                    pp_enabled=parallel_dims.pp_enabled,
                    cpu_offload=training.enable_cpu_offload,
                    reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
                )
                logger.info("Applied module-group conditioned FSDP to the model")
        else:
            apply_fsdp(
                model,
                dp_mesh,
                param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
                reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
                pp_enabled=parallel_dims.pp_enabled,
                cpu_offload=training.enable_cpu_offload,
                reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
                ep_degree=parallel_dims.ep,
                edp_mesh=edp_mesh,
                gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
            )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
    cp_enabled: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    positions_sharding = Replicate() if cp_enabled else None
    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), Replicate(), None, positions_sharding),
                desired_input_layouts=(
                    Replicate(),
                    Replicate(),
                    None,
                    positions_sharding,
                ),
            ),
            "attention.wq": colwise_parallel(use_local_output=False),
            "attention.wk": colwise_parallel(use_local_output=False),
            "attention.wv": colwise_parallel(use_local_output=False),
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )
