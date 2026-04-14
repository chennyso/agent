# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import atexit
import contextlib
import json
import os
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
    fine_grained_offloading_disable_offload,
    fine_grained_offloading_enable_offload,
    fine_grained_offloading_flush_delayed_groups,
    fine_grained_offloading_prefetch,
)
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.transformer.cuda_graphs import create_cudagraphs, set_current_microbatch
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    nvtx_range_pop,
    nvtx_range_push,
)

from .combined_1f1b import (
    combined_1f1b_schedule_for_interleaved_pipelining,
    combined_1f1b_schedule_for_no_pipelining,
)
from .hybrid_cp_schedule import hybrid_context_parallel_forward_backward

# Types
Shape = Union[List[int], torch.Size]


_PRODUCTION_SCHEDULE_FAMILIES = {"fixed_1f1b", "interleaved", "zero_bubble", "dualpipe_v"}
_ADAPTER_SCHEDULE_FAMILIES = {"zbv", "v_half", "v_min", "custom"}
_ALL_SCHEDULE_FAMILIES = _PRODUCTION_SCHEDULE_FAMILIES | _ADAPTER_SCHEDULE_FAMILIES
_SCHEDULE_RUNTIME_HOOKS = (
    "before_forward_hook",
    "after_forward_hook",
    "before_backward_hook",
    "after_backward_hook",
    "before_send_recv_hook",
    "after_send_recv_hook",
    "before_optimizer_tail_hook",
    "memory_action_hook",
)
_SCHEDULE_ACTION_RUNNER_SINGLETON: Optional["ScheduleActionRunner"] = None
_SCHEDULE_ACTION_RUNNER_LOCK = threading.Lock()


def get_forward_backward_func(pp_size: Optional[int] = None, vp_size: Optional[int] = None):
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step.

    collect_non_loss_data (optional, bool, default=False): TODO.

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    adjust_tensor_shapes_fn (Callable, optional): A function that adjusts the receive and send
        tensor shapes. Only applicable in forward_backward_pipelining_without_interleaving for now.
        Takes in a list of receive shapes and a list of send shapes and returns the adjusted
        respective list of shapes. Thus it is not used in the other forward-backward functions
        which have different shape handling.

    force_all_reduce (bool, optional): If true, force use of all-reduce for gradient reduction
        instead of reduce-scatter (if using distributed optimizer) in this iteration to ensure all
        data-parallel ranks have fully reduced gradients. This is useful for easier wgrad saving
        (can just inspect DP replica 0 to get full set of wgrads for entire model).

    Args:
        pp_size (Optional[int]): Pipeline model parallel size to use.
        vp_size (Optional[int]): Virtual pipeline model parallel size to use.
            If both pp_size and vp_size are None, both values fall back to parallel_state.
            Otherwise, provided values are used as-is and None is treated as an explicit input.

    """
    if pp_size is None and vp_size is None:
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    if pp_size > 1:
        if vp_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.

    Supports multiple formats:
    - torch.Tensor: Deallocates the tensor directly
    - List[Tensor]: Recursively deallocates each element
    - Dict[str, Tensor]: Recursively deallocates each value (for multi-module pipelines)
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return

    # Handle dict format (multi-module pipelines)
    if isinstance(out, dict):
        for value in out.values():
            deallocate_output_tensor(value, deallocate_pipeline_outputs)
        return

    # Handle list format
    if isinstance(out, list):
        for item in out:
            deallocate_output_tensor(item, deallocate_pipeline_outputs)
        return

    # Base case: deallocate tensor
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step_calc_loss(
    model,
    output_tensor,
    loss_func,
    config,
    vp_stage,
    collect_non_loss_data,
    num_microbatches,
    forward_data_store,
    cp_group_size=None,
    is_last_stage=None,
):
    """Calculate the loss and number of tokens for forward_step()"""

    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    model_vp_stage = getattr(model, "vp_stage", None)
    if vp_stage is not None and model_vp_stage is not None:
        assert (
            vp_stage == model_vp_stage
        ), f"vp_stage ({vp_stage}) doesn't match model_vp_stage ({model_vp_stage})"

    if cp_group_size is None and is_last_stage is None:
        # fallback to parallel state
        cp_group_size = parallel_state.get_context_parallel_world_size()
        is_last_stage = parallel_state.is_pipeline_last_stage(
            ignore_virtual=False, vp_stage=vp_stage
        )
    else:
        assert is_last_stage is not None, "is_last_stage must be provided"
        if is_last_stage:
            assert cp_group_size is not None, "cp_group_size must be provided on last stage"

    num_tokens = torch.tensor(0, dtype=torch.int)
    if is_last_stage:
        if loss_func is None:
            forward_data_store.append(output_tensor)
        elif not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    # Protect against division by zero when all tokens are masked
                    #   in a microbatch.
                    output_tensor /= torch.clamp(num_tokens, min=1)
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor *= cp_group_size
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            cp_size_for_scaling = cp_group_size if cp_group_size is not None else 1
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale * cp_size_for_scaling / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    return output_tensor, num_tokens


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    cp_group_size,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    vp_stage=None,
    is_last_stage=True,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.
        vp_stage (int, optional):
            The virtual pipeline stage. Defaults to None.
        is_last_stage (bool, optional):
            Whether it is the last stage. Defaults to True.
            Also considering virtual stages.
            In case of PP/VPP, is_last_stage/is_vp_last_stage.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    runtime_runner = get_schedule_action_runner()
    runtime_runner.set_context(
        phase=str(runtime_runner.current_context().get("phase") or "steady"),
        microbatch_id=current_microbatch if current_microbatch is not None else runtime_runner.current_context().get("microbatch_id", -1),
        vchunk_id=vp_stage if vp_stage is not None else runtime_runner.current_context().get("vchunk_id", 0),
        lane_id=0,
    )
    invoke_schedule_runtime_hook(
        "before_forward_hook",
        {
            "microbatch_id": current_microbatch,
            "vchunk_id": vp_stage,
            "is_last_stage": bool(is_last_stage),
        },
    )
    invoke_schedule_runtime_hook(
        "memory_action_hook",
        {
            "trigger_hook": "before_forward_hook",
            "microbatch_id": current_microbatch,
            "vchunk_id": vp_stage,
            "is_last_stage": bool(is_last_stage),
            "fine_grained_activation_offloading": bool(getattr(config, "fine_grained_activation_offloading", False)),
        },
    )
    runtime_token = runtime_runner.begin_action(
        "FWD",
        microbatch_id=current_microbatch,
        vchunk_id=vp_stage,
        phase=str(runtime_runner.current_context().get("phase") or "steady"),
        lane_id=0,
        metadata={"is_last_stage": bool(is_last_stage)},
    )

    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )
    output_tensor, num_tokens = forward_step_calc_loss(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size,
        is_last_stage,
    )
    runtime_runner.end_action(
        runtime_token,
        metadata={
            "num_tokens": int(num_tokens) if num_tokens is not None else 0,
            "is_last_stage": bool(is_last_stage),
        },
    )
    invoke_schedule_runtime_hook(
        "after_forward_hook",
        {
            "microbatch_id": current_microbatch,
            "vchunk_id": vp_stage,
            "num_tokens": int(num_tokens) if num_tokens is not None else 0,
        },
    )
    invoke_schedule_runtime_hook(
        "memory_action_hook",
        {
            "trigger_hook": "after_forward_hook",
            "microbatch_id": current_microbatch,
            "vchunk_id": vp_stage,
            "num_tokens": int(num_tokens) if num_tokens is not None else 0,
            "fine_grained_activation_offloading": bool(getattr(config, "fine_grained_activation_offloading", False)),
        },
    )

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def backward_step(input_tensor, output_tensor, output_tensor_grad, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first stage)."""

    runtime_runner = get_schedule_action_runner()
    current_context = runtime_runner.current_context()
    invoke_schedule_runtime_hook(
        "before_backward_hook",
        {
            "microbatch_id": current_context.get("microbatch_id"),
            "vchunk_id": current_context.get("vchunk_id"),
        },
    )
    invoke_schedule_runtime_hook(
        "memory_action_hook",
        {
            "trigger_hook": "before_backward_hook",
            "microbatch_id": current_context.get("microbatch_id"),
            "vchunk_id": current_context.get("vchunk_id"),
            "fine_grained_activation_offloading": bool(getattr(config, "fine_grained_activation_offloading", False)),
        },
    )
    runtime_token = runtime_runner.begin_action(
        "BWD_ACT",
        microbatch_id=current_context.get("microbatch_id"),
        vchunk_id=current_context.get("vchunk_id"),
        phase=str(current_context.get("phase") or "steady"),
        lane_id=0,
    )

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # In multi-modal models like VLM, some batches may not have images.
    # When no image is present, the vision encoder (as a separate pipeline stage)
    # will not participate in the computation.
    # This results in a tensor that does not require gradients.
    # In such cases, we intentionally skip the backward pass while preserving zero gradients.
    if output_tensor[0].requires_grad:
        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    runtime_runner.end_action(
        runtime_token,
        metadata={
            "had_output_grad": bool(output_tensor_grad and output_tensor_grad[0] is not None),
        },
    )
    invoke_schedule_runtime_hook(
        "after_backward_hook",
        {
            "microbatch_id": current_context.get("microbatch_id"),
            "vchunk_id": current_context.get("vchunk_id"),
        },
    )

    return input_tensor_grad


def backward_step_multimodule(
    input_tensor: Dict[str, torch.Tensor],
    output_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output_tensor_grad: Optional[Dict[str, torch.Tensor]],
    config,
    language_model_module_name: str,
) -> Dict[str, torch.Tensor]:
    """Backward step for multi-module pipelines.

    In multi-module pipelines, tensors are organized as dictionaries with
    module names as keys. Each module's backward pass is performed independently.
    """
    # Retain gradients on all input tensors.
    for module_name, tensor in input_tensor.items():
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor is not None:
            tensor.retain_grad()

    # Last stage: output_tensor is a scalar loss from the language model.
    # Associate it with the language_model_module_name.
    if not isinstance(output_tensor, dict):
        output_tensor = {language_model_module_name: output_tensor}

    # Handle output_tensor_grad: None (last stage) or dict (intermediate stages).
    if not output_tensor_grad:
        output_tensor_grad = {key: None for key in output_tensor.keys()}

    # Apply grad scaling if needed (for last stage only).
    for module_name in output_tensor.keys():
        if output_tensor_grad[module_name] is None and config.grad_scale_func is not None:
            output_tensor[module_name] = config.grad_scale_func(output_tensor[module_name])

    # Perform backward pass for each module.
    for module_name in output_tensor.keys():
        output_tensor_module = output_tensor[module_name]
        output_tensor_grad_module = output_tensor_grad[module_name]

        # In multi-modal models like VLM, some batches may not have images.
        # In such cases, skip backward while preserving zero gradients.
        if output_tensor_module is not None and output_tensor_module.requires_grad:
            if config.deallocate_pipeline_outputs:
                custom_backward(output_tensor_module, output_tensor_grad_module)
            else:
                torch.autograd.backward(
                    output_tensor_module, grad_tensors=output_tensor_grad_module
                )

    # Collect gradients for input tensors.
    input_tensor_grad = {}
    for module_name, tensor in input_tensor.items():
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor is None:
            input_tensor_grad[module_name] = None
        else:
            input_tensor_grad[module_name] = tensor.grad

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    """Check if it is the first validation step."""
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: Optional[int] = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,  # unused
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run forward and backward passes with no pipeline parallelism"""

    runtime_runner = get_schedule_action_runner()
    runtime_runner.set_phase("steady")

    if pg_collection is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif pg_collection is not None:
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for non-pipeline-parallel schedule"

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if config.overlap_moe_expert_parallel_comm and not forward_only:
        forward_data_store, total_num_tokens = combined_1f1b_schedule_for_no_pipelining(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            output_tensor_grad,
            forward_data_store,
            config,
            collect_non_loss_data,
            first_val_step,
            forward_only,
            no_sync_func,
            total_num_tokens,
            partial(check_first_val_step, first_val_step, forward_only),
        )
    elif config.hybrid_context_parallel:
        forward_data_store, total_num_tokens = hybrid_context_parallel_forward_backward(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            output_tensor_grad,
            forward_data_store,
            config,
            collect_non_loss_data,
            first_val_step,
            forward_only,
            no_sync_func,
            total_num_tokens,
            check_first_val_step,
            model_type,
        )
    else:
        with no_sync_func():
            for i in range(num_microbatches - 1):
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    pg_collection.cp.size(),
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                    current_microbatch=i,
                )
                total_num_tokens += num_tokens
                if not forward_only:
                    backward_step(input_tensor, output_tensor, output_tensor_grad, config)
        # Run computation for last microbatch out of context handler (want to
        # synchronize gradients).
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            pg_collection.cp.size(),
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, num_microbatches == 1
            ),
            current_microbatch=num_microbatches - 1,
        )

        total_num_tokens += num_tokens

        if not forward_only:
            backward_step(input_tensor, output_tensor, output_tensor_grad, config)

    if config.finalize_model_grads_func is not None and not forward_only:
        invoke_schedule_runtime_hook(
            "before_optimizer_tail_hook",
            {"op_name": "finalize_model_grads"},
        )
        finalize_token = runtime_runner.begin_action(
            "WGRAD_OPT",
            phase="cooldown",
            lane_id=0,
            metadata={"op_name": "finalize_model_grads"},
        )
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )
        runtime_runner.end_action(finalize_token, metadata={"op_name": "finalize_model_grads"})

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()

    runtime_runner.flush()

    return forward_data_store


def clear_embedding_activation_buffer(config, model, is_last_stage):
    """Clear embedding activation buffer."""

    if is_last_stage and config.defer_embedding_wgrad_compute:
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(
                model[-1], 'post_process', return_model_obj=True
            )
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)

        # Need to ensure no stray activations exists in this buffer
        embedding_module.embedding_activation_buffer.clear()

        return embedding_module
    else:
        return None


def finish_embedding_wgrad_compute(config, embedding_module, is_last_stage, tp_group):
    """Finish embedding wgrad compute."""
    if is_last_stage and config.defer_embedding_wgrad_compute:
        runtime_runner = get_schedule_action_runner()
        invoke_schedule_runtime_hook(
            "before_optimizer_tail_hook",
            {"op_name": "finish_embedding_wgrad_compute"},
        )
        runtime_token = runtime_runner.begin_action(
            "WGRAD_OPT",
            phase=str(runtime_runner.current_context().get("phase") or "cooldown"),
            lane_id=0,
            metadata={"op_name": "finish_embedding_wgrad_compute"},
        )
        embedding_activation_buffer = embedding_module.embedding_activation_buffer
        grad_output_buffer = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )

        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight, tp_group
        )
        runtime_runner.end_action(runtime_token, metadata={"op_name": "finish_embedding_wgrad_compute"})


def get_pp_rank_microbatches(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    forward_only=False,
    overlap_moe_expert_parallel_comm=False,
    p2p_communicator: Optional[P2PCommunicator] = None,
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
    if p2p_communicator is not None:
        pipeline_parallel_size = p2p_communicator.pp_group.size()
        pipeline_parallel_rank = p2p_communicator.pp_group.rank()
        virtual_pipeline_parallel_size = p2p_communicator.virtual_pipeline_model_parallel_size
    else:
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        virtual_pipeline_parallel_size = (
            parallel_state.get_virtual_pipeline_model_parallel_world_size()
        )

    total_num_microbatches = num_microbatches * num_model_chunks
    are_all_microbatches_in_warmup = False

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    elif pipeline_parallel_size > 1:
        if virtual_pipeline_parallel_size is None:
            # forward_backward_pipelining_without_interleaving
            num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:
            # forward_backward_pipelining_with_interleaving
            # Run (num_model_chunks-1)*microbatch_group_size_per_vp_stage on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
            # When enabling overlap_moe_expert_parallel_comm, we schedule one extra micro-batch
            # forward step before the 1f1b stages. This is needed to ensure the forward
            # and backward computations are independent in all 1f1b steps.
            if overlap_moe_expert_parallel_comm:
                num_warmup_microbatches = num_warmup_microbatches + 1
    else:
        # forward_backward_no_pipelining
        # This path is only used for cuda graph capturing compatibility for the PP=1 case.
        num_warmup_microbatches = 0

    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )


def _edge_interleave(values: List[int]) -> List[int]:
    ordered: List[int] = []
    left = 0
    right = len(values) - 1
    while left <= right:
        ordered.append(values[left])
        left += 1
        if left <= right:
            ordered.append(values[right])
            right -= 1
    return ordered


def _center_out_order(num_items: int) -> List[int]:
    if num_items <= 0:
        return []
    middle = num_items // 2
    right = list(range(middle, num_items))
    left = list(range(middle - 1, -1, -1))
    return right + left


def _resolve_group_phase(group_index: int, num_groups: int) -> str:
    if num_groups <= 1:
        return "steady"
    if group_index == 0:
        return "warmup"
    if group_index == num_groups - 1:
        return "cooldown"
    return "steady"


def _base_model_chunk_order(num_model_chunks: int, template: str) -> List[int]:
    if template == "pp4_middle_relief" and num_model_chunks > 1:
        if num_model_chunks == 2:
            return [1, 0]
        return _center_out_order(num_model_chunks)
    return list(range(num_model_chunks))


def _parse_structure_priority_hints(raw: str, num_model_chunks: int) -> Optional[List[int]]:
    if not str(raw or "").strip():
        return None
    parsed: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            return None
    if len(parsed) != num_model_chunks:
        return None
    return parsed


def _pipeline_parallel_location() -> tuple[Optional[int], Optional[int]]:
    if not parallel_state.model_parallel_is_initialized():
        return None, None
    try:
        pp_rank = int(parallel_state.get_pipeline_model_parallel_rank())
        pp_world_size = int(parallel_state.get_pipeline_model_parallel_world_size())
    except Exception:
        return None, None
    if pp_world_size <= 0:
        return None, None
    return pp_rank, pp_world_size


def _parse_stage_chunk_priority_hints(raw: str, num_model_chunks: int) -> Dict[int, List[int]]:
    parsed: Dict[int, List[int]] = {}
    text = str(raw or "").strip()
    if not text:
        return parsed
    for stage_entry in text.split(";"):
        stage_entry = stage_entry.strip()
        if not stage_entry or ":" not in stage_entry:
            continue
        stage_token, priorities_token = stage_entry.split(":", 1)
        try:
            stage_id = int(stage_token.strip())
        except ValueError:
            continue
        priorities = _parse_structure_priority_hints(priorities_token, num_model_chunks)
        if priorities is not None:
            parsed[int(stage_id)] = priorities
    return parsed


def _parse_json_hint_dict(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _parse_json_hint_list(raw: str) -> List[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _dedupe_json_dict_list(items: Any) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        try:
            encoded = json.dumps(item, sort_keys=True, separators=(",", ":"))
        except Exception:
            continue
        if encoded in seen:
            continue
        seen.add(encoded)
        deduped.append(dict(item))
    return deduped


def _parse_stage_chunk_priority_hint_map(raw: str) -> Dict[int, List[int]]:
    parsed: Dict[int, List[int]] = {}
    text = str(raw or "").strip()
    if not text:
        return parsed
    for stage_entry in text.split(";"):
        stage_entry = stage_entry.strip()
        if not stage_entry or ":" not in stage_entry:
            continue
        stage_token, priorities_token = stage_entry.split(":", 1)
        try:
            stage_id = int(stage_token.strip())
        except ValueError:
            continue
        priorities: List[int] = []
        for raw_priority in priorities_token.split(","):
            token = str(raw_priority).strip()
            if not token:
                continue
            try:
                priorities.append(int(token))
            except ValueError:
                continue
        if priorities:
            parsed[int(stage_id)] = list(dict.fromkeys(priorities))
    return parsed


def _normalize_runtime_state_migration_hints(items: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip().lower()
        target_stage_ids: List[int] = []
        for raw_stage in list(item.get("target_stage_ids") or []):
            try:
                target_stage_ids.append(int(raw_stage))
            except Exception:
                continue
        target_stage_ids = list(dict.fromkeys(target_stage_ids))
        target_layer_group_ids = [
            str(raw).strip()
            for raw in list(item.get("target_layer_group_ids") or [])
            if str(raw).strip()
        ]
        target_state_ids = [
            str(raw).strip()
            for raw in list(item.get("target_state_ids") or [])
            if str(raw).strip()
        ]
        payload: Dict[str, Any] = {
            "action": action,
            "target_stage_ids": target_stage_ids,
            "target_layer_group_ids": list(dict.fromkeys(target_layer_group_ids)),
            "target_state_ids": list(dict.fromkeys(target_state_ids)),
            "direction": str(item.get("direction") or "").strip(),
        }
        shift_unit = str(item.get("shift_unit") or "").strip()
        if shift_unit:
            payload["shift_unit"] = shift_unit
        try:
            offset_slots = int(item.get("offset_slots", 0) or 0)
        except Exception:
            offset_slots = 0
        if offset_slots:
            payload["offset_slots"] = offset_slots
        try:
            prefetch_distance_slots = int(item.get("prefetch_distance_slots", 0) or 0)
        except Exception:
            prefetch_distance_slots = 0
        if prefetch_distance_slots:
            payload["prefetch_distance_slots"] = prefetch_distance_slots
        insert_before = [
            str(raw).strip()
            for raw in list(item.get("insert_before") or [])
            if str(raw).strip()
        ]
        if insert_before:
            payload["insert_before"] = list(dict.fromkeys(insert_before))
        if payload["action"] and payload["target_stage_ids"]:
            normalized.append(payload)
    return _dedupe_json_dict_list(normalized)


def _normalize_stage_chunk_priority_hint_map(payload: Any) -> Dict[int, List[int]]:
    normalized: Dict[int, List[int]] = {}
    for raw_stage_id, raw_hints in dict(payload or {}).items():
        try:
            stage_id = int(raw_stage_id)
        except Exception:
            continue
        parsed_hints: List[int] = []
        for raw_hint in list(raw_hints or []):
            try:
                parsed_hints.append(int(raw_hint))
            except Exception:
                continue
        if parsed_hints:
            normalized[int(stage_id)] = list(dict.fromkeys(parsed_hints))
    return normalized


def _merge_stage_chunk_priority_hint_maps(*maps: Dict[int, List[int]]) -> Dict[int, List[int]]:
    merged: Dict[int, List[int]] = {}
    for mapping in maps:
        for stage_id, raw_hints in dict(mapping or {}).items():
            try:
                parsed_stage_id = int(stage_id)
            except Exception:
                continue
            existing = list(merged.get(parsed_stage_id) or [])
            parsed_hints = list(existing)
            for raw_hint in list(raw_hints or []):
                try:
                    parsed_hints.append(int(raw_hint))
                except Exception:
                    continue
            if parsed_hints:
                merged[parsed_stage_id] = list(dict.fromkeys(parsed_hints))
    return merged


def _parse_runtime_repair_action_contracts(raw: str) -> List[Dict[str, Any]]:
    return _parse_json_hint_list(raw)


def _aggregate_runtime_repair_contracts(runtime_repair_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {
        "state_migration_hints": [],
        "window_overrides": [],
        "operator_cluster_overrides": [],
        "chunk_priority_hints": {},
        "overlap_channels": [],
        "optimizer_runtime": {},
        "memory_intents": {},
        "state_plan_patch": {},
    }
    for action_contract in list(runtime_repair_actions or []):
        lowering = dict(action_contract.get("compile_lowering") or {})
        aggregated["state_migration_hints"].extend(list(lowering.get("state_migration_hints") or []))
        aggregated["window_overrides"].extend(list(lowering.get("window_overrides") or []))
        aggregated["operator_cluster_overrides"].extend(list(lowering.get("operator_cluster_overrides") or []))
        aggregated["overlap_channels"].extend(
            [
                str(item).strip()
                for item in list(lowering.get("overlap_channels") or [])
                if str(item).strip()
            ]
        )
        aggregated["memory_intents"].update(dict(lowering.get("memory_intents") or {}))
        aggregated["state_plan_patch"].update(dict(lowering.get("state_plan_patch") or {}))
        if dict(lowering.get("optimizer_runtime") or {}):
            aggregated["optimizer_runtime"].update(dict(lowering.get("optimizer_runtime") or {}))
        aggregated["chunk_priority_hints"] = _merge_stage_chunk_priority_hint_maps(
            aggregated.get("chunk_priority_hints") or {},
            _normalize_stage_chunk_priority_hint_map(lowering.get("chunk_priority_hints") or {}),
        )
    aggregated["state_migration_hints"] = _normalize_runtime_state_migration_hints(
        aggregated.get("state_migration_hints") or []
    )
    aggregated["window_overrides"] = _dedupe_json_dict_list(aggregated.get("window_overrides") or [])
    aggregated["operator_cluster_overrides"] = _dedupe_json_dict_list(
        aggregated.get("operator_cluster_overrides") or []
    )
    aggregated["overlap_channels"] = sorted(
        {
            str(item).strip()
            for item in list(aggregated.get("overlap_channels") or [])
            if str(item).strip()
        }
    )
    return aggregated


def _parse_stage_tag_tokens(raw: str) -> List[str]:
    return [
        token.strip()
        for token in str(raw or "").replace(",", "|").split("|")
        if token.strip()
    ]


def _merge_string_hint_payload(
    target: Dict[str, str],
    source: Dict[str, str],
    *,
    overwrite: bool,
) -> Dict[str, str]:
    merged = dict(target or {})
    for key, value in dict(source or {}).items():
        token_key = str(key).strip()
        token_value = str(value or "").strip()
        if not token_key or not token_value:
            continue
        if token_key == "stage_tags":
            merged["stage_tags"] = "|".join(
                sorted(
                    {
                        *set(_parse_stage_tag_tokens(merged.get("stage_tags") or "")),
                        *set(_parse_stage_tag_tokens(token_value)),
                    }
                )
            )
            continue
        if overwrite or not str(merged.get(token_key) or "").strip():
            merged[token_key] = token_value
    return merged


def _runtime_state_migration_semantics_by_stage(
    state_migration_hints: List[Dict[str, Any]]
) -> Dict[int, Dict[str, str]]:
    semantics: Dict[int, Dict[str, str]] = {}
    for hint in list(state_migration_hints or []):
        action = str(hint.get("action") or "").strip().lower()
        direction = str(hint.get("direction") or "").strip()
        offset_slots = int(hint.get("offset_slots") or 0)
        prefetch_distance_slots = int(hint.get("prefetch_distance_slots") or 0)
        for stage_id in list(hint.get("target_stage_ids") or []):
            try:
                parsed_stage_id = int(stage_id)
            except Exception:
                continue
            entry = dict(semantics.get(parsed_stage_id) or {})
            stage_tags = set(_parse_stage_tag_tokens(entry.get("stage_tags") or ""))
            stage_tags.add("state_migration_active")
            if action == "offload_timing_shift":
                stage_tags.add("memory_hotspot")
                entry.setdefault("memory_policy_mode", "selective")
                if direction:
                    entry["offload_shift_direction"] = direction
                if offset_slots:
                    entry["offload_shift_offset_slots"] = str(offset_slots)
            elif action == "selective_reload_prefetch":
                stage_tags.add("reload_sensitive")
                entry.setdefault("reload_policy", "selective_prefetch")
                entry.setdefault("prefetch_policy", "selective")
                if prefetch_distance_slots:
                    entry["prefetch_distance_slots"] = str(prefetch_distance_slots)
            entry["stage_tags"] = "|".join(sorted(stage_tags))
            semantics[parsed_stage_id] = entry
    return semantics


def _merge_stage_semantic_maps(
    base: Dict[int, Dict[str, str]],
    extra: Dict[int, Dict[str, str]],
) -> Dict[int, Dict[str, str]]:
    merged: Dict[int, Dict[str, str]] = {int(key): dict(value) for key, value in dict(base or {}).items()}
    for raw_stage_id, payload in dict(extra or {}).items():
        try:
            stage_id = int(raw_stage_id)
        except Exception:
            continue
        merged[stage_id] = _merge_string_hint_payload(
            dict(merged.get(stage_id) or {}),
            {str(key): str(value) for key, value in dict(payload or {}).items()},
            overwrite=False,
        )
    return merged


def _parse_int_vector_hint(raw: str) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, list):
        parsed: List[int] = []
        for item in payload:
            try:
                parsed.append(max(int(item), 1))
            except Exception:
                continue
        return parsed
    parsed: List[int] = []
    for token in text.replace("|", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(max(int(token), 1))
        except Exception:
            continue
    return parsed


def _parse_schedule_grid_spec(raw: str) -> Dict[str, Any]:
    payload = _parse_json_hint_dict(raw)
    if not payload:
        return {}
    try:
        payload["lanes"] = max(int(payload.get("lanes", 1) or 1), 1)
        payload["time_slots"] = max(int(payload.get("time_slots", 0) or 0), 0)
        payload["stage_count"] = max(int(payload.get("stage_count", 1) or 1), 1)
        payload["vstage_count"] = max(int(payload.get("vstage_count", 1) or 1), 1)
        payload["microbatch_count"] = max(int(payload.get("microbatch_count", 1) or 1), 1)
        payload["family"] = str(payload.get("family", "fixed_1f1b") or "fixed_1f1b").strip()
        payload["weight_version_policy"] = str(payload.get("weight_version_policy", "default") or "default").strip()
        payload["constraints"] = dict(payload.get("constraints") or {})
        payload["notes"] = [str(item) for item in (payload.get("notes") or []) if str(item).strip()]
    except Exception:
        return {}
    parsed_cells: List[Dict[str, Any]] = []
    for item in list(payload.get("cells") or []):
        if not isinstance(item, dict):
            continue
        parsed_cells.append(
            {
                "kind": str(item.get("kind") or "BUBBLE").strip().upper(),
                "stage_id": max(int(item.get("stage_id", 0) or 0), 0),
                "lane_id": max(int(item.get("lane_id", 0) or 0), 0),
                "microbatch_id": int(item.get("microbatch_id", -1) or -1),
                "vchunk_id": max(int(item.get("vchunk_id", 0) or 0), 0),
                "time_slot": max(int(item.get("time_slot", 0) or 0), 0),
                "stream_or_channel": str(item.get("stream_or_channel") or "").strip() or None,
                "weight_version_tag": str(item.get("weight_version_tag") or "").strip() or None,
            }
        )
    payload["cells"] = parsed_cells
    return payload


def _parse_schedule_action_specs(raw: str) -> List[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    parsed: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        parsed.append(
            {
                "action_type": str(item.get("action_type") or "WAIT").strip().upper(),
                "stage_id": max(int(item.get("stage_id", 0) or 0), 0),
                "lane_id": max(int(item.get("lane_id", 0) or 0), 0),
                "microbatch_id": max(int(item.get("microbatch_id", 0) or 0), 0),
                "vchunk_id": max(int(item.get("vchunk_id", 0) or 0), 0),
                "time_slot": max(int(item.get("time_slot", 0) or 0), 0),
                "duration_hint": max(float(item.get("duration_hint", 0.0) or 0.0), 0.0),
                "dependency_ids": [str(dep) for dep in list(item.get("dependency_ids") or []) if str(dep).strip()],
                "memory_delta": float(item.get("memory_delta", 0.0) or 0.0),
                "stream_or_channel": str(item.get("stream_or_channel") or "").strip() or None,
                "weight_version_tag": str(item.get("weight_version_tag") or "").strip() or None,
            }
        )
    return parsed


def _parse_stage_family_hints(raw: str) -> Dict[int, Dict[str, str]]:
    parsed: Dict[int, Dict[str, str]] = {}
    text = str(raw or "").strip()
    if not text:
        return parsed
    for stage_entry in text.split(";"):
        stage_entry = stage_entry.strip()
        if not stage_entry:
            continue
        tokens = [token.strip() for token in stage_entry.split(",") if token.strip()]
        if not tokens:
            continue
        try:
            stage_id = int(tokens[0])
        except ValueError:
            continue
        payload: Dict[str, str] = {}
        for token in tokens[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                payload[key] = value
        if payload:
            parsed[int(stage_id)] = payload
    return parsed


def _parse_stage_semantic_hints(raw: str) -> Dict[int, Dict[str, str]]:
    parsed: Dict[int, Dict[str, str]] = {}
    payload = _parse_json_hint_dict(raw)
    if payload:
        for key, value in payload.items():
            try:
                stage_id = int(key)
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            parsed[int(stage_id)] = {
                str(item_key): str(item_value)
                for item_key, item_value in dict(value).items()
                if str(item_key).strip() and str(item_value).strip()
            }
        return parsed
    text = str(raw or "").strip()
    if not text:
        return parsed
    try:
        payload_list = json.loads(text)
    except Exception:
        return parsed
    if not isinstance(payload_list, list):
        return parsed
    for item in payload_list:
        if not isinstance(item, dict):
            continue
        try:
            stage_id = int(item.get("stage_id"))
        except Exception:
            continue
        parsed[int(stage_id)] = {
            str(item_key): str(item_value)
            for item_key, item_value in dict(item).items()
            if str(item_key).strip()
            and str(item_key) != "stage_id"
            and str(item_value).strip()
        }
    return parsed


def _parse_schedule_runtime_hints() -> Dict[str, Any]:
    runtime_repair_actions = _parse_runtime_repair_action_contracts(
        os.environ.get("RUNTIME_REPAIR_ACTIONS", "")
    )
    runtime_repair_recommendations = _parse_runtime_repair_action_contracts(
        os.environ.get("RUNTIME_REPAIR_RECOMMENDATIONS", "")
    )
    runtime_repair_summary = _parse_json_hint_dict(os.environ.get("RUNTIME_REPAIR_SUMMARY", ""))
    runtime_repair_score_weights = _parse_json_hint_dict(
        os.environ.get("RUNTIME_REPAIR_SCORE_WEIGHTS", "")
    )
    runtime_repair_policy_table = _parse_json_hint_dict(
        os.environ.get("RUNTIME_REPAIR_POLICY_TABLE", "")
    )
    runtime_repair_aggregated = _aggregate_runtime_repair_contracts(runtime_repair_actions)
    telemetry_budget = _parse_json_hint_dict(os.environ.get("TELEMETRY_BUDGET", ""))
    if not isinstance(telemetry_budget, dict):
        telemetry_budget = {}
    explicit_level = str(os.environ.get("SCHEDULE_RUNTIME_TRACE_LEVEL", "") or "").strip()
    explicit_max_mb = str(os.environ.get("SCHEDULE_RUNTIME_TRACE_MAX_MB", "") or "").strip()
    explicit_max_events = str(os.environ.get("SCHEDULE_RUNTIME_TRACE_MAX_EVENTS", "") or "").strip()
    explicit_sampled_windows = str(os.environ.get("SCHEDULE_RUNTIME_TRACE_SAMPLED_WINDOWS", "") or "").strip()
    if explicit_level:
        telemetry_budget["level"] = explicit_level
    if explicit_max_mb:
        try:
            telemetry_budget["max_trace_mb"] = int(explicit_max_mb)
        except Exception:
            pass
    if explicit_max_events:
        try:
            telemetry_budget["max_events_per_rank"] = int(explicit_max_events)
        except Exception:
            pass
    if explicit_sampled_windows:
        try:
            telemetry_budget["sampled_windows"] = int(explicit_sampled_windows)
        except Exception:
            pass
    if not telemetry_budget:
        telemetry_budget = {
            "level": "summary",
            "max_trace_mb": 128,
            "max_events_per_rank": 20000,
            "sampled_windows": 2,
            "emit_compare_svg": False,
        }
    state_plan = _parse_json_hint_dict(os.environ.get("STATE_PLAN", ""))
    offload_plan = _parse_json_hint_dict(os.environ.get("OFFLOAD_PLAN", ""))
    reload_plan = _parse_json_hint_dict(os.environ.get("RELOAD_PLAN", ""))
    comm_chunk_plan = _parse_json_hint_dict(os.environ.get("COMM_CHUNK_PLAN", ""))
    overlap_hints = _parse_json_hint_dict(os.environ.get("SCHEDULE_OVERLAP_HINTS", ""))
    memory_hints = _parse_json_hint_dict(os.environ.get("SCHEDULE_MEMORY_HINTS", ""))
    partition_hints = _parse_json_hint_dict(os.environ.get("SCHEDULE_PARTITION_HINTS", ""))
    state_plan.update(dict(runtime_repair_aggregated.get("state_plan_patch") or {}))

    explicit_state_migration_hints = _parse_json_hint_list(
        os.environ.get("SCHEDULE_STATE_MIGRATION_HINTS", "")
    )
    state_migration_hints = _normalize_runtime_state_migration_hints(
        [
            *list(explicit_state_migration_hints or []),
            *list(state_plan.get("runtime_state_migration_hints") or []),
            *list(offload_plan.get("runtime_state_migration_hints") or []),
            *list(reload_plan.get("runtime_state_migration_hints") or []),
            *list(runtime_repair_aggregated.get("state_migration_hints") or []),
        ]
    )
    if state_migration_hints:
        state_plan["runtime_state_migration_hints"] = list(state_migration_hints)
        offload_plan["runtime_state_migration_hints"] = [
            dict(item)
            for item in state_migration_hints
            if str(item.get("action") or "").strip() == "offload_timing_shift"
        ]
        reload_plan["runtime_state_migration_hints"] = [
            dict(item)
            for item in state_migration_hints
            if str(item.get("action") or "").strip() == "selective_reload_prefetch"
        ]
    if "reload_prefetch_window" in state_plan and "prefetch_window" not in reload_plan:
        try:
            reload_plan["prefetch_window"] = int(state_plan.get("reload_prefetch_window") or 0)
        except Exception:
            pass

    merged_window_overrides = _parse_window_override_hints(
        json.dumps(
            _dedupe_json_dict_list(
                [
                    *_parse_json_hint_list(os.environ.get("SCHEDULE_WINDOW_OVERRIDE_HINTS", "")),
                    *list(runtime_repair_aggregated.get("window_overrides") or []),
                ]
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    merged_operator_cluster_hints_by_stage = _parse_operator_cluster_hints(
        json.dumps(
            _dedupe_json_dict_list(
                [
                    *_parse_json_hint_list(os.environ.get("SCHEDULE_OPERATOR_CLUSTER_HINTS", "")),
                    *list(runtime_repair_aggregated.get("operator_cluster_overrides") or []),
                ]
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    merged_operator_cluster_hints = [
        dict(item)
        for hints in merged_operator_cluster_hints_by_stage.values()
        for item in list(hints or [])
    ]
    runtime_chunk_priority_hints = _normalize_stage_chunk_priority_hint_map(
        dict(comm_chunk_plan.get("runtime_chunk_priority_hints") or {})
    )
    stage_chunk_priority_hints = _merge_stage_chunk_priority_hint_maps(
        _parse_stage_chunk_priority_hint_map(os.environ.get("SCHEDULE_STAGE_CHUNK_PRIORITY_HINTS", "")),
        runtime_chunk_priority_hints,
        dict(runtime_repair_aggregated.get("chunk_priority_hints") or {}),
    )
    if stage_chunk_priority_hints:
        comm_chunk_plan["runtime_chunk_priority_hints"] = {
            str(stage_id): list(hints) for stage_id, hints in stage_chunk_priority_hints.items()
        }

    overlap_channels = {
        str(item).strip()
        for item in list(runtime_repair_aggregated.get("overlap_channels") or [])
        if str(item).strip()
    }
    if "reload" in overlap_channels:
        overlap_hints["enable_reload_overlap"] = True
    if "optimizer_tail" in overlap_channels:
        overlap_hints["enable_optimizer_tail_overlap"] = True
        overlap_hints["enable_grad_reduce_overlap"] = True
        overlap_hints["enable_param_gather_overlap"] = True
    if overlap_channels:
        existing_pairs = [
            str(item).strip()
            for item in list(overlap_hints.get("priority_frontier_pairs") or [])
            if str(item).strip()
        ]
        overlap_hints["priority_frontier_pairs"] = list(
            dict.fromkeys(existing_pairs + [f"channel:{item}" for item in sorted(overlap_channels)])
        )
        overlap_hints["status"] = "runtime_repair"

    memory_hints.update(dict(runtime_repair_aggregated.get("memory_intents") or {}))
    if state_migration_hints:
        existing_stage_policies = [
            dict(item)
            for item in list(memory_hints.get("per_stage_policies") or [])
            if isinstance(item, dict)
        ]
        for hint in state_migration_hints:
            existing_stage_policies.append(
                {
                    "action": str(hint.get("action") or ""),
                    "target_stage_ids": [int(item) for item in list(hint.get("target_stage_ids") or [])],
                    "target_layer_group_ids": [
                        str(item) for item in list(hint.get("target_layer_group_ids") or []) if str(item).strip()
                    ],
                    "target_state_ids": [
                        str(item) for item in list(hint.get("target_state_ids") or []) if str(item).strip()
                    ],
                    "direction": str(hint.get("direction") or ""),
                    "offset_slots": int(hint.get("offset_slots") or 0),
                    "prefetch_distance_slots": int(hint.get("prefetch_distance_slots") or 0),
                }
            )
        memory_hints["per_stage_policies"] = _dedupe_json_dict_list(existing_stage_policies)
        memory_hints["status"] = "runtime_repair"

    stage_semantic_hints = _merge_stage_semantic_maps(
        _parse_stage_semantic_hints(os.environ.get("SCHEDULE_STAGE_SEMANTIC_HINTS", "")),
        _runtime_state_migration_semantics_by_stage(state_migration_hints),
    )
    return {
        "family": str(os.environ.get("SCHEDULE_FAMILY", "") or "").strip(),
        "dispatch_order": str(os.environ.get("SCHEDULE_DISPATCH_ORDER", "") or "").strip(),
        "lane_policy": str(os.environ.get("SCHEDULE_LANE_POLICY", "") or "").strip(),
        "group_size_vector": _parse_int_vector_hint(os.environ.get("SCHEDULE_GROUP_SIZE_VECTOR", "")),
        "schedule_grid_spec": _parse_schedule_grid_spec(os.environ.get("SCHEDULE_GRID_SPEC", "")),
        "schedule_action_specs": _parse_schedule_action_specs(os.environ.get("SCHEDULE_ACTION_SPECS", "")),
        "schedule_node_specs": _parse_json_hint_list(os.environ.get("SCHEDULE_NODE_SPECS", "")),
        "schedule_edge_specs": _parse_json_hint_list(os.environ.get("SCHEDULE_EDGE_SPECS", "")),
        "runtime_repair_summary": runtime_repair_summary,
        "runtime_repair_score_weights": runtime_repair_score_weights,
        "runtime_repair_policy_table": runtime_repair_policy_table,
        "runtime_repair_actions": runtime_repair_actions,
        "runtime_repair_recommendations": runtime_repair_recommendations,
        "state_plan": state_plan,
        "offload_plan": offload_plan,
        "reload_plan": reload_plan,
        "comm_chunk_plan": comm_chunk_plan,
        "window_reconfig_plan": _parse_json_hint_dict(os.environ.get("WINDOW_RECONFIG_PLAN", "")),
        "telemetry_budget": telemetry_budget,
        "stage_semantic_hints": stage_semantic_hints,
        "state_migration_hints": state_migration_hints,
        "window_override_hints": merged_window_overrides,
        "operator_cluster_hints": merged_operator_cluster_hints,
        "operator_cluster_hints_by_stage": merged_operator_cluster_hints_by_stage,
        "stage_chunk_priority_hints": stage_chunk_priority_hints,
        "overlap_hints": overlap_hints,
        "memory_hints": memory_hints,
        "partition_hints": partition_hints,
        "stage_local_vpp_vector": _parse_int_vector_hint(os.environ.get("STAGE_LOCAL_VPP_VECTOR", "")),
        "supported_families": sorted(_ALL_SCHEDULE_FAMILIES),
        "available_hooks": list(_SCHEDULE_RUNTIME_HOOKS),
    }


def _get_local_stage_family_hint() -> Dict[str, str]:
    pp_rank, _ = _pipeline_parallel_location()
    if pp_rank is None:
        return {}
    parsed = _parse_stage_family_hints(os.environ.get("SCHEDULE_STAGE_FAMILY_HINTS", ""))
    merged = dict(parsed.get(int(pp_rank)) or {})
    runtime_hints = _parse_schedule_runtime_hints()
    semantic_hints = dict(runtime_hints.get("stage_semantic_hints") or {})
    local_semantic = semantic_hints.get(int(pp_rank))
    if isinstance(local_semantic, dict):
        merged = _merge_string_hint_payload(
            merged,
            {str(key): str(value) for key, value in dict(local_semantic).items()},
            overwrite=True,
        )
    overlap_hints = dict(runtime_hints.get("overlap_hints") or {})
    memory_hints = dict(runtime_hints.get("memory_hints") or {})
    partition_hints = dict(runtime_hints.get("partition_hints") or {})
    if "enable_p2p_overlap" in overlap_hints:
        merged.setdefault(
            "p2p_policy",
            "overlap" if bool(overlap_hints.get("enable_p2p_overlap")) else "serial",
        )
    if bool(overlap_hints.get("enable_optimizer_tail_overlap")):
        merged.setdefault("optimizer_runtime_mode", "tail_guarded_overlap")
    checkpoint_policy = str(memory_hints.get("checkpoint_policy") or "").strip()
    if checkpoint_policy:
        merged.setdefault("checkpoint_policy", checkpoint_policy)
    offload_policy = str(memory_hints.get("offload_policy") or "").strip()
    if offload_policy:
        merged.setdefault("memory_policy_mode", offload_policy)
    reload_policy = str(memory_hints.get("reload_policy") or "").strip()
    if reload_policy:
        merged.setdefault("reload_policy", reload_policy)
    prefetch_policy = str(memory_hints.get("prefetch_policy") or "").strip()
    if prefetch_policy:
        merged.setdefault("prefetch_policy", prefetch_policy)
    partition_mode = str(partition_hints.get("partition_mode") or "").strip()
    if partition_mode:
        merged.setdefault("partition_mode", partition_mode)
    return merged


def _resolve_local_group_size(default_group_size: int) -> int:
    group_size_vector = [
        max(int(item), 1)
        for item in list(_parse_schedule_runtime_hints().get("group_size_vector") or [])
    ]
    if not group_size_vector:
        return max(int(default_group_size or 1), 1)
    pp_rank, _ = _pipeline_parallel_location()
    if pp_rank is None:
        return int(group_size_vector[0])
    if int(pp_rank) < len(group_size_vector):
        return int(group_size_vector[int(pp_rank)])
    return int(group_size_vector[-1])


def get_schedule_runtime_policy() -> Dict[str, Any]:
    policy = _parse_schedule_runtime_hints()
    local_stage_hint = _get_local_stage_family_hint()
    pp_rank, _ = _pipeline_parallel_location()
    policy["local_stage_index"] = int(pp_rank) if pp_rank is not None else None
    policy["local_stage_hint"] = dict(local_stage_hint or {})
    return policy


def get_schedule_family_registry() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for family in sorted(_ALL_SCHEDULE_FAMILIES):
        if family == "fixed_1f1b":
            warmup_rule = "fill_pipeline"
            steady_rule = "1f1b"
            cooldown_rule = "drain_pipeline"
        elif family == "interleaved":
            warmup_rule = "fill_interleaved"
            steady_rule = "grouped_interleave"
            cooldown_rule = "drain_interleaved"
        elif family in {"zero_bubble", "zbv", "v_half", "v_min"}:
            warmup_rule = "bubble_fill"
            steady_rule = "zero_bubble_proxy"
            cooldown_rule = "optimizer_tail_hide"
        elif family == "dualpipe_v":
            warmup_rule = "dualpipe_fill"
            steady_rule = "dualpipe_overlap"
            cooldown_rule = "dualpipe_drain"
        else:
            warmup_rule = "custom"
            steady_rule = "custom"
            cooldown_rule = "custom"
        registry[family] = {
            "family": family,
            "warmup_rule": warmup_rule,
            "steady_rule": steady_rule,
            "cooldown_rule": cooldown_rule,
            "weight_version_policy": "delayed_wgrad" if family in {"zero_bubble", "zbv", "v_half", "v_min", "dualpipe_v"} else "default",
            "family_specific_constraints": [],
            "execution_mode": "production" if family in _PRODUCTION_SCHEDULE_FAMILIES else "adapter",
        }
    return registry


def get_schedule_action_view() -> Dict[str, Any]:
    policy = get_schedule_runtime_policy()
    grid_spec = dict(policy.get("schedule_grid_spec") or {})
    action_specs = list(policy.get("schedule_action_specs") or [])
    return {
        "family": str(policy.get("family") or ""),
        "grid_spec": grid_spec,
        "action_specs": action_specs,
        "action_count": int(len(action_specs)),
        "available_hooks": list(policy.get("available_hooks") or []),
    }


def _memory_action_runtime_enabled(payload: Optional[Dict[str, Any]] = None) -> bool:
    event = dict(payload or {})
    explicit = event.get("fine_grained_activation_offloading")
    if explicit is not None:
        return bool(explicit)
    raw = str(os.environ.get("ENABLE_FINE_GRAINED_ACTIVATION_OFFLOADING", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on", "enabled"}


def _execute_memory_action_hook(policy: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    event = dict(payload or {})
    event["hook_name"] = "memory_action_hook"
    if not _memory_action_runtime_enabled(event):
        event["status"] = "inactive"
        event["reason"] = "fine_grained_activation_offloading_disabled"
        return event

    local_stage_hint = dict(policy.get("local_stage_hint") or {})
    state_migration_hints = list(policy.get("state_migration_hints") or [])
    trigger_hook = str(event.get("trigger_hook") or "").strip().lower()
    microbatch_id = max(int(event.get("microbatch_id") or 0), 0)
    applied_actions: List[str] = []

    offload_shift_direction = str(local_stage_hint.get("offload_shift_direction") or "").strip().lower()
    offload_shift_offset_slots = max(int(local_stage_hint.get("offload_shift_offset_slots") or 0), 0)
    if trigger_hook in {"before_forward_hook", "after_forward_hook"} and any(
        str(item.get("action") or "").strip() == "offload_timing_shift" for item in state_migration_hints
    ):
        if offload_shift_direction == "later" and microbatch_id < offload_shift_offset_slots:
            fine_grained_offloading_disable_offload()
            applied_actions.append("disable_offload")
        else:
            fine_grained_offloading_enable_offload()
            applied_actions.append("enable_offload")
        if trigger_hook == "after_forward_hook":
            flushed = int(fine_grained_offloading_flush_delayed_groups() or 0)
            event["flushed_delayed_groups"] = flushed
            if flushed > 0:
                applied_actions.append("flush_delayed_groups")

    reload_policy = str(local_stage_hint.get("reload_policy") or "").strip().lower()
    prefetch_policy = str(local_stage_hint.get("prefetch_policy") or "").strip().lower()
    prefetch_distance_slots = max(int(local_stage_hint.get("prefetch_distance_slots") or 0), 0)
    if trigger_hook == "before_backward_hook" and (
        reload_policy == "selective_prefetch"
        or prefetch_policy == "selective"
        or any(str(item.get("action") or "").strip() == "selective_reload_prefetch" for item in state_migration_hints)
    ):
        requested = max(prefetch_distance_slots, 1)
        prefetched = int(fine_grained_offloading_prefetch(distance_slots=requested) or 0)
        event["prefetch_distance_slots"] = requested
        event["prefetched_groups"] = prefetched
        if prefetched > 0:
            applied_actions.append("prefetch_reload_groups")

    event["applied_actions"] = applied_actions
    event["status"] = "applied" if applied_actions else "noop"
    return event


def invoke_schedule_runtime_hook(hook_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    event = dict(payload or {})
    token = str(hook_name or "").strip()
    if token not in _SCHEDULE_RUNTIME_HOOKS:
        event["status"] = "unknown_hook"
        return event
    if token == "memory_action_hook":
        return _execute_memory_action_hook(get_schedule_runtime_policy(), event)
    policy = get_schedule_runtime_policy()
    event["status"] = "ready"
    event["hook_name"] = token
    event["schedule_family"] = str(policy.get("family") or "")
    event["dispatch_order"] = str(policy.get("dispatch_order") or "")
    event["lane_policy"] = str(policy.get("lane_policy") or "")
    event["stage_semantics"] = dict(policy.get("local_stage_hint") or {})
    event["overlap_hints"] = dict(policy.get("overlap_hints") or {})
    event["memory_hints"] = dict(policy.get("memory_hints") or {})
    event["partition_hints"] = dict(policy.get("partition_hints") or {})
    event["schedule_grid_spec"] = dict(policy.get("schedule_grid_spec") or {})
    event["schedule_action_specs"] = list(policy.get("schedule_action_specs") or [])
    event["telemetry_budget"] = dict(policy.get("telemetry_budget") or {})
    event["state_plan"] = dict(policy.get("state_plan") or {})
    event["runtime_repair_summary"] = dict(policy.get("runtime_repair_summary") or {})
    event["runtime_repair_actions"] = list(policy.get("runtime_repair_actions") or [])
    event["runtime_repair_recommendations"] = list(policy.get("runtime_repair_recommendations") or [])
    event["state_migration_hints"] = list(policy.get("state_migration_hints") or [])
    return event


class ScheduleActionRunner:
    """Runtime-facing action tracker for schedule-driven execution.

    The current Megatron execution loops are still mostly schedule-template based.
    This runner bridges the new ScheduleGridSpec / ScheduleActionSpec contract into
    runtime-observed telemetry so we can progressively migrate toward a full
    action-driven executor without losing execution visibility in the meantime.
    """

    def __init__(self) -> None:
        self.policy = get_schedule_runtime_policy()
        self.action_view = get_schedule_action_view()
        self.stage_id = self.policy.get("local_stage_index")
        self.family = str(self.policy.get("family") or "fixed_1f1b")
        self.trace_dir = str(os.environ.get("SCHEDULE_RUNTIME_TRACE_DIR") or "").strip()
        telemetry_budget = dict(self.policy.get("telemetry_budget") or {})
        self.telemetry_level = str(telemetry_budget.get("level") or "full_debug").strip().lower() or "full_debug"
        self.max_trace_mb = max(int(telemetry_budget.get("max_trace_mb") or 256), 1)
        self.max_events_per_rank = max(int(telemetry_budget.get("max_events_per_rank") or 20000), 1)
        self.sampled_windows = max(int(telemetry_budget.get("sampled_windows") or 2), 1)
        self.emit_compare_svg = bool(telemetry_budget.get("emit_compare_svg", True))
        self.rank = self._resolve_global_rank()
        self._trace_start = time.perf_counter()
        self._context: Dict[str, Any] = {
            "phase": "steady",
            "microbatch_id": -1,
            "vchunk_id": 0,
            "lane_id": 0,
        }
        self._events: List[Dict[str, Any]] = []
        self._sampled_microbatches: List[int] = []
        self._lock = threading.Lock()
        self._flushed = False
        self._expected_local_actions = self._filter_local_actions()
        self._metrics: Dict[str, float] = {
            "comm_ms": 0.0,
            "wait_ms": 0.0,
            "reload_stall_ms": 0.0,
            "offload_overlap_success_ratio": 0.0,
            "observed_action_count": 0.0,
        }

    def _resolve_global_rank(self) -> int:
        env_rank = os.environ.get("RANK")
        if env_rank is not None:
            try:
                return int(str(env_rank).strip())
            except Exception:
                pass
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_rank())
        except Exception:
            pass
        if self.stage_id is not None:
            return int(self.stage_id)
        return 0

    def _filter_local_actions(self) -> List[Dict[str, Any]]:
        action_specs = list(self.action_view.get("action_specs") or [])
        if self.stage_id is None:
            return [dict(item) for item in action_specs]
        return [
            dict(item)
            for item in action_specs
            if int(item.get("stage_id") or 0) == int(self.stage_id)
        ]

    def _actions_by_phase(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {"warmup": [], "steady": [], "cooldown": []}
        grid_spec = dict(self.action_view.get("grid_spec") or {})
        cells = [
            dict(item)
            for item in list(grid_spec.get("cells") or [])
            if self.stage_id is None or int(item.get("stage_id") or 0) == int(self.stage_id)
        ]
        cell_index: Dict[tuple[int, int, int, int, str], Dict[str, Any]] = {}
        for cell in cells:
            key = (
                int(cell.get("time_slot") or 0),
                int(cell.get("lane_id") or 0),
                int(cell.get("microbatch_id") or 0),
                int(cell.get("vchunk_id") or 0),
                str(cell.get("kind") or "BUBBLE").upper(),
            )
            cell_index[key] = dict(cell)
        for action in list(self._expected_local_actions or []):
            slot = int(action.get("time_slot") or 0)
            lane_id = int(action.get("lane_id") or 0)
            microbatch_id = int(action.get("microbatch_id") or 0)
            vchunk_id = int(action.get("vchunk_id") or 0)
            action_type = str(action.get("action_type") or "WAIT").upper()
            phase = "steady"
            if action_type == "WAIT":
                matched = cell_index.get((slot, lane_id, microbatch_id, vchunk_id, "BUBBLE"))
                if matched is not None:
                    phase = str(matched.get("phase") or "steady")
            else:
                matched = cell_index.get((slot, lane_id, microbatch_id, vchunk_id, action_type))
                if matched is not None:
                    phase = str(matched.get("phase") or "steady")
            grouped.setdefault(phase, []).append(dict(action))
        for phase in list(grouped.keys()):
            grouped[phase] = sorted(
                grouped[phase],
                key=lambda item: (
                    int(item.get("time_slot") or 0),
                    int(item.get("lane_id") or 0),
                    int(item.get("microbatch_id") or 0),
                    int(item.get("vchunk_id") or 0),
                    str(item.get("action_type") or ""),
                ),
            )
        return grouped

    def _planned_grid_trace(self) -> Dict[str, Any]:
        grid_spec = dict(self.action_view.get("grid_spec") or {})
        if not grid_spec:
            return {}
        cells = [
            dict(item)
            for item in list(grid_spec.get("cells") or [])
            if self.stage_id is None or int(item.get("stage_id") or 0) == int(self.stage_id)
        ]
        return {
            "format": "pipeline_grid_trace",
            "source": "schedule_action_plan",
            "family": str(grid_spec.get("family") or self.family),
            "lanes": int(grid_spec.get("lanes") or 1),
            "time_slots": int(grid_spec.get("time_slots") or 0),
            "stage_count": 1,
            "vstage_count": int(grid_spec.get("vstage_count") or 1),
            "microbatch_count": int(grid_spec.get("microbatch_count") or 1),
            "weight_version_policy": str(grid_spec.get("weight_version_policy") or "default"),
            "constraints": dict(grid_spec.get("constraints") or {}),
            "cells": sorted(
                cells,
                key=lambda item: (
                    int(item.get("lane_id") or 0),
                    int(item.get("time_slot") or 0),
                    str(item.get("kind") or ""),
                ),
            ),
            "notes": ["planned local stage grid from ScheduleGridSpec"],
        }

    def _observed_grid_trace(self) -> Dict[str, Any]:
        with self._lock:
            events = [dict(item) for item in self._events]
        if not events:
            return {}
        positive_durations = [
            max(float(item.get("duration_ms") or 0.0), 0.0)
            for item in events
            if float(item.get("duration_ms") or 0.0) > 0.0
        ]
        slot_ms = min(positive_durations) if positive_durations else 1.0
        slot_ms = max(float(slot_ms), 1e-6)
        lanes = max((int(item.get("lane_id") or 0) for item in events), default=0) + 1
        max_end_slot = 0
        cells: List[Dict[str, Any]] = []
        occupied: set[tuple[int, int]] = set()
        for event in events:
            lane_id = int(event.get("lane_id") or 0)
            start_slot = max(int(round(float(event.get("start_ms") or 0.0) / slot_ms)), 0)
            span_slots = max(int(round(float(event.get("duration_ms") or 0.0) / slot_ms)), 1)
            kind = str(event.get("action_type") or "WAIT").upper()
            if kind == "WAIT":
                kind = "BUBBLE"
            for slot in range(start_slot, start_slot + span_slots):
                occupied.add((lane_id, slot))
                cells.append(
                    {
                        "stage_id": int(event.get("stage_id") or 0),
                        "lane_id": lane_id,
                        "time_slot": int(slot),
                        "kind": kind,
                        "microbatch_id": int(event.get("microbatch_id") or 0),
                        "vchunk_id": int(event.get("vchunk_id") or 0),
                        "phase": str(event.get("phase") or "steady"),
                        "evidence_source": "runtime_observed",
                    }
                )
            max_end_slot = max(max_end_slot, start_slot + span_slots)
        for lane_id in range(max(lanes, 1)):
            for slot in range(max(max_end_slot, 1)):
                if (lane_id, slot) in occupied:
                    continue
                cells.append(
                    {
                        "stage_id": int(self.stage_id) if self.stage_id is not None else 0,
                        "lane_id": int(lane_id),
                        "time_slot": int(slot),
                        "kind": "BUBBLE",
                        "microbatch_id": -1,
                        "vchunk_id": 0,
                        "phase": "steady",
                        "evidence_source": "runtime_observed",
                    }
                )
        return {
            "format": "pipeline_grid_trace",
            "source": "runtime_observed",
            "family": self.family,
            "lanes": max(lanes, 1),
            "time_slots": max(max_end_slot, 1),
            "stage_count": 1,
            "vstage_count": 1,
            "microbatch_count": max((int(item.get("microbatch_id") or -1) for item in events), default=-1) + 1,
            "weight_version_policy": str((self.action_view.get("grid_spec") or {}).get("weight_version_policy") or "default"),
            "constraints": {"slot_ms": round(float(slot_ms), 4)},
            "cells": sorted(
                cells,
                key=lambda item: (
                    int(item.get("lane_id") or 0),
                    int(item.get("time_slot") or 0),
                    str(item.get("kind") or ""),
                ),
            ),
            "notes": ["observed local stage grid synthesized from runtime action events"],
        }

    def _compact_grid_trace(self, grid_trace: Dict[str, Any]) -> Dict[str, Any]:
        if not grid_trace:
            return {}
        counts: Dict[str, int] = {}
        for cell in list(grid_trace.get("cells") or []):
            token = str(cell.get("kind") or "BUBBLE").upper()
            counts[token] = int(counts.get(token, 0)) + 1
        compact = {key: value for key, value in dict(grid_trace).items() if key != "cells"}
        compact["counts"] = counts
        compact["cell_count"] = int(sum(counts.values()))
        compact["notes"] = list(compact.get("notes") or []) + ["cells elided by telemetry budget"]
        return compact

    def _should_keep_event(self, event: Dict[str, Any]) -> bool:
        if self.telemetry_level == "summary":
            return False
        if len(self._events) >= self.max_events_per_rank:
            return False
        if self.telemetry_level == "full_debug":
            return True
        microbatch_id = int(event.get("microbatch_id") or -1)
        if microbatch_id < 0:
            return len(self._events) < self.max_events_per_rank
        if microbatch_id in self._sampled_microbatches:
            return True
        if len(self._sampled_microbatches) < self.sampled_windows:
            self._sampled_microbatches.append(microbatch_id)
            return True
        return False

    @property
    def enabled(self) -> bool:
        return bool(self.trace_dir)

    def set_context(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in {"microbatch_id", "vchunk_id", "lane_id", "stage_id"}:
                try:
                    self._context[key] = int(value)
                except Exception:
                    self._context[key] = value
            else:
                self._context[key] = value

    def set_phase(self, phase: str) -> None:
        token = str(phase or "").strip() or "steady"
        self._context["phase"] = token

    def current_context(self) -> Dict[str, Any]:
        return dict(self._context)

    def begin_action(
        self,
        action_type: str,
        *,
        microbatch_id: Optional[int] = None,
        vchunk_id: Optional[int] = None,
        phase: Optional[str] = None,
        lane_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        kind = str(action_type or "").strip().upper() or "BUBBLE"
        if lane_id is None:
            lane_id = 1 if kind in {"COMM", "OFFLOAD", "RELOAD"} else 0
        token = {
            "action_type": kind,
            "stage_id": int(self.stage_id) if self.stage_id is not None else int(self._context.get("stage_id") or 0),
            "lane_id": int(lane_id),
            "microbatch_id": int(microbatch_id if microbatch_id is not None else self._context.get("microbatch_id", -1)),
            "vchunk_id": int(vchunk_id if vchunk_id is not None else self._context.get("vchunk_id", 0)),
            "phase": str(phase or self._context.get("phase") or "steady"),
            "metadata": dict(metadata or {}),
            "_started_at": time.perf_counter(),
        }
        return token

    def end_action(self, token: Optional[Dict[str, Any]], *, metadata: Optional[Dict[str, Any]] = None) -> None:
        if token is None:
            return
        ended_at = time.perf_counter()
        started_at = float(token.pop("_started_at", ended_at))
        start_ms = max((started_at - self._trace_start) * 1000.0, 0.0)
        end_ms = max((ended_at - self._trace_start) * 1000.0, start_ms)
        event = {
            "stage_id": int(token.get("stage_id") or 0),
            "lane_id": int(token.get("lane_id") or 0),
            "microbatch_id": int(token.get("microbatch_id") or -1),
            "vchunk_id": int(token.get("vchunk_id") or 0),
            "phase": str(token.get("phase") or "steady"),
            "action_type": str(token.get("action_type") or "BUBBLE"),
            "start_ms": round(start_ms, 4),
            "end_ms": round(end_ms, 4),
            "duration_ms": round(max(end_ms - start_ms, 0.0), 4),
            "metadata": {**dict(token.get("metadata") or {}), **dict(metadata or {})},
            "evidence_source": "runtime_observed",
        }
        with self._lock:
            self._metrics["observed_action_count"] += 1.0
            if event["action_type"] == "COMM":
                self._metrics["comm_ms"] += float(event["duration_ms"])
            elif event["action_type"] == "WAIT":
                self._metrics["wait_ms"] += float(event["duration_ms"])
            elif event["action_type"] == "RELOAD":
                self._metrics["reload_stall_ms"] += float(event["duration_ms"])
            elif event["action_type"] == "OFFLOAD":
                overlap_mode = str((event.get("metadata") or {}).get("overlap_mode") or "").strip().lower()
                total_count = self._metrics.get("offload_overlap_total_count", 0.0) + 1.0
                self._metrics["offload_overlap_total_count"] = total_count
                if overlap_mode in {"async", "overlapped", "nonblocking"}:
                    success_count = self._metrics.get("offload_overlap_success_count", 0.0) + 1.0
                    self._metrics["offload_overlap_success_count"] = success_count
            if self._should_keep_event(event):
                self._events.append(event)

    def build_trace_payload(self) -> Dict[str, Any]:
        with self._lock:
            events = sorted(
                [dict(item) for item in self._events],
                key=lambda item: (
                    float(item.get("start_ms") or 0.0),
                    int(item.get("stage_id") or 0),
                    int(item.get("lane_id") or 0),
                ),
            )
            metrics = dict(self._metrics)
        total_overlap = float(metrics.get("offload_overlap_total_count") or 0.0)
        metrics["offload_overlap_success_ratio"] = (
            float(metrics.get("offload_overlap_success_count") or 0.0) / total_overlap
            if total_overlap > 0.0
            else 0.0
        )
        action_type_counts: Dict[str, int] = {}
        for event in events:
            token = str(event.get("action_type") or "WAIT").upper()
            action_type_counts[token] = int(action_type_counts.get(token, 0)) + 1
        action_type_duration_breakdown: Dict[str, float] = {}
        for event in events:
            token = str(event.get("action_type") or "WAIT").upper()
            action_type_duration_breakdown[token] = float(action_type_duration_breakdown.get(token, 0.0)) + float(
                event.get("duration_ms") or 0.0
            )
        max_end_ms = max((float(item.get("end_ms") or 0.0) for item in events), default=0.0)
        local_stage_hint = dict(self.policy.get("local_stage_hint") or {})
        planned_grid_trace = self._planned_grid_trace()
        observed_grid_trace = self._observed_grid_trace()
        planned_slot_count = sum(
            1
            for item in list(planned_grid_trace.get("cells") or [])
            if str(item.get("kind") or "").upper() != "BUBBLE"
        )
        observed_slot_count = sum(
            1
            for item in list(observed_grid_trace.get("cells") or [])
            if str(item.get("kind") or "").upper() != "BUBBLE"
        )
        metrics["planned_action_count"] = int(len(self._expected_local_actions))
        metrics["planned_slot_count"] = int(planned_slot_count)
        metrics["observed_slot_count"] = int(observed_slot_count)
        metrics["planned_vs_observed_event_count_delta"] = int(len(events) - len(self._expected_local_actions))
        metrics["planned_vs_observed_slot_delta"] = int(observed_slot_count - planned_slot_count)
        metrics["action_type_duration_breakdown"] = {
            str(key): round(float(value), 4) for key, value in sorted(action_type_duration_breakdown.items())
        }
        effective_level = self.telemetry_level
        action_plan: Dict[str, Any] = {
            "action_count": int(len(self._expected_local_actions)),
            "actions_by_phase_counts": {
                str(phase): int(len(items))
                for phase, items in sorted(self._actions_by_phase().items())
            },
        }
        if effective_level == "full_debug":
            action_plan["actions"] = list(self._expected_local_actions)
            action_plan["actions_by_phase"] = self._actions_by_phase()

        payload: Dict[str, Any] = {
            "format": "schedule_runtime_event_trace",
            "timing_basis": "runtime_observed",
            "family": self.family,
            "rank": int(self.rank),
            "stage_id": int(self.stage_id) if self.stage_id is not None else None,
            "stage_semantics": local_stage_hint,
            "telemetry": {
                "requested_level": self.telemetry_level,
                "effective_level": effective_level,
                "max_trace_mb": int(self.max_trace_mb),
                "max_events_per_rank": int(self.max_events_per_rank),
                "sampled_windows": int(self.sampled_windows),
                "emit_compare_svg": bool(self.emit_compare_svg),
                "sampled_microbatches": list(self._sampled_microbatches),
            },
            "action_plan": action_plan,
            "metrics": metrics,
            "summary": {
                "projected_timeline_span_ms": round(float(max_end_ms), 4),
                "available_hooks": list(self.policy.get("available_hooks") or []),
                "action_type_counts": action_type_counts,
            },
        }
        if effective_level == "summary":
            payload["planned_grid_trace"] = self._compact_grid_trace(planned_grid_trace)
            payload["observed_grid_trace"] = self._compact_grid_trace(observed_grid_trace)
            payload["events"] = []
        elif effective_level == "aggregated_grid":
            payload["planned_grid_trace"] = self._compact_grid_trace(planned_grid_trace)
            payload["observed_grid_trace"] = observed_grid_trace
            payload["events"] = events
        else:
            payload["planned_grid_trace"] = planned_grid_trace
            payload["observed_grid_trace"] = observed_grid_trace
            payload["events"] = events

        encoded = json.dumps(payload, ensure_ascii=False)
        budget_bytes = int(self.max_trace_mb) * 1024 * 1024
        if len(encoded.encode("utf-8")) > budget_bytes and payload.get("events"):
            payload["telemetry"]["effective_level"] = "aggregated_grid"
            payload["telemetry"]["downgrade_reason"] = "trace_size_budget"
            payload["planned_grid_trace"] = self._compact_grid_trace(planned_grid_trace)
            payload["observed_grid_trace"] = self._compact_grid_trace(observed_grid_trace)
            payload["events"] = []

        encoded = json.dumps(payload, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > budget_bytes:
            payload["telemetry"]["effective_level"] = "summary"
            payload["telemetry"]["downgrade_reason"] = "trace_size_budget_summary"
            payload["observed_grid_trace"] = self._compact_grid_trace(observed_grid_trace)
            payload["planned_grid_trace"] = self._compact_grid_trace(planned_grid_trace)
            payload["events"] = []
            payload["action_plan"] = {
                "action_count": int(len(self._expected_local_actions)),
                "actions_by_phase_counts": {
                    str(phase): int(len(items))
                    for phase, items in sorted(self._actions_by_phase().items())
                },
            }
        return payload

    def flush(self) -> Optional[str]:
        if not self.enabled:
            return None
        with self._lock:
            if self._flushed and self._events:
                return None
            self._flushed = True
        trace_dir = Path(self.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"rank{int(self.rank):03d}"
        if self.stage_id is not None:
            suffix += f"_stage{int(self.stage_id):03d}"
        path = trace_dir / f"schedule_runtime_trace_{suffix}.json"
        path.write_text(
            json.dumps(self.build_trace_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)


class _ScheduleRuntimeCommunicatorProxy:
    def __init__(self, communicator: Any, runner: ScheduleActionRunner) -> None:
        self._communicator = communicator
        self._runner = runner

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._communicator, name)
        if not callable(attr) or not (name.startswith("send_") or name.startswith("recv_")):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            invoke_schedule_runtime_hook("before_send_recv_hook", {"op_name": name})
            token = self._runner.begin_action(
                "COMM",
                phase=self._runner.current_context().get("phase"),
                metadata={"op_name": name},
            )
            try:
                return attr(*args, **kwargs)
            finally:
                self._runner.end_action(token, metadata={"op_name": name})
                invoke_schedule_runtime_hook("after_send_recv_hook", {"op_name": name})

        return _wrapped


def get_schedule_action_runner(force_reset: bool = False) -> ScheduleActionRunner:
    global _SCHEDULE_ACTION_RUNNER_SINGLETON
    with _SCHEDULE_ACTION_RUNNER_LOCK:
        if force_reset or _SCHEDULE_ACTION_RUNNER_SINGLETON is None:
            _SCHEDULE_ACTION_RUNNER_SINGLETON = ScheduleActionRunner()
        return _SCHEDULE_ACTION_RUNNER_SINGLETON


def _instrument_p2p_communicator_for_schedule_runtime(
    communicator: Any,
    runner: ScheduleActionRunner,
) -> Any:
    if communicator is None or not runner.enabled:
        return communicator
    if isinstance(communicator, _ScheduleRuntimeCommunicatorProxy):
        return communicator
    return _ScheduleRuntimeCommunicatorProxy(communicator, runner)


def _flush_schedule_action_runner() -> None:
    global _SCHEDULE_ACTION_RUNNER_SINGLETON
    runner = _SCHEDULE_ACTION_RUNNER_SINGLETON
    if runner is None:
        return
    try:
        runner.flush()
    except Exception:
        return


atexit.register(_flush_schedule_action_runner)


def _parse_window_override_hints(raw: str) -> List[Dict[str, str]]:
    parsed: List[Dict[str, str]] = []
    text = str(raw or "").strip()
    if not text:
        return parsed
    try:
        payload = json.loads(text)
    except Exception:
        return parsed
    if not isinstance(payload, list):
        return parsed
    for item in payload:
        if not isinstance(item, dict):
            continue
        phase = str(item.get("phase") or "").strip()
        window = str(item.get("window") or "").strip()
        stage_selector = str(item.get("stage_selector") or "").strip()
        chunk_order_policy = str(item.get("chunk_order_policy") or "").strip()
        if phase not in {"steady", "cooldown"}:
            continue
        if window not in {"last_1_group", "last_2_groups", "cooldown_all", "cooldown_first_group"}:
            continue
        if stage_selector not in {"tail_stage", "hotspot_stage", "optimizer_sensitive_stage"}:
            continue
        if chunk_order_policy not in {"reverse_chunk_order", "target_chunk_first", "center_out", "edge_interleave"}:
            continue
        entry = {
            "phase": phase,
            "window": window,
            "stage_selector": stage_selector,
            "chunk_order_policy": chunk_order_policy,
        }
        for key in (
            "combined_policy",
            "p2p_policy",
            "flush_policy",
            "checkpoint_policy",
            "optimizer_target_chunk",
        ):
            value = str(item.get(key) or "").strip()
            if value:
                entry[key] = value
        parsed.append(entry)
    return parsed


def _parse_operator_cluster_hints(raw: str) -> Dict[int, List[Dict[str, object]]]:
    parsed: Dict[int, List[Dict[str, object]]] = {}
    text = str(raw or "").strip()
    if not text:
        return parsed
    try:
        payload = json.loads(text)
    except Exception:
        return parsed
    if not isinstance(payload, list):
        return parsed
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            stage_index = int(item.get("stage_index"))
        except Exception:
            continue
        cluster_role = str(item.get("cluster_role") or "").strip()
        semantic_role = str(item.get("semantic_role") or "").strip()
        local_priority = str(item.get("local_priority") or "normal").strip()
        overlap_policy = str(item.get("overlap_policy") or "guarded").strip()
        memory_policy = str(item.get("memory_policy") or "resident").strip()
        if cluster_role not in {
            "attention_comm",
            "backward_critical",
            "memory_hotspot",
            "optimizer_sensitive",
            "embedding_loss_anchor",
            "mlp_compute",
        }:
            continue
        if local_priority not in {"high", "normal", "protected"}:
            continue
        if overlap_policy not in {"aggressive", "guarded", "disabled"}:
            continue
        if memory_policy not in {"resident", "checkpoint", "offload_guarded"}:
            continue
        phases: List[str] = []
        for raw_phase in list(item.get("phases") or []):
            token = str(raw_phase).strip()
            if token in {"warmup", "steady", "cooldown"} and token not in phases:
                phases.append(token)
        if not phases:
            phases = ["steady", "cooldown"]
        entry: Dict[str, object] = {
            "stage_index": int(stage_index),
            "cluster_role": cluster_role,
            "semantic_role": semantic_role or "decoder",
            "local_priority": local_priority,
            "overlap_policy": overlap_policy,
            "memory_policy": memory_policy,
            "phases": phases,
        }
        for key in ("subgraph", "unit_name", "optimizer_target_chunk", "reason"):
            value = str(item.get(key) or "").strip()
            if value:
                entry[key] = value
        parsed.setdefault(int(stage_index), []).append(entry)
    return parsed


def _get_local_operator_cluster_hints() -> List[Dict[str, object]]:
    pp_rank, _ = _pipeline_parallel_location()
    if pp_rank is None:
        return []
    parsed = dict(_parse_schedule_runtime_hints().get("operator_cluster_hints_by_stage") or {})
    return list(parsed.get(int(pp_rank)) or [])


def _cluster_hint_matches_phase(hint: Dict[str, object], phase: str) -> bool:
    phases = [str(item).strip() for item in list(hint.get("phases") or []) if str(item).strip()]
    if not phases:
        phases = ["steady", "cooldown"]
    return phase in phases


def _apply_operator_cluster_hint_chunk_order(
    order: List[int],
    *,
    num_model_chunks: int,
    phase: str,
    local_stage_hint: Dict[str, str],
) -> List[int]:
    if not order:
        return order
    hints = [
        item
        for item in _get_local_operator_cluster_hints()
        if _cluster_hint_matches_phase(item, phase)
    ]
    if not hints:
        return order
    for hint in hints:
        cluster_role = str(hint.get("cluster_role") or "")
        local_priority = str(hint.get("local_priority") or "normal")
        overlap_policy = str(hint.get("overlap_policy") or "guarded")
        memory_policy = str(hint.get("memory_policy") or "resident")
        optimizer_target_chunk = str(hint.get("optimizer_target_chunk") or "").strip().lower()
        if cluster_role in {"optimizer_sensitive", "backward_critical"} and local_priority == "high":
            override = {
                "chunk_order_policy": "target_chunk_first",
                "optimizer_target_chunk": optimizer_target_chunk or "tail",
            }
            order = _apply_chunk_order_policy(
                order,
                num_model_chunks=num_model_chunks,
                local_stage_hint=local_stage_hint,
                override=override,
            )
        elif cluster_role == "memory_hotspot" and memory_policy in {"checkpoint", "offload_guarded"}:
            override = {"chunk_order_policy": "center_out"}
            order = _apply_chunk_order_policy(
                order,
                num_model_chunks=num_model_chunks,
                local_stage_hint=local_stage_hint,
                override=override,
            )
        elif cluster_role == "attention_comm" and overlap_policy in {"guarded", "disabled"}:
            override = {"chunk_order_policy": "edge_interleave"}
            order = _apply_chunk_order_policy(
                order,
                num_model_chunks=num_model_chunks,
                local_stage_hint=local_stage_hint,
                override=override,
            )
        elif cluster_role == "embedding_loss_anchor" and phase == "cooldown" and local_priority in {"high", "protected"}:
            override = {"chunk_order_policy": "reverse_chunk_order"}
            order = _apply_chunk_order_policy(
                order,
                num_model_chunks=num_model_chunks,
                local_stage_hint=local_stage_hint,
                override=override,
            )
    return order


def _cluster_hint_for_phase_policy(phase: str, field: str) -> str:
    for hint in _get_local_operator_cluster_hints():
        if not _cluster_hint_matches_phase(hint, phase):
            continue
        if field == "checkpoint" and str(hint.get("memory_policy") or "") in {"checkpoint", "offload_guarded"}:
            return "disable_full"
        if field == "p2p" and str(hint.get("cluster_role") or "") == "attention_comm":
            overlap_policy = str(hint.get("overlap_policy") or "")
            if overlap_policy in {"guarded", "disabled"}:
                return "serial"
        if field == "combined":
            overlap_policy = str(hint.get("overlap_policy") or "")
            memory_policy = str(hint.get("memory_policy") or "")
            if overlap_policy == "disabled" or memory_policy in {"checkpoint", "offload_guarded"}:
                return "serial"
    return ""


def _local_stage_matches_selector(local_stage_hint: Dict[str, str], stage_selector: str) -> bool:
    stage_tags = _local_stage_tags(local_stage_hint)
    family = str(local_stage_hint.get("family") or "").strip()
    pp_rank, pp_world_size = _pipeline_parallel_location()
    if stage_selector == "tail_stage":
        return (
            (pp_rank is not None and pp_world_size is not None and pp_rank == (pp_world_size - 1))
            or "tail_sensitive" in stage_tags
            or family in {"tail_guarded", "optimizer_guarded_tail"}
        )
    if stage_selector == "hotspot_stage":
        return "memory_hotspot" in stage_tags or family in {"memory_hotspot", "checkpoint_guarded"}
    if stage_selector == "optimizer_sensitive_stage":
        return (
            "optimizer_sensitive" in stage_tags
            or family == "optimizer_guarded_tail"
            or bool(str(local_stage_hint.get("optimizer_runtime_mode") or "").strip())
        )
    return False


def _phase_group_indices(phase: str, num_groups: int) -> List[int]:
    return [group_index for group_index in range(max(int(num_groups), 0)) if _resolve_group_phase(group_index, num_groups) == phase]


def _window_matches_group(phase: str, window: str, group_index: int, num_groups: int) -> bool:
    phase_groups = _phase_group_indices(phase, num_groups)
    if not phase_groups:
        return False
    if phase == "steady":
        if window == "last_1_group":
            return group_index == phase_groups[-1]
        if window == "last_2_groups":
            return group_index in phase_groups[-2:]
        return False
    if phase == "cooldown":
        if window == "cooldown_all":
            return group_index in phase_groups
        if window == "cooldown_first_group":
            return group_index == phase_groups[0]
        return False
    return False


def _matching_window_overrides(
    *,
    phase: str,
    group_index: int,
    num_groups: int,
    local_stage_hint: Dict[str, str],
) -> List[Dict[str, str]]:
    overrides = list(_parse_schedule_runtime_hints().get("window_override_hints") or [])
    if not overrides:
        return []
    return [
        item
        for item in overrides
        if str(item.get("phase") or "") == phase
        and _window_matches_group(phase, str(item.get("window") or ""), group_index, num_groups)
        and _local_stage_matches_selector(local_stage_hint, str(item.get("stage_selector") or ""))
    ]


def _resolve_optimizer_runtime_value(
    local_stage_hint: Dict[str, str],
    stage_key: str,
    env_key: str,
    default: str = "",
) -> str:
    value = str(local_stage_hint.get(stage_key) or "").strip()
    if value:
        return value
    return str(os.environ.get(env_key, default) or default).strip()


def _local_stage_tags(local_stage_hint: Dict[str, str]) -> set[str]:
    raw = str(local_stage_hint.get("stage_tags") or "").strip()
    if not raw:
        return set()
    return {
        token.strip()
        for token in raw.replace(",", "|").split("|")
        if token.strip()
    }


def _get_local_pipeline_layout_tokens(num_model_chunks: int) -> List[str]:
    layout = str(os.environ.get("PIPELINE_LAYOUT", "") or "").strip()
    pp_rank, pp_world_size = _pipeline_parallel_location()
    if not layout or pp_rank is None or pp_world_size is None or num_model_chunks <= 0:
        return []
    stages = [stage.strip() for stage in layout.split("|")]
    if len(stages) != pp_world_size * num_model_chunks:
        return []
    return [
        stages[(chunk_id * pp_world_size) + pp_rank]
        for chunk_id in range(num_model_chunks)
        if (chunk_id * pp_world_size) + pp_rank < len(stages)
    ]


def _resolve_local_optimizer_target_chunk(
    num_model_chunks: int,
    local_stage_hint: Dict[str, str],
) -> Optional[int]:
    if num_model_chunks <= 1:
        return None
    runtime_mode = _resolve_optimizer_runtime_value(
        local_stage_hint,
        "optimizer_runtime_mode",
        "SCHEDULE_OPTIMIZER_RUNTIME_MODE",
    )
    if runtime_mode != "tail_guarded_overlap":
        return None

    target_policy = _resolve_optimizer_runtime_value(
        local_stage_hint,
        "optimizer_target_policy",
        "SCHEDULE_OPTIMIZER_TARGET_POLICY",
        "tail_stage_first",
    )
    family = str(local_stage_hint.get("family") or "").strip()
    stage_tags = _local_stage_tags(local_stage_hint)
    if (
        target_policy == "tail_stage_first"
        and family not in {"optimizer_guarded_tail", "tail_guarded"}
        and not {"tail_sensitive", "optimizer_sensitive"}.issubset(stage_tags)
    ):
        return None

    explicit_target = str(local_stage_hint.get("optimizer_target_chunk") or "").strip().lower()
    if explicit_target.isdigit():
        return max(min(int(explicit_target), num_model_chunks - 1), 0)

    layout_tokens = _get_local_pipeline_layout_tokens(num_model_chunks)
    if explicit_target in {"tail", "last"} or target_policy == "tail_stage_first":
        for chunk_id, tokens in enumerate(layout_tokens):
            if "L" in tokens:
                return int(chunk_id)
        return num_model_chunks - 1

    return None


def _apply_optimizer_windowed_chunk_order(
    order: List[int],
    *,
    num_model_chunks: int,
    phase: str,
    group_index: int,
    num_groups: int,
    local_stage_hint: Dict[str, str],
) -> List[int]:
    if not order or phase == "warmup":
        return order
    window_policy = _resolve_optimizer_runtime_value(
        local_stage_hint,
        "optimizer_window_policy",
        "SCHEDULE_OPTIMIZER_WINDOW_POLICY",
    )
    if window_policy != "tail_flush_aligned":
        return order
    if group_index < max(num_groups - 2, 0):
        return order
    target_chunk = _resolve_local_optimizer_target_chunk(num_model_chunks, local_stage_hint)
    if target_chunk is None or target_chunk not in order:
        return order
    return [target_chunk] + [chunk_id for chunk_id in order if chunk_id != target_chunk]


def _apply_chunk_order_policy(
    order: List[int],
    *,
    num_model_chunks: int,
    local_stage_hint: Dict[str, str],
    override: Dict[str, str],
) -> List[int]:
    if not order:
        return order
    policy = str(override.get("chunk_order_policy") or "").strip()
    if policy == "reverse_chunk_order":
        return list(reversed(order))
    if policy == "target_chunk_first":
        target_chunk_hint = str(override.get("optimizer_target_chunk") or "").strip().lower()
        target_chunk = _resolve_local_optimizer_target_chunk(num_model_chunks, local_stage_hint)
        if target_chunk_hint.isdigit():
            target_chunk = max(min(int(target_chunk_hint), num_model_chunks - 1), 0)
        elif target_chunk_hint in {"tail", "last"} and target_chunk is None:
            target_chunk = num_model_chunks - 1
        elif target_chunk_hint in {"head", "first"}:
            target_chunk = 0
        if target_chunk is None or target_chunk not in order:
            return order
        return [target_chunk] + [chunk_id for chunk_id in order if chunk_id != target_chunk]
    if policy == "center_out":
        center_order = _center_out_order(num_model_chunks)
        return [chunk_id for chunk_id in center_order if chunk_id in order]
    if policy == "edge_interleave":
        return _edge_interleave(order)
    return order


def _apply_window_override_chunk_order(
    order: List[int],
    *,
    num_model_chunks: int,
    phase: str,
    group_index: int,
    num_groups: int,
    local_stage_hint: Dict[str, str],
) -> List[int]:
    matched_overrides = _matching_window_overrides(
        phase=phase,
        group_index=group_index,
        num_groups=num_groups,
        local_stage_hint=local_stage_hint,
    )
    for override in matched_overrides:
        order = _apply_chunk_order_policy(
            order,
            num_model_chunks=num_model_chunks,
            local_stage_hint=local_stage_hint,
            override=override,
        )
    return order


def _resolve_window_override_value(
    *,
    phase: str,
    group_index: int,
    num_groups: int,
    local_stage_hint: Dict[str, str],
    key: str,
) -> str:
    for override in _matching_window_overrides(
        phase=phase,
        group_index=group_index,
        num_groups=num_groups,
        local_stage_hint=local_stage_hint,
    ):
        value = str(override.get(key) or "").strip()
        if value:
            return value
    return ""


def _get_structure_aware_chunk_metadata(num_model_chunks: int) -> Optional[List[Dict[str, int]]]:
    pp_rank, pp_world_size = _pipeline_parallel_location()
    policy_stage_hints = dict(_parse_schedule_runtime_hints().get("stage_chunk_priority_hints") or {})
    per_stage_hints: Dict[int, List[int]] = {}
    for raw_stage_id, raw_hints in policy_stage_hints.items():
        try:
            stage_id = int(raw_stage_id)
        except Exception:
            continue
        parsed_hints: List[int] = []
        for raw_hint in list(raw_hints or []):
            try:
                parsed_hints.append(int(raw_hint))
            except Exception:
                continue
        if len(parsed_hints) == num_model_chunks:
            per_stage_hints[int(stage_id)] = parsed_hints
    if pp_rank is not None:
        explicit_stage_hints = per_stage_hints.get(int(pp_rank))
        if explicit_stage_hints is not None:
            return [
                {
                    "chunk_id": int(chunk_id),
                    "compute_weight": int(priority),
                    "criticality": 0,
                }
                for chunk_id, priority in enumerate(explicit_stage_hints)
            ]

    raw_hints = os.environ.get("SCHEDULE_CHUNK_PRIORITY_HINTS", "")
    explicit_hints = _parse_structure_priority_hints(raw_hints, num_model_chunks)
    if explicit_hints is not None:
        return [
            {
                "chunk_id": int(chunk_id),
                "compute_weight": int(priority),
                "criticality": 0,
            }
            for chunk_id, priority in enumerate(explicit_hints)
        ]

    layout = str(os.environ.get("PIPELINE_LAYOUT", "") or "").strip()
    if not layout or pp_rank is None or pp_world_size is None:
        return None

    stages = [stage.strip() for stage in layout.split("|")]
    if len(stages) != pp_world_size * num_model_chunks:
        return None
    metadata: List[Dict[str, int]] = []
    for chunk_id in range(num_model_chunks):
        stage_index = chunk_id * pp_world_size + pp_rank
        if stage_index >= len(stages):
            return None
        stage_tokens = stages[stage_index]
        decoder_count = stage_tokens.count("t")
        criticality = 0
        if "E" in stage_tokens:
            criticality += 2
        if "L" in stage_tokens:
            criticality += 2
        if chunk_id in {0, num_model_chunks - 1}:
            criticality += 1
        metadata.append(
            {
                "chunk_id": int(chunk_id),
                "compute_weight": int(decoder_count),
                "criticality": int(criticality),
            }
        )
    return metadata


def _apply_structure_aware_chunk_order(order: List[int], phase: str) -> List[int]:
    metadata = _get_structure_aware_chunk_metadata(len(order))
    if not metadata:
        return order

    by_chunk_id = {int(item["chunk_id"]): item for item in metadata}

    def _warm_key(chunk_id: int):
        item = by_chunk_id.get(int(chunk_id), {})
        # Warmup / steady front-load structurally critical chunks, then heavier chunks.
        return (
            -int(item.get("criticality", 0)),
            -int(item.get("compute_weight", 0)),
            order.index(chunk_id),
        )

    def _cool_key(chunk_id: int):
        item = by_chunk_id.get(int(chunk_id), {})
        # Cooldown drains structurally critical chunks first, but prefers lighter
        # chunks within the same criticality bucket to shorten tail exposure.
        return (
            -int(item.get("criticality", 0)),
            int(item.get("compute_weight", 0)),
            order.index(chunk_id),
        )

    if phase == "cooldown":
        return sorted(order, key=_cool_key)
    return sorted(order, key=_warm_key)


def _resolve_model_chunk_order(
    num_model_chunks: int,
    *,
    template: str,
    dispatch_order: str,
    phase: str,
    warmup_policy: str,
    cooldown_policy: str,
    group_index: int,
    num_groups: int,
    local_stage_hint: Dict[str, str],
) -> List[int]:
    order = _base_model_chunk_order(num_model_chunks, template)

    if dispatch_order in {"structure_aware_critical_first"}:
        return _apply_structure_aware_chunk_order(order, phase)

    if dispatch_order in {"balanced_round_robin"}:
        order = _edge_interleave(order)
    elif dispatch_order in {
        "middle_stage_relief",
        "stage_local_nonuniform_vpp",
        "boundary_localized",
        "boundary_comm_aware",
        "optimizer_tail_guarded",
    } and num_model_chunks > 2:
        order = _center_out_order(num_model_chunks)

    if phase == "warmup":
        if warmup_policy in {"balanced_fill", "grouped", "bubble_fill"}:
            order = _edge_interleave(order)
        elif warmup_policy in {"tail_prefill"}:
            order = list(reversed(order))
    elif phase == "cooldown":
        if cooldown_policy in {
            "tail_min",
            "opt_prioritized",
            "tail_drain",
            "drain_with_w",
            "staggered_wgrad",
            "late_wait",
            "optimizer_tail_hide",
            "tail_checkpoint_guard",
        } or dispatch_order in {"tail_boundary_rewrite", "zero_bubble_proxy"}:
            order = list(reversed(order))

    order = _apply_optimizer_windowed_chunk_order(
        order,
        num_model_chunks=num_model_chunks,
        phase=phase,
        group_index=group_index,
        num_groups=num_groups,
        local_stage_hint=local_stage_hint,
    )
    order = _apply_operator_cluster_hint_chunk_order(
        order,
        num_model_chunks=num_model_chunks,
        phase=phase,
        local_stage_hint=local_stage_hint,
    )
    return _apply_window_override_chunk_order(
        order,
        num_model_chunks=num_model_chunks,
        phase=phase,
        group_index=group_index,
        num_groups=num_groups,
        local_stage_hint=local_stage_hint,
    )


def _parse_explicit_flush_microbatch_ids(raw: str, valid_ids: List[int]) -> List[int]:
    if not raw.strip():
        return []
    valid_set = set(valid_ids)
    parsed: List[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value not in valid_set or value in seen:
            continue
        parsed.append(value)
        seen.add(value)
    if len(parsed) == len(valid_ids):
        return parsed
    return []


def _resolve_microbatch_order(
    microbatch_ids: List[int],
    *,
    phase: str,
    dispatch_order: str,
    cooldown_policy: str,
    flush_order_policy: str,
    explicit_flush_ids: str,
    local_stage_hint: Optional[Dict[str, str]] = None,
    group_index: int = 0,
    num_groups: int = 1,
) -> List[int]:
    if not microbatch_ids:
        return []
    local_stage_hint = dict(local_stage_hint or {})

    if phase == "cooldown":
        override_flush_policy = _resolve_window_override_value(
            phase=phase,
            group_index=group_index,
            num_groups=num_groups,
            local_stage_hint=local_stage_hint,
            key="flush_policy",
        ).lower()
        if override_flush_policy in {
            "reverse_last_group",
            "tail_checkpoint_guard",
            "optimizer_tail_hide",
            "tail_min",
            "opt_prioritized",
        }:
            return list(reversed(microbatch_ids))
        explicit = _parse_explicit_flush_microbatch_ids(explicit_flush_ids, microbatch_ids)
        if explicit:
            return explicit
        if flush_order_policy in {"reverse_last_group", "tail_min", "opt_prioritized", "optimizer_tail_hide", "tail_checkpoint_guard"}:
            return list(reversed(microbatch_ids))
        if cooldown_policy in {
            "tail_min",
            "opt_prioritized",
            "tail_drain",
            "drain_with_w",
            "staggered_wgrad",
            "late_wait",
            "optimizer_tail_hide",
            "tail_checkpoint_guard",
        } or dispatch_order in {"tail_boundary_rewrite", "zero_bubble_proxy"}:
            return list(reversed(microbatch_ids))

    if dispatch_order == "balanced_round_robin":
        return _edge_interleave(microbatch_ids)

    return microbatch_ids


def _default_checkpoint_activations_microbatch(
    virtual_microbatch_id: int,
    max_outstanding_backprops: Optional[int],
    config,
):
    if max_outstanding_backprops is None:
        return None
    return (
        virtual_microbatch_id % max_outstanding_backprops
        >= config.num_microbatches_with_partial_activation_checkpoints
    )


def _resolve_phase_checkpoint_policy(phase: str, default_value):
    local_stage_hint = _get_local_stage_family_hint()
    raw = str(local_stage_hint.get("checkpoint_policy") or "").strip().lower()
    if not raw:
        raw = str(
            os.environ.get(f"SCHEDULE_{phase.upper()}_CHECKPOINT_POLICY", "default") or "default"
        ).strip().lower()
    if raw in {"", "default", "inherit"}:
        cluster_hint = _cluster_hint_for_phase_policy(phase, "checkpoint")
        if cluster_hint == "disable_full":
            return False
        return default_value
    if raw in {"full", "all", "force_full"}:
        return True
    if raw in {"partial", "off", "none", "disable", "false", "prefer_partial", "selective", "tail_selective", "guarded_selective", "hotspot_selective"}:
        return False
    return default_value


def _phase_uses_p2p_overlap(phase: str, default_enabled: bool) -> bool:
    if not default_enabled:
        return False
    local_stage_hint = _get_local_stage_family_hint()
    raw = str(local_stage_hint.get("p2p_policy") or "").strip().lower()
    if not raw:
        raw = str(os.environ.get(f"SCHEDULE_{phase.upper()}_P2P_POLICY", "default") or "default").strip().lower()
    if raw in {"", "default", "inherit", "overlap", "on", "enabled"}:
        cluster_hint = _cluster_hint_for_phase_policy(phase, "p2p")
        if cluster_hint == "serial":
            return False
        return True
    if raw in {"serial", "off", "none", "defer", "disabled"}:
        return False
    return default_enabled


def _resolve_execution_phase_for_virtual_microbatches(
    f_virtual_microbatch_id=None, b_virtual_microbatch_id=None
) -> str:
    has_forward = f_virtual_microbatch_id is not None
    has_backward = b_virtual_microbatch_id is not None
    if has_forward and has_backward:
        return "steady"
    if has_forward:
        return "warmup"
    if has_backward:
        return "cooldown"
    return "steady"


def _phase_uses_combined_overlap(phase: str, default_enabled: bool = True) -> bool:
    if not default_enabled:
        return False
    local_stage_hint = _get_local_stage_family_hint()
    raw = str(local_stage_hint.get("combined_policy") or "").strip().lower()
    if not raw:
        raw = str(
            os.environ.get(f"SCHEDULE_{phase.upper()}_COMBINED_POLICY", "default") or "default"
        ).strip().lower()
    if raw in {"", "default", "inherit", "combined", "overlap", "enabled", "on"}:
        cluster_hint = _cluster_hint_for_phase_policy(phase, "combined")
        if cluster_hint == "serial":
            return False
        return True
    if raw in {"serial", "off", "none", "disabled"}:
        return False
    return default_enabled


def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """Get the schedule table for PP scheduling."""
    runtime_policy = get_schedule_runtime_policy()
    local_stage_hint = dict(runtime_policy.get("local_stage_hint") or {})
    template = str(os.environ.get("SCHEDULE_TEMPLATE", "fixed_1f1b") or "fixed_1f1b").strip()
    schedule_family = str(runtime_policy.get("family") or "").strip()
    dispatch_order = str(os.environ.get("DISPATCH_ORDER", "default") or "default").strip()
    warmup_policy = str(
        os.environ.get("SCHEDULE_WARMUP_POLICY", "default") or "default"
    ).strip()
    cooldown_policy = str(
        os.environ.get("SCHEDULE_COOLDOWN_POLICY", "default") or "default"
    ).strip()
    flush_order_policy = str(
        os.environ.get("SCHEDULE_FLUSH_ORDER_POLICY", "default") or "default"
    ).strip()
    explicit_flush_ids = str(os.environ.get("SCHEDULE_FLUSH_MICROBATCHES", "") or "").strip()
    if schedule_family in _ALL_SCHEDULE_FAMILIES:
        if schedule_family == "interleaved":
            template = "interleaved"
        elif schedule_family in {"zero_bubble", "zbv", "v_half", "v_min"} and dispatch_order in {"", "default"}:
            dispatch_order = "zero_bubble_proxy"
        elif schedule_family == "dualpipe_v" and dispatch_order in {"", "default"}:
            dispatch_order = "balanced_round_robin"
        template = str(schedule_family or template).strip()
    template = str(local_stage_hint.get("preferred_template") or template).strip()
    dispatch_order = str(
        runtime_policy.get("dispatch_order")
        or local_stage_hint.get("dispatch_order")
        or local_stage_hint.get("local_dispatch_hint")
        or dispatch_order
    ).strip()
    warmup_policy = str(local_stage_hint.get("warmup_policy") or warmup_policy).strip()
    cooldown_policy = str(local_stage_hint.get("cooldown_policy") or cooldown_policy).strip()
    microbatch_group_size_per_vp_stage = _resolve_local_group_size(microbatch_group_size_per_vp_stage)

    schedule_table = []
    group_starts = list(range(0, num_microbatches, microbatch_group_size_per_vp_stage))
    for group_index, min_microbatch_id_in_group in enumerate(group_starts):
        phase = _resolve_group_phase(group_index, len(group_starts))
        max_microbatch_id = min(
            min_microbatch_id_in_group + microbatch_group_size_per_vp_stage, num_microbatches
        )
        model_chunk_order = _resolve_model_chunk_order(
            num_model_chunks,
            template=template,
            dispatch_order=dispatch_order,
            phase=phase,
            warmup_policy=warmup_policy,
            cooldown_policy=cooldown_policy,
            group_index=group_index,
            num_groups=len(group_starts),
            local_stage_hint=local_stage_hint,
        )
        ordered_microbatch_ids = _resolve_microbatch_order(
            list(range(min_microbatch_id_in_group, max_microbatch_id)),
            phase=phase,
            dispatch_order=dispatch_order,
            cooldown_policy=cooldown_policy,
            flush_order_policy=flush_order_policy,
            explicit_flush_ids=explicit_flush_ids,
            local_stage_hint=local_stage_hint,
            group_index=group_index,
            num_groups=len(group_starts),
        )
        for model_chunk_id in model_chunk_order:
            for microbatch_id in ordered_microbatch_ids:
                schedule_table.append((microbatch_id, model_chunk_id))
    return schedule_table


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    runtime_runner = get_schedule_action_runner()

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    config = get_model_config(model[0])
    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
        cp_size = cp_group.size()
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection"
            " provide none or provide all the process groups"
        )
    p2p_communicator = _instrument_p2p_communicator_for_schedule_runtime(
        p2p_communicator,
        runtime_runner,
    )

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for interleaved pipeline parallelism"

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        # vp is ignored for clear_embedding_activation_buffer
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()

    if (
        config.microbatch_group_size_per_vp_stage > num_microbatches
        or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)

    # If the final micro-batch group has fewer micro-batches than pipeline-parallel size,
    # the pipeline will have dependency bubbles.
    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

    # Compute number of warmup and remaining microbatches.
    # seems only used for vpp
    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
    )

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )

    # Decouple individual lookup table for microbatch_id and model_chunk_id.
    # For example, the micro-batch table for PP2 N3M5 with VP2 is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # Similarly, the model chunk table is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    # Both tables are indexed with virtual_microbatch_id.
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            is_pp_first_stage(p2p_communicator.pp_group)
            if forward
            else is_pp_last_stage(p2p_communicator.pp_group)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id

    def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
        """Preprocess for forward_step_helper"""
        runtime_runner.set_context(
            microbatch_id=microbatch_id,
            vchunk_id=model_chunk_id,
            lane_id=0,
        )
        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if _is_vp_first_stage(vp_stage=model_chunk_id) and is_pp_first_stage(pp_group):
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)

        # For non-depth-first pipeline schedules, the first rank would buffer multiple received
        # activation tensors for a model chunk until accessed during warmup.
        # This input buffering is needed to overlap the computation with the receipt of
        # the next inputs. To index the proper buffered inputs for forword_step, we use
        # microbatch_id offset with number of released microbatches that have completed backprop.
        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        return input_tensor

    def forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens):
        """Postprocess for forward_step_helper"""
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens

        # If forward-only, no need to save tensors for a backward pass.
        if forward_only:
            # Release the tensor that have completed forward step.
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return

    def forward_step_helper(virtual_microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks"""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=True)

        input_tensor = forward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id
        )

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
            is_last_stage=_is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group),
        )

        forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens)

        return output_tensor

    def backward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id):
        """Preprocess for backward_step_helper"""
        backward_microbatch_id = -1
        try:
            backward_index = max(int(virtual_microbatch_id) - int(num_warmup_microbatches), 0)
            backward_microbatch_id = int(
                microbatch_id_table[min(backward_index, max(len(microbatch_id_table) - 1, 0))]
            )
        except Exception:
            backward_microbatch_id = -1
        runtime_runner.set_context(
            microbatch_id=backward_microbatch_id,
            vchunk_id=model_chunk_id,
            lane_id=0,
        )
        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        # pylint: disable=E0606
        if _is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group):
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        return input_tensor, output_tensor, output_tensor_grad

    def backward_step_helper_postprocess(virtual_microbatch_id):
        """Postprocess for backward_step_helper"""
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks"""
        nonlocal output_tensor_grads
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)

        input_tensor, output_tensor, output_tensor_grad = backward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id
        )

        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)

        backward_step_helper_postprocess(virtual_microbatch_id)

        return input_tensor_grad

    def forward_backward_helper_wrapper(
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        checkpoint_activations_microbatch=None,
    ):
        """
        wrap forward_helper, backward_helper, and combined_forward_backward_helper in a unified way
        """
        execution_phase = _resolve_execution_phase_for_virtual_microbatches(
            f_virtual_microbatch_id=f_virtual_microbatch_id,
            b_virtual_microbatch_id=b_virtual_microbatch_id,
        )
        combined_overlap_enabled = _phase_uses_combined_overlap(
            execution_phase,
            default_enabled=bool(config.overlap_moe_expert_parallel_comm and not forward_only),
        )
        # The combined MoE overlap path does not support per-microbatch activation
        # checkpoint selection. Fall back to the conventional interleaved path when a
        # phase-local checkpoint policy requests an explicit True/False decision.
        combined_overlap_enabled = (
            combined_overlap_enabled and checkpoint_activations_microbatch is None
        )
        if combined_overlap_enabled:  # Combined 1F1B path
            return combined_1f1b_schedule_for_interleaved_pipelining(
                config,
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                forward_data_store,
                forward_step_helper_preprocess,
                forward_step_helper_postprocess,
                backward_step_helper_preprocess,
                backward_step_helper_postprocess,
                get_microbatch_id_in_model_chunk,
                get_model_chunk_id,
                partial(check_first_val_step, first_val_step, forward_only),
                is_first_microbatch_for_model_chunk,
                collect_non_loss_data,
                f_virtual_microbatch_id=f_virtual_microbatch_id,
                b_virtual_microbatch_id=b_virtual_microbatch_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
            )
        else:  # Conventional interleaved 1F1B path
            forward_output_tensor = None
            backward_input_tensor_grad = None
            # forward pass
            if f_virtual_microbatch_id is not None:
                forward_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
                if pre_forward is not None:
                    pre_forward()
                forward_output_tensor = forward_step_helper(
                    f_virtual_microbatch_id, checkpoint_activations_microbatch
                )
                if post_forward is not None:
                    forward_output_tensor = post_forward(forward_output_tensor)

            # Backward pass.
            if b_virtual_microbatch_id is not None:
                backward_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
                if pre_backward is not None:
                    pre_backward()
                backward_input_tensor_grad = backward_step_helper(b_virtual_microbatch_id)
                if post_backward is not None:
                    backward_input_tensor_grad = post_backward(backward_input_tensor_grad)
            return forward_output_tensor, backward_input_tensor_grad

    # ==============================main logic=========================================
    _is_vp_first_stage = partial(
        is_vp_first_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    _is_vp_last_stage = partial(
        is_vp_last_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    pp_group = p2p_communicator.pp_group

    # Run warmup forward passes.
    nvtx_range_push(suffix="warmup")
    runtime_runner.set_phase("warmup")
    input_tensors[0].append(
        p2p_communicator.recv_forward(
            tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
        )
    )

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if is_pp_first_stage(p2p_communicator.pp_group):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if is_pp_last_stage(p2p_communicator.pp_group):
        bwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []
    warmup_overlap_p2p = _phase_uses_p2p_overlap(
        "warmup", bool(config.overlap_p2p_comm_warmup_flush)
    )
    cooldown_overlap_p2p = _phase_uses_p2p_overlap(
        "cooldown", bool(config.overlap_p2p_comm_warmup_flush)
    )

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        runtime_runner.set_phase("warmup")
        runtime_runner.set_context(
            microbatch_id=get_microbatch_id_in_model_chunk(k, forward=True),
            vchunk_id=cur_model_chunk_id,
            lane_id=0,
        )

        if warmup_overlap_p2p:
            if (
                not (
                    _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group)
                )
                and k != 0
            ):
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if warmup_overlap_p2p and not is_pp_first_stage(
            p2p_communicator.pp_group
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communicator.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        checkpoint_activations_microbatch = _resolve_phase_checkpoint_policy(
            "warmup",
            _default_checkpoint_activations_microbatch(
                k, max_outstanding_backprops, config
            ),
        )

        output_tensor, _ = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # Don't send tensor downstream if on last stage.
        if _is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group):
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not warmup_overlap_p2p:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communicator.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not is_pp_first_stage(p2p_communicator.pp_group):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=False, tensor_shape=tensor_shape, overlap_p2p_comm=True
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])
    nvtx_range_pop(suffix="warmup")

    # Run 1F1B in steady state.
    nvtx_range_push(suffix="steady")
    runtime_runner.set_phase("steady")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        runtime_runner.set_phase("steady")
        runtime_runner.set_context(
            microbatch_id=get_microbatch_id_in_model_chunk(forward_k, forward=True),
            vchunk_id=get_model_chunk_id(forward_k, forward=True),
            lane_id=0,
        )

        # Decide to checkpoint all layers' activations of the current micro-batch.
        checkpoint_activations_microbatch = _resolve_phase_checkpoint_policy(
            "steady",
            _default_checkpoint_activations_microbatch(
                forward_k, max_outstanding_backprops, config
            ),
        )

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        if config.overlap_p2p_comm:

            backward_k = k

            # Sync forward recv
            def pp_pre_forward(vp_stage=None):
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                if not (_is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_prev_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                            'should have registered recv handle'
                        )
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()
                    else:
                        if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                            recv_prev_wait_handle.wait()

                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Async forward send / receive
            def pp_post_forward(output_tensor, vp_stage=None):
                nonlocal send_next_wait_handle
                nonlocal fwd_recv_buffer
                nonlocal fwd_wait_handles
                nonlocal recv_prev_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                # Last virtual stage no activation tensor to send.
                if _is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group):
                    output_tensor = None

                recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                    forward_k, forward=True
                )

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_next_wait_handle is not None:
                    send_next_wait_handle.wait()
                if fwd_wait_handles is not None:
                    send_next_wait_handle = (
                        fwd_wait_handles.pop("send_next")
                        if "send_next" in fwd_wait_handles
                        else None
                    )
                    if "recv_prev" in fwd_wait_handles:
                        recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
                # assert fwd_wait_handles is not None

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.
                if recv_prev:
                    input_tensors[next_forward_model_chunk_id].append(
                        fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                    )
                    fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None

                return output_tensor

            # Sync backward recv
            def pp_pre_backward(vp_stage=None):
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if not (_is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_next_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                            'should have registered recv next handle'
                        )
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()
                    else:
                        if recv_next_wait_handles is not None and recv_next_wait_handles:
                            recv_next_wait_handle = recv_next_wait_handles.pop(0)
                            recv_next_wait_handle.wait()

            # Async backward send / receive
            def pp_post_backward(input_tensor_grad, vp_stage=None):
                nonlocal send_prev_wait_handle
                nonlocal bwd_wait_handles
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                # First virtual stage no activation gradient tensor to send.
                if _is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group):
                    input_tensor_grad = None

                recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                    backward_k, forward=False
                )

                (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
                return input_tensor_grad

            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                pre_forward=pp_pre_forward,
                pre_backward=pp_pre_backward,
                post_forward=pp_post_forward,
                post_backward=pp_post_backward,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

        else:  # No p2p overlap.
            backward_k = k
            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )
            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            if _is_vp_last_stage(vp_stage=forward_model_chunk_id) and is_pp_last_stage(pp_group):
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            if _is_vp_first_stage(vp_stage=backward_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communicator.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="steady")

    # Run cooldown backward passes (flush out pipeline) for the last model chunk.
    nvtx_range_push(suffix="cooldown")
    runtime_runner.set_phase("cooldown")
    curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communicator.recv_backward(
                    tensor_shape,
                    is_last_stage=(
                        _is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                    ),
                )
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            runtime_runner.set_phase("cooldown")
            runtime_runner.set_context(
                microbatch_id=-1,
                vchunk_id=cur_model_chunk_id,
                lane_id=0,
            )
            if (
                not (_is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group))
                and k != 0
            ):
                if cooldown_overlap_p2p:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if cooldown_overlap_p2p and not is_pp_last_stage(
                p2p_communicator.pp_group
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            if cooldown_overlap_p2p:
                if not is_pp_last_stage(p2p_communicator.pp_group):
                    _, bwd_wait_handles = p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communicator.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communicator.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
    nvtx_range_pop(suffix="cooldown")

    nvtx_range_push(suffix="misc")
    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        invoke_schedule_runtime_hook(
            "before_optimizer_tail_hook",
            {"op_name": "finalize_model_grads"},
        )
        finalize_token = runtime_runner.begin_action(
            "WGRAD_OPT",
            phase="cooldown",
            lane_id=0,
            metadata={"op_name": "finalize_model_grads"},
        )
        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )
        runtime_runner.end_action(finalize_token, metadata={"op_name": "finalize_model_grads"})

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()
    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()
    nvtx_range_pop(suffix="misc")

    runtime_runner.flush()

    return forward_data_store


def get_tensor_shapes(
    *,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Determine tensor shapes for pipeline communication.

    Returns [()] for variable_seq_lengths mode (shapes exchanged dynamically),
    or computed shapes for fixed sequence length mode.
    """
    tensor_shapes = []

    if config.variable_seq_lengths:
        # Shapes exchanged dynamically during P2P communication
        tensor_shapes.append(())
        return tensor_shapes

    # Fixed sequence lengths - compute shape
    effective_seq_length = decoder_seq_length if decoder_seq_length is not None else seq_length
    effective_seq_length = effective_seq_length // cp_group.size()

    if config.sequence_parallel:
        effective_seq_length = effective_seq_length // tp_group.size()

    tensor_shapes.append((effective_seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""
    runtime_runner = get_schedule_action_runner()

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    tp_group, cp_group, cp_size = None, None, None

    # Determine if this is a multi-module pipeline
    # (used for validation and backward function selection)
    is_multimodule = isinstance(pg_collection, MultiModuleProcessGroupCollection) or isinstance(
        p2p_communicator, MultiModulePipelineCommunicator
    )

    if p2p_communicator is None and pg_collection is None:
        # Default: single-module with parallel_state groups
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"

        if is_multimodule:
            # Multi-module: use language model's CP size for loss scaling
            if not config.variable_seq_lengths:
                raise ValueError(
                    "config.variable_seq_lengths=True required for multi-module pipelines"
                )
            if pg_collection.has_language_model():
                cp_size = pg_collection.get_language_model_cp_size()
            else:
                # Encoder-only ranks should not use CP loss scaling.
                cp_size = None

        elif isinstance(pg_collection, ProcessGroupCollection):
            # Single-module: extract tp/cp groups and cp_size
            assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
            assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
            tp_group = pg_collection.tp
            cp_group = pg_collection.cp
            cp_size = cp_group.size()

        else:
            raise TypeError(
                f"pg_collection must be ProcessGroupCollection or "
                f"MultiModuleProcessGroupCollection, got {type(pg_collection)}"
            )
    else:
        raise ValueError("Provide both p2p_communicator and pg_collection, or neither")
    p2p_communicator = _instrument_p2p_communicator_for_schedule_runtime(
        p2p_communicator,
        runtime_runner,
    )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, p2p_communicator.is_pp_last_stage
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Select backward function based on whether multi-module or single-module
    if is_multimodule:
        backward_func = partial(
            backward_step_multimodule,
            language_model_module_name=pg_collection.language_model_module_name,
        )
    else:
        backward_func = backward_step

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        runtime_runner.set_phase("warmup")
        runtime_runner.set_context(microbatch_id=i, vchunk_id=0, lane_id=0)
        # Decide to checkpoint all layers' activations of the current micro-batch
        checkpoint_activations_microbatch = _resolve_phase_checkpoint_policy(
            "warmup",
            _default_checkpoint_activations_microbatch(
                i, max_outstanding_backprops, config
            ),
        )

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        runtime_runner.set_phase("steady")
        runtime_runner.set_context(microbatch_id=num_warmup_microbatches, vchunk_id=0, lane_id=0)
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)
        runtime_runner.set_phase("steady")
        runtime_runner.set_context(
            microbatch_id=i + num_warmup_microbatches,
            vchunk_id=0,
            lane_id=0,
        )

        # Decide to checkpoint all layers' activations of the current micro-batch
        checkpoint_activations_microbatch = _resolve_phase_checkpoint_policy(
            "steady",
            _default_checkpoint_activations_microbatch(
                i + num_warmup_microbatches, max_outstanding_backprops, config
            ),
        )

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            runtime_runner.set_context(microbatch_id=i, vchunk_id=0, lane_id=0)
            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            runtime_runner.set_phase("cooldown")
            runtime_runner.set_context(microbatch_id=i, vchunk_id=0, lane_id=0)

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            p2p_communicator.send_backward(input_tensor_grad, p2p_communicator.is_pp_first_stage)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        invoke_schedule_runtime_hook(
            "before_optimizer_tail_hook",
            {"op_name": "finalize_model_grads"},
        )
        finalize_token = runtime_runner.begin_action(
            "WGRAD_OPT",
            phase="cooldown",
            lane_id=0,
            metadata={"op_name": "finalize_model_grads"},
        )
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )
        runtime_runner.end_action(finalize_token, metadata={"op_name": "finalize_model_grads"})

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()

    runtime_runner.flush()

    return forward_data_store
