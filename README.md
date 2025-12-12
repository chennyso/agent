# FSDP2 Agent 调优骨架（默认示例 Qwen-7B / 4×A800，可换模型）

这是一个**可落地**的 Agent 调优框架：LLM 做策略生成，PyTorch FSDP2 真实跑、profiling、打分，然后迭代。默认示例用 Qwen-7B + 4×A800，但模型名称可通过参数改成任意 Hugging Face Causal LM。硬件不固定：可自动探测 GPU/显存，也可通过 --hardware-json 传入自定义拓扑（节点数/每节点 GPU/带宽类型/mesh_shape）。

## 三层架构（我们的设计，不是抄 CUDAForge）
- Level 3：**Strategy Controller**（实验经理）
  管预算/退出条件/非法策略过滤（显存越界、mesh 无效等），并协调 Judge/Coder 与执行。
- Level 2：**Dual-Agent Reasoning**
  - Agent-Judge：读 profiler metrics，做瓶颈归因（compute/comm/memory/overlap），给出调优方向。
  - Agent-Coder：根据 Judge + 历史生成新策略 JSON（结构化 DSL）。
- Level 1：**Executor**
  真实 torchrun 子进程执行 FSDP2 训练 + profiling（comm/compute/显存），输出统一 JSON。

流程：profile → Judge 诊断 → Coder 生成策略 → Controller 校验/止损 → torchrun 执行 → 迭代。

## 目录
- src/fsdp_agent/config.py：策略 dataclass、默认/启发式/随机种子、JSON 序列化。
- src/fsdp_agent/fsdp_apply.py：将策略映射到 FSDP2 包装逻辑。
- src/fsdp_agent/dataloaders.py：合成 Causal LM 数据加载。
- src/fsdp_agent/dataset_stats.py：数据集统计工具。
- src/fsdp_agent/hardware_info.py：硬件/拓扑探测或自定义 JSON。
- src/fsdp_agent/train.py：加载模型、训练与 profiling、打分。
- src/fsdp_agent/trial_runner.py：单次策略的 torchrun 入口（rank0 写 metrics）。
- src/fsdp_agent/agent_loop.py：LLM 驱动的搜索循环。

## 依赖与环境
- 默认单机 4×A800 (80GB) + NVLink；也可通过 --hardware-json 覆盖硬件描述（节点数、每节点 GPU 数/显存、interconnect、mesh_shape）。代码内置自动探测 GPU/显存作为兜底。
- Python 3.10+，	orch>=2.4（FSDP2 组合式），	ransformers，openai（或替换为自有 LLM 客户端）。
- NCCL 正常，使用 	orchrun 启动。
- OPENAI_API_KEY 需设置（或修改 call_llm）。

安装示例（按需固定版本）：
`ash
pip install "torch>=2.4" transformers openai
`

## 跑单个策略试验
`ash
torchrun --nproc_per_node=4 -m fsdp_agent.trial_runner \
  --strategy-file path/to/strategy.json \
  --output ./runs/metrics_0.json \
  --model-name Qwen/Qwen-7B \
  --global-batch-size 8 --seq-len 2048 --num-steps 30 \
  --dataset-stats-file dataset_stats.json \
  --hardware-json hardware.json
`
metrics_0.json 中包含 tokens/s、通信/计算时间、显存峰值和得分。

## 跑 Agent 循环（LLM 驱动）
`ash
python -m fsdp_agent.agent_loop \
  --rounds 5 \
  --model-name Qwen/Qwen-7B \
  --global-batch-size 8 --seq-len 2048 \
  --num-steps 30 --num-warmup 5 \
  --mem-limit-gb 70 \
  --workdir ./runs \
  --llm-model gpt-4o-mini \
  --dataset-stats-file dataset_stats.json \
  --hardware-json hardware.json
`
流程：
1) 先跑 default、heuristic、andom 三个种子策略。
2) 每轮用最近历史构造 prompt，让 LLM 输出 JSON 策略，校验后 torchrun 执行并打分。
3) 结果写入 ./runs/metrics_*.json、策略存 strategy_*.json、trace 在 ./runs/traces/，总结在 summary.json。

## 说明与注意
- sdp_apply 触达 FSDP2 内部字段（_get_fsdp_state 等），正式实验需固定 PyTorch 版本。
- 目前 throughput 由 step 时间估算，如需更精确可解析 profiler trace 中的 tokens 计数。
- 如不想用 OpenAI，替换 gent_loop.py 里的 call_llm 即可。
- 建议在真实实验中增加验证损失和多次重复以平滑噪声；异构/多节点可通过 hardware.json + mesh_shape 描述。
