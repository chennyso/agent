# AutoFSDP-Agent（可真实运行的 FSDP2 调优框架）

AutoFSDP-Agent 是一个以 LLM 为“对比式策略裁判”的闭环系统：在真实硬件上运行 FSDP2，采集通信/计算/显存证据，构造少量策略变更假设，让 LLM 选择最合理的行为改变，并通过受限算子落地，试验、接受或回滚。

核心分工：
- Executor（Level 1）：真实 torchrun 子进程执行 FSDP2，采集 profiler 证据。
- Judge/Coder（Level 2）：LLM 只做对比判断“哪种策略改变更合理”，不直接生成代码。
- Controller（Level 3）：负责预算/过滤/止损、调用 LLM、调用算子库与 validator，组织 trial。
- Profiler：提供“发生了什么”的结构化证据（通信占比、overlap、显存、collective 次数等）。
- Diagnoser（轻）：将原始指标转成可读事实描述，供 LLM 判断。
- Action Library：将 LLM 选中的“行为假设”翻译成安全、可回滚的 FSDP2 配置修改。
- Validator：拦截非法/危险组合，保守估算显存。

流程（每轮）：profile → 证据描述 → 候选行为假设 → LLM 选择 → 动作应用 → 试验 → 接受/回滚。

## 模式说明
- Config-level 模式（默认）：LLM 在若干完整 Fsdp2Strategy 方案中选择。
- Action-level 模式（推荐，≤10 次试验）：LLM 在少量“行为假设/动作”中做对比选择（如提前 all-gather、合并 group），由动作库落地最小配置修改。本文方法论默认以 action-level 视角描述。

## 仓库结构（可运行主线）
- src/fsdp_agent/config.py：FSDP2 策略定义、默认/启发式/随机种子。
- src/fsdp_agent/fsdp_apply.py：策略到 FSDP2 包装映射。
- src/fsdp_agent/utils/：数据、统计、硬件、指标工具。
- src/fsdp_agent/train.py：加载模型、训练与 profiling、打分（含 trace_summary 和 MFU）。
- src/fsdp_agent/trial_runner.py：torchrun 入口（单策略试验、写 metrics，含 strategy_hash）。
- src/fsdp_agent/agent_loop.py：Controller + LLM 比较裁判 + 子进程执行（默认内网 LLM）。
- docs/AutoFSDP-Agent.md：方法细节、LLM prompt、动作映射表、模式说明。

## 依赖
- Python 3.10+
- PyTorch >= 2.4（FSDP2 组合式）
- transformers（加载 Qwen 等 HF 模型）
- requests（内网 LLM HTTP 调用；如需 openai 风格可自行改）
- NCCL 可用（Linux + GPU 环境）

安装示例：
`ash
pip install "torch>=2.4" transformers requests
`

## 运行示例（A800 4 卡，模型路径 /public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen2.5-14B）
### 单策略试验
`ash
torchrun --nproc_per_node=4 -m fsdp_agent.trial_runner \
  --strategy-file path/to/strategy.json \
  --output ./runs/metrics_0.json \
  --model-name /public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen2.5-14B \
  --global-batch-size 8 --seq-len 2048 --num-steps 30 \
  --dataset-stats-file dataset_stats.json \
  --hardware-json hardware.json
`
metrics_0.json 将包含 tokens/s、通信/计算时间、显存峰值、MFU、trace_summary、得分。

### Agent 循环（LLM 使用内网接口）
`ash
python -m fsdp_agent.agent_loop \
  --rounds 5 \
  --model-name /public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen2.5-14B \
  --global-batch-size 8 --seq-len 2048 \
  --num-steps 30 --num-warmup 5 \
  --mem-limit-gb 70 \
  --workdir ./runs \
  --llm-model /models/Qwen2.5-72B-Instruct \
  --llm-endpoint http://10.100.1.93:12365/v1/chat/completions \
  --dataset-stats-file dataset_stats.json \
  --hardware-json hardware.json
`
流程：跑 default/heuristic/random 种子 → 每轮生成“执行证据 + 候选行为假设” → LLM 选择 → 校验 → torchrun 试验 → 接受/回滚。

## LLM 调用说明
- gent_loop.py 默认使用内网 HTTP 接口 http://10.100.1.93:12365/v1/chat/completions，模型 /models/Qwen2.5-72B-Instruct。如需更换，修改 --llm-endpoint/--llm-model 或 call_llm。
- LLM 不直接生成配置，只在候选假设/策略中做对比选择并给出理由。

## 真实执行与安全
- 训练/试验使用真实 FSDP2 包装（sdp_apply.py），torchrun 多进程，profiler 采集真实通信与显存；trace 写入 --trace-dir（默认 ./runs/traces，可用 TensorBoard 查看）。
- 可用 utils.hardware_info 自动探测 GPU/显存，或用 --hardware-json 覆盖。
- Validator/动作库逻辑参考 docs/AutoFSDP-Agent.md，按需接入。

## 可选增强
- 将动作库真正接入 agent_loop，默认使用 action-level 搜索。
- 加强 validator（OOM/timeout 风险）、历史记忆（避免重复无效动作）。
- 扩展 profiler 提取更丰富的 overlap/collective 特征，丰富 trace_summary。
