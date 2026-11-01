# ParaMeld：执行证据驱动的数值搜索研究模块

本目录为独立研究实现，暂时不修改当前 AutoFSDP-Agent 的真实训练执行路径。

## 实现状态

- fixed_order_path.py：对于固定 DAG、已确认的资源先后关系及模块成本，使用混合整数主问题与最长路径分离求解模块边界。提供模块冻结、非空连续区间约束、路径约束增量添加。
- exclusive_schedule.py：对任务依赖、排他资源和已核实的跨 Rank 通信顺序进行混合整数调度优化。
- tests/test_optimizers.py：使用穷举分区和 Google OR-Tools 官方作业调度实例检查最优性；测试非法输入与资源互斥约束。

## 现有工程接口

src/fsdp_agent/config.py 已支持 Fsdp2Strategy、GroupingConfig、ParallelSpec、LayerOverride、策略序列化和验证。
src/fsdp_agent/train.py 与 trial_runner.py 包含 FSDP2 真实训练、Profiler、CUDA 计时及运行指标。
src/fsdp_agent/agent_loop.py 负责模型调用和 Trial 结果记录。

当前仓库尚未确认具有 ParaMeld 论文完整的微批 DAG 路径恢复器和任意模块边界求解器，因此本目录尚未与主线训练流程集成。

## 测试命令

安装 numpy、scipy、networkx、pytest 后，在仓库根目录执行：

    PYTHONPATH=research/parameld pytest -q research/parameld/tests

## 必须完成的 GPU 集成

1. 从真实执行事件获得 F、B、P2P、TP/CP/EP Collective、FSDP AllGather/ReduceScatter、参数释放以及设备资源顺序。
2. 校验跨 Rank 通信一致性、P2P 匹配、梯度与参数版本，以及后端支持的 FSDP 单元和流水线布局。
3. 将合法候选转成 Fsdp2Strategy 后运行实际 GPU 测量，验证预测时间、动态显存和训练正确性。
4. 在模块分区或调度引发资源执行顺序变化时，重新构造 DAG；旧路径约束仅在其有效条件成立时可用。
5. 再连接预算感知的 Agent 搜索，并根据真实费用和收益选择模型调用、数值求解或 GPU 测量。

## 数学证明范围

固定执行图下，路径约束增量生成可求解对应的路径代理目标。资源顺序、执行成本或者通信竞争发生改变时，该性质不能直接推广到完整训练搜索。

调度模型目前将指定资源视为排他资源，未表达真实 GPU 通信和计算同时执行时的减速关系。优化结果需要完整执行模型和 GPU 训练验证。求解器不能证明最优时会明确报错。