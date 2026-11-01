# ParaMeld：执行条件邻域与混合并行搜索研究

该目录提供真实数值求解与数学测试，位于独立研究分支，不改变现有 AutoFSDP-Agent 的训练主线。

详细算法、文献对照、正确性范围与完整 GPU 验证条件见 [DESIGN.md](DESIGN.md)。

## 已实现的六个模块

| 文件 | 功能 |
| --- | --- |
| fixed_order_path.py | 固定执行依赖与模块费用下，用 SciPy MILP 与 DAG 最长路径实现增量路径约束分区 |
| execution_neighborhood.py | 根据事件时间余量、资源影响及配置变量依赖生成有证据的联合搜索邻域 |
| exclusive_schedule.py | 在已验证跨 Rank 通信次序时，以排他资源模型求解局部事件安排 |
| capacity_schedule.py | OR-Tools CP-SAT 的累计资源、每 Rank 动态参数缓冲与计算通信调度 |
| decision_aware_measurement.py | 利用共享成本系数的候选时间差区间，选择有机会改变排序的补测对象 |
| budgeted_portfolio.py | 上下文搜索行动选择；使用已观察的改善率和墙钟/GPU 小时预算 |

## 测试

安装 Python 3.11、numpy、scipy、networkx、ortools、pytest，在仓库根目录执行：

    PYTHONPATH=research/parameld python -m pytest -q research/parameld/tests

GitHub Actions 工作流为 .github/workflows/parameld-research.yml。

测试使用真实数学求解器、精确小规模分区与公开作业调度示例，验证最优性、资源约束、显存存活和测量选择。没有使用 mock。测试结果只证明相应数学模型，在 GPU 上的运行行为需要单独实测。

## 与已有仓库的关系

src/fsdp_agent/config.py 支持 Fsdp2Strategy、GroupingConfig、ParallelSpec 和 LayerOverride；train.py 和 trial_runner.py 提供真实 FSDP2 训练、性能计时与 profiler；agent_loop.py 提供模型调用与试验记录。

目前尚未确认该仓库具有 ParaMeld 论文描述的完整微批 DAG 回放器及对应任意模块边界的 CP-SAT 代码，因此这些研究模块仍是独立可运行的求解器。不能把示例时间解释成真实训练加速。

## 真实集成的必要条件

1. 从训练后端获得受验证的任务依赖、通信组、参数/激活存活及资源先后关系。
2. 对任何 PP/VPP、FSDP、通信和微批配置变更重新检查任务图与成本适用范围。
3. 由后端核验跨 Rank Collective、P2P、参数版本和梯度更新顺序。
4. 用 profiler 校准并行通信干扰与动态显存，再将候选提交真实训练与正确性验证。
5. 在同预算实验中比较原确定性搜索、原 Agent、BALANS、OptiCo、Primus、ATTUNE 和新增算法。

只有固定结构路径子问题及输入资源模型上的最优性可以依据当前实现声明；完整混合并行训练的收益仍须验证。
