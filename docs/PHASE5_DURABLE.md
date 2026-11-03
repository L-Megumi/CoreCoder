# Phase 5：任务级持久化与恢复

Linux 同机、单逻辑 Controller 模式；保留原 Agent Loop、CoreCoderAdapter、Harness、DAGScheduler、JobBudget 和 Context Engine。新层不是另一个 Runtime。默认旧调用不启用持久化。

```python
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.durable import DurableStore
from researchcoder.durable.recovery import RecoveryCoordinator

io = ArtifactIO('/absolute/job-artifacts')
with DurableStore('/absolute/job-state/state.sqlite', io) as store:
    with RecoveryCoordinator(store) as runner:
        result = runner.execute(plan, trusted_scheduler_config)

# 重启后，从应用程序的受信任配置重建模板、模型设置、验证命令和权限。
# 不从 SQLite 或旧 WorkerRequest 创建模型客户端/权限。
with DurableStore('/absolute/job-state/state.sqlite', io) as store:
    with RecoveryCoordinator(store) as runner:
        report = runner.recover(job_id, trusted_scheduler_config)
        # report.status: resumed / completed / stopped / needs_review
        # runner.last_result 是实际 SchedulerResult；不能仅凭 resumed 认定业务成功。
```

数据库与 ArtifactIO 目录都在候选工作区外。数据库保存 Job、PlanRevision、Task、Attempt、JobBudget、Operation、Worker lease、事件和引用索引；所有记录都有 schema_version。状态更新、CAS 版本和事件在同一事务提交，重复完成与旧版本被拒绝。内容先经原 ArtifactIO 写入、文件和目录 fsync，再校验完整引用图后提交数据库。孤立 Artifact 只报告，不自动删除。JSON 摘要用于损坏检测，不是对抗同 UID 攻击者的加密认证。

目标 ext4 已执行真实 SQLite 写锁竞争测试，使用 WAL + synchronous=FULL。未确认文件系统不自动启用；显式 delete 模式是供经过环境验证的部署使用，不代表所有 NFS/共享文件系统均已支持。

单 Controller 进程持有排他租约；Scheduler 和其受控服务线程在进程内锁下写 SQLite。Worker 不获得数据库对象或写接口。Unix socket 以目录/文件权限及内核 SO_PEERCRED、受监督进程组、PID 启动时间和 boot_id 校验来源，仅接受受信任 TaskSpec 授权的操作提案。socket 协议没有模型密钥或通用文件路径执行接口。

操作日志状态：prepared → started → applied/verified/failed/unknown。编辑意图记录修改前字节、预计修改后字节及摘要，实际写入仍由原 EditFileTool/WriteFileTool 执行。验证仍走原公共 run_verification。确认完成的操作只有在当前源码与结果仍匹配时才可复用；unknown、旧版本或缺结果不会直接重放。

Controller SIGKILL 后，Linux guard 的父退出信号清理受监督进程组；重启再核对 boot_id、PID/start_ticks、guard 命令、原请求身份及租约。它不是 OS 安全沙箱，不承诺处理逃离会话的恶意进程。PID 身份不符时拒绝终止，不误杀其他进程。

恢复策略：

- 已提交完成任务，核对 Artifact、Context/Evidence 引用及工作区后直接装载原结果；已完成补丁不重新执行。
- 没有执行的任务按原依赖继续。已丢失、可证明无副作用的尝试在新的 PlanRevision 中重新运行，沿用历史预算和 Attempt，不恢复旧 Token Loop 的中间帧。
- 丢失 Worker 的 pending 预算完整结算为 unknown 负债，旧结算不删除；原 Job 单调时钟截止时间、恢复次数、计划代数和总任务数不重置。恢复入口先提交计数，防止再次崩溃丢失次数。
- 补丁已落盘但 TaskResult 未提交，即使字节匹配，也只确认 applied；工作流结果仍不完整，隔离为 needs_review。部分字节与意图不匹配、验证运行中失联、未知外部副作用均不自动重放。
- 已提交补丁任务、尚未执行验证的情况，可由原 Scheduler 继续授权验证。验证失败、用户取消、预算终止保留终止语义，不作为“未执行任务”重启。
- 验证命令返回 0 但改变源码版本时，保留真实命令结果，拒绝将任务提交为当前源码已验证；恢复进入人工对账且不重复运行命令。
- 损坏/不兼容 Checkpoint 拒绝，不通过回退旧计划掩盖可能已发生的副作用。对持久化路径、工具、预算的读取也重新施加受信任配置范围。

新的已批准 PlanRevision 可通过同一 runner.execute(child_plan, same_trusted_config) 提交，必须满足 parent/generation 和累计限制。模板注册表需预先授权所需工作流；本阶段不从数据库自动生成新 Planner 客户端，也不恢复 Planner/模型流式输出内部状态。

离线真实崩溃矩阵：

```bash
cd /home/example/ResearchCoder
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
/home/example/miniforge3/envs/researchcoder/bin/python -B -m benchmarks.recovery.run_faults \
  --output-dir /tmp/researchcoder-recovery-new-run
```

矩阵包含 19 个场景/对照，以独立 Controller/Worker 和真实 SIGKILL 注入。文件修改由 inotify 计数，验证由独立外部计数器记录。受控数值文件用于验证恢复机制，不是训练诊断准确率基准；另有原 PyTorch Autograd + Context + 独立梯度 Oracle 集成测试。没有付费模型调用。

将需要人工处理的案例与预先定义的自动恢复案例分别统计，原始失败/拒绝保留。不可把安全拒绝率当业务成功率，也不能声称任意 Shell、外部服务或所有工具都有 exactly-once 保证。长期 Memory、分布式调度、OS 沙箱、FastAPI 和新 MCP 均未实现。
