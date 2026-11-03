# M20：有界修复闭环实现与四例真实验证

2026-11-03，协议 `m20-repair-cycle-v1`。完整回归继续暂缓。

新版本的四例真实执行均通过：**4 planned、4 started、4 archived、4 E2E succeeded**。每例仅一次，保留原有候选源码、公共输入、独立 Oracle 和 Job 总预算，没有在试验中调整策略或重跑替换样本。

| 案例 | E2E | 独立 F2P | 独立 P2P | 实际模型请求 | 实际 Token |
|---|---|---:|---:|---:|---:|
| Packaging | 通过 | 5/5 | 10/10 | 11 | 148,060 |
| SAM | 通过 | 3/3 | 3/3 | 12 | 127,144 |
| JSON Pointer | 通过 | 4/4 | 7/7 | 12 | 88,604 |
| ItsDangerous | 通过 | 3/3 | 7/7 | 7 | 62,111 |

总计 **42 次真实请求、50 次工具调用、425,919 实际 Token**，其中输入 416,709、输出 9,210；缓存命中 64,896、未命中 351,813。传输及 Runtime 的 unknown、pending 均为 0。Job wall time 合计 144.59 秒。所有受监督 Worker 进程组已结束，专用代理连接已关闭。

按检查时的 [DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/) 和实际缓存/非高峰 Usage，费用估算 **$0.058492638**；保守准入计量 **$0.1360647**，低于本轮 $1 范围，均不是账单金额。

## 实际修改

### 1. 短纠错预算

`ProjectPolicy.repair_cycle_version="repair-cycle-v1"` 是显式可选策略，默认 `off`。启用时要求 explicit completion、progress accounting 及一次反馈重试。

`researchcoder/evaluation/projects/repair_cycle.py` 在原 `single_limits` 已扣除 Job charged/pending 的结果上切分角色额度。本轮首个写入任务最多 8 次模型请求；保留余量用于最多 6 次请求的纠错任务，仍受原 22 次模型、60 次工具、400,000 Token、600 秒的 Job 上限约束。每次实际请求继续经过原 BudgetedModel，额外调用、未知成本和上下文开销没有独立免计量通道。

`routing.py` 对已完成专家交接后的公开失败采用短纠错准入：至少剩余 4 次模型、3 次工具、20,000 Token、70 秒。取消、权限拒绝、源码变化、pending、unknown、重试耗尽仍优先停止。没有重新创建 Job、重复调度专家或将首轮失败改为成功。

### 2. 调查到修复的阶段提示

`runtime/budget.py`、`runtime/coding.py` 与 `corecoder/workflow.py` 在剩余调用趋紧或连续调查达到阈值时发出具有版本 SHA 的 `coding.action_phase`。此阶段继续保留原权限内的工具，提示复用证据、只做必要的定点读取、提交有依据的修改或报告未完成。原最终报告及验证预留规则仍在。

这是受预算约束的执行阶段信号；不会强迫没有依据的写入，也没有新增工具权限。新增的项目 Prompt 版本为 `project-repair-cycle-v1`，与 M19 区分。

### 3. 取消后的计量归档

旧实验控制器对 `model_requests=None` 和 `unknown_requests=None` 执行数字比较，可能在权限取消之后再次抛出 TypeError。新增 `provider_accounting_unknown` 明确处理 null、非法计数、传输数量不匹配及未知 Usage，新的实验控制器适配使用此函数。原停止原因与未知预算责任保持不变。

对 M19 SAM 第三次真实归档离线重放：旧表达式复现 TypeError，新函数返回 accounting stop，原 200,000 Token 等责任额度未释放，原记录未修改。见 [accounting-replay.json](accounting-replay.json)。

进一步按摘要比对，旧拒绝目标是允许修改清单内的 `sam.py`。旧收据没有保留原 Guard 的细分原因，**不能直接把它断言为跨目录越权，也不能证明它是可安全自动重放的写入**。本轮保持原 Guard，只修复次级归档错误；更细的拒绝原因归档仍是后续改进项。

## 测试与证据边界

- 26 项新增测试通过，包括合法/非法短纠错、计量 null、预算切分、真实 Worker 的错误补丁→公开失败→纠错→F2P/P2P、行动阶段保留工具，以及越权写入继续拒绝。
- 70 项原路由测试在最终源码通过；33 项原权限测试通过，其后改动为路由规则摘要注册和新测试记录字段修正，不影响这些权限路径。合计 129 个不同的相关节点通过，未运行全量回归。
- 初次开发门禁的 16 个失败保留在私有日志：新增规则 SHA 尚未注册、以及新测试误读请求包字段。修复实现注册和测试字段访问后，重跑受影响的新测试；没有删除、skip 或放宽行为断言。
- 四例真实执行均在首次写入 Attempt 完成；**真实纠错次数为 0，真实行动阶段事件为 0**。短纠错和行动阶段由离线真实 Worker 固定动作测试覆盖，本轮真实结果没有单独证明这两项机制带来因果收益。
- M19 是每例三次，本轮每例一次，且 Prompt 与预算分配策略发生变化。因此不将 7/12→4/4 直接包装为稳定成功率提升、统计显著改善或 Multi-Agent 优于 Single。
- 模型根因解释和代码定位仍未独立人工评分，不能把 F2P/P2P 当作根因准确率。

## 文件和版本

正式仓库更新六个既有模块，新增 `repair_cycle.py` 与 `tests/test_project_repair_cycle.py`，共八个文件。原分支 `researchcoder-evidence`、HEAD、历史未提交/未跟踪成果保留；修改前 599 文件快照已逐项验证。发布仓库另更新首页及来源说明。

旧策略默认行为保留；原 M19 成绩与源码恢复点为 Git 提交 `3acd19f172616fd4ceb2b8b6f5ab540200282312`。M20 运行快照包含 601 文件，源码与 Prompt 身份见 [source-identity.json](source-identity.json)、[protocol.json](protocol.json)。发布时仅新增的结果链接、来源说明与运行快照文档不同；两个运行包的字节与本轮 Worker 使用的实现相同。

启用新策略时，在可信控制器配置的 `policy` 中设置：

```json
{
  "repair_cycle_version": "repair-cycle-v1",
  "completion_version": "explicit-v1",
  "progress_version": "progress-v1",
  "max_feedback_rounds": 1
}
```

候选范围、公共命令与独立 Oracle 继续使用原版本。新策略没有 SQLite 模式变更、第二套 Scheduler、额外 Agent 或 RSI。

公开 Trial 是脱敏数值/来源投影，原始私有 Artifact SHA 指向已验证的服务器及本地归档，不代表公开投影文件的摘要。完整模型输入、代码正文 Trace、密钥与本地环境文件没有上传。

```bash
python results/m20/recompute.py
```

该命令只读取记录重算，不访问模型或网络。
