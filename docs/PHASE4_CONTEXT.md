# Evidence-Aware Context（Phase 4）

Linux 正式接口是 `ExecutionConfig.context_policy`。默认 `None` 保留 Phase 3 行为；启用新上下文不替换 CoreCoder Agent Loop、Adapter、Worker 或 Scheduler。

```python
from dataclasses import replace
from researchcoder.context.policy import ContextPolicy

# config 是控制器创建的 ExecutionConfig，角色取自已有五类角色。
config = replace(config, context_policy=ContextPolicy(
    strategy="evidence", role="diagnosis", allow_recovery=True,
))
result = harness.execute(task_spec, config)
```

DAG 模式对已有各 TaskTemplate 的 config 作相同配置。Scheduler 根据受信任的任务角色重新绑定 policy.role，将授权 EvidenceBundle 的 ArtifactRef 传给 Worker；不在模型提示中复制完整上游报告。`allow_recovery` 默认关闭，控制器明确开启后才会注册只读 `restore_context_evidence`，计入原工具预算。它只接受已登记的引用或原始证据 ID，不接受文件路径。跨生产者有同名 ID 时使用唯一 DocumentRef；Store 层还支持 task/attempt 查询。

原始日志、工具 JSON、源码页、验证 stdout/stderr 和假设分别保存，摘要不覆盖原文。内容复用位于原 ArtifactIO 的 Job 目录，SHA256 校验、不可覆盖写入和 Linux 文件锁保证并行 Worker 不读到半写入内容。Catalog 是可序列化引用目录，独立于运行态 TaskState；本阶段没有任务恢复或数据库。

管线在模型请求前执行：完整工具组校验 → 原文归档 → 重复内容/数值告警聚合 → 异常因果链、应用栈帧、源码及失败尝试提取 → 角色/任务关键词排序 → 预算视图。工具调用与响应成组保留或移除。每次构建保存 catalog、receipt、估算统计，Trace 只记录元数据。Agent 内原 ContextManager 由透传 facade 接管，历史原文不会在模型入口之前丢失；旧 CLI 未启用此配置时保持原行为。

模型调用仍经过 BudgetedModel / BudgetLedger，Worker 的计量仍由原 JobBudget 结算。上下文构建占用原墙钟预算；指令、工具 schema、输出预留先扣除，再按角色权重和实际需求分配证据、源码和历史空间，未用额度归还其他分区。Token 为字符数/3 的估算，与供应商 Usage 独立；缺失 Usage 保持 unknown。没有严格计费上限保证。

原文恢复返回摘要、来源与 current_eligible。源码变化后必须显式请求 historical，历史内容继续标记为历史观察。真实读操作在产生时归档版本，避免同一轮先读后改造成版本错绑。观察、历史观察和假设始终区分；外部日志、源码注释、上游假设和旧反馈 JSON 作为 user/tool 数据处理。

复现实验（已有 Linux 环境，无付费 API）：

```bash
export PYTHONDONTWRITEBYTECODE=1
export RESEARCHCODER_EXPERIMENT_PYTHON=/home/example/miniforge3/envs/M2D/bin/python
/home/example/miniforge3/envs/researchcoder/bin/python -B -m benchmarks.context.run_ablation \
  --output-dir /tmp/researchcoder-context-ablation-new --include-dag
```

10 个固定案例各运行 A 原有 EvidenceContextManager、B 普通首尾截断、C 原通用摘要的确定性回退、D 证据管线，输入上限均为 4000 估算 Token。独立 `cases.json` 规定证据与故障/位置答案；固定规则 probe 无权访问原文或答案。另有四组真实 Worker/Adapter/工具的 Autograd DAG，以原独立梯度 Oracle 验证修复。ScriptedLLM 只提供固定动作，不能据此声称真实模型诊断准确率提升。所有原始视图、失败、计时和恢复引用均写入 results.json。

限制：有限证据卡片和应用栈帧选择不保证无限日志的每条证据都在提示中，省略部分通过原文引用恢复；原核心工具的分页/帧数限制仍存在。通用摘要组使用离线回退，不代表付费模型摘要能力。短上下文可能因来源元数据而增加 Token；构建耗时和单次 CPU 测量不构成在线性能结论。恢复响应有大小上限，长原文应按行读取。Linux 为正式执行平台，未新增 Windows 多进程/文件锁兼容承诺。
