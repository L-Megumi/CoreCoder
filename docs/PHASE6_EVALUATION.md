# Phase 6A–6B：Benchmark 与评测基础设施

新模块 `researchcoder.evaluation` 复用既有合同、ArtifactIO、CoreCoderAdapter、ResearchHarness、DAGScheduler、Context Engine 和 RecoveryCoordinator。它不实现第二套 Agent Loop 或 Verifier。Linux 是正式执行平台。

## 运行与重算

```bash
python -B -m researchcoder.evaluation list
python -B -m researchcoder.evaluation run --output-dir /tmp/new-evaluation \
  --torch-python "$RESEARCHCODER_EXPERIMENT_PYTHON" --offline-observability
python -B -m researchcoder.evaluation report /tmp/new-evaluation \
  --output /tmp/new-evaluation-recomputed.json
python -B -m researchcoder.evaluation.pilot
```

默认 ScriptedLLM，无付费模型请求、无 Langfuse 网络发送。输出目录必须是新的目录，且位于源码安装目录外。`report` 只读取保存的身份、原始记录及 Artifact，不调用模型。先保存全部 planned trials，再逐条归档结果；缺失、失败、超时、取消、预算停止样本均保留在分母中。重算标明原始与当前 scorer 摘要，不隐瞒评分器版本变化。

## 案例与 Oracle

包内 data 提供 8 个 CPU 案例：autograd_detach、shape_transpose、dtype_features、nonfinite_logsigmoid、label_indices、optimizer_omission、normalization_scale、crossfile_batch。开发集 4、验证集 2、留出集 2；公开的小型自建数据集不等同于第三方权威基准，公开源文件也不构成保密留出集。

每例声明项目版本、源码 SHA256、公共目标/日志、根因词组规则、文件级定位答案、允许修改范围及独立 F2P/P2P 条件。Oracle、答案、健康源码和 seeds 留在评测控制器侧；Agent 工具只访问自己的候选工作区。`public_input` 是显式白名单，不能把完整 BenchmarkCase 传给 Worker。ScriptedLLM 的固定动作由控制器构造，仅用于机制验证，不是推理效果测量或 live few-shot。

外部 CPU Oracle 复用公共 Verifier。Preflight 检查健康 F2P/P2P 均通过，故障 F2P 失败但已有 P2P 通过；修复后分别验证 F2P/P2P。原 autograd 的源码及 Oracle 字节复用，并保留有意冻结 backbone 的负对照。其他案例独立检查形状、值、梯度、参数身份/更新、输入不变性和跨文件行为。候选限制为小型训练代码，静态审查拒绝明显文件/反射执行，外部 Oracle 有摘要保护；这不是恶意代码 OS 沙箱或防御所有 Python 逃逸的证明。

## 四种模式与公平边界

- single：原 CodingTaskSpec/ResearchHarness/RepairAgent 路径。
- sequential：相同 5 节点 DAG，max_workers=1。
- parallel：相同 DAG，两个独立只读 Worker 并行，单写入修复。
- evidence：parallel DAG 加原 Evidence-Aware Context Engine；其他策略可以通过配置覆盖做消融。

模式共享模型、输出上限、全局 JobLimits、源码读写范围、公共日志、Oracle 和任务目标。Single 把同一全局额度交给一个 Worker，Multi 分配给角色；额外角色不能增加 Job 总预算。Single 使用原编程工具，Multi 使用诊断角色接口，因此 Single/Multi 比较还包含角色 API/Prompt 差异，不能作为只改变并行度的纯消融；sequential/parallel 的图和角色相同。各次复制独立候选目录、Artifact 与 SQLite。案例/模式顺序可轮换并支持重复试验。

Multi 模式调用原 RecoveryCoordinator 持久化 DAG。Single 复用原 Harness，不声称增加了 Single CodingTask 的自动持久化恢复。评测批次自身不自动恢复执行中的试验；可对保存的完整/部分记录重算，未知副作用仍按 Phase 5 处理，不能重跑未知任务冒充幂等。

## 指标口径

端到端成功要求原工作流完成、允许范围完整、独立 F2P 和 P2P 同时通过。根因指标为独立词组规则匹配，定位为模型明确结论中的文件级匹配，均不能冒充语义评审或行号精度。原始源码观测不自动获得定位得分。

实际 Usage 不完整则实际总 Token 与 Cost 为 null；保留预算原记录的已知子集、估算输入和 unknown。只有完整 Usage 与被确认适用的统一费率才能计算 Cost；缓存、时段及推理计费信息不足时仅给规划情景。ScriptedLLM 的调用数是工程调用数，非供应商付费请求数。

上下文保留率按 Model Gateway 入口前后公共证据签名的保留机会计算。只保存哈希，不记录原始 Prompt/日志；它不覆盖在 Gateway 前已经被旧 ContextManager 删除的内容。审计量有大小上限，不完整时返回 unknown。没有发生恢复/没有物理副作用计数器的样本，恢复与重复副作用指标为 null，不默认填 0。Phase 5 测量记录可通过 recovery_metrics 单独汇总。

批次准入读取原 JobBudget 的消耗/unknown 预占，不创建第二套任务预算；支持全批模型请求/Token 责任额度、失败次数和 unknown Usage 停止。停止后剩余 planned trials 记录为未执行的取消样本。准入估算和事后实际结算不能被描述为严格美元费用上限。报告含每例、宏观、覆盖率、重复标准差及描述性 Wilson 区间；重复同一案例具有相关性，不应直接宣称 IID 推断成立。

## 可选 Langfuse

默认关闭；offline 模式产生本地白名单投影。外部投影仅含数值、固定枚举、稳定伪名身份与父子关联，不传源码、Prompt、日志、环境变量、路径或 Artifact 正文。历史时间作为 metadata 保存，离线回放的 SDK span 时间是导出时间，不能冒充原运行瀑布图。

可选依赖为 Python 3.11+ 的 Langfuse v5；使用 `start_observation`、`create_score` 和 `flush`，不使用旧 trace()/generation()。依据[官方兼容说明](https://langfuse.com/docs/compatibility)及[Python API 参考](https://python.reference.langfuse.com/langfuse)。当前仅以 Mock SDK 验证签名、关联、Usage、评分和连接失败隔离，未安装 SDK、未上传到真实服务。实际服务端兼容、处理完成与可见性仍需独立验证。

```bash
# 默认仍是离线导出
python -B -m researchcoder.evaluation export /tmp/new-evaluation --output /tmp/offline-projection.json
# 仅在获准外发后才使用 --enable-network，并由运行环境提供凭据。
```

导出发生在本地结果归档后；禁用、缺依赖、缺凭据、网络异常均不改变已完成 Agent 任务。本阶段不实现可靠导出队列或 exactly-once 上传，SDK 异步 flush 不等于已验证服务端入库。

## 真实模型准备

现有 Worker 支持 ScriptedLLM 和 OpenAI Chat Completions 兼容接口；所有模型请求走 BudgetedModel，框架重试/stream_options 回退的每次 transport 均计量，SDK 自动重试关闭。新增 ModelRequestOptions 只允许显式 DeepSeek 非 thinking 选项，通过受信任配置进入原 Worker；默认 None 不进入原 Checkpoint authority/fingerprint，旧默认身份不变。完整 thinking reasoning_content 历史未实现。

[DeepSeek 官方非 thinking 工具调用文档](https://api-docs.deepseek.com/guides/tool_calls/)规定对应请求参数；当前只有 Mock 验证，尚未进行真实 Provider 兼容试验。Pilot 方案由 `evaluation.pilot` 生成，包括 3 个开发案例、四模式、12 trials、请求/Token 预算与失败/unknown 停止。价格情景引用[官方价格表](https://api-docs.deepseek.com/quick_start/pricing/)，执行前应再次核对。

在用户明确批准模型、API 预算和范围之前，不执行 live run。CLI 对 live 另需显式 --allow-live；RSI、自动策略优化和 Phase 7 不在本阶段范围。
