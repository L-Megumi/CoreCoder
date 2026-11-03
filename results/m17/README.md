# ResearchCoder M17：测试收尾记录

本目录发布 2026-11-03 M17 收尾的评测数据。**离线工程验证已完成，真实 LLM 补测未完成**；真实失败、中断和未启动样本全部保留。

## 1. 上下文消融

同一批 10 个离线案例、4 种策略，共 40 行结果、80 个角色视图。运行入口为 `python -B -m benchmarks.context.run_ablation --output-dir <isolated-output> --input-tokens 4000`。不调用付费模型。

| 策略 | 关键证据保留 | 平均压缩前估算 Token | 平均压缩后估算 Token | 构建失败 | 恢复文档 |
|---|---:|---:|---:|---:|---:|
| A：项目原有上下文策略 | 23/32 | 16,145.6 | 717.9 | 2 | 116 |
| B：普通截断 | 21/32 | 16,145.6 | 888.8 | 0 | 140 |
| C：通用摘要回退 | 17/32 | 16,145.6 | 163.0 | 0 | 140 |
| D：Evidence-Aware | 32/32 | 16,145.6 | 1,392.7 | 0 | 140 |

D 相对未压缩输入的估算 Token 减少 **91.37%**，保留 32/32 预期证据，140 份原文恢复摘要校验通过。D 比 A/B/C 保留更多内容，也使用更多 Token；这里不声称 D 是最省 Token 的策略。

Token 采用字符数/3 估算，不是供应商实际计量。A 是项目已有 EvidenceContextManager 路径；C 是 ContextManager 的确定性提取式摘要回退。固定规则诊断探针只能检查证据可用性，不能代表真实模型诊断准确率。原始消息体未公开，数值、原始记录摘要及引用保留在 `context-records.json`。

## 2. 新增历史缺陷与独立对照

新增 JSON Pointer 标量遍历和 ItsDangerous 未来时间戳两个历史缺陷，使用固定上游版本的源码切片及独立合成 Oracle。不是完整上游仓库基准成绩。

- [python-json-pointer 修复提交](https://github.com/stefankoegl/python-json-pointer/commit/16c4c8e33a30e364d94f7a870a6c79916b5f870d)：错误地将字符串当作序列继续遍历。
- [ItsDangerous 修复提交](https://github.com/pallets/itsdangerous/commit/c30678d19e37011890e2374cca04f7789e101793)：设置 max_age 后仍接受相对验证时钟为未来的签名。

健康、故障、参考修复、故意绕过的负对照 × public/F2P/P2P 共 **24/24 符合预期处置**。故障版本和负对照的预期失败计入对照正确性，不冒充 24 次任务修复成功。负对照能通过 F2P，但被 P2P 拒绝。来源、版本、Oracle 与许可证摘要见 `case-provenance.json`，逐项结果见 `oracle-controls.json`。

## 3. 离线专家交接

Packaging、SAM、JSON Pointer、ItsDangerous 四例使用固定动作 ScriptedLLM，实际运行专家 Worker、写入 Worker、证据交换、文件 Guard 和独立 F2P/P2P。最终 **4/4 通过**；每例为同一 Job 内两个独立 Worker。补丁采用已知参考，不能算真实模型修复率。

初次 SAM 因脚本补丁上下文不足被 Guard 拒绝，F2P 失败；补充三行上下文后通过。初次 Packaging 通过结果被复用，没有重复执行。初始失败和最终结果分别保留在 `scripted-execution-history.json` 与 `scripted-e2e.json`，没有修改 Oracle 或放宽 Guard。

## 4. 真实模型补测：因 Usage unknown 停止

| 案例 | 计划 | 启动 | 最终状态 | F2P / P2P |
|---|---|---|---|---|
| Packaging | 是 | 是 | cancelled / interrupted | 未执行 / 未执行 |
| SAM | 是 | 否 | cancelled_before_start | 未执行 / 未执行 |
| JSON Pointer | 是 | 否 | cancelled_before_start | 未执行 / 未执行 |
| ItsDangerous | 是 | 否 | cancelled_before_start | 未执行 / 未执行 |

计划 4、启动 1、归档 1、修复完成 0；发生 1 次 HTTP 请求、0 次工具调用。模型为 DeepSeek Flash 非 Thinking，输出上限 2,048 Token。专家 Attempt 归档为 timeout，首个模型流中记录 KeyboardInterrupt；未获得实际 Usage，未产生源码修改或最终 F2P/P2P。

这不能归因成模型不会修复或引用失败，也不能据此断言供应商服务器超时。原始记录并未提供 SDK APITimeoutError。完整阶段边界与异常事实见 `interruption-review.json`。

批次按 usage_unknown 门禁停止，保留 **6,853 Token / $0.0038991 准入责任额度**。实际 Token 和费用均为 unknown/null，不能把传输汇总中的已知用量累加器 0 当成实际零成本。此前 M15 的 unknown 责任额度独立保留，没有冲抵。其余三例没有自动重放或另建 Job 绕过停止规则。受监督 Worker 与本轮专用网络通道已清理。

## 5. 版本身份与复算

执行源码为未提交的服务器工作区快照：595 文件；保留 M16 的 543 文件，新增 52 个案例文件，未改已有 runtime、Prompt、测试或 Oracle。`source-identity.json` 记录完整源码清单摘要和可恢复快照摘要。

本结果分支从 GitHub main `7a11892cd31ad6490ab1090b7b8d9c71f90418cb` 创建；它仅发布结果，**不是完整 ResearchCoder 源码发布**。运行时工作区 HEAD 不足以重建那份未提交源码，不能将本次结果的运行身份冒充为此 GitHub commit。

下载后的 9,063,463 字节原始归档及其中 2,559 份记录均已校验 SHA256；完整归档及源码快照本地保留。公开文件采用白名单提取，省略原始模型消息、私有路径和启动环境；这些派生文件的摘要不等于原始 Artifact 摘要，公开目录不提供原 Artifact 的直接恢复服务。

在本目录执行 `python -B recompute.py`，可从逐案例记录重新生成并核对汇总，不调用模型。各文件摘要见 `publication-manifest.json`。

按开发约定，本轮没有执行集中全量 pytest 或安装验收。M16 的 45 个已通过定向节点仅作为相同 runtime/test 字节的继承记录，没有称为 M17 新跑结果。本目录也不覆盖 Phase 6E 或其他历史版本成绩，不能用于宣称 Multi-Agent 相对 Single 的稳定优势。
