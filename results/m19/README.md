# M19：统一最终配置的重复验证与完整源码发布

2026-11-03。协议：`m19-unified-final-v1`。

四个案例各运行三次，12 次全部启动、归档；E2E 成功 **7/12（58.3%）**。此前 M18 的四例成功记录来自不同轮次，本轮没有借用那些成功记录，也没有替换失败样本。

| 案例 | 重复 1 | 重复 2 | 重复 3 | 本轮成功 | 实际请求 | 实际 Token |
|---|---|---|---|---:|---:|---:|
| Packaging | 成功 | 成功 | 未完成 | 2/3 | 34 | 438,006 |
| SAM | 修复失败 | 成功 | 权限停止、取消 | 1/3 | 34 | 380,006 |
| JSON Pointer | 修复失败 | 成功 | 预算停止 | 1/3 | 38 | 322,463 |
| ItsDangerous | 成功 | 成功 | 成功 | 3/3 | 30 | 260,928 |

结论：可以确认四例在同一配置下均有真实成功记录，但**不能确认四例均稳定成功**。每例仅三次，均属于 development 案例；本轮没有 Single 对照，不能据此宣称 Multi-Agent 带来净收益、压缩在线降本或泛化提升。

## 冻结条件

- `deepseek-flash`，非 Thinking，流式 Tool Calling，每请求输出上限 2,048 Token。
- 每 Job：22 次模型调用、60 次工具调用、400,000 Token、600 秒。
- Code 专家：6 次模型、10 次工具、80,000 Token、180 秒；写入任务：12 次模型、24 次工具、200,000 Token、300 秒。
- `code-first-v1`、`readiness-v3`、`streaming-v1`、`focused-repair-v1`、`bounded-excerpts-v1`；每次使用独立候选工作区与 Artifact。
- 三轮轮换案例顺序；整个试验中没有调整 Prompt、角色权限、预算、源码或 Oracle。
- 12 次均有成功的 Code 专家任务及交接记录；专家只读，后续写入任务消费交接。此处是显式专家优先策略，不能当作自适应触发优越性的证据。
- 公共复现、候选源码、独立 F2P/P2P 与 M18 使用的案例身份保持一致。保护性 Oracle 与参考材料位于候选目录之外。

原始模型调用共有 **136 次**，输入 **1,369,319**、输出 **32,084**、合计 **1,401,403 Token**。缓存命中 223,104、未命中 1,146,215；传输层所有请求 Usage 已知，无 pending。

按检查时的 [DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/) 和实际缓存、非高峰时段 Usage，费用估算为 **$0.191851962**；控制器不计缓存优惠的保守准入计量为 **$0.4492965**，均低于 $3 范围。这不是账单金额。

**传输 Usage 已知不代表 Runtime 已完整结算。** SAM 第三次被权限门禁终止，Worker 未交回完整计量；JobBudget 保留该任务 12 次模型、24 次工具、200,000 Token 的 unknown 责任额度。未把它改为零或按旁路观测强制释放。详见 [summary.json](summary.json)。

## 任务状态与独立验证

原工作流返回 7 succeeded、4 incomplete、1 cancelled。试验控制器将最后一次取消保留为 incomplete Trial。11 次完成最终独立检查，1 次因取消未完成检查；F2P 通过 7 次，P2P 通过 10 次。最终 E2E 分母仍为全部 12 次，未将取消或失败移出统计。

根因准确率和定位准确率仍为 `null / not_scored_model_hypothesis_only`，没有将模型解释当作独立人工评分。Job wall time 合计 468.31 秒，不等于整个批次的总耗时。

五次未成功的具体原因见 [failure-analysis.md](failure-analysis.md)。最后一次发生 `write_not_allowed` 的可信拒绝，停止记录同时归档了 `TypeError`；未知目标类别和次级异常尚未完全定位，没有据此声称发生了越界写入，工具拒绝记录为 `tool_executed=false`。

## 完整源码和复现材料

本分支发布 **786 个源码、测试、文档、配置及案例文件**：保留原有 599 个文件，新增 187 个可移植复现材料。`corecoder/`、`researchcoder/` 与冻结试验实现逐文件相同；文档和示例日志中的个人路径已匿名化，两个精确断言同步使用匿名化路径，未放宽断言。

四个历史案例的固定故障源代码、公开复现和独立 Oracle 位于 [`benchmarks/repositories/`](../../benchmarks/repositories/)。源码版本、许可证及 Oracle 原始/发布摘要见各案例 `provenance.json`。只改用了相对路径入口；底层 Oracle 和适配器原文没有修改。

原正式开发仓库的 599 个文件、分支、HEAD 和全部未提交/未跟踪状态保持不变。发布从独立目录生成，保留可恢复快照，未对正式工作区执行 reset、clean、pull、merge 或 commit。

本轮按用户要求**停止完整回归，并未重新运行**。实际完成的必要检查：

- 匿名化证据素材相关测试：28 passed。
- 四例故障候选 public/F2P/P2P：12 次检查符合预期（public/F2P 失败、P2P 通过）。
- 四例真实成功候选使用可移植 Oracle 复核 F2P/P2P：8/8 通过。
- Wheel 构建、独立目录安装、两个包的导入及原 CLI、Research CLI、Evaluation CLI 帮助检查通过。

服务器缺少构建后端的首次 Wheel 检查失败被保留；随后只在独立临时目录安装项目已声明的 hatchling 及其构建依赖，再次构建通过。未修改模型运行环境或升级项目依赖。

## 公开记录与离线重算

- [protocol.json](protocol.json)：冻结的轮换顺序、角色配置和预算。
- [trials/](trials/)：12 个 Trial 的数值、状态、Oracle、请求 Usage、身份和预算投影。
- [source-identity.json](source-identity.json)：786 文件清单、摘要、发布差异及私有归档身份。
- [release-checks.json](release-checks.json)、[release-verification.json](release-verification.json)：必要检查及原工作区保持证明。
- [publication-review.json](publication-review.json)：发布范围审查。

```bash
python results/m19/recompute.py
```

重算只读取已保存记录，不调用模型。公开文件是脱敏的数值与来源投影；`original_raw_artifact` 等摘要指向保留在私有归档中的原始字节，不能拿公开投影重新计算原始 Artifact SHA。私有日志、源码正文 Trace、环境配置、密钥、SQLite、权重和数据集不作为发布结果上传。

M17/M18 历史记录保持独立，未被覆盖或重新标为本协议结果。
