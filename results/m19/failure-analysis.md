# M19 未成功样本复核

本轮只测量冻结版本，没有在试验中修复失败或重跑替换结果。以下复核使用已有 Artifact、候选代码、独立检查和预算账本，不新增模型请求。

| Trial | 原始结果 | 独立检查 | 已确认机制 |
|---|---|---|---|
| 02 / SAM 重复 1 | coding verification_failed → routing_stopped | F2P 0/3；P2P 0/3 | 将 base optimizer 构造错误改为额外传入 `defaults`，没有解决恢复后的 param_groups 同步；错误补丁被公共验证和独立检查拒绝。后续 12 次调用预留要求超过剩余 11 次，控制器没有扩大额度重试。 |
| 03 / JSON Pointer 重复 1 | coding verification_failed → routing_stopped | F2P 0/4；P2P 7/7 | 字符串拒绝分支被放在 Sequence 分支之后，字符串先按 Sequence 处理，新增拒绝条件不可达。剩余 7 次模型调用不满足下一写入任务 12 次预留。 |
| 09 / JSON Pointer 重复 3 | coding budget_exhausted → routing_stopped | F2P 未通过；P2P 通过 | 专家已完成交接，但写入任务在既有预算/调查边界终止；不把专家成功当作整个任务完成。 |
| 11 / Packaging 重复 3 | coding incomplete / unknown | F2P 0/5；P2P 10/10 | 写入任务报告调查窗口关闭，未产生源码修改。此前已读到 hash 与 equality 不一致，但持续读取源码后没有落地补丁。此处 `failure_kind=unknown` 是任务完成分类；传输 Usage 实际已知。 |
| 12 / SAM 重复 3 | permission_violation → cancelled；Trial incomplete | 最终 Oracle 未运行，保留 null | 收到可信 `write_not_allowed`、target_category=unknown 的拒绝，工具未执行。门禁终止 Worker，归档次级 TypeError；Worker 计量不完整，JobBudget 保留 unknown 责任。目标原文及次级异常不能从公开投影完整确定，标为待进一步定位。 |

所有 12 次的专家任务均 succeeded，并生成交接引用。因此本轮主要剩余问题位于“写入落地及失败后的预算协调”，而非没有触发专家。交接存在不证明因果收益：没有同条件取消交接的对照，也没有独立评定专家每个假设的正确性。

可在后续单独版本中考虑小范围调查/写入预算协调、公共失败反馈后的有界纠错和次级异常归档修复。不能据本轮修改原 Oracle、关闭权限校验、回填成功报告或将最后一次取消剔除。任何新版本成绩都应使用独立协议记录。
