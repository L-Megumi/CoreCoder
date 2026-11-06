# ResearchCoder M23 v2 · 历史缺陷重复修复验收

Packaging 和 SAM 各登记三次独立自然启动，沿用公开任务和独立检查。E2E 同时要求工作流完成、有实际授权修改、公开复现和 F2P/P2P 通过。

| 案例 | 次数 | 状态 | E2E | 公开复现 | F2P | P2P | 请求 | 实际 Token | 保守费用估算（美元） |
| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| packaging-requirement-hash | 1 | incomplete / safety_or_permission_violation（复核数据不足） | — | — | — | — | 5 | — | — |
| sam-checkpoint-resume | 1 | cancelled_before_start | — | — | — | — | — | — | — |
| packaging-requirement-hash | 2 | cancelled_before_start | — | — | — | — | — | — | — |
| sam-checkpoint-resume | 2 | cancelled_before_start | — | — | — | — | — | — | — |
| packaging-requirement-hash | 3 | cancelled_before_start | — | — | — | — | — | — | — |
| sam-checkpoint-resume | 3 | cancelled_before_start | — | — | — | — | — | — | — |

每例目标为同一冻结配置连续三份独立 E2E 成功。
- packaging-requirement-hash 计划 3 次，启动 1 次，独立确认 E2E 成功 0/3。
- sam-checkpoint-resume 计划 3 次，启动 0 次，独立确认 E2E 成功 0/3。

本轮未达到两例各三次独立修复成功的验收目标。

每个案例仅三次，结果仅描述这些开发案例与本轮配置。表中费用取自计量器的保守计费估算；缺失数据用横杠标明。

本轮沿用 v1 通过 82 项定向测试的源码版本。对 0 份具有可核验最终源码的候选，在隔离副本重新运行冻结的公开、F2P 和 P2P 检查。

[逐次验收与补丁索引](acceptance.json) · [冻结协议](protocol.json) · [测试、变更及私有归档摘要](verification.json)

[v1 首次请求中断记录](../m23/README.md)单独保留。此前 1 次未知请求只计入累计准入预算，未并入 v2 修复样本。

本次控制器收到 5 次带完整用量的模型响应。专家交付了 24 条证据，随后验证器因自身安装路径经过符号链接，拒绝加载公共检查脚本。独立复核确认候选项目的 132 个文件内容一致，写入任务尚未启动。实体路径与链接路径的源码清单一致（608 个文件），同一份公共检查脚本在实体路径下通过原检查。此记录保留首个 Packaging 未完成及其余五项未启动时的状态。
