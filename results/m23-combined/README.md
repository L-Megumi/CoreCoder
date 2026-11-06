# ResearchCoder · Packaging 与 SAM 六次重复修复验收

每个案例预先登记三次。第一份 Packaging 试验保留原来的未完成结果，其他五个槽位由续跑批次各执行至多一次。

| 案例 | 次数 | 运行状态 | E2E | 公开检查 | F2P | P2P | 模型请求 | 实际 Token | 费用估算（美元） |
| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| packaging-requirement-hash | 1 | incomplete | 未完成端到端交付 | — | — | — | 5 | 48103 | 0.015219 |
| sam-checkpoint-resume | 1 | needs_review | 未完成端到端交付 | — | — | — | 9 | 87476 | 0.028312 |
| packaging-requirement-hash | 2 | cancelled_before_start | — | — | — | — | 0 | — | — |
| sam-checkpoint-resume | 2 | cancelled_before_start | — | — | — | — | 0 | — | — |
| packaging-requirement-hash | 3 | cancelled_before_start | — | — | — | — | 0 | — | — |
| sam-checkpoint-resume | 3 | cancelled_before_start | — | — | — | — | 0 | — | — |

- packaging-requirement-hash 计划 3 次，启动 1 次，独立确认 E2E 修复 0/3。
- sam-checkpoint-resume 计划 3 次，启动 1 次，独立确认 E2E 修复 0/3。

本轮未达到两个案例各三次全部通过的预设验收。

| 终评检查 | 线上已确认执行的命令 | 执行情况待核实的槽位 | 有效线上回执 | 离线独立复跑的命令 |
| --- | ---: | ---: | ---: | ---: |
| F2P | 0 | 0 | 0 | 0 |
| P2P | 0 | 0 | 0 | 0 |

终评次数按候选版本上的检查器进程计数。检查器中的命名断言、基线预检和事后复跑分开统计。横杠表示缺少该项回执。费用列为已知请求的保守估算。

[逐次结果与实测补丁](acceptance.json) · [两份冻结协议与归档摘要](protocol.json) · [独立复验摘要](verification.json)

### SAM 首次修改的独立复核

主修复 Agent 对授权文件 `sam.py` 的第一次编辑已写入。第二次编辑依据的读取记录已过期，守卫拒绝了这次操作。正式试验停在 `needs_review`，没有进入终评。下列结果单列在正式成绩之外。

在隔离副本上重放已写入的第一份补丁，并运行冻结的检查器。公开复现通过，故障检查 3/3、回归检查 3/3。补丁可应用到仓库中的原始故障源码，应用后的摘要与试验现场一致。[中间补丁](sam-intermediate.patch) · [独立复验回执](sam-intermediate-review.json)。
