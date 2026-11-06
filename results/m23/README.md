# ResearchCoder M23 · 历史缺陷重复修复验收

Packaging 和 SAM 各登记三次独立自然启动，沿用公开任务和独立检查。E2E 同时要求工作流完成、有实际授权修改、公开复现和 F2P/P2P 通过。

| 案例 | 次数 | 状态 | E2E | 公开复现 | F2P | P2P | 请求 | 实际 Token | 保守费用估算（美元） |
| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| packaging-requirement-hash | 1 | cancelled / usage_unknown（复核数据不足） | — | — | — | — | 1 | — | — |
| sam-checkpoint-resume | 1 | cancelled_before_start | — | — | — | — | — | — | — |
| packaging-requirement-hash | 2 | cancelled_before_start | — | — | — | — | — | — | — |
| sam-checkpoint-resume | 2 | cancelled_before_start | — | — | — | — | — | — | — |
| packaging-requirement-hash | 3 | cancelled_before_start | — | — | — | — | — | — | — |
| sam-checkpoint-resume | 3 | cancelled_before_start | — | — | — | — | — | — | — |

两例各三次的在线稳定性还有待验证。六个登记项中，Packaging 启动一次，SAM 尚未启动。SSH 转发在首个请求前约二十分钟断开。请求记录为连接异常，用量未知。控制器保留 6,812 Token 和 0.0038868 美元的最高预留，实际费用未知。本次没有产出候选补丁。

历史 Packaging 交接的离线复放中，旧格式即使移除全部摘录仍需 64,084 字符。新格式使用 49,907 字符交付 34 条可恢复证据，全部出现在修复 Agent 的首轮输入。真实消息的输入估算为 15,737 Token，模型由 ScriptedLLM 替代。

SAM 的新策略将当前源码引用的确认安排在编辑前，并将最近一次实际改动的有界差异绑定到源码摘要。独立故障检查和回归检查继续决定最终修复判定。

Linux 定向测试通过 82 项。本轮按约定仅执行改动相关测试。

[逐次验收与补丁索引](acceptance.json) · [冻结协议](protocol.json) · [测试、变更及私有归档摘要](verification.json)
