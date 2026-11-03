# 任务路由与会话中的模式切换

新增 `researchcoder run` 和 `researchcoder chat`。保留原 `code`、`diagnose`、`repair` 入口；不写子命令仍进入原编程模式。新入口在现有 Agent 工作流之上增加任务路由和会话状态。

## 先看离线完整演示

```bash
conda activate researchcoder
cd /home/example/ResearchCoder
python -m corecoder.routing_demo --output-dir .tmp/routing-demo-new
```

输出目录须不存在。演示实际执行断言脚本得到失败日志，依次完成：

1. 编程：通过原文件工具写项目说明。
2. 诊断：从日志提取证据、定位并读取源码，生成报告。
3. 解读：解释已有报告，不开放工具。
4. 修复：重新核对报告和源码版本，编辑整数除法，再执行原断言命令。
5. 返回编程：重新读取修改后的源码。

模型决定由脚本固定，日志、文件读写、报告交接和命令复验真实执行；不调用 API，不代表真实模型修复成功率。`summary.json` 保存模式及结果序列，`session/session.json` 关联逐轮执行文件。

## 实际会话

```bash
researchcoder chat --project /path/to/project --log /path/to/failure.log \
  --verify python -m pytest tests/test_related.py -q
```

`--verify` 必须放在最后。进入会话后依次输入：

```text
排查这次训练报错，核对相关源码
解释报告，为什么会失败？
请修复这个问题
帮我写一个相关函数的使用示例
/status
/quit
```

前三轮分别使用只读调查、无工具报告解读和报告驱动修复。第四轮回到通用编程。报告生成后不会自动修改文件；必须有新的明确修复任务。普通编程任务仍能使用七个原工具，包含文件修改、命令执行和子 Agent；普通函数修复也可直接在 code 模式完成，不强制先生成实验诊断报告。

已有报告可启动新的修复或解读会话：

```bash
researchcoder chat --project /path/to/project --report /path/to/diagnosis.json
```

每次会话只绑定一个项目。相对日志、报告、会话目录以项目为基准；`--log` 与 `--report` 不同时使用。运行配置沿用已有密钥配置；真实运行会调用模型服务。诊断继续使用现有 DeepSeek 诊断接口。

## 手动指定与绑定命令

| 输入 | 行为 |
| --- | --- |
| `/code 任务` | 本轮进入原编程能力 |
| `/diagnose 任务` | 本轮重新调查已绑定日志 |
| `/review 问题` | 核对并解读已有报告；不执行工具 |
| `/repair 任务` | 本轮显式进入修复；要求有效报告 |
| `/log /path/to/new.log` | 绑定日志并清除旧修复上下文 |
| `/report /path/to/diagnosis.json` | 校验并绑定已有报告；清除日志绑定 |
| `/verify ["python","-m","pytest","-q"]` | 设置仅在修复后运行的验证参数数组 |
| `/verify null` | 清除验证命令 |
| `/status`、`/help`、`/quit` | 查看状态、帮助和退出 |

路径包含空格时，可在 `/log`、`/report` 后用一对引号包住完整路径。切换命令只指定本轮；后续仍按入口的默认模式路由。`chat --mode code` 等参数设置该会话的默认模式，单轮 `/...` 可覆盖。

缺少日志或报告时返回 `needs_input`，不构造模型、不调用工具。“继续”这样的含糊输入同样要求明确任务；本入口不会据此自动修复。

## 一次性执行和路由预览

```bash
researchcoder run "帮我写一个排序函数" --project /path/to/project
researchcoder run "排查训练报错" --project /path/to/project --log failure.log
researchcoder run "请修复比例计算" --project /path/to/project --report diagnosis.json \
  --verify python check_ratio.py
researchcoder run "排查训练报错" --project /path/to/project --log failure.log --dry-run
```

`--mode code|diagnose|review|repair` 可覆盖自动规则。`--dry-run` 仅显示基于输入文字及绑定参数的路由，不读日志或报告、不创建会话、不调用 API；因此不证明输入文件存在或报告有效。新入口的诊断默认开启 guided 工具选择、evidence 上下文和 checklist，原独立诊断入口的默认值保持不变。

## 实现与边界

- 路由使用用户本轮请求、显式模式和日志/报告绑定状态的确定性规则，不增加分类模型调用。常见开发、日志排查、修复命令、只读要求及报告咨询分别处理；记录选择原因。它不具备任意自然语言理解保证，可用显式命令处理误分。
- 日志或源码返回的文字不能让控制器切换模式。诊断只开放五个证据工具；报告解读不开放工具，即使模型返回工具调用也不会执行。
- 编程轮次保留同一编码 Agent 的对话；诊断使用独立调查上下文。修复后清空旧编程 Agent，回到 code 时重新读取当前文件。报告解读基于已有报告，不等于新增调查。
- 报告驱动修复沿用项目、引用、报告内容哈希和已观察源码哈希校验。新日志或不完整的新诊断不会保留旧报告作为活动修复上下文。报告使用一次后清除；后续修复需重新调查或显式绑定有效报告。
- 验证失败时，run/chat 会保存新的失败证据，并在 `--max-feedback-rounds` 预算内自动只读重查；通过校验的新报告可供下一条明确修复任务使用，详见 [验证反馈闭环](verification-feedback.md)。
- 没有验证命令时返回 `completed_unverified`；指定命令实际退出零才返回 `verified`。非零退出和超时不记为验证成功。这只说明指定命令的结果，不保证训练恢复。
- 每轮输出独立执行文件，`session.json` 记录任务、路由原因、模式变化、状态和文件关联。编程中途 API 失败会保留已执行工具记录，并隐藏提供方异常正文；不会自动撤销修改。
- 单会话至多 100 个任务，每项任务最多 6000 字符；编程沿用上下文压缩并设置 12 轮上限，诊断和修复沿用已有预算。七个原编程工具的行为保留，子 Agent 内部工具调用不在顶层审计覆盖范围内。
- 会话记录用于审计，尚不能重启后完整恢复新 chat 的编码历史；可重新绑定报告。原 code 会话保存与恢复入口仍可用。新入口也不提供操作系统沙箱或训练调度。

## 回归检查

```bash
python -m pytest tests/test_routing.py -q
python -m pytest tests/ -q
```

覆盖规则路由、显式覆盖、缺少输入、模式隔离、跨轮上下文、过期报告、实际修复及复验、失败审计、API 错误隐藏、命令入口和完整离线会话。测试数量属于工程验证，不能当作真实任务路由准确率或诊断成功率。
