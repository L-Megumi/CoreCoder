# ResearchCoder 统一入口与修复交接

ResearchCoder 面向机器学习研发，保留通用代码开发能力，并把只读故障调查交接给代码修改和验证。
显式子命令和报告交接保持可用；新增 run/chat 支持确定性任务路由和会话模式切换，见 [任务路由与会话指南](task-routing.md)。

## 入口

在服务器项目目录执行：

```bash
conda activate researchcoder
cd /home/example/ResearchCoder
python -m corecoder.researchcoder --help
```

安装当前源码后也可使用 `researchcoder` 命令，原 `corecoder` 入口保持兼容。

| 命令 | 用途 |
| --- | --- |
| `researchcoder code` | 原编程 Agent，保留读写、编辑、搜索、命令执行、会话和子 Agent 能力 |
| `researchcoder diagnose` | 现有只读诊断，输出证据、观察和修复建议 |
| `researchcoder repair` | 显式进入修改阶段，使用已完成的诊断报告和用户任务 |
| `researchcoder demo` | 无 API 的真实文件闭环演示 |
| `researchcoder run` | 一次性任务路由，可显式覆盖模式 |
| `researchcoder chat` | 连续会话，衔接编程、诊断、报告解读和修复 |

不写子命令时进入原编程模式。模块入口与命令入口的参数相同。
统一入口的 DeepSeek 编程模式复用已配置的密钥和接口，并使用普通文本输出，不继承诊断报告的 JSON 输出约束。

## 先看完整流程

输出目录须不存在：

```bash
python -m corecoder.researchcoder demo --output-dir .tmp/workflow-demo-new
```

演示会创建独立小项目，实际运行断言脚本得到失败 traceback，调用只读诊断工具，生成诊断报告，再通过原文件编辑工具修改整数除法，重新执行原断言脚本。
模型的调查与修改决定由脚本固定，文件读取、修改和验证命令真实执行；没有 API 调用，也不代表真实模型效果。
输出包含 `failure.log`、`diagnosis.json`、`repair.json` 和 `summary.json`，保留修改后的项目。

## 用于实际项目

通用编程入口支持原参数，新增项目目录选择：

```bash
researchcoder code --project /path/to/project -p "阅读项目并说明训练入口，不修改文件"
```

先诊断，保存完整 JSON：

```bash
researchcoder diagnose /path/to/failure.log --project /path/to/project \
  --completion-mode checklist --output .tmp/diagnosis-new.json
```

准备好任务后，显式启动修复；这一步会允许修改代码和执行命令：

```bash
researchcoder repair .tmp/diagnosis-new.json --project /path/to/project \
  --instruction "核对诊断假设，修复相关实现，保留原测试" \
  --output .tmp/repair-new.json \
  --verify python -m pytest tests/test_related.py -q
```

`--verify` 必须放在最后，后面全部作为命令参数；由程序在项目目录执行，不经过 shell 展开。
未提供 `--verify` 时，模型总结不能将结果升级为验证通过。实际 API 模式使用配置的服务并产生调用开销。

## 交接与结果

- 修复前重新校验报告结构和引用，要求项目一致，并核对已观察文件的内容哈希。
- 报告或文件在准备交接后发生变化时拒绝执行，要求重新调查；旧报告不会悄悄套用于另一份源码。
- 修复阶段复用六个直接编程工具，不开放子 Agent，以保持本次工具预算和审计完整；通用编程模式仍保留全部七个工具。
- 默认修复最多 12 轮、24 次工具请求；验证命令默认超时 30 秒。失败或中断时保存已执行的工具记录和可追踪文件变化，不自动回滚。
- 输出文件必须是新文件，避免覆盖旧报告或源码。

| status | 含义 |
| --- | --- |
| `verified` | 用户指定的验证命令实际退出 0 |
| `completed_unverified` | 修改阶段结束，但没有执行独立验证命令 |
| `verification_failed` | 验证命令非零退出或超时 |
| `incomplete` | 模型请求失败、预算耗尽或执行异常 |
| `interrupted` | 用户中断，需检查已发生的修改 |

`verified` 只代表该命令通过，不证明所有测试通过或训练已恢复。文件变化列表覆盖已观察文件和文件工具目标，不能穷举 shell 的所有副作用。
这些工具不是操作系统沙箱；修复建议仍需根据任务核查。当前尚未实现自动补丁审批、事务回滚或通用训练调度。

## 回归检查

```bash
python -m pytest tests/test_workflow.py -q
python -m pytest tests/ -q
```

开发继续围绕完整工作流推进。无堆栈源码搜索仍待实现，正式效果测评和系统讲解后置。
