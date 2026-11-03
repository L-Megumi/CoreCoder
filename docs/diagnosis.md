# ResearchCoder 第三阶段：证据驱动诊断 Agent

本阶段把证据提取、源码定位和当前文件/目录检查接入 CoreCoder 的 Agent Loop。
模型根据工具返回内容决定下一步调查，并输出带实际引用的结构化报告。
使用 DeepSeek 的非思考模式工具调用。后续工具选择与证据上下文优化已接入，详见 [优化和对照实验指南](optimization.md)。

## 1. 先运行离线闭环

```bash
source /home/example/miniforge3/etc/profile.d/conda.sh
conda activate researchcoder
cd /home/example/ResearchCoder
mkdir -p .tmp
python -m corecoder.diagnose --demo --output .tmp/diagnosis-demo.json --markdown .tmp/diagnosis-demo.md
python -m pytest tests/test_diagnostic_tools.py tests/test_diagnose.py -q
```

离线演示会在服务器临时目录创建小型项目和一份链式异常日志，依次调用四个真实工具。
最终报告的路径状态和引用 ID 来自实际工具返回；调用顺序和结论模板由脚本固定。
输出标记 `mode=offline_scripted` 和模型 `scripted-diagnosis-demo`。
该演示验证工程闭环，不代表 DeepSeek 的诊断能力、准确率或优化收益。演示不会访问 API。
演示目录保留在结果的 `demo_workspace` 中，便于检查其中的日志和源码。

## 2. 在服务器交互配置 DeepSeek

在项目目录执行：

```bash
python -m corecoder.diagnose --configure-deepseek
```

请在交互式 SSH 终端运行；非交互式环境会拒绝输入密钥。程序会隐藏输入，写入项目 `.env` 中的 `DEEPSEEK_API_KEY`、`CORECODER_MODEL`、`DEEPSEEK_BASE_URL`，保留其他已有配置项。
文件权限设置为仅当前用户读写；Git 已忽略 `.env`。密钥不写入报告，不通过命令行参数传递。
配置命令不会请求 API，也不会修改当前进程以外终端的环境变量；后续运行会从项目目录加载 `.env`。
如果已在终端导出同名变量，终端变量优先于 `.env`。

默认模型为 `deepseek-flash`，接口为 `https://api.deepseek.com`。可以在配置命令后用 `--model`、`--base-url` 覆盖。
默认值依据 2026-10-20 核对的 [DeepSeek 官方接入说明](https://api-docs.deepseek.com/)；模型别名指向的版本可能变化。
为便于后续对照实验，应记录运行日期、模型名称和配置，正式实验时尽可能固定可用版本。

适配层明确传入 `thinking.type=disabled`。开启思考模式需要额外处理多轮工具调用中的推理字段，本版没有启用。
接口格式依据 [官方工具调用说明](https://api-docs.deepseek.com/guides/tool_calls/)。
同时启用 `response_format={"type":"json_object"}`，提示中提供 JSON 结构示例，依据 [官方 JSON 输出说明](https://api-docs.deepseek.com/guides/json_mode/)。这只约束 JSON 语法，字段和引用仍需本地校验。
不会把无关的 `OPENAI_API_KEY` 自动发送给 DeepSeek；优先使用 `DEEPSEEK_API_KEY`，其次为显式 `CORECODER_API_KEY`。

## 3. 真实案例调用

```bash
python -m corecoder.diagnose examples/logs/dronevehicle_missing_path.log \
  --project /home/example/M2D-LIF \
  --allow-path-root /data/example/M2D-LIF \
  --output .tmp/dronevehicle-diagnosis.json \
  --markdown .tmp/dronevehicle-diagnosis.md
```

`--allow-path-root` 仅增加路径是否存在的元数据检查范围，不允许读取项目外文件内容。
这里明确允许检查历史日志中的 `/data/example/M2D-LIF` 路径，便于和当前配置进行比较。
项目外的 Ultralytics settings 文件仍会标记为范围外，而不是被误认为缺失。
`--map-root OLD=NEW` 用于源码定位和读取的显式迁移；`inspect_paths` 始终检查传入的字面路径，不把映射后存在当成旧路径存在。

本案例是历史失败日志。模型应将过去的异常与当前 YAML、目录状态分开描述，不能声称训练已经修复或已重跑验证。
真实调用会发送相关日志和源码片段给你配置的模型接口，并产生该接口的调用开销。

## 五个调查工具

| 工具 | 功能 |
| --- | --- |
| `extract_log_evidence` | 提取绑定日志的证据，支持分页；返回稳定 ID、异常消息、堆栈和因果关系 |
| `locate_source` | 根据已提取的证据 ID 核验当前源码位置和关联配置 |
| `search_project_source` | 按已展示证据的关键词，在有界 Python 源码范围搜索候选；随后读取核查 |
| `read_project_file` | 按行读取项目内支持的代码或配置，返回内容哈希与观察 ID |
| `inspect_paths` | 在允许范围内检查目录/文件是否存在，返回实际观察 ID |

没有通用 shell、文件修改、训练执行和子 Agent 工具。模型提出的修复步骤仅作为建议输出。
同一调查中的工具状态更新按顺序串行化，以兼容框架可能并行分发工具的行为。
依赖先前结果的操作仍需由模型放在后续轮次。

## 报告和引用

JSON 结果包含：

- `status`：`completed` 或 `incomplete`。
- `mode`、`model`、项目目录和日志来源。
- `diagnosis`：摘要、历史故障、当前观察、原因假设、验证方法、建议动作和局限。
- `evidence`：完整证据原文和结构；工具展示截断不会修改该存储。
- `observations`：当前源码、配置或目录检查的实际结果，以 `obs_...` 编号。
- `tool_events`：调用名称、参数、顺序、耗时、成功/失败及观察引用。
- `metrics`：模型调用数、工具调用数、工具错误数、提供方返回的 Token 用量和总耗时。
- `report_attempts`：各次报告原文及校验错误；不静默删除模型返回的额外字段。

`ev_...` 引用历史日志证据；`obs_...` 引用本次实际调查的观察。Markdown 报告附引用索引，完整内容在 JSON 中回查。

报告校验要求：至少提取到一条日志证据、取得可用的当前文件/路径观察、字段结构完整、所引 ID 必须真实存在。
`current_state` 自身必须引用至少一个可用的当前观察，不能仅引用范围外的路径或没有源码片段的定位记录。
校验不评判模型结论的语义正确性，也不能证明某个实际存在的引用充分支持该结论。
`completed` 表示通过结构与引用校验，不等同于“根因已证实”或“训练故障已解决”。

如果模型返回无效 JSON、伪造引用、调用失败或预算耗尽，结果标记 `incomplete`，保存已有证据和调用记录，退出码为 2。
API 错误正文不保存，以免服务端错误消息回显凭据片段。报告会给出错误类型供排查。
已有日志证据和可用当前观察时，无效报告默认额外请求一次修正，只使用原调查结果，不开放工具。
修正后仍须通过全部字段和引用校验；失败则保留记录并返回 `incomplete`。
可用 `--max-report-repairs 0` 关闭；`metrics.report_repairs` 明确记录额外请求次数，相关 Token 和耗时计入总量。

## 预算与度量范围

- 默认最多 8 轮 Agent Loop，最多 16 次已注册工具请求，可用 `--max-rounds`、`--max-tool-calls` 调整。
- 报告修正预算独立于调查轮数，最多额外 1 次模型请求，不重新执行调查。
- 模型单次最大输出默认 4096 Token，可通过 `CORECODER_MAX_TOKENS` 配置。
- SDK 请求超时为 30 秒，关闭 SDK 自动重试；框架已有瞬时错误重试仍会生效。流式请求的实际总时间可能更长。
- `llm_calls` 统计模型接口封装调用，包含上下文摘要调用，不代表底层重试后的 HTTP 请求总数。
- Token 值来自接口响应；`provider_usage_complete=false` 时不能视作完整计费统计。离线演示不伪造提供方 Token 用量。
- 报告校验失败与用量缺失分开处理；失败报告也可能有完整的响应 Token。完整标志不证明底层重试请求的账单完整。
- 不根据旧价格表给新模型估算金额。
- 日志最多 5 MB，超限拒绝而不静默截断；提取结果展示支持分页，完整证据保存在运行记录。
- 新增证据上下文和动态工具选择，可用 `--context-mode evidence --tool-selection guided` 显式开启；默认保留原始诊断策略。两项优化的机制、边界与实验入口见 [优化指南](optimization.md)。
- 当前工具集主要围绕日志给出的堆栈、配置与路径展开；无堆栈、无配置或文件线索的数值异常可能需要补充源码入口，后续将扩展内容搜索能力。

## 离线测试与真实 API 验证分开

```bash
python -m pytest tests/ -q
```

单元测试隔离本机模型环境变量和项目 `.env` 对原框架配置默认值的影响；不执行真实模型调用。
适配测试检查 DeepSeek 请求设置，演示测试检查真实工具流与报告引用。
只有真实诊断命令成功执行后，才能报告 DeepSeek 实际联通情况和该次诊断表现。

现有流程的验收步骤、截断行引用限制及新增回归说明见 [核心流程验收](acceptance.md)。
