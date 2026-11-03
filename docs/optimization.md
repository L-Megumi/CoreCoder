# ResearchCoder：工具选择、证据上下文与对照实验

本阶段在既有诊断 Agent 上加入两项独立开关，并提供四组对照运行入口。
两项优化均为可解释的规则与工程实现，没有训练工具选择模型。

## 快速运行

```bash
source /home/example/miniforge3/etc/profile.d/conda.sh
conda activate researchcoder
cd /home/example/ResearchCoder

# 显式启用两项优化；只读调查，不执行训练或修改目标项目
python -m corecoder.diagnose examples/logs/dronevehicle_missing_path.log \
  --project /home/example/M2D-LIF --allow-path-root /data/example/M2D-LIF \
  --tool-selection guided --context-mode evidence \
  --output .tmp/optimized.json --markdown .tmp/optimized.md

# 同一版代码关闭两项优化，作为基线
python -m corecoder.diagnose examples/logs/dronevehicle_missing_path.log \
  --project /home/example/M2D-LIF --allow-path-root /data/example/M2D-LIF \
  --tool-selection all --context-mode baseline \
  --output .tmp/baseline.json --markdown .tmp/baseline.md

# 无需密钥、不调用模型 API 的机制检查
python -m corecoder.experiments --offline --output-dir .tmp/offline-new

# 真实模型四组对照，默认每组一次；目录必须尚不存在
python -m corecoder.experiments --live \
  --log examples/logs/dronevehicle_missing_path.log --project /home/example/M2D-LIF \
  --allow-path-root /data/example/M2D-LIF --output-dir .tmp/live-new --repeats 1 --seed 42

python -m pytest tests/ -q
```

CLI、Python `run_diagnosis()` 和 `run_demo()` 默认均使用 `all/baseline`；用同名参数显式启用优化。
首次真实四组对照未证明总开销下降，因此暂不将实验策略设为默认。
真实对照命令会产生 API 调用开销；`--repeats` 支持 1 至 3，分别对应 4 至 12 次完整诊断。

## Evidence-Guided Tool Selection

`EvidenceToolPolicy` 在每轮模型请求前，结合本次调查的状态选择工具 schema：

| 调查状态 | 开放工具 |
| --- | --- |
| 尚未提取日志 | 仅 extract_log_evidence |
| 已展示的证据含待定位栈帧或引用配置 | 四个调查工具 |
| 没有待定位线索，或已有定位返回 | 证据提取、项目读取、路径检查 |
| 已展示无堆栈且无配置引用的 ERROR/数值异常 | 额外开放 search_project_source |

证据分页与重读始终保留；新一页展示新的栈帧/配置证据后，可以重新开放定位。
无 traceback 的数值异常不会强制调用堆栈定位；可按证据关键词搜索源码，或直接读取用户提供的源码入口。
调查清单会跟踪一次搜索及至少一个候选的实际读取；未命中和限额截断作为未知项保留。
模型仍负责选择具体调用及参数；策略没有承诺减少模型需要的调查轮数。

同一轮中未开放的工具调用会被阻止，并记录 PolicyError；不会因模型请求绕过前置步骤。
定位返回过的证据暂不再次开放 locate_source，包括未匹配或范围外的定位结果；可通过直接读取继续调查。
这适用于一次有界调查，尚不是支持持续文件变化的监控策略。

各工具 schema 也明确给出分页/行数/数组上限，运行时校验继续生效。
`selection_events` 记录每轮工具列表与规则原因，`request_metrics` 记录各次请求的 schema 数与字符开销。

## Evidence-Aware Context Management

完整原始证据与观察保存在 Investigation 中；最终 JSON 仍包含原文、完整栈帧、源码片段和文件哈希。
`EvidenceContextManager` 只变换发给模型的上下文：

1. 将工具 JSON 投影为紧凑、可解析的 JSON，保留证据/观察 ID、因果关系、路径、状态、行号、校验结果和文件哈希。
2. 较长消息与代码片段用 head/tail/omitted_chars 明确标出省略；额外保留有限数量的引号路径线索，以及 NaN/Inf 信号。栈帧保留首尾各两帧并标明省略数量。
   配置中的 path/train/val/test/data/dataset_dir 等关键源码行另外保留在 config_path_lines，附原始行号；这是词法提取，不代表完成 YAML 语义解析或变量展开。
3. 接近估计预算时，从完整存储生成证据记忆，恢复已展示证据的 cause/context 祖先，保留有限当前观察；只删除完整的旧工具调用与返回组。
4. 记忆明确区分历史日志与当前观察，显示未展示证据、未保留观察的数量，并提示按页重提取证据或重读项目文件。

这条路径不请求模型生成摘要。完整存储未被上下文压缩修改。
关键证据及其祖先无法装入记忆预算时显式返回 incomplete，提示提高 `--max-context-tokens`，不宣称所有证据已成功保留。
观察过多时允许省略部分观察，因此不能承诺任意规模输入都无信息损失。

预算包含系统提示和全部工具 schema 的字符估计；采用约 3 字符/Token 的近似计数，不能保证与提供方 tokenizer 一致。
`--max-context-tokens` 覆盖环境变量 CORECODER_MAX_CONTEXT。过小预算可能连系统提示和工具描述都装不下。
`context_events` 记录压缩前后估计量、是否删除旧组以及记忆大小。

## 对照实验与结果解释

| 组名 | 工具策略 | 上下文策略 |
| --- | --- | --- |
| baseline | all | baseline |
| tools | guided | baseline |
| context | all | evidence |
| both | guided | evidence |

四组使用同一版代码、同一输入、相同模型接口配置、调查和报告纠正预算。
离线四组共享同一个真实临时项目与日志；模型选择和报告由脚本固定，检验工程机制。
真实运行按 seed 随机化组顺序；seed 只控制顺序，不控制模型生成，也不控制服务端缓存。
默认上下文预算为 128000，最多 8 轮调查、16 个已注册工具请求、额外一次报告纠正。

输出目录包含 manifest.json、每组完整 JSON/Markdown、progress.json 和 summary.json。
已有输出目录会拒绝运行，避免覆盖先前结果；如果中途退出，已完成的组记录仍保留。
报告会记录失败组并纳入统计，不通过只筛选成功结果来改善平均值。

真实结果重点比较：结构/引用校验完成率、工具请求/错误数、模型调用数、输入/输出 Token 和耗时。
Token 优先使用提供方实际返回的累计值，并检查 provider_usage_complete；估计量与实际用量分开命名。
离线演示的 API Token 为 0，不能将字符估算量当作计费 Token。

离线压力测试是人工构造的三种转录：异常因果链、无堆栈数值异常、长异常消息，并加入无关工具结果触发压缩。
每种预先检查 5 项结构化事实：证据 ID、因果边/NaN/Inf、存在与缺失路径状态。
路径状态同时与独立构造的“当前目录存在、历史目录缺失”预期核对。
该指标只计算压缩后仍可从 JSON 读到的精确事实，不衡量模型能否从自然语言摘要推断这些事实。
baseline 压力测试使用不调用模型的摘要回退；不能据此宣称真实模型语义记忆或诊断准确率提升。

单个历史案例、每组一次的真实结果只能作为功能验证与初步开销观察。
后续需要多个独立、具有根因标注的失败案例及重复运行，再计算根因准确率、源码定位准确率、证据引用质量和开销分布。
现有工具对没有源码/配置线索的数值异常仍缺少内容搜索入口；本阶段没有补齐自动搜索或执行训练验证。
