# 无堆栈日志的证据驱动源码搜索

对于 `step=1 loss=nan`，日志没有文件和行号。现在可先提取证据，按 `loss` 搜索 Python 源码，读取候选，再形成带引用的原因假设。原 code 入口及七个编程工具保持可用。

## 离线演示

在服务器执行；输出目录须不存在：

```bash
conda activate researchcoder
cd /home/example/ResearchCoder
python -m corecoder.source_search_demo --output-dir .tmp/source-search-demo-new
```

演示创建独立小项目，实际执行产生 `loss=nan` 的日志。脚本模型依次选择提取、搜索、读取三个真实工具，再生成报告，不调用 API。结果在输出目录的 `diagnosis.json`。此演示验证工具流程，不代表真实模型诊断准确率。

## 实际调查

```bash
researchcoder diagnose /path/to/failure.log --project /path/to/project \
  --tool-selection guided --context-mode evidence --completion-mode checklist \
  --output .tmp/frameless-diagnosis-new.json
```

实际调查使用已配置的模型接口。`search_project_source(query, evidence_ids, paths=None, max_results=8)` 要求引用已经展示的证据 ID。关键词是大小写不敏感的字面量，不执行正则或项目代码；仅在项目内搜索 `.py`。模型可以缩小 `paths` 或更换关键词；初始规则会从 `loss=nan` 这样的键值提取 `loss`。

无堆栈、无配置引用的关键证据，没有用户指定源码入口时，调查清单要求一次搜索。存在候选时至少实际读取一个同版本候选的命中行；显式源码入口则直接读取。未命中或不可读的检查保留为未知，不自动编造根因。

## 范围与结果

- 每次至多检查 2000 个目录条目、80 个 Python 文件、总共读取 2 MB；单文件上限 200 KB、目录深度 24、结果上限 20（默认 8）。
- 排除隐藏路径、虚拟环境、依赖与构建目录，以及符号链接；`data` 等普通源码目录仍可搜索。权限错误、文件过大、限额触发等进入 `partial` 和 `reasons`。
- `source_candidate` 记录证据关联、查询、命中行、匹配列、截取的匹配文本、函数范围、源码片段和文件哈希。它是调查候选，不能单独满足报告的当前文件观察或主要源码位置。
- `source_search` 记录范围、预算、实际扫描量、候选 ID 和未完成原因；零命中只说明当次扫描范围，不能证明项目内不存在相关代码。
- 后续读取观察到文件新版本时，旧候选仍保留供审计，但不再作为当前版本。当前性是最新一次观察，并非持续文件监控。
- 上下文投影与证据记忆保留查询、命中行、哈希、候选关联和截断原因；完整数据保存在结果文件。预算不足时仍可能省略部分观察，并显示省略数量。

这一步提供有界关键词调查能力；尚未实现语义索引、调用图推理或大型仓库的全量检索保证。训练是否恢复仍需显式修复和实际验证命令确认。

## 回归验证

```bash
python -m pytest tests/test_source_search.py -q
python -m pytest tests/ -q
```

覆盖候选引用、字面量查询、访问范围、扫描预算、文件版本、调查清单、上下文保留以及四种工具/上下文配置下的完整离线流程。测试数量属于工程验证，不是诊断效果指标。
