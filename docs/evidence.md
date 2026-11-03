# ResearchCoder 第一阶段：结构化日志证据

本阶段在已有 `extract_evidence(text)` 上扩展证据格式，提供独立离线入口，为源码定位和诊断 Agent 提供输入。
不需要模型、API Key 或深度学习框架；目前尚未接入自动诊断和源码定位。

## 运行

服务器的项目目录为 `/home/example/ResearchCoder`，Conda 环境为 `researchcoder`。

```bash
conda activate researchcoder
cd /home/example/ResearchCoder
mkdir -p .tmp
python -m corecoder.evidence examples/logs/dronevehicle_missing_path.log --output .tmp/dronevehicle-evidence.json
python -m pytest tests/test_evidence.py tests/test_evidence_structured.py -q
```

省略 `--output` 时向标准输出写入 UTF-8 JSON。输入支持 UTF-8、UTF-8 BOM 和 CRLF。
输入不存在、编码错误或输出失败时退出码为 2；输出文件不能与输入日志指向同一文件。

Python 调用保持兼容：

```python
from corecoder.evidence import extract_evidence

evidence = extract_evidence(log_text, source="experiment-001.log")
```

## 证据契约

所有记录共有以下字段，保持按日志出现顺序返回：

| 字段 | 含义 |
| --- | --- |
| `id` | 由日志来源、行号范围、类型和原文生成的确定性 ID |
| `source` | 调用者提供的日志来源；省略时为 `null` |
| `line_no` / `end_line_no` | 日志中的起止行号，从 1 开始，包含结束行 |
| `type` | `ERROR`、`WARNING` 或 `NUMERIC_ANOMALY` |
| `content` | 兼容旧接口；独立日志行去除首尾空白，traceback 保留原始行 |
| `raw_content` | 对应范围的原始行内容，换行统一为 LF，用于回查与审计 |

异常证据还包含：

| 字段 | 含义 |
| --- | --- |
| `exception_type` | 如 `FileNotFoundError`；不完整堆栈未找到异常行时为 `null` |
| `message` | 异常头后的消息及其后续行，不包含调用栈；保留消息内部空行 |
| `frames` | 按打印顺序保存调用栈帧 |
| `truncated` | 未找到异常结束行时为 `true`；为 `false` 不代表能检测消息中途被截断 |
| `cause_id` | 显式因果链中上一异常的 ID，来自 `raise ... from ...` 的标准分隔语句 |
| `context_id` | 隐式异常处理上下文的上一异常 ID，不等同于已经证实的故障原因 |
| `chain_id` | 当前异常链第一条证据的 ID；无关联时为自身 ID |

每个 frame 包含 `file_path`、源码 `line_no`、`function`、日志 `log_line_no` 和日志打印的 `source_line`。
frame 中的代码是日志快照，尚未核验当前磁盘文件；源码行号和日志行号不可混用。

ID 对同一来源、同一内容和相同行号范围保持稳定；日志被编辑、前方插入行或来源改变时 ID 会改变。
调用者应为不同实验提供不同 `source`。同一行的不同证据标签具有不同 ID。
完整原文用于存储和回查，后续上下文管理应选择必要字段，避免将 `content` 和 `raw_content` 重复发送给模型。

## DroneVehicle 回归案例

`examples/logs/dronevehicle_missing_path.log` 根据用户提供的历史日志还原了原始 traceback 排版，
移除了聊天中的 HTML 空格、Markdown 转义和行末反斜线；解析器本身不执行这些格式替换。

预期结果恰好为两条异常证据：

1. 第 1–8 行：`FileNotFoundError`，保留 2 个调用栈帧和完整缺失路径消息。
2. 第 12–25 行：`RuntimeError`，保留 5 个调用栈帧，`cause_id` 指向第一条证据。

日志中的缺失目录是 `/data/example/M2D-LIF/dataset/DroneVehicle_train_val_crop/images/val`。
人工核查日期为 2026-10-20：当前服务器的 `DroneVehicle.yaml` 已使用 `/home/example/M2D-LIF/dataset/DroneVehicle_train_val_crop`，
且对应 `images/val` 目录存在。因此这是历史回归样例，不能据此声称当前服务器仍有该故障。
这些目录检查是本次开发时的人工核查，尚非证据提取模块的自动诊断能力，也未验证数据内容完整性。

## 边界和限制

- 支持顺序打印的普通 Python traceback、标准显式因果链和隐式异常上下文。
- 下一 traceback、异常链分隔语句、明确的日志级别前缀（例如 `INFO:`）和常见训练指标行作为边界。
- 消息中的空行会保留；边界前或文件尾的分隔空行不计入证据。
- 无分隔符的任意文本无法可靠区分为异常消息或后续普通日志。本版优先保留在消息中，避免静默丢失。
- 消息自身如果以已识别的日志级别或指标格式开头，也可能被拆开；原始日志仍应保留供回查。
- 尚不支持 Python ExceptionGroup 的树状堆栈、逐行带 logger/rank 前缀的堆栈，以及多进程交错输出。
- 普通日志沿用现有 ERROR/WARNING 关键词规则，仍可能误报，例如 `error rate = 0`。该问题独立于本阶段的 traceback 修复。
- 本阶段没有诊断准确率、工具调用或 Token 优化收益结论；这些需要后续完整 Agent 的对照实验。

## 下一阶段

使用结构化 frames 定位项目源码，安全处理路径映射，提取所在函数和相关配置；
将日志快照与当前文件核查结果分开记录，明确文件或行号已经变化的情况。
