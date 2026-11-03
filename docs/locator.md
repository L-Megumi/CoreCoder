# ResearchCoder 第二阶段：证据驱动的源码与配置定位

本阶段接收第一阶段的结构化证据，读取当前项目文件，定位对应代码和异常中引用的配置。
整个过程使用路径解析、文本核验和 Python AST，不需要 LLM，也不会导入或执行被调查项目的代码。

## 在服务器运行

```bash
source /home/example/miniforge3/etc/profile.d/conda.sh
conda activate researchcoder
cd /home/example/ResearchCoder
mkdir -p .tmp
python -m corecoder.locator examples/logs/dronevehicle_missing_path.log --project /home/example/M2D-LIF --output .tmp/dronevehicle-locations.json
python -m pytest tests/test_locator.py -q
```

输入为原始日志；CLI 内部先调用 `extract_evidence`，再调用定位器。
输出为 UTF-8 JSON。省略 `--output` 时输出到终端。输出文件不能覆盖输入日志或报告关联的项目文件。

Python 接口：

```python
from corecoder.evidence import extract_evidence
from corecoder.locator import locate_evidence

evidence = extract_evidence(log_text, source="failed.log")
report = locate_evidence(evidence, "/home/example/M2D-LIF")
```

## 路径迁移

默认使用日志中的绝对路径，或相对于项目根目录的相对路径。
旧路径需要显式映射，不通过同名文件或后缀猜测位置。例如：

```bash
python -m corecoder.locator failed.log --project /home/example/M2D-LIF --map-root /data/example/M2D-LIF=/home/example/M2D-LIF
```

`--map-root` 可以重复提供，优先采用最长的路径前缀，并按路径组件匹配。
支持将 Windows 格式的日志路径映射到 Linux 项目目录。映射后的文件仍需位于指定项目根目录内。
`logged_frame` 和 `requested_path` 始终保留日志中的位置，映射结果记录在 `resolved_path`，两者不会被混同。

## 输出含义

`locations` 中每个对象对应一个调用栈帧：

| 字段 | 来源或含义 |
| --- | --- |
| `evidence_id` / `frame_index` | 对应哪条日志证据、该证据中的哪个帧；index 从 0 开始 |
| `logged_frame` | 第一阶段提取的日志快照，保持不变 |
| `requested_path` | 日志中的文件路径 |
| `resolved_path` / `relative_path` | 当前项目内解析出的路径及相对路径 |
| `path_resolution` | `absolute`、`relative` 或显式 `mapped` |
| `current_line_no` | 当前选中的源码行；没有可靠定位时为 `null` |
| `current_line_at_logged_line` | 当前文件在日志原行号处的文本，可能已经不同 |
| `logged_line_matches` | 日志原行号处的代码文本是否一致；无代码快照时为 `null` |
| `function` | AST 提取的当前函数名称、限定名与定义范围，含装饰器 |
| `function_verified` | 当前所选位置与日志函数名是否一致；无法判断时为 `null` |
| `candidate_lines` / `candidate_count` | 重定位候选行，列表最多保留 20 个 |
| `function_candidates` | 无可靠定位时，最多给出 5 个同名函数线索，不自动选定 |
| `snippet` / `snippet_basis` | 当前文件片段及选取依据 |
| `file_sha256` | 本次实际读取文件的内容哈希，便于核对代码快照 |
| `parse_status` / `parse_error` | Python AST 是否成功解析，失败时保留原因 |

定位状态由机械核验产生，不是模型诊断：

| `status` | 含义 |
| --- | --- |
| `matched` | 原行号代码与日志一致，且可核验的函数名未冲突；不代表文件整体版本相同 |
| `relocated` | 原位置不一致，但找到一处代码快照匹配且函数名未冲突的新位置 |
| `ambiguous` | 找到多处候选，保留候选而不猜测 |
| `source_changed` | 未找到满足条件的代码快照，可能发生了代码变化 |
| `unverified` | 日志无代码快照，只能按原行号取上下文 |
| `function_mismatch` | 无代码快照且原位置与函数名冲突 |
| `line_out_of_range` | 原行号超出当前文件范围且无法重定位 |
| `missing` | 项目内目标文件不存在 |
| `outside_project` | 文件或解析后的符号链接目标位于项目外，跳过读取 |
| `unmapped_path` | 不属于当前系统可直接访问的绝对路径，需要显式映射 |
| `virtual_path` | `<stdin>` 等虚拟文件位置 |
| `invalid_path` / `invalid_line` | 路径或行号不合法 |
| `not_file` / `too_large` / `read_error` / `decode_error` | 文件无法按当前限制读取，原因分别记录 |

`current_line_no=null` 时仍可能给出 `snippet_basis=logged_line_unverified` 的参考片段，
它仅展示原行号附近的当前文本，不能被当作已经确认的故障位置。
同一代码行存在多处时，仅当日志原行号本身仍匹配时接受原位置；重定位时保留歧义。

`config_files` 关联异常消息或普通证据正文中被引号包裹的 YAML、JSON、TOML、INI、CFG 路径。
同一个路径的重复引用合并 `evidence_ids`，配置内容按预算预览。
这一阶段不解析配置语义，不执行下载脚本，也不验证数据集内容。

## 读取预算

同一解析路径的成功读取可复用，但每次命中前核对文件签名；文件变化时失效并重新读取。缺失等失败结果不缓存，恢复后可以再次读取。
读取前后发现文件版本变化时返回 `file_changed_during_read`，不提供不一致片段。具体边界见 [文件变化与调查快照](file-versions.md)。
Python 文件先按编码声明解码，再解析 AST。语法错误不会让整个定位过程失败，但函数核验可能不可用。

默认限制：

- 最多定位 50 个栈帧、处理 10 个不同配置路径引用。
- 单文件最多读取 1,000,000 字节；超大文件跳过。
- 单片段最多 40 行、8,000 个字符；长函数默认取目标行前后各 3 行。
- 超出栈帧或配置预算的数量在 `stats` 中明确记录。

API 可以通过 `context_lines`、`max_snippet_lines`、`max_snippet_chars`、`max_file_bytes`、`max_frames`、`max_config_files` 调整。
CLI 提供 `--max-frames`。`stats.files_read` 和 `stats.cache_hits` 是本次定位的实际文件读取与缓存计数，不是 LLM 工具调用次数或 Token 节省量。
日志证据的完整原文独立保存在第一阶段结果和原始日志中；定位输出只保留帧引用和必要当前片段。

## 当前案例的验收目标

DroneVehicle 样例包含 7 个帧、5 个不同 Python 文件，异常正文引用了数据集 YAML 和用户目录下的 settings YAML。
以 `/home/example/M2D-LIF` 为项目根目录时：

1. 核验 7 个栈帧的当前代码与函数，并保留每条结果的证据 ID。
2. 读取项目内 `data/DroneVehicle.yaml`。
3. 保留项目外 `settings.yaml` 的引用并标记 `outside_project`，不读取其内容。
4. 明确这是历史失败日志，不据此断言当前训练仍会失败。

## 已知边界与后续

- 精确行匹配忽略首尾空白，不做任意语义相似匹配；内部格式变化仍可能被标记为代码变化。
- 文件哈希标记当前读取的快照；没有记录历史源码提交时，不能证明当前文件就是报错时的版本。
- AST 提供静态函数范围；lambda、推导式等特殊运行时函数名可能无法验证，结果会保留不确定性。
- 不自动跨项目搜索文件，不递归追踪任意配置引用，不加载模型权重或运行训练。
- 路径范围检查用于控制调查范围，不是执行不可信项目代码的安全沙箱；本模块不执行项目代码。

下一阶段把证据提取和定位作为工具接入 Agent Loop，让模型根据已核实事实决定进一步调查动作并生成诊断报告。
