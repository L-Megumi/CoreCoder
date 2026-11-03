# ResearchCoder 核心流程验收

本阶段收敛现有功能：证据提取、源码定位、只读调查、调查清单和报告校验。验收使用脚本模型和真实文件工具，不需要 API Key；它验证工程行为，不给出模型诊断准确率或降本结论。

## 本轮修复

1. **当前状态必须有可用引用**：仅取得过可用观察还不够，`current_state.observation_ids` 本身必须包含实际文件内容或实际路径检查。范围外、未读取到内容的源码记录只能用于说明局限。
2. **源码引用按可见行校验**：片段增加 `char_truncated` 和 `visible_end_line`。保留原 `end_line` 表示原选取范围，但不能引用超出完整可见行的源码。字符截断的最后一行保守地排除；可以从目标行重新读取后引用。
3. **无效参数也计入预算**：非 JSON 对象参数、参数类型错误和未知工具都留下失败审计记录，不能绕过工具请求预算。非字符串读取路径在写入观察前拒绝，避免破坏调查清单。
4. **用量与报告状态分开**：报告无效或调查预算耗尽，不会抹去已经完整收到的 Token 统计。请求异常或缺少用量时仍标记不完整。该标记只表示封装调用收到的响应是否带用量，不代表底层重试 HTTP 请求的完整账单。
5. **复用已完成检查**：清单提供有界的完成项与观察 ID，并提示复用结果和停止重复范围外请求。这是调查提示，尚未证明减少真实模型调用。过长配置行阻止分页前进时标为 blocked，不反复建议同一页，不推断当前路径。

源码可见行校验描述原始工具片段。上下文投影可能继续缩短内容；本轮没有实现对每一轮模型实际可见文本的逐行跟踪。
报告仍只做结构、引用可用性和源码范围校验，没有实现自然语言结论的全面事实核验。

## 在你的服务器运行

```bash
source /home/example/miniforge3/etc/profile.d/conda.sh
conda activate researchcoder
cd /home/example/ResearchCoder
```

先检查回归测试：

```bash
python -m pytest tests/test_acceptance.py tests/test_completion.py tests/test_diagnose.py -q
```

生成一份可阅读的离线诊断。输出文件名可以自行更换，避免覆盖旧记录：

```bash
mkdir -p .tmp
python -m corecoder.diagnose --demo \
  --tool-selection guided --context-mode evidence --completion-mode checklist \
  --output .tmp/acceptance-demo.json --markdown .tmp/acceptance-demo.md
```

预期 `status=completed`、`mode=offline_scripted`，调查清单没有 pending，四次工具请求都有审计记录，提供方 Token 完整标志为 false（本来就没有 API 请求）。

然后用新的输出目录执行两个离线入口；目录必须不存在：

```bash
python -m corecoder.experiments --offline --output-dir .tmp/acceptance-optimization-new
python -m corecoder.evaluation --offline --output-dir .tmp/acceptance-evaluation-new
python -m pytest tests/ -q
```

前者验证四个优化组合和构造的上下文压力场景；后者回放 7 个受控案例的两组流程，共 14 次。
`fixture_validation_passed=true` 只表明回放和评分管线一致；答案由标注回放模型生成，因此 `summary=null`，不能当成模型成绩。

## 建议亲手检查

- 在演示 Markdown 中找到历史路径与当前路径，回到 JSON 确认对应 `ev_`、`obs_` 和实际检查状态。
- 用 `--max-rounds 1` 再运行演示，应该退出 2、状态 incomplete，同时保留已提取证据。
- 用 `--max-tool-calls 1` 再运行演示，后续工具失败应计入审计，不能悄悄继续调查。
- 理解 `completed` 只表示报告通过校验；`blocked` 只表示未能确认。二者都不能说明故障已经修复。

## 当前验收边界

已实现的是诊断和修复建议，没有自动修改实验项目或执行训练验证。
文件读取器复用缓存前会核对文件签名，已观测到新版本时排除过期的当前引用；详细行为见 [文件变化与调查快照](file-versions.md)。这仍不保证文件并发变化时始终反映最新磁盘状态。
仍需后续独立真实案例、人工事实审阅和重复实验，判断诊断效果和优化收益。本阶段不修改此前实验结果。
