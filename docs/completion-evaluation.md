# 调查完成条件与多类型评测

本阶段解决“报告格式有效，但关键检查尚未完成”的问题，并加入可复查的受控样例评分。
完成清单、报告引用校验、模型结论正确性是三个不同层次。

## 1. 对真实项目启用清单

```bash
source /home/example/miniforge3/etc/profile.d/conda.sh
conda activate researchcoder
cd /home/example/ResearchCoder

python -m corecoder.diagnose examples/logs/dronevehicle_missing_path.log \
  --project /home/example/M2D-LIF --allow-path-root /data/example/M2D-LIF \
  --completion-mode checklist \
  --output .tmp/checklist.json --markdown .tmp/checklist.md

# 无堆栈日志可以提供入口路径；内容仍通过只读工具读取
python -m corecoder.diagnose /path/to/run.log --project /path/to/project \
  --source-hint entry.py --completion-mode checklist --structured-findings
```

`--completion-mode off` 保留旧行为，也是默认值。
它可与 `--tool-selection guided`、`--context-mode evidence` 独立组合；真实评测先单独比较清单关闭/开启。
源码提示最多 10 项，提示只提供路径，不能越过项目读取范围。
未提供线索的无堆栈数值异常仍缺少自动内容搜索入口，可能无法完成源码检查。

## 2. 清单如何工作

清单只读取本次 Investigation 存储，不额外读取源码、执行训练或隐式检查文件系统。
它根据已提取证据及实际工具观察生成检查项：

- 尚未展示的 ERROR/NUMERIC_ANOMALY 证据：继续提取对应日志页。
- 有 traceback 的证据：尝试定位当前源码。
- 日志明确声明的缺失路径：对该字面路径进行当前元数据检查。
- 日志引用的配置：读取内容，或明确记录范围外、缺失等原因。
- 识别为 `images not found` 的 YAML 数据集配置：从已读取文本中识别绝对 path 与标量 val，检查当前解析出的验证集路径。
- 用户提供的源码入口：取得可见源码片段；已有定位片段可复用，避免要求重复读取。

状态为 observed、pending、blocked。blocked 表示无法确认，不表示不存在；有 blocked 项时可以形成带局限的报告。
ready_for_report 表示没有 pending 项，不等于根因已确认、源码位置一定匹配或训练已经恢复。

每轮模型请求附当前待检查项、已有阻碍和建议工具参数；建议优先批量检查具体目标，避免逐个检查无关祖先目录。
模型过早提交报告时，最多允许两次继续调查，仍使用原有调查轮数和工具执行预算。
没有额外创建无限循环；预算耗尽后标记 incomplete，保留早期报告、待检查项与调用记录。
报告格式纠正仍单独最多一次，不开放新工具。

配置解析是保守的 YAML 标量子集：支持普通/引号路径、绝对根目录加相对 val。
关键字段未出现在截断预览时，会要求继续分页读取，并只合并同一文件哈希下的已观察源码行。
相对根目录、列表、别名、标签、重复 path/val 等会记录 blocked，不猜测框架的工作目录或 DATASETS_DIR。
该路径约定只用于上述识别出的数据集配置，不把所有 YAML 都按 Ultralytics 语义解释。
清单最多处理 64 条关键证据；超限要求提供更小且完整的日志片段，不静默宣称调查完成。

结果新增 investigation_checklist、investigation_continuations、context_memory，以及独立的调查轮数统计。
原始日志和观察继续完整保存；即使历史被压缩，也能查看最终证据记忆和完整审计记录。
报告 JSON 中的重复键现在会被拒绝，避免后一个同名字段静默覆盖前一个。

## 3. 结构化预测与评分

`--structured-findings` 在原报告上增加：

- failure_category：数据路径、CUDA OOM、张量形状、数值异常、依赖缺失或 unknown。
- cause_code：预定义原因代码或 unknown；它是模型预测，不是规则引擎给出的答案。
- primary_locations：实际源码观察 ID 与行号，行号必须位于已取得的源码范围内；无法定位时允许空列表。

这些字段只便于评分，不保证文字解释与引用语义一致。
评分器分别比较类别、标注原因代码、标注源码行，以及指定路径的实际检查覆盖。
失败报告保留在分母中，不只筛选成功结果。
源码行指标是受控样例的标注故障/异常位置命中，不代表任意真实实验的根因定位准确率。

## 4. 七个受控样例

| ID | 情况 | 类别 |
| --- | --- | --- |
| c01 | 历史数据路径缺失，当前验证集路径存在 | 数据路径 |
| c02 | 当前配置对应的验证集仍缺失 | 数据路径 |
| c03 | 申请 GPU 显存超过日志显示的空闲量 | CUDA OOM |
| c04 | 卷积需要 3 通道，输入为 1 通道 | 张量形状 |
| c05 | view 目标形状与元素总量不相容 | 张量形状 |
| c06 | 零概率取对数后出现 NaN 的受控代码/日志 | 数值异常 |
| c07 | 导入未找到的运行时模块 | 依赖缺失 |

它们是手工构造的说明性源码和日志，没有运行 PyTorch/NumPy 训练来产生日志，也不需要安装这两个库才能调查。
每个样例的入口路径统一为 entry.py；评分标注 annotation.json 位于 project/ 之外，真实模型的文件工具不能读取。
真实模型不会收到该样例的期望类别、原因代码或正确行号。
示例本身短且明确，不能当作独立真实测试集或泛化能力证明。

```bash
# 不调用模型 API：用已知标注回放，检查工具、引用和评分器能否正确对接
python -m corecoder.evaluation --offline --output-dir .tmp/eval-offline-new

# 真实模型：选 5 类各一个样例，每个比较清单关闭/开启，共 10 次诊断
python -m corecoder.evaluation --live --cases c01 c03 c04 c06 c07 \
  --output-dir .tmp/eval-live-new --seed 42

# 不指定 --cases 时评测全部 7 个样例，共 14 次诊断
python -m pytest tests/ -q
```

输出目录必须不存在，避免覆盖先前结果。包含生成的项目/日志、外部标注、每次完整报告、manifest、progress 与 summary。
seed 只控制样例/组的运行顺序，不控制模型生成和服务端缓存。
默认最多 8 轮调查、16 个已注册工具执行预算、一次格式纠正；评测 CLI 支持 --max-rounds 1 至 8。
真实调用会产生模型费用，Token 使用提供方返回值，字符估算另外记录。

离线模式明确使用 GoldReplayLLM：它有意读取答案，以验证评分器和管线，不能拿其结果声称模型准确率。
因此离线 summary 不输出性能评分聚合，只输出 fixture_validation_passed 与各次回放记录。
真实模式才给出模型预测命中率和成本汇总；每例每组只有一次，结果只作小样本工程观察。
退出码 0 表示各次报告完成，不表示类别或原因预测全部正确。

下一步应收集多个独立真实故障，保留故障时配置、修复结果及标注，再测量真实根因诊断效果和开销分布。
