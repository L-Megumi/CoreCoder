# ResearchCoder

面向 Python 工程与深度学习实验排错的 Multi-Agent Coding System。系统将日志和源码观察组织为可追溯证据，在受控预算内完成调查、专家交接、代码修改与独立验证，同时保留通用编程入口。

[M21 真实重复试验、自然纠错与两项新开发案例](results/m21/README.md)

## 核心机制

- **Multi-Agent Orchestration**：独立 Worker 进程、DAG 依赖与并行只读调查、角色工具约束、有限重规划，以及 Code 专家向单写入 Worker 的可信交接。
- **Evidence-Aware Context**：结构化异常因果链、源码版本与 Artifact SHA256 绑定、角色上下文隔离、分层压缩、原文卸载与按引用恢复。跨任务证据与当前任务本地观察使用不同引用范围。
- **Reliable Harness**：Job/Task/Attempt 状态机、请求与工具预算、实际/估算/unknown 用量区分、SQLite Checkpoint、Operation Journal 和崩溃后的副作用对账。
- **Evaluation-Driven Optimization**：独立 F2P/P2P Oracle、冻结协议、真实模型 Trial/Trace/Usage 归档、失败归因与可选择的版本化策略。系统不会将部分报告或单个测试通过自动判成整体修复成功。

进程隔离与应用级 Guard 不等同于完整 OS 沙箱；未知副作用需要停止并对账。当前优化闭环包含人工分析与版本化采纳，未实现自动 RSI。

## 安装与入口

正式执行平台为 Linux；开发和发布检查使用 Python 3.11。包同时包含 `corecoder` 与 `researchcoder`，保留原 CLI。

```bash
python -m pip install -e '.[dev]'
corecoder --help
researchcoder --help
researchcoder-eval --help
```

模型密钥只通过本机环境配置，勿提交 `.env`。联网调用需要用户显式选择；离线示例不需要模型密钥。训练案例需要一个已经安装 CPU PyTorch 的解释器：

```bash
export RESEARCHCODER_EXPERIMENT_PYTHON=/path/to/torch-environment/bin/python
python -m examples.multi_agent_autograd_demo --output-dir outputs/autograd-demo
```

该示例使用 ScriptedLLM、实际独立 Worker、真实工具与梯度 Oracle，验证工程机制，不代表真实模型准确率。

## 四个可复现历史缺陷

`benchmarks/repositories/` 包含固定候选源码、上游版本与许可证、公开复现以及独立 F2P/P2P 检查：

| 案例 | 缺陷 | 独立检查 |
|---|---|---|
| packaging-requirement-hash | 相等 Requirement 的哈希不一致 | F2P 5 / P2P 10 |
| sam-checkpoint-resume | 恢复后外层与底层优化器参数组不同步 | F2P 3 / P2P 3 |
| jsonpointer_scalar | 错将 JSON 字符串作为序列继续遍历 | F2P 4 / P2P 7 |
| itsdangerous_future | max_age 检查接受未来时间戳 | F2P 3 / P2P 7 |

以下命令只生成受信任配置，核验源码和检查器 SHA，不调用模型：

```bash
python -m examples.prepare_historical \
  --case packaging-requirement-hash \
  --output outputs/packaging-config.json
```

对 SAM 使用同一命令并添加 `--torch-python /path/to/torch-environment/bin/python`。在可联网环境配置 `DEEPSEEK_API_KEY` 后，显式执行单个真实 Trial：

```bash
python -m researchcoder.evaluation.projects \
  --config outputs/packaging-config.json \
  --output outputs/packaging-trial \
  --allow-live
```

每次输出目录必须不存在。候选只包含故障项目，Oracle、参考答案与控制器材料位于候选之外；Agent 的工具权限不会随日志内容扩大。

## 实验记录

- [M17 上下文消融](results/m17/README.md)：10 个离线案例、四种上下文策略，完整保留逐例结果、来源恢复及估算 Token 语义。
- [M18 失败驱动修正](results/m18/README.md)：四个案例取得真实通过记录，完整保留中间失败和版本差异。
- [M19 统一最终配置重复验证](results/m19/README.md)：四例各三次独立 Trial，轮换顺序，统一模型、角色策略和预算；不在试验中调参。
- [M20 有界修复闭环](results/m20/README.md)：新增短纠错预算、调查到修复的阶段信号和取消后的计量归档处理；四例新版本真实检查均通过，各运行一次。

通过率以所有计划与启动样本为基础报告，E2E、F2P/P2P 与模型解释评分分开。历史调试成功记录不当作统一协议的一次性成功率。小型 development 集结果不代表公开大规模基准或留出集泛化。

项目测试位于 `tests/`。本轮按用户要求暂缓完整回归，只执行必要的新增入口、匿名化素材与安装导入检查；结果报告记录实际执行范围。

## 目录

```text
corecoder/                 Agent Loop、诊断与修复内核、通用工具和 CLI
researchcoder/contracts.py  版本化任务、证据与验证合同
researchcoder/runtime/      Harness、Worker、预算、隔离、Tracing
researchcoder/orchestration/ DAG、角色、证据交换与重规划
researchcoder/context/      证据存储、压缩、角色视图与恢复
researchcoder/durable/      Checkpoint、Journal、Recovery
researchcoder/evaluation/   Benchmark、Runner、Oracle 结果及指标
benchmarks/                 PyTorch、上下文与历史缺陷夹具
tests/                     项目单元与集成测试
examples/                  离线演示及可移植案例配置入口
results/                   可复算的公开实验记录
```

上游 CoreCoder 的原始说明保留于 [README_CORECODER.md](README_CORECODER.md)。项目保留原 MIT 许可证与归属；第三方案例遵守各自许可证，来源见案例 `provenance.json`。发布源码不包含密钥、数据集、权重、SQLite 或原始私有运行 Trace。路径匿名化与发布身份说明见 [RELEASE_PROVENANCE.md](docs/RELEASE_PROVENANCE.md)。
