# M2D-LIF 项目源码故障注入验收

本轮从真实项目复制 `ultralytics` 的 Python/YAML 源码和 DroneVehicle 配置，在独立目录使用 CPU 小输入执行真实模块。原项目、数据集、权重和训练任务均不作为修改目标。

## 六类案例

| ID | 注入类型 | 真实执行位置 | 校验要点 |
| --- | --- | --- | --- |
| m01 | 验证集目录配置错误 | check_det_dataset | 配置指向正确的独立样例目录 |
| m02 | nc 与 names 数量不一致 | check_det_dataset | 保持原类别集合、恢复配置一致性 |
| m03 | 卷积输入通道偏移 | Conv | 多种通道/尺寸下参数、形状和前向结果正确 |
| m04 | 拼接轴错误 | Concat | 默认通道拼接及显式维度参数均正确 |
| m05 | IoU 分母缺少稳定项 | box_iou | 退化框无 NaN，正常框重叠值仍正确 |
| m06 | 依赖导入名称错误 | conv 模块导入 | 正常导入真实依赖并执行卷积 |

每例先对未注入的副本运行独立校验，要求通过；再注入修改并确认同一校验失败。m05 入口直接输出 `iou=nan`，没有人为附加 traceback，校验器独立检查非有限值。

配置案例沿用原类别和通道信息，将数据路径适配为独立的空目录；只验证配置检查行为，不验证真实图像质量。关闭无关字体下载，数据检查传入 `autodownload=False`。其他案例直接调用复制模块中的完整实现，不用替代算法生成错误。

## 准备案例：不调用 API

在 ResearchCoder 项目目录运行，输出目录须不存在且位于原 M2D 项目之外：

```bash
python examples/m2d_fault_acceptance.py \
  --source /home/example/M2D-LIF \
  --experiment-python /home/example/miniforge3/envs/M2D/bin/python \
  --output-dir .tmp/m2d-injections-new
```

使用已有 M2D 环境运行被测模块；ResearchCoder 环境负责调用 Agent，不需要在其中安装 Torch。子进程使用 CPU、单线程、独立运行设置，并关闭字节码写入原目录。

## 调用已配置模型验收

确认允许向模型接口发送这些测试副本的必要源码片段、日志与诊断上下文后，复用尚未修复的注入：

```bash
python examples/m2d_fault_acceptance.py \
  --source /home/example/M2D-LIF \
  --experiment-python /home/example/miniforge3/envs/M2D/bin/python \
  --output-dir .tmp/m2d-injections-new --prepared --live \
  --env-file .env --max-requests 72 --max-case-requests 14
```

每例进行一轮初始诊断和一条明确修复任务。本运行器关闭自动反馈调查，以控制请求预算；产品控制器的反馈功能不受影响。提供方请求总上限 72、默认每例累计上限 14（可配置至 20），单例 300 秒后不再发起新请求。框架及 SDK 重试在验收中关闭，接口兼容性回退也计入请求尝试数。`--direct` 仅在本次进程忽略代理配置。

调用模型会产生 API 用量。源码副本仅含代码、相关 YAML 和最小入口，不复制 `.env`、数据集或权重。模型请求不包含外部注入标注，研究用 API 密钥不传递给模块校验子进程。

## 外部校验与记录

- `cases/mXX/project/` 是模型可读取的项目，包含注入后的正常文件名；不包含注入配方、正确源码、预期故障行或校验器。
- `evaluator/` 保存注入配方、原/修改后哈希、注入行、健康对照与故障对照，以及独立语义检查，位于 Agent 项目根目录之外。
- `logs/` 保存程序实际输出；`runs/mXX/session/` 保留诊断、修复和反馈记录。
- 本轮验收向修复模型开放 read_file、edit_file、write_file、grep、glob 五个工具，提示、接口声明与执行注册表一致；禁止命令工具、写入入口、修改外部校验器及新建替代依赖。该限制属于验收运行器，产品的七个原编程工具不变。
- 原入口、场景输入、校验器及标注核对哈希；修复后由外部校验再次执行。变更不能通过改弱检查或改变输入来获得通过。
- 所有案例保留在结果中，包括格式失败、范围违规、预算耗尽和复验失败。被拦截的工具请求作为可恢复错误进入审计，不因请求本身提前终止整个工作流；独立检查结合受保护文件哈希判断实际效果。已修复的副本不能通过 `--prepared` 静默复用为全新测试。
- 运行前后核对原项目所复制源码的哈希。用户并行修改原项目也会被记录为发生变化，不据此覆盖或还原用户文件。

`injected_line_observed` 表示工具实际返回的源码范围覆盖注入行；`hypothesis_cites_that_observation` 表示原因假设引用过该观察。二者是调查覆盖指标，不能直接当作语义根因准确率。

这是一组从项目源码构造的小规模功能验收，初始尝试与修正运行器后的续接分开保留，没有独立自然故障集或基线对照，不能据此推断通用诊断准确率或优化降本幅度。完整训练恢复仍需实际训练任务的专门验证。

## 本地回归

```bash
python -m pytest tests/test_fault_injection.py -q
python -m pytest tests/ -q
```

这些回归检查副本隔离、标注位置、原项目保留、API 密钥排除、健康对照失败时停止注入以及超时输出处理，不调用模型。

## 保留历史的续接

`--continue-repairs --prepared --live` 读取 `summary.json`，要求副本尚未改变且诊断仍有效，写入独立的 `continuations/` 与 `continuation-summary.json`。

`--recover-repairs --prepared --live` 读取上一阶段结果，先检查所有当前文件与已记录修改哈希一致，再写入 `recoveries/` 与 `recovery-summary.json`。已有修改只重新执行独立检查，不复用失效诊断；尚未修改的案例复用有效报告继续修复。全部阶段的请求累计计数，不重置预算，也不覆盖旧结果。

独立检查通过与修复工作流完成分别记录：模型在补丁写入后中断，补丁仍可能通过独立检查，但其工作流状态继续保留 incomplete。新增运行器逻辑的回归使用脚本模型，不调用付费 API。

本次实测记录见 [M2D 故障注入结果](m2d-acceptance-results.md)。

后续版本新增了活动案例的预留配额、请求前持久化账本及已有阶段拒绝重跑：详见 [修复效率调整](repair-efficiency.md)。默认全新六案例在总预算 72 下各预留 12 次，每例累计上限 14 同时生效；未使用配额不自动转移。本次旧实测结果保持原样，没有应用新策略后重测。
