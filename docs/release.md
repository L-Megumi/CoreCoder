# ResearchCoder 交付版本

版本：`0.4.0+researchcoder.1`。源码包导入名和发行包名仍为corecoder，统一命令为researchcoder；同时保留corecoder入口。版本号用于标识当前交付，不代表PyPI发布或GitHub标签已经创建。

## 安装与环境

已测运行环境为Linux、Python 3.11.17；项目声明支持Python 3.10及以上，其他版本未完成同等验收。标准功能不依赖Torch；真实M2D故障注入、三项Torch校验器测试需要单独的实验Python。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' -c requirements-tested.txt
researchcoder --help
```

Windows激活命令为`.venv\Scripts\Activate.ps1`。无需在本机重建服务器目录。requirements-tested.txt是当前已测依赖版本约束，不是所有平台的通用锁文件；如果索引中不存在某个版本，应记录新环境版本并重新验证。

下载wheel后可用`python -m pip install /path/to/corecoder-0.4.0+researchcoder.1-py3-none-any.whl`安装。源码包还包含examples、测试与实验说明；wheel只提供应用模块和命令。

## 三个无密钥演示

```bash
researchcoder demo --output-dir .tmp/demo-workflow-new
python -m corecoder.symbol_demo --output-dir .tmp/demo-symbol-new
python -m corecoder.feedback_demo --output-dir .tmp/demo-feedback-new
```

所有输出目录须不存在。演示使用脚本模型，实际创建故障日志、读取和修改文件，并执行原验证命令；不调用API。它们用于检查安装与工作流，不计为真实模型成功率。通用编程演示可执行`researchcoder code --demo`。

## 配置和使用

从项目根目录复制`.env.example`为`.env`并在服务器本地填入密钥；也可运行`researchcoder diagnose --configure-deepseek`，交互输入不会显示密钥。启动目录的环境变量优先于.env。已有CORECODER_API_KEY或OPENAI_API_KEY可能优先用于编程，请在自己的终端确认选用接口，勿把密钥发到聊天。

```bash
# 只预览路由，不调用模型
researchcoder run '排查训练日志异常' --project /path/to/project --log /path/to/train.log --dry-run

# 只读诊断，显式开启两项优化和调查清单
researchcoder diagnose /path/to/train.log --project /path/to/project --completion-mode checklist --tool-selection guided --context-mode evidence --output .tmp/diagnosis-new.json --markdown .tmp/diagnosis-new.md

# 明确修复任务；--verify必须放最后
researchcoder repair .tmp/diagnosis-new.json --project /path/to/project --instruction '修复报告指出的问题，保持接口语义' --output .tmp/repair-new.json --verify python -m pytest -q

# 通用编程、持续会话
researchcoder code --project /path/to/project
researchcoder chat --project /path/to/project --log /path/to/train.log
```

真实诊断与编程会产生API用量。修复会修改指定项目并执行所给命令；建议在版本控制下工作。报告必须与当前项目及源码版本一致，文件变化后需重新调查。通用编程继承原框架命令执行能力，不是操作系统沙箱。

## 最终验证

```bash
python -m pytest tests -q
# 配置单独Torch环境时启用另外三项集成检查
RESEARCHCODER_EXPERIMENT_PYTHON=/path/to/M2D/bin/python python -m pytest tests -q
```

全部条件满足时636项通过；无实验Python时预计633项通过、3项明确跳过。发布验收另外验证wheel从源码目录之外安装运行，具体检查见交付记录。

## 实验版本与已知限制

| 记录 | 结果 | 解释范围 |
| --- | --- | --- |
| stage15修复验收 | 六类最终通过 | 含报告恢复、人工契约澄清；不是首次自动6/6 |
| stage16四组对照 | 48次诊断，双开输入Token约降20% | 该冻结版本、六类开发案例；不能外推到当前版本 |
| stage17可靠性 | 初始化核查、有限归因一致性检查 | 动态/继承未知；中文词法规则存在漏检 |
| stage18压缩续诊 | 证据组6/6报告有效、31/38严格字段匹配 | 人工长转录；包含协议修正，不能称无损记忆 |

当前仍会混淆配置路径与配置值，也存在信号字段格式误述；结构引用校验不保证语义正确。未做公开基准或独立自然故障集测评，没有自动训练调度、全量训练恢复或通用诊断准确率保证。交付冻结后新增修改需要单独记录版本与验证。

## 归属与资料

保留原MIT许可证和作者署名。原框架首页见[README_CORECODER.md](../README_CORECODER.md)，上游为[he-yufeng/CoreCoder](https://github.com/he-yufeng/CoreCoder)。Agent Loop、模型接口、通用工具、原上下文能力来自CoreCoder；新增工作集中在证据提取与定位、诊断与修复交接、路由反馈、两类证据优化、可靠性检查及受控实验。详细功能与各阶段链接见[功能索引](../README_RESEARCHCODER.md)。
