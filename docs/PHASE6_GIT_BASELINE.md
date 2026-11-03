# Git 稳定基线整理建议（尚未暂存或提交）

先保留 Phase 5/6 原始快照、最终源码清单、HEAD、tracked.diff 和测试节点记录，再审查全量候选清单。不能仅提交新 evaluation/durable 目录：Phase 1～5 的合同、诊断和 Context/DAG/Runtime 仍有大量未跟踪依赖。

应纳入所有必要 corecoder/researchcoder 源码、tests、benchmarks、examples、小型 JSON/日志素材、docs、pyproject.toml、requirements-tested.txt、CI、README、许可证和必要 assets。.env.example 只允许占位值。

建议的 .gitignore 增量如下，需审查目录命名并确认后应用，本轮未改变旧忽略规则：

```gitignore
# Credentials and environment overrides
.env.*
!.env.example
*.pem
*.key
id_rsa
id_ed25519
# Explicit runtime output locations
/runs/
/artifacts/
/traces/
/eval-runs/
/checkpoints/
*.sqlite
*.sqlite-wal
*.sqlite-shm
*.sqlite.controller
# Large training inputs/outputs
/datasets/
/weights/
*.pt
*.pth
*.ckpt
```

不全局忽略 *.json/*.log 或 benchmarks/。必要异常日志、案例 manifest 和独立 Oracle 是测试输入，不能误排除。运行结果、Trace、SQLite 和大体积 Artifact 通过交付快照单独保管，不提交到代码仓库。

发布前检查文档和历史测试素材中的真实用户目录、项目路径与实验细节；保持原行为测试通过后再决定如何去标识化。不要为了 Git 美观自动改写历史测试或丢弃未提交成果。

拟提交说明：`feat: establish ResearchCoder Phase 1–6B reproducible baseline`；正文说明包含历史未提交的诊断内核、合同/Adapter、Harness、DAG、Context、Durable Recovery 和新增 Evaluation/Benchmark，记录最终测试、源码摘要及离线测量范围。

用户确认后，按审核后的显式清单暂存，检查 staged 名称和 diff/敏感信息，再建立稳定提交。push、拆分历史提交或分支整理另行授权。当前不执行 git add/commit/push/reset/clean。
