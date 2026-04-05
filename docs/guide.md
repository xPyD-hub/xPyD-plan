# xPyD-plan 使用指南

xPyD-plan 是 [xPyD-proxy](https://github.com/xPyD-hub/xPyD-proxy) 的 benchmark 数据分析工具包，用于从真实 benchmark 结果中找到最优的 **Prefill:Decode 实例比例**。

> **核心原则：** 不猜测、不建模、不仿真——一切基于实际 benchmark 数据。

---

## 安装

```bash
# 基础安装
pip install xpyd-plan

# 含 HTML 报告生成
pip install "xpyd-plan[report]"

# 开发环境
pip install "xpyd-plan[dev]"
```

---

## 核心子命令

### 分析类

| 子命令 | 说明 |
|--------|------|
| `analyze` | SLA 合规检查、利用率分析、最优 P:D ratio 搜索 |
| `sensitivity` | P:D ratio 与 SLA 满足率曲线，含 cliff 检测 |
| `confidence` | Bootstrap 置信区间（延迟百分位） |
| `decompose` | 按请求分解延迟为 prefill/decode/overhead 阶段 |
| `tail` | 扩展百分位分析（P99.9, P99.99） |

### 对比与测试

| 子命令 | 说明 |
|--------|------|
| `compare` | 两次 benchmark 对比，回归检测 |
| `ab-test` | 统计 A/B 测试（Welch's t-test, Mann-Whitney U） |
| `model-compare` | 多模型延迟和成本效率并排对比 |
| `drift` | 分布漂移检测（Kolmogorov-Smirnov） |

### 规划与优化

| 子命令 | 说明 |
|--------|------|
| `recommend` | 综合 SLA、成本、Pareto、趋势数据的排序建议 |
| `plan-capacity` | 线性扩展模型的容量规划 |
| `what-if` | 场景模拟——扩缩 QPS 或实例数并对比 |
| `fleet` | 多 GPU 类型 fleet 规模计算（含预算约束） |
| `pareto` | 跨延迟、成本、浪费的 Pareto 前沿分析 |
| `interpolate` | 对未测试 P:D ratio 做性能插值/外推 |
| `forecast` | 基于历史趋势的容量预测 |
| `threshold-advisor` | SLA 阈值调优 |

### 成本与预算

| 子命令 | 说明 |
|--------|------|
| `budget` | SLA 预算在 TTFT/TPOT 阶段的分配 |
| `scorecard` | 综合效率评分（SLA + 利用率 + 浪费） |
| `sla-tier` | 多 SLA 层级分析 |

### 数据管理

| 子命令 | 说明 |
|--------|------|
| `validate` | 数据质量评分、异常值检测 |
| `filter` | 按 token/延迟/时间窗过滤和采样 |
| `merge` | 合并多个 benchmark 文件 |
| `discover` | 递归扫描目录查找 benchmark 文件 |
| `generate` | 生成合成 benchmark 数据（测试用） |
| `export` | 批量导出分析结果（JSON/CSV/table） |

### 监控与告警

| 子命令 | 说明 |
|--------|------|
| `dashboard` | Rich TUI 实时仪表盘 |
| `alert` | YAML 定义的告警规则，CI/CD 友好退出码 |
| `trend` | 历史趋势追踪（SQLite 存储） |
| `metrics` | Prometheus/OpenMetrics 格式导出 |
| `timeline` | 时间窗分析，含 warmup 检测 |

---

## 典型工作流

### Step 1: 收集 Benchmark 数据

使用 [xpyd-bench](https://github.com/xPyD-hub/xPyD-bench) 对不同 P:D ratio 配置进行性能测试：

```bash
# 在不同 P:D 配置下跑 benchmark
xpyd-bench run --config cluster-2p6d.yaml --output results/2p6d.json
xpyd-bench run --config cluster-3p5d.yaml --output results/3p5d.json
xpyd-bench run --config cluster-4p4d.yaml --output results/4p4d.json
```

### Step 2: 分析最优 P:D Ratio

```bash
xpyd-plan analyze \
  --benchmark results/2p6d.json results/3p5d.json results/4p4d.json \
  --sla-ttft 200 --sla-tpot 50
```

这会输出每种 P:D 配置的 SLA 合规率、利用率和资源浪费，标出最优 ratio。

### Step 3: 获取部署建议

```bash
xpyd-plan recommend \
  --benchmark results/ \
  --sla-ttft 200 --sla-tpot 50 \
  --cost-model gpu-costs.yaml
```

`recommend` 综合 SLA 合规、成本效率、Pareto 最优和趋势数据，给出排序后的部署建议。

### Step 4: 模拟扩容场景

```bash
# 假设 QPS 翻倍，看哪个 ratio 还能撑住
xpyd-plan what-if \
  --benchmark results/3p5d.json \
  --scale-qps 2.0 \
  --sla-ttft 200 --sla-tpot 50
```

### Step 5: 导出结果

```bash
# 导出为 JSON（供自动化消费）
xpyd-plan export --benchmark results/ --format json --output plan-results.json

# 导出为 CSV（供电子表格分析）
xpyd-plan export --benchmark results/ --format csv --output plan-results.csv

# 生成 Markdown 报告（供 PR/Wiki）
xpyd-plan report --format markdown --benchmark results/ --output report.md
```

---

## 结果解读

### P:D Ratio

Prefill 实例数与 Decode 实例数的比值。例如 `2:6` 表示 2 个 Prefill 实例 + 6 个 Decode 实例（总共 8 个）。不同的 ratio 在 TTFT（首 token 延迟）和 TPOT（每 token 生成延迟）上表现不同。

### Throughput

给定 P:D 配置下实际测得的吞吐量（QPS）。更高的 throughput 意味着同等硬件处理更多请求。

### Cost

基于 GPU 小时费率计算的每请求成本或总运行成本。通过 `--cost-model` 提供 GPU 价格配置。

### Pareto Frontier

在延迟、成本和资源浪费三个维度上，不存在被其他方案完全支配的配置集合。Pareto 前沿上的点代表「没有免费午餐」——改善一个指标必然牺牲另一个。`pareto` 子命令可视化这条边界，帮助做权衡决策。

### SLA Compliance

给定百分位（P95/P99）下，TTFT 和 TPOT 是否满足目标阈值。合规率 = 满足 SLA 的请求占比。

### Resource Waste

实际使用率与理论满载之间的差距。最优 ratio 在满足 SLA 的前提下最小化这个浪费。

---

## 更多信息

- [README](../README.md) — 项目总览与快速上手
- [设计原则](DESIGN_PRINCIPLES.md) — 架构与设计决策
- [开发循环](DEV_LOOP.md) — 开发流程指南
- [Roadmap](../ROADMAP.md) — 完整里程碑列表
