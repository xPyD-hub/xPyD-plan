# xPyD-plan 当前迭代状态

> 最后更新：2026-04-05

---

## 当前里程碑：M104 — Latency Baseline Manager ✅

项目已完成 **104 个里程碑**，覆盖从核心分析到高级优化的完整功能链。

---

## 主要功能列表

### 核心分析
- Benchmark 数据加载、校验与 SLA 合规分析
- 最优 P:D ratio 搜索（多场景支持）
- 敏感度分析与 cliff 检测
- Bootstrap 置信区间
- 延迟分解（prefill/decode/overhead）
- 尾延迟分析（P99.9, P99.99）

### 对比与测试
- Benchmark 对比与回归检测
- A/B 统计测试（Welch's t-test, Mann-Whitney U）
- 多模型对比
- 分布漂移检测（KS 检验）

### 规划与优化
- 容量规划（线性扩展模型）
- What-if 场景模拟
- 多 GPU 类型 fleet 规模计算
- Pareto 前沿分析
- 综合部署建议（recommend）
- 性能插值/外推
- 容量预测（历史趋势）
- SLA 阈值调优

### 成本与预算
- 成本感知优化
- SLA 预算分配
- 综合效率评分卡
- 多 SLA 层级分析

### 数据管理
- 数据质量评分与异常值检测
- 过滤、合并、注释、发现
- 合成数据生成
- 批量导出（JSON/CSV/table）
- SQLite 导出
- Schema 迁移
- Benchmark 会话管理

### 监控与告警
- Rich TUI 实时仪表盘
- YAML 告警规则
- 历史趋势追踪
- Prometheus 指标导出
- 时间线分析
- 饱和点检测
- 健康检查

### 高级分析
- 工作负载聚类
- 相关性分析
- 延迟热力图
- 根因分析
- 扩展效率与拐点检测
- QPS 曲线拟合
- 延迟方差分解
- 延迟预测集成
- 延迟异常分类
- SLA 余量计算
- 延迟基线管理

### 报告与配置
- HTML/Markdown 报告生成
- YAML 配置系统
- 多步 Pipeline 执行器
- Replay 调度

---

## 已知限制

1. **仅支持离线分析** — 不支持实时流式接入生产流量数据
2. **线性扩展假设** — `plan-capacity` 使用线性模型，在高并发下可能偏离实际
3. **单集群视角** — 暂不支持跨集群/跨区域联合优化
4. **GPU 型号覆盖** — 内置 profile 仅包含 A100/H100，其他型号需手动配置
5. **Benchmark 格式** — 仅支持 xpyd-bench 原生格式和兼容 JSON，不支持其他 benchmark 工具的直接导入

---

## 下一步计划

- 扩展 GPU profile 库（L40S, B200 等新硬件）
- 在线/流式分析模式探索
- 跨集群联合优化
- 与 xPyD-proxy 自动调参的闭环集成
- Web UI dashboard（替代 TUI）
- 更丰富的可视化（交互式图表）
