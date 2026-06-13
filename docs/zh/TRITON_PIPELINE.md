# `torch_npu/_inductor` — Triton Codegen & Autotune Pipeline 文档

## 1. 整体架构一览

```
用户模型 (nn.Module / torch.compile)
    │
    ▼
[Dynamo] FX Graph  (trace + capture)
    │
    ▼
[Inductor] GraphLowering          ← graph.py  (patch_run_node)
    │  FX Node → IR Node (Pointwise / Reduction / ExternKernel …)
    │
    ▼
[Scheduler]                        ← torch._inductor.scheduler (patch_scheduler)
    │  节点融合决策 (can_fuse_vertical / horizontal)
    │
    ▼
[NPUCombinedScheduling]            ← codegen/npu_combined_scheduling.py
    │  路由: CATLASS template  vs  NPUTritonScheduling
    │
    ▼
[NPUTritonScheduling]              ← codegen/scheduling.py
    │  select_tiling → codegen_node_schedule → define_kernel
    │
    ▼
[NPUIndexTritonKernel]             ← codegen/triton.py
    │  生成 Triton Python kernel 源码
    │
    ▼
[cached_autotune / TileGenerator]  ← npu_triton_heuristics.py + fasta_autotune.py
    │  生成候选 tiling config 列表
    │
    ▼
[NPUCachingAutotuner]              ← npu_triton_heuristics.py
    │  _precompile_config → triton.compile (每个 config)
    │  _make_launchers → make_launcher (每个 binary)
    │
    ▼
[autotune_to_one_config]           ← NPUCachingAutotuner.autotuner()
    │  benchmark_all_configs → bench(launcher)
    │  选出最优 launcher
    │
    ▼
[最优 launcher 执行]               ← NPUCachingAutotuner.run()
    │  可选: save_npu_kernel / check_accuracy / debug_kernel
    └──→ 实际 NPU 执行
```

---

## 2. 入口：模块注册 (`__init__.py`)

`torch_npu/_inductor/__init__.py` 在 import 时通过 monkey-patch 的方式接管 PyTorch inductor 的多个关键位置：

| Patch 函数 | 目标 | 作用 |
|---|---|---|
| `register_backend_for_device('npu', NPUCombinedScheduling, ...)` | inductor 后端注册 | 让 inductor 对 NPU device 使用 NPU 调度 |
| `patch_triton_heuristics_cached_autotune()` | `torch._inductor.runtime.triton_heuristics.cached_autotune` | 替换为 NPU 版本，使用 `NPUCachingAutotuner` |
| `patch_gen_common_triton_ext_imports()` | `triton.gen_common_triton_imports` | 注入 `torch_npu` 的 import，让生成的 Triton 代码包含 NPU helpers |
| `patch_run_node()` | `GraphLowering.run_node` | NPU 特定的 IR 节点处理（stride 对齐等） |
| `patch_scheduler()` | Scheduler | 覆盖 `are_long_distant_nodes`（融合距离阈值 20 vs GPU 的 64） |
| `patch_tuning_process()` + `patch_tuning_process_pool()` | `autotune_process` | 将 `CUDA_VISIBLE_DEVICES` 替换为 `ASCEND_RT_VISIBLE_DEVICES` |
| `_replace_precompile()` (可选) | `NPUCachingAutotuner.precompile` | 替换为并行预编译版本 `precompile_parallel` |
| `_replace_benchmark_all_configs()` (可选) | `CachingAutotuner.benchmark_all_configs` | 侵入式替换 benchmark 函数 |

---

## 3. GraphLowering — FX → Inductor IR

**文件：** `graph.py`

`patch_run_node()` 覆盖 `GraphLowering.run_node`，在原始 inductor 逻辑之上增加：

- **Stride 对齐**：对 output 节点按 eager model 的 stride 调整内存布局（`require_stride_order` / `require_exact_strides`）
- **Channels-last 优化**：4D tensor 的 `channels_last` stride 优化
- **内存实现控制**：多用户节点、超过 max_reads 的 buffer 强制 `realize()`
- **Lowering axis 限制**：按 `npu_config.lowering_axis_count` 截断维度

---

## 4. Scheduler — 节点融合

**文件：** `torch._inductor.scheduler`（被 `patch_scheduler` 修改）

关键修改：

```python
# codegen/scheduling.py
def are_long_distant_nodes(node1, node2) -> bool:
    proximity_score = max(
        abs(node1.min_order - node2.max_order),
        abs(node2.min_order - node1.max_order),
    )
    return proximity_score > 20  # NPU: 20, GPU default: 64
```

更保守的融合策略，防止大 kernel 导致 NPU 内存溢出。

---

## 5. NPUCombinedScheduling — 调度路由

**文件：** `codegen/npu_combined_scheduling.py`

持有两个子调度器：

- **`CATLASSScheduling`** — CATLASS 算子库路径（GEMM 等模板算子）
- **`NPUTritonScheduling`** — Triton 生成路径

路由逻辑（`choose_node_backend`）：

```python
def choose_node_backend(self, node):
    if self._catlass_scheduling.is_catlass_template(node):
        return self._catlass_scheduling
    return self._triton_scheduling
```

---

## 6. NPUTritonScheduling — Tiling 与 Kernel 代码生成

**文件：** `codegen/scheduling.py`

核心流程 `codegen_node_schedule`：

```
1. select_tiling(node_schedule, numel, reduction_numel)
       ↓  返回 tiling dict: {"x": numel_x, "r": numel_r, ...}
2. NPUIndexTritonKernel(tiling, features=kernel_features)
       ↓  构造 kernel 对象
3. decide_codegen_dims_in_kernel(node_schedule, kernel)
       ↓  分析 index，决定哪些轴是 split / tiling / no-loop
4. codegen_node_schedule_with_kernel(node_schedule, kernel)
       ↓  遍历 schedule node，调用 kernel.codegen_*
5. kernel.codegen_kernel()
       ↓  返回 Triton Python 源码字符串
6. define_kernel(src_code, ...)
       ↓  注册到 wrapper，返回 kernel_name
7. kernel.call_kernel(kernel_name)
       ↓  在 wrapper 中生成调用语句
```

---

## 7. NPUIndexTritonKernel — Triton 源码生成

**文件：** `codegen/triton.py`

继承链：`NPUIndexTritonKernel → TritonKernel → SIMDKernel`

### 关键特性

- **多轴 tiling**：支持 `w/v/t/z/y/x` 多个 pointwise 轴 + `r` reduction 轴，映射为 `XBLOCK/YBLOCK/RBLOCK` 等 constexpr
- **自定义 IterationRangesEntry**：`IterationRangesEntryNPUIndex` 带有 `is_tiling_axis`、`is_split_axis`、`is_no_loop_axis` 标记
- **NPU 算子覆写**：`NPUTritonKernelOverrides` 将 `exp/sqrt/tanh/floor/erf/ceil` 替换为 `tl_math.*`（NPU 扩展实现）
- **index_select / gather / scatter 特化**：生成 `extension.custom("__builtin_index_select", ...)` / `extension.gather_out_to_ub(...)` 等 NPU 原语
- **生成的 import 替换**：`gen_common_triton_imports()` 在标准 import 后追加：

```python
import torch_npu
from torch_npu._inductor import npu_triton_heuristics as triton_heuristics
from torch_npu._inductor.npu_triton_helpers import libdevice, extension, math as tl_math
```

### 生成 kernel 的结构示例

```python
@triton_heuristics.cached_autotune(
    size_hints=[...],
    configs=[Config({...}, num_warps=1, num_stages=1), ...],
    triton_meta={...},
    inductor_meta={
        "split_axis": [...], "tiling_axis": [...],
        "no_loop_axis": [...], "low_dims": [...],
        "npu_kernel_type": "simd",
        "kernel_name": "triton_kernel_0",
        ...
    },
    heuristic_type=HeuristicType.DEFAULT,
    filename=__file__,
)
@triton.jit
def triton_kernel_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, ...):
    ...
```

---

## 8. Tiling Config 生成

**文件：** `codegen/tile_generator.py` | `fasta_autotune.py`

由 `triton_config_npu_index()` 触发（在 `cached_autotune` 内部），根据 `npu_config.fasta_autotune` 开关选择不同策略：

### 标准路径：`TileGenerator`

```
TileGenerator.descend_split_tiling()
    ↓
按 split_axis / tiling_axis / no_loop_axis 枚举所有合法的 (XBLOCK, XBLOCK_SUB, ...) 组合
    ↓
根据 npu_kernel_type (SIMD / SIMT / SIMD_SIMT_MIX) 调整范围
    ↓
返回 List[triton.Config]  （每个 Config 包含 kwargs + num_warps + num_stages）
```

对于 `SIMD_SIMT_MIX` 类型，会分别生成三组 config：

```python
tile_generator.set_kernel_type(NPUKernelType.SIMT_ONLY)
configs.extend(tile_generator.descend_split_tiling())
tile_generator.set_kernel_type(NPUKernelType.SIMT_TEMPLATE)
configs.extend(tile_generator.descend_split_tiling())
tile_generator.set_kernel_type(NPUKernelType.SIMD)
configs.extend(tile_generator.descend_split_tiling())
```

### FastA 路径：`FastATileGenerator` + `NPUFastAutotuner`

两种方法（由 `FastASetting.autotune_method` 控制）：

- **SampleStack**：分桶采样，每桶取最多 `bucket_max_config_num` 个配置，减少 benchmark 次数
- **Expert**：专家规则生成 + 基于 UB（Unified Buffer）使用量剪枝

所有 config 在最后被附加 `split_axis` 和 `split_blocks` 字段，用于运行时 grid 计算：

```python
cfg.kwargs["split_axis"] = tuple(split_axis)
cfg.kwargs["split_blocks"] = tuple(split_blocks)
```

---

## 9. `cached_autotune` — 装饰器工厂

**文件：** `npu_triton_heuristics.py:1201`

替换 `torch._inductor.runtime.triton_heuristics.cached_autotune`，对 `@triton.jit` 函数进行包装：

```
cached_autotune(size_hints, configs, triton_meta, ...)
    │
    ├─ 1. unique_configs(configs)  去重
    │
    ├─ 2. AutotuneCache.create(...)  尝试读取磁盘/远程缓存
    │       如命中 best_config → configs 缩减为 [best_config]
    │
    ├─ 3. 根据条件选择 Autotuner 类型：
    │       profile_bandwidth=True  → NPUDebugAutotuner
    │       fasta_autotune=True     → NPUFastAutotuner
    │       else                    → NPUCachingAutotuner
    │
    └─ 4. return decorator(fn)  → Autotuner 实例（替换原始 @jit fn）
```

---

## 10. NPUCachingAutotuner — 预编译与 Launcher 创建

**文件：** `npu_triton_heuristics.py:420`

### 10.1 预编译阶段 `precompile()`

```
precompile()
    │
    └─ _precompile_worker()  [串行] 或 _precompile_worker_parallel() [并行线程池]
            │
            ├─ for each Config c:
            │       _precompile_config(c)
            │           │
            │           ├─ copy triton_meta，注入 cfg.kwargs (XBLOCK=128 等)
            │           ├─ 构造 ASTSource(fn, signature, constants)
            │           ├─ 构造 GPUTarget(device_type, cc, warp_size=32)
            │           ├─ options = {num_warps, num_stages, compile_mode,
            │           │             enable_vf_fusion, multibuffer, ...}
            │           ├─ binary = triton.compile(ASTSource, target, options)
            │           │       ↑ 调用 Triton 编译器，生成 NPU 二进制 (.npubin)
            │           │       ↑ binary.metadata.required_ub_bits  ← UB 使用量反馈
            │           └─ return TritonCompileResultNpu(binary, cfg, compile_meta, inductor_meta)
            │
            └─ self.compile_results = [TritonCompileResultNpu, ...]
```

若所有 config 编译失败，自动开启 `enable_vf_fusion=True` 重试一轮。

### 10.2 Launcher 创建 `_make_launchers()`

```
_make_launchers()
    │
    └─ for result in compile_results:
            result.make_launcher()  [TritonCompileResultNpu]
                │
                ├─ 生成 GridExprNpu（从 inductor_meta["grid_type"] 动态获取）
                │       grid.generate(cfg)
                │       → x_grid = f"({xnumel} + {XBLOCK} - 1) // {XBLOCK}"
                │
                ├─ 动态 exec() 生成 launcher 函数：
                │       def launcher(in_ptr0, ..., stream):
                │           grid_0 = ...
                │           runner(grid_0, grid_1, grid_2, stream, function, ...)
                │
                └─ launcher.config = cfg  (保留 Config 对象用于后续筛选)
```

---

## 11. 运行时 Autotune：`run()` → `autotuner()` → `autotune_to_one_config()`

**文件：** `npu_triton_heuristics.py:1046`

```
NPUCachingAutotuner.run(*args, stream, ...)
    │
    ├─ [快速路径] 若 xnumel == 0，直接 return（空 kernel 优化）
    │
    ├─ [解释模式] triton_interpret=True → fn[grid](*args, **cfg.kwargs)
    │
    ├─ [单 launcher 已有 fallback] 直接调用
    │
    └─ autotuner(*args, stream, ...)
            │
            ├─ len(launchers) == 0:
            │       precompile()           ← 触发编译（lazy compilation）
            │
            ├─ len(launchers) > 1:
            │       autotune_to_one_config(*args, **kwargs)
            │           │
            │           └─ benchmark_all_configs(*args, **kwargs)
            │                   │
            │                   └─ for launcher in launchers:
            │                           bench(launcher, *args)
            │                               │
            │                               └─ benchmarker.benchmark_gpu(kernel_call, rep=1)
            │                                  或 do_bench_using_profiling_npu (profiler 模式)
            │                   → timings: Dict[launcher, ms]
            │                   → self.launchers = [min(timings)]
            │                   → save_cache_hook(best_config)  ← 写磁盘/远程 cache
            │
            └─ [可选] coordinate_descent_tuning()  ← 坐标下降微调
    │
    ├─ [debug 模式] maybe_run_debug() → check_accuracy / fallback_to_fx / data_dump
    │
    ├─ [保存 kernel] save_npu_kernel()
    │       CudaKernelParamCache.set(key, params, binary.asm["npubin"])
    │
    └─ launcher(*args, stream=stream)  ← 最终执行
```

---

## 12. AutotuneCache — 磁盘/远程缓存

**文件：** `runtime/autotune_cache.py`（re-export from `torch._inductor.runtime.autotune_cache`）

```
AutotuneCache.create(inductor_meta, filename, configs_hash)
    │
    ├─ 计算 cache key = hash(filename + configs_hash + inductor_meta)
    │
    ├─ read_best(inductor_meta, configs)
    │       → 若缓存命中，返回 [best_config]，跳过全量 benchmark
    │
    └─ save(best_config, autotune_time_ns)
            → 持久化最优 config
```

Cache 触发条件（满足全部才启用）：

- `not force_disable_caches`
- `filename is not None`（非 inline kernel）
- `len(configs) > 1`（有多个候选，才值得 cache）
- `not TRITON_INTERPRET=1`

---

## 13. 关键数据流：`inductor_meta` 字段

`inductor_meta` 是贯穿整条 pipeline 的 metadata 字典，各阶段关键字段：

| 字段 | 生成位置 | 用途 |
|---|---|---|
| `kernel_name` | `scheduling.py:define_kernel` | 唯一标识 kernel |
| `split_axis` | `scheduling.py:codegen_node_schedule` | 哪些轴是 grid split 轴 |
| `tiling_axis` | 同上 | 哪些轴是 block-level tiling 轴 |
| `no_loop_axis` | 同上 | 哪些轴无需循环（size==1） |
| `low_dims` | 同上 | 低维轴信息，供 TileGenerator 剪枝 |
| `npu_kernel_type` | `NPUIndexTritonKernel` | SIMD / SIMT / MIX |
| `axis_names` | `NPUIndexTritonKernel` | 轴名映射（x/y/r 等） |
| `grid_type` | `NPUIndexTritonKernel` | GridExpr 子类名，决定 grid 计算方式 |
| `split_axis_dtype` | `NPUIndexTritonKernel` | 轴的数据类型，影响 UB 计算 |
| `traced_graph_hash` / `traced_graph_dir` | `scheduling.py`（dump mode） | FX graph fallback 路径 |
| `dual_reduction` | `NPUIndexTritonKernel` | 双 reduction 轴标记 |

---

## 14. NPUKernelType — 编译模式

**文件：** `codegen/triton_utils.py`

```python
class NPUKernelType(Enum):
    SIMD          # 向量化模式（默认）
    SIMT_ONLY     # 标量模式（等价 CUDA SIMT）
    SIMT_TEMPLATE # SIMT 模板模式
    SIMD_SIMT_MIX # 混合模式，TileGenerator 同时生成三类 config
```

`compile_mode` 字段直接传入 `triton.compile(options={..., "compile_mode": ...})`，控制 Triton 后端的编译策略。

---

## 15. 完整调用时序图

```
torch.compile(model)(inputs)
    │
    ▼
Dynamo trace → FX Graph
    │
    ▼
GraphLowering.run()              [graph.py: patch_run_node]
    每个 FX node → Inductor IR
    │
    ▼
Scheduler.finalize_schedule()    [scheduler.py: patch_scheduler]
    融合节点 → SchedulerNode / FusedSchedulerNode
    │
    ▼
NPUCombinedScheduling.codegen_node(fused_node)
    │
    ├─[CATLASS]→ CATLASSScheduling.codegen_template()
    │
    └─[Triton]→ NPUTritonScheduling.codegen_node_schedule()
                    │
                    ├─ 1. select_tiling → tiling dict
                    ├─ 2. NPUIndexTritonKernel(tiling)
                    ├─ 3. codegen_node_schedule_with_kernel → body
                    ├─ 4. kernel.codegen_kernel() → src_code (Triton Python)
                    └─ 5. define_kernel → kernel_name
                              │ wrapper 中生成:
                              │   @cached_autotune(configs=[...], inductor_meta={...})
                              │   @triton.jit
                              │   def triton_kernel_N(...): ...
                              │
                              ▼ (运行时首次调用)
                    NPUCachingAutotuner.run()
                        │
                        ├─ precompile(): triton.compile(config) × N  →  launchers
                        ├─ autotune_to_one_config(): bench(launcher) × N  →  best
                        └─ launcher(*args, stream)  →  NPU 执行
```

---

## 16. 环境变量 & 配置开关速查

| 变量/配置 | 文件 | 作用 |
|---|---|---|
| `TORCHINDUCTOR_NPU_BACKEND` | `__init__.py` | `default`=Triton路径, `tilelang`=TileLang路径, `mlir`/`dvm` |
| `npu_config.fasta_autotune` | `fasta_autotune.py` | 启用 FastA 快速调优（减少 benchmark 次数） |
| `npu_config.fasta_autotune_method` | `fasta_autotune.py` | `SampleStack` 或 `Expert` |
| `npu_config.max_precompiled_thread_num` | `__init__.py` | 并行预编译线程数（>1 时启用并行） |
| `npu_config.aggresive_autotune` | `__init__.py` | 替换 `benchmark_all_configs` 为 NPU 实现 |
| `INDUCTOR_ASCEND_DEBUG` | `npu_triton_heuristics.py` | 开启 Triton 编译 debug 模式 |
| `TRITON_INTERPRET` | `npu_triton_heuristics.py` | 软件解释执行，跳过真实编译 |
| `FAST_RUN_WITH_MAX_TILING_NUM` | `npu_triton_heuristics.py` | 限制候选 config 数量（快速跑测使用） |
| `ASCEND_RT_VISIBLE_DEVICES` | `autotune_process.py` | 多卡 autotune 时可见设备列表 |
| `npu_config.dump_fx_graph` | `scheduling.py` | 导出 FX graph 用于精度比对/fallback |
| `npu_config.check_accuracy` | `npu_triton_heuristics.py` | 运行时对比 Triton kernel 与 FX graph 输出 |





# 融合策略

## `NPUTritonScheduling.can_fuse`（scheduling.py）

完整的融合策略，分四种情况：



```
1. Reduction + Reduction
   → numel1 == numel2 AND rnumel1 == rnumel2 才能融合

2. Pointwise + Pointwise  
   → numel/rnumel 匹配
   → 检查合并后的 tiling 是否一致（防止 tiling 冲突）

3. Pointwise → Reduction（标量/向量 epilogue）
   → pointwise_numel == reduction_numel：scalar epilogue（每行一个值）
   → pointwise_numel == reduction_numel × rnumel：vector epilogue（每个元素）
   → 检查 SIMDKernel.is_compatible 确保 ranges 兼容

4. Reduction → Pointwise
   → 等价于 case 3（swap 参数）
```

------

## `TileLangScheduling.can_fuse`（tilelang.py）

在 NPUTriton 的基础上加了 **TileLang 专属的限制层**：



```python
def can_fuse(self, node1, node2):
    # ① Reduction + Reduction：只允许特定组合
    if node1.is_reduction() and node2.is_reduction():
        if not _is_supported_reduction_fusion(node1, node2):
            return False       # 只允许 [max,sum] 或 [sum,sum]
        return super().can_fuse(...)

    # ② Reduction + Pointwise（或反向）
    if node1.is_reduction() != node2.is_reduction():
        # vector epilogue 只允许 max+sum / sum+sum / sum
        if is_vector_epilogue and reduction_types not in [...]:
            return False
        # scalar epilogue：pointwise_numel == reduction_numel 且 rnumel==1
        if same_numel(pointwise_numel, reduction_numel):
            return SIMDKernel.is_compatible(...)

    # ③ 其余情况全部交给 NPUTriton
    return super().can_fuse(node1, node2)
```

再加上 `codegen_node` 里的额外拦截：



```python
def _unsupported_tilelang_reason(nodes, ...):
    # 超过 2 个 reduction → 回退 Triton
    # 含 template 节点 → 回退 Triton
    # 含 split-scan 节点 → 回退 Triton
```

------

## 两者差异对比

| 场景                  | NPUTriton               | TileLang                            |
| --------------------- | ----------------------- | ----------------------------------- |
| Reduction + Reduction | numel/rnumel 匹配即可   | 只允许 `[max,sum]` / `[sum,sum]`    |
| Vector epilogue       | 支持所有 reduction 类型 | 仅限 max+sum / sum+sum / sum        |
| Scalar epilogue       | 支持所有                | 同上                                |
| Template 节点         | 支持（TritonTemplate）  | 直接回退 Triton                     |
| Split-scan            | 有条件支持              | 直接回退 Triton                     |
| 兜底                  | 自己决定                | `super().can_fuse()` 交给 NPUTriton |

**本质区别**：NPUTriton 是完整实现，TileLang 是在它之上做了"白名单"过滤——只有 TileLang 已经支持的 reduction 模式才自己处理，其余全部 fallback 到 NPUTriton。



## 融合限制总览（从外到内，按拦截顺序）

------

### 第一层：距离限制（全局，scheduling.py:63）



```python
proximity_score = max(
    abs(node1.min_order - node2.max_order),
    abs(node2.min_order - node1.max_order),
)
# NPU A5 阈值 20（GPU 是 64）
return proximity_score > 20   # True → 阻止融合
```

目的：防止 horizontal fusion 拉长 tensor 的 live interval，增加内存峰值。

------

### 第二层：NPUTritonScheduling.can_fuse（scheduling.py:474）

按场景分支：



```
① ForeachKernel 节点
   → 交给 ForeachKernelSchedulerNode.can_fuse

② node1 含 cat_store
   → 直接 False

③ Split-scan + Reduction
   → False

④ Reduction + Reduction
   → numel1 == numel2 AND rnumel1 == rnumel2 才允许

⑤ Pointwise + Pointwise
   → numel/rnumel 必须匹配（除非是 template prologue 特殊路径）
   → 只允许 TritonTemplate，不允许 CUDATemplate
   → config.triton.tiling_prevents_pointwise_fusion 开启时：
       合并后的 tiling 必须与各自 tiling 一致

⑥ Pointwise → Reduction（或反向）
   → vector epilogue（numel_pw == numel_r × rnumel_r）：
       - SIMDKernel.is_compatible 检查 ranges 兼容
       - config.triton.tiling_prevents_reduction_fusion 开启时检查 tiling
   → scalar epilogue（numel_pw == numel_r）：
       - 直接允许
   → 其他 numel 关系：不允许
```

------

### 第三层：TileLangScheduling.can_fuse（tilelang.py:1572）

在 NPUTriton 基础上叠加更严格的限制：



```
① Reduction + Reduction
   → 只允许 ["max","sum"] 或 ["sum","sum"] 序列
   → 其余（如 ["min","sum"]）直接 False

② Reduction + Pointwise（或反向）
   → 已融合的 reduction 若超过 1 个，序列必须是 ["max","sum"] 或 ["sum","sum"]
   → vector epilogue 时，reduction 类型只允许 ["max","sum"] / ["sum","sum"] / ["sum"]
   → scalar/vector epilogue：pointwise_rnumel 必须为 1
   → SIMDKernel.is_compatible 检查所有 pointwise 节点

③ 其余情况
   → 交给 super().can_fuse()（NPUTritonScheduling）
```

------

### 第四层：codegen_node 拦截（tilelang.py:1620）

`can_fuse` 通过后，真正生成代码前的最后一道门：



```python
def _unsupported_tilelang_reason(nodes, ...):
    if len(reduction_nodes) > 2:      # _MAX_REDUCTIONS_PER_KERNEL = 2
        → fallback Triton
    if any node.is_template():
        → fallback Triton
    if any node.is_split_scan():
        → fallback Triton
```

------

### 第五层：kernel 运行时 NotImplementedError → fallback Triton

`SIMDScheduling.codegen_node` 用 `try/except NotImplementedError` 包住整个 kernel 生成：



```
TileLangKernel.load()
   → dtype 不在 _ANY_SUPPORTED_DTYPE → NotImplementedError

TileLangKernel.store()
   → mode == "atomic_add" → NotImplementedError
   → _check_op_graph_dtype() 发现 op 不支持该 dtype → NotImplementedError

TileLangKernel.reduction()
   → value 是 tuple → NotImplementedError
   → reduction_type 不在 {"sum","max","min"} → NotImplementedError
   → 超过 2 个 reduction → NotImplementedError
   → 序列不是 ["max","sum"] / ["sum","sum"] → NotImplementedError

TileLangKernel._emit_vec_op()
   → op 没有对应的 T.v* → NotImplementedError
```

------

### 汇总表

| 层次        | 位置                   | 判断依据                          | 结果            |
| ----------- | ---------------------- | --------------------------------- | --------------- |
| 距离        | scheduling.py:85       | `order` 差值 > 20                 | 拒绝融合        |
| NPUTriton   | scheduling.py:474      | numel/rnumel/tiling/template      | 拒绝融合        |
| TileLang    | tilelang.py:1572       | reduction 类型白名单              | 拒绝融合        |
| codegen拦截 | tilelang.py:1620       | template/split-scan/超量reduction | fallback Triton |
| 运行时      | tilelang.py kernel方法 | dtype/op支持、tuple reduction     | fallback Triton |

---

# TileLang Matmul (T.gemm) Codegen

## 1. 为什么 matmul 需要单独处理

Pointwise/Reduction op（`torch.add`、`x.sum()`）经过 inductor lowering 后产出 `ComputedBuffer`，被 scheduler 包成 `SchedulerNode`，最终由 `TileLangKernel.codegen_kernel()` 生成 `T.vadd/T.reduce` 代码。

`aten.mm` 走的是完全不同的路径：

```
aten.mm
  → _register_npu_inductor_mm()  ← 必须显式注册
  → autotune_select_algorithm([TileLangGemmCaller])
  → TemplateBuffer (TemplateSchedulerNode)
  → TileLangScheduling.codegen_template()
  → T.gemm @T.prim_func
```

- 产出的是 `TemplateBuffer`，不是 `ComputedBuffer`
- scheduler 把它包成 `TemplateSchedulerNode`，调 `codegen_template()` 而不是 `codegen_nodes()`
- 如果没有注册 lowering，inductor 会尝试 `make_fallback(aten.mm)`，但 `aten.mm` 同时在 `decompositions` 里，触发断言：`both a fallback and a decomp for same op`

因此 `__init__.py` 的 tilelang 分支必须额外调用：

```python
_register_npu_inductor_mm()
_register_npu_inductor_addmm()   # nn.Linear(bias=True) → aten.addmm
_register_npu_inductor_bmm()     # torch.bmm
```

---

## 2. 完整 Pipeline

```
torch.mm(A, B)                        # 用户调用
    │
    ▼ Dynamo trace
aten.mm.default(arg0, arg1)           # FX graph 节点
    │
    ▼ GraphLowering.run_node()
_register_npu_inductor_mm() 里的 tuned_mm()
    │
    ├─ mm_args()                       # 规范化输入，得到 layout(dtype=fp16)
    │
    ├─ use_tilelang_template() == True?
    │       TORCHINDUCTOR_NPU_BACKEND=tilelang
    │       AND layout.dtype in {fp16, int8}
    │
    ├─ add_tilelang_gemm_choices(choices=[TileLangGemmCaller])
    │       ↑ 覆盖 layout.dtype 为 accum_dtype(fp32)
    │       ↑ 用 FixedLayout(dtype=fp32) 替换原 fp16 layout
    │
    ▼ autotune_select_algorithm(choices=[TileLangGemmCaller])
    │       len(choices)==1 → choices[0].output_node()  (不做 benchmark)
    │
    ▼ TileLangGemmCaller.output_node()
    │       TemplateBuffer(
    │           layout=FixedLayout(fp32, size=[M,N]),
    │           inputs=[mat1, mat2],
    │           make_kernel_render=_TileLangGemmRender(params)
    │       )
    │
    ▼ 调度阶段
Scheduler.codegen()
    │   node.is_template() == True
    │
    ▼ TileLangScheduling.codegen_template(template_node, epilogue_nodes)
    │
    ├─ 检测 isinstance(render, _TileLangGemmRender)  ← TileLang 标记
    │       否 → 转发给 _triton_scheduling.codegen_template()
    │
    ├─ codegen_tilelang_mm_src(kernel_name, ...)    ← 生成 @T.prim_func 源码
    │
    ├─ define_kernel_matmul(src_code, ...)           ← 注入 wrapper 到 header
    │
    ├─ n.mark_run() for n in [template_node, *epilogue_nodes]
    │
    └─ wrapper.writeline(
           "tilelang_mm_0(mat1, mat2, out, M, N, K)"
       )
```

---

## 3. 关键类与函数

### `_TileLangGemmParams`（dataclass）

存储生成 T.gemm kernel 所需的全部参数：

```python
@dataclasses.dataclass
class _TileLangGemmParams:
    M, N, K: Any           # sympy 表达式，工厂函数运行时解析为整数
    dtype: torch.dtype     # 输入 dtype，e.g. float16
    accum_dtype: torch.dtype # 累加 dtype，e.g. float32
    block_M: int = 128
    block_N: int = 128
    block_K: int = 64
```

---

### `_TileLangGemmRender`（marker 类）

存储在 `TemplateBuffer.make_kernel_render` 字段，作为 TileLang matmul 的**类型标记**：

```python
class _TileLangGemmRender:
    params: _TileLangGemmParams
    def __call__(self, buf): raise NotImplementedError  # 不会被调用
```

`TileLangScheduling.codegen_template()` 用 `isinstance(render, _TileLangGemmRender)` 检测它，触发 T.gemm 代码生成路径。若不是该类型，则转发给 NPU Triton 处理。

---

### `TileLangGemmCaller(ChoiceCaller)`

```python
class TileLangGemmCaller(ChoiceCaller):
    def output_node(self) -> TensorBox:
        buf = TemplateBuffer(
            layout=...,            # FixedLayout(fp32)
            inputs=[mat1, mat2],
            make_kernel_render=_TileLangGemmRender(params),
        )
        return TensorBox.create(buf)

    def benchmark(self, *args, out=None) -> float:
        return float("inf")        # 不参与 benchmark，确保单选时直接走 output_node()
```

---

### `add_tilelang_gemm_choices()`

注册 TileLang 为 `aten.mm` 的唯一候选（在 `mm.py` 的 `tuned_mm()` 里调用）：

```python
def add_tilelang_gemm_choices(choices, layout, input_nodes, ...):
    dtype = mat1.get_dtype()           # fp16
    accum_dtype = {fp16: fp32, int8: int32}[dtype]

    # 关键：覆盖 layout 为 fp32，否则 inductor 分配 fp16 buffer
    # 导致 T.gemm 写入 fp32 值被误读为 fp16 → 产生 nan/溢出
    layout = FixedLayout(
        device=layout.device,
        dtype=accum_dtype,            # fp32
        size=layout.size,
        stride=[N, 1],                # 行优先 contiguous
    )
    choices.append(TileLangGemmCaller(...))
```

**为什么必须用 `FixedLayout` 而不是 `FlexibleLayout`：**
当 mm 后跟 epilogue（`relu/sigmoid/scale`）时，inductor 在 `SchedulerNode._compute_attrs()` 里调用 `make_indexer()` 读取 mm 输出的访问模式。`FlexibleLayout.allow_indexing = False` 会触发断言。

---

### `codegen_tilelang_mm_src()`

生成 `@T.prim_func` 源码字符串（形状通过工厂函数的 `_M/_N/_K` 变量注入）：

```python
import tilelang.language as T

_block_M = 128
_block_N = 128
_block_K = 64

@T.prim_func
def tilelang_mm_0_prim_fn(
    A: T.Tensor((_M, _K), 'float16'),
    B: T.Tensor((_K, _N), 'float16'),
    C: T.Tensor((_M, _N), 'float32'),
):
    with T.Kernel(T.ceildiv(_N, _block_N) * T.ceildiv(_M, _block_M),
                  is_npu=True) as (cid, _):
        by = cid // T.ceildiv(_N, _block_N)   # M tile 索引
        bx = cid % T.ceildiv(_N, _block_N)    # N tile 索引

        A_shared = T.alloc_shared((_block_M, _block_K), 'float16')
        B_shared = T.alloc_shared((_block_K, _block_N), 'float16')
        C_local  = T.alloc_fragment((_block_M, _block_N), 'float32')  # 片上累加器

        for k in T.Pipelined(T.ceildiv(_K, _block_K), num_stages=2):
            T.copy(A[by * _block_M, k * _block_K], A_shared)
            T.copy(B[k * _block_K, bx * _block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local, initC=(k == 0))  # 昇腾 AIC cube 指令

        T.copy(C_local, C[by * _block_M, bx * _block_N])
```

**分块映射：**

| 变量 | 含义 |
|---|---|
| `cid` | 当前 AIC 核的 block 编号（`blockIdx.x`） |
| `by` | C 矩阵 M 方向的 tile 编号 |
| `bx` | C 矩阵 N 方向的 tile 编号 |
| `A_shared` / `B_shared` | L1 缓冲（`shared.dyn`，FFTS 管理） |
| `C_local` | 片上 fragment（`local.fragment`，cube 累加器） |
| `T.gemm(..., initC=(k==0))` | k=0 时初始化 C_local，否则累加 |
| `T.Pipelined(..., num_stages=2)` | 双缓冲流水，隐藏 DMA 延迟 |

---

### `define_kernel_matmul()`

将 prim_func 包装为带形状缓存的调用函数，注入 `wrapper.header`：

```python
# 工厂函数：接受运行时 M/N/K，动态构造 prim_func
def _prim_factory_tilelang_mm_0(M, N, K):
    _M, _N, _K = M, N, K
    import tilelang.language as T
    @T.prim_func
    def tilelang_mm_0_prim_fn(A, B, C): ...
    return tilelang_mm_0_prim_fn

# 形状缓存：相同形状复用已编译 binary
_tilelang_mm_0_cache = {}

def tilelang_mm_0(A, B, C, M, N, K):
    _key = (int(M), int(N), int(K))
    if _key not in _tilelang_mm_0_cache:
        _tilelang_mm_0_cache[_key] = _tilelang_0.compile(
            _prim_factory_tilelang_mm_0(*_key), target='npuir'
        )
    _tilelang_mm_0_cache[_key](A, B, C)
```

在 `output_code.py` 中对应的调用语句：

```python
tilelang_mm_0(arg1_1, arg0_1, buf0, 256, 512, 128)
```

---

## 4. Dtype 处理

T.gemm 在硬件上使用 AIC cube 单元计算，**输入 fp16，累加 fp32**。这与 `aten.mm(fp16, fp16)` 的默认返回 dtype（fp16）不同：

| 阶段 | dtype |
|---|---|
| 输入 A / B | `float16`（原始用户数据） |
| C_local（片上累加） | `float32`（防止精度损失） |
| 输出 buffer C | `float32`（由 FixedLayout 保证） |
| 用户可见输出 | `float32`（与 eager 对比时自动 cast） |

---

## 5. 与 NPU Triton / CATLASS 的对比

| 特性 | NPU Triton | CATLASS | TileLang T.gemm |
|---|---|---|---|
| 触发条件 | 默认路径 | `use_catlass_template()` | `TORCHINDUCTOR_NPU_BACKEND=tilelang` |
| 硬件指令 | AIV 向量 | AIC cube (CATLASS lib) | AIC cube (`hivm.hir.mmadL1`) |
| 输出 dtype | fp16 | fp16 | fp32 |
| Epilogue 融合 | ✅ | ✅ | ✅（fused mm+relu 等） |
| 代码生成方式 | Triton JIT | C++ template | TileLang prim_func |
| 调度路径 | `codegen_nodes()` | `codegen_template()` | `codegen_template()` |