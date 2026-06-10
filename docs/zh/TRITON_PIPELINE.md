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