# TileLang Inductor Backend for Ascend NPU

## 概述

`tilelang.py` 实现了一个 PyTorch Inductor 的自定义 codegen 后端，将 Inductor 的 IR 编译为 TileLang `@T.prim_func` 内核，通过 `tilelang.compile(target='npuir')` 编译后在昇腾 NPU 上执行。

激活方式：

```bash
TORCHINDUCTOR_NPU_BACKEND=tilelang python your_script.py
```

或在代码中：

```python
import os
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "tilelang"
import torch
import torch_npu

model = torch.compile(model, backend="inductor")
```

---

## 架构概览

### 继承层次

```
上游 PyTorch Inductor
├── SIMDKernel
│   └── TritonKernel
│       └── NPUIndexTritonKernel     (triton.py)   ← NPU Triton kernel
│           └── TileLangKernel       (tilelang.py)  ← TileLang kernel
│
└── SIMDScheduling
    └── TritonScheduling
        └── NPUTritonScheduling      (scheduling.py) ← NPU Triton scheduler
            └── TileLangScheduling   (tilelang.py)   ← TileLang scheduler
```

`TileLangKernel` 继承 `NPUIndexTritonKernel` 的目的是复用其 range tree 基础设施和 axis 分析管道（`decide_codegen_dims_in_kernel`），同时通过完整覆写 `load/store/reduction/codegen_kernel` 来生成 TileLang 代码而非 Triton 代码。

`TileLangScheduling` 继承 `NPUTritonScheduling` 以复用完整的 scheduling 流程（包括 SplitTiling、ReductionAnalysis），仅覆写 `__init__`（强制 `self.kernel_type = TileLangKernel`）、`define_kernel` 和 `codegen_sync`。

---

## 完整 Pipeline

```
torch.compile(fn, backend="inductor")
        │
        ▼
Dynamo trace → FX graph → Inductor graph
        │
        ▼
NPUTritonScheduling.codegen_node_schedule()          ← 继承自 NPUTritonScheduling
  │
  ├─①  select_tiling()
  │       确定初始 tile shape（xnumel / rnumel）
  │
  ├─②  TileLangKernel.__init__(tiling, features)
  │       初始化 range tree、axis 字段、TileLang op graph 字段
  │
  ├─③  decide_codegen_dims_in_kernel()              ← 复用 NPUTritonScheduling 实现
  │    ├─ _mark_store_index_keys                    记录 Store index key，保护不被替换
  │    ├─ _transform_schedule_indexing
  │    │   ├─ split_and_set_ranges()               节点 var_ranges → kernel range tree
  │    │   ├─ transform_dims_in_indexing()         线性化拆分 index 表达式
  │    │   └─ substituted_dims_in_indexing()       替换冗余父轴
  │    ├─ _record_store_unified_indexing           收集 Store index
  │    ├─ _remove_substituted_dims_from_kernel     删掉无用轴
  │    └─ _finalize_kernel_codegen_dims
  │        ├─ SplitTiling.select_split_tiling_axis()
  │        │   ├─ find_lowest_dimension()          stride=1 的低维轴
  │        │   ├─ select_split_axis()              分配给 AI Core 的轴（≤3个）
  │        │   └─ select_tiling_axis()             在 SRAM 内 tile 的轴
  │        ├─ ReductionAnalysis()                  确定 reduced_dim
  │        └─ select_no_loop_axis()               ≤4KB 的 tiling 轴 → 无显式 for 循环
  │
  ├─④  codegen_node_schedule_with_kernel()         ← 上游 TritonScheduling 实现
  │       对每个 node 调 node._body(*index_vars)：
  │       ├─ TileLangKernel.load()                记录 _tl_inputs / _var_bufs
  │       ├─ TileLangKernel.store()               记录 _output_vars
  │       ├─ TileLangKernel.store_reduction()     记录 _is_reduction_output
  │       ├─ TileLangKernel.reduction()           记录 _var_ops[var] = ("reduce", mode, src)
  │       └─ TileLangKernel.codegen_body()        NOP（清空 buffer，不生成 Triton 代码）
  │
  ├─⑤  TileLangKernel.codegen_kernel()
  │    ├─ [pointwise]  _codegen_pointwise_kernel()
  │    │   ├─ 读 split_axis / tiling_axis → 确定 grid 和 T.copy offset
  │    │   ├─ T.alloc_shared(XBLOCK) × 每个输入输出
  │    │   ├─ T.copy(gm[offset], local)
  │    │   ├─ T.vadd/vsub/vexp/...               向量 intrinsic
  │    │   └─ T.copy(local, gm[offset])
  │    │
  │    └─ [reduction]  _codegen_reduction_kernel()
  │        ├─ T.Tensor((_xnumel, _rnumel)) 2D 张量签名
  │        ├─ T.alloc_shared(ROW_BLOCK, rnumel)   输入 block（cbuf）
  │        ├─ T.alloc_shared(ROW_BLOCK, 1)        输出 block（cbuf）
  │        ├─ T.copy(in[cid*ROW_BLOCK, 0], local, size=[ROW_BLOCK, N])
  │        ├─ T.reduce(src, dst, dims=1, reduce_mode='sum')
  │        └─ T.copy(out_local, out[cid*ROW_BLOCK, 0], size=[ROW_BLOCK, 1])
  │
  ├─⑥  TileLangScheduling.define_kernel()
  │       生成 Python wrapper（写入 wrapper.header）：
  │
  │       def _prim_factory_<name>(xnumel, rnumel):
  │           _xnumel = xnumel; _rnumel = rnumel
  │           @T.prim_func
  │           def <name>_prim_fn(...): ...
  │           return <name>_prim_fn
  │
  │       _<name>_cache = {}
  │
  │       def <name>(in_ptr0, ..., xnumel, rnumel):
  │           _key = (int(xnumel), int(rnumel))
  │           if _key not in _<name>_cache:
  │               _<name>_cache[_key] = tilelang.compile(
  │                   _prim_factory_<name>(*_key), target='npuir')
  │           _<name>_cache[_key](in_ptr0, ...)
  │
  └─⑦  TileLangKernel.call_kernel()
          wrapper.writeline("<name>(in_ptr0, ..., xnumel, rnumel)")
```

---

## 关键组件

### TileLangKernel

#### Op Graph 追踪机制

TileLang 不像 Triton 那样逐行 emit 代码，而是先建立 op graph，最后在 `codegen_kernel()` 里统一生成：

| 字段 | 类型 | 含义 |
|---|---|---|
| `_tl_inputs` | `dict[name → (var, local_name, dtype)]` | 每个输入 tensor 的本地 buffer 信息 |
| `_tl_input_indices` | `dict[name → sympy.Expr]` | 每个输入的 sympy index 表达式 |
| `_tl_outputs` | `dict[name → (var, local_name, dtype)]` | 每个输出 tensor 的本地 buffer 信息 |
| `_tl_output_indices` | `dict[name → sympy.Expr]` | 每个输出的 sympy index 表达式 |
| `_var_bufs` | `dict[var_name → local_buf_name]` | CSE 变量 → 读自哪个 local buffer |
| `_var_ops` | `dict[var_name → (op, operands)]` | CSE 变量 → 产生它的 op |
| `_var_consts` | `dict[var_name → literal]` | 常量 CSE 变量 |
| `_output_vars` | `dict[local_buf → (result_var, dtype)]` | 输出 buffer → 产生它的 CSE var |
| `_is_reduction_output` | `dict[local_buf → bool]` | 是否是 reduction 结果 |

#### 覆写逻辑

```
NPUIndexTritonKernel 方法          TileLangKernel 行为
─────────────────────────────────────────────────────────────────
load(name, index)                  记录 _tl_inputs[name] 和 _var_bufs[cse_var]
store(name, index, value)          记录 _output_vars 和 _check_op_graph_dtype
store_reduction(name, index, val)  记录 _output_vars + _is_reduction_output
reduction(dtype, src_dtype, rt, v) 记录 _var_ops[result] = ("reduce", mode, src)
codegen_body()                     NOP（只做 CSE 失效 + buffer 清理）
codegen_kernel()                   从 _tl_inputs/_var_ops/_output_vars 重建 prim_func
call_kernel(name)                  emit: name(tensors..., xnumel, rnumel)
```

### TileLangScheduling

```
NPUTritonScheduling 方法           TileLangScheduling 行为
──────────────────────────────────────────────────────────────────
__init__(input_scheduler)          super().__init__() 后强制 self.kernel_type = TileLangKernel
                                   （覆盖父类 __init__ 中设置的 NPUIndexTritonKernel）
codegen_node_schedule()            完全继承（含 decide_codegen_dims_in_kernel）
define_kernel(src, sched, k, h)    TileLang factory wrapper 模式（见 Pipeline ⑥）
codegen_sync()                     torch.npu.synchronize()
```

---

## Pointwise Kernel 生成

### 1-D 情形（单轴）

```python
_XBLOCK = 128

@T.prim_func
def kernel_prim_fn(
    in_ptr0: T.Tensor((_xnumel,), 'float32'),
    out_ptr0: T.Tensor((_xnumel,), 'float32'),
):
    with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):
        _in_ptr0_local  = T.alloc_shared((_XBLOCK,), 'float32')
        _out_ptr0_local = T.alloc_shared((_XBLOCK,), 'float32')

        T.copy(in_ptr0[cid * _XBLOCK], _in_ptr0_local)
        T.vadd(_in_ptr0_local, _in_ptr1_local, _out_ptr0_local)  # 由 _var_ops 生成
        T.copy(_out_ptr0_local, out_ptr0[cid * _XBLOCK])
```

### 2-D 情形（split_axis=[x0], tiling_axis=[x1]）

```python
@T.prim_func
def kernel_prim_fn(
    in_ptr0: T.Tensor((_xnumel,), 'float32'),
    out_ptr0: T.Tensor((_xnumel,), 'float32'),
):
    with T.Kernel(M, T.ceildiv(N, _XBLOCK), is_npu=True) as (_gs0, _gi):
        _in_ptr0_local  = T.alloc_shared((_XBLOCK,), 'float32')
        _out_ptr0_local = T.alloc_shared((_XBLOCK,), 'float32')

        T.copy(in_ptr0[_gs0 * N + _gi * _XBLOCK], _in_ptr0_local)
        T.vadd(...)
        T.copy(_out_ptr0_local, out_ptr0[_gs0 * N + _gi * _XBLOCK])
```

grid 和 copy offset 由 `_build_copy_offset()` 通过 sympy 替换 axis 符号计算得出：
- `split_axis[i].symbol()` → `sympy.Symbol(f"_gs{i}")`（AI Core 行号）
- `inner_axis.symbol()` → `sympy.Symbol("_gi") * _XBLOCK`（列 tile 偏移）

---

## Reduction Kernel 生成

### 设计约束

昇腾 NPU 的 `hivm.hir.store` 只支持：
- `gm → ub` / `ub → gm` / `ub → ub`

`T.alloc_fragment` 对应 `#hivm.address_space<zero>`（非 UB），`zero → gm` 的 `memref.copy` 会失败。

正确做法：
1. 输出 buffer 用 `T.alloc_shared((_ROW_BLOCK, 1))` → `cbuf (= UB)`
2. `_ROW_BLOCK × 1 × 4B = 128B` 满足 DMA 对齐（单 float 4B 过小）
3. 用 `T.reduce(..., dims=1)` 做批量规约，避免标量写

```python
_ROW_BLOCK = 32   # 每个 AI Core 处理的行数

@T.prim_func
def kernel_prim_fn(
    in_ptr0:  T.Tensor((_xnumel, _rnumel), 'float32'),  # 2-D: M × N
    out_ptr0: T.Tensor((_xnumel, 1),       'float32'),  # 2-D: M × 1
):
    with T.Kernel(T.ceildiv(_xnumel, _ROW_BLOCK), is_npu=True) as (cid, _):
        _in_ptr0_local  = T.alloc_shared((_ROW_BLOCK, _rnumel), 'float32')
        _out_ptr0_local = T.alloc_shared((_ROW_BLOCK, 1),       'float32')

        T.copy(in_ptr0[cid * _ROW_BLOCK, 0], _in_ptr0_local,
               size=[_ROW_BLOCK, _rnumel])

        T.reduce(_in_ptr0_local, _out_ptr0_local,
                 dims=1, reduce_mode='sum', clear=True)

        T.copy(_out_ptr0_local, out_ptr0[cid * _ROW_BLOCK, 0],
               size=[_ROW_BLOCK, 1])
```

> `T.copy` 的具体硬件指令由 TileLang 编译器在 NPUIR 降级时自动选择（`gm→cbuf` 降为 `hivm.hir.nd2nz`，`cbuf→gm` 降为 `hivm.hir.store: ub→gm`），不出现在 TileLang 源码层。

### dtype 提升

Inductor 对所有 reduction 调用 `upcast_acc_dtype()` 将 fp16 → fp32 accumulation。
即使输入 tensor 是 fp16，TileLang reduction kernel 的 `in_ptr0` 也是 fp32（inductor 在 kernel 外插入了 cast）。Pointwise kernel 不受此影响，保持原始 dtype。

---

## 向量 Op 映射

### Binary Ops（T.vXXX(A, B, C)）

| inductor op | TileLang | 支持 dtype |
|---|---|---|
| add | T.vadd | fp16, fp32 |
| sub | T.vsub | fp16, fp32 |
| mul | T.vmul | fp16, fp32, int16, int32, int64 |
| truediv | T.vdiv | fp16, fp32, int64 |
| maximum | T.vmax | fp16, fp32 |
| minimum | T.vmin | fp16, fp32, bf16, int16, int32, int64 |
| pow | T.vpow | int32 only |
| bitwise_and | T.vand | int8, int64, fp16, fp32, bool |

### Unary Ops（T.vXXX(A, B)）

| inductor op | TileLang | 支持 dtype |
|---|---|---|
| exp | T.vexp | fp16, fp32 |
| log | T.vln | fp16, fp32 |
| sqrt | T.vsqrt | fp16, fp32 |
| rsqrt | T.vrsqrt | fp16, fp32 |
| relu | T.vrelu | fp16, fp32 |
| sigmoid | T.vsigmoid | fp16, fp32 |
| cos | T.vcos | fp16, fp32 |
| sin | T.vsin | fp16, fp32 |
| tanh | T.vtanh | fp16, fp32 |
| erf | T.verf | fp16, fp32 |
| abs | T.vabs | fp16, fp32, uint8, int32, int64 |
| neg | T.vmul(x, -1.0) | fp16, fp32 |

不支持的 op/dtype 组合会抛 `NotImplementedError`，触发 inductor 自动 fallback 到 Triton。

### Reduction Ops（T.reduce）

| inductor reduction_type | TileLang reduce_mode |
|---|---|
| sum | 'sum' |
| max | 'max' |
| min | 'min' |
| prod | 'prod' |
| xor_sum | 'xor' |

---

## NPU Triton Pipeline

NPU Triton 后端（`triton.py` + `scheduling.py`）的完整流程，作为对比基准：

```
NPUTritonScheduling.codegen_node_schedule()
  │
  ├─①  select_tiling()                          ← 同 TileLang：完全相同
  │
  ├─②  NPUIndexTritonKernel.__init__(tiling)    ← 初始化 range tree / axis 字段
  │
  ├─③  decide_codegen_dims_in_kernel()          ← 同 TileLang：完全相同
  │       SplitTiling / ReductionAnalysis / no_loop_axis
  │
  ├─④  codegen_node_schedule_with_kernel()      ← 同 TileLang：调用路径相同
  │       对每个 node 调 node._body(*index_vars)：
  │       ├─ NPUIndexTritonKernel.load()
  │       │     ① IndexAnalysis 分析 index（BlockPtr / IndexingOptions）
  │       │     ② emit: tl.load(ptr + offset, mask)  → self.loads buffer
  │       ├─ NPUIndexTritonKernel.store()
  │       │     ① 分析 index
  │       │     ② emit: tl.store(ptr + offset, val, mask) → self.stores buffer
  │       ├─ NPUIndexTritonKernel.reduction()
  │       │     persistent: emit tl.sum/max(val, dim) → self.compute buffer
  │       │     loop:       emit accumulator 更新    → self.compute buffer
  │       │                 emit tl.sum(acc, dim)    → self.post_loop_store buffer
  │       └─ NPUIndexTritonKernel.codegen_body()    ← ★ Triton 独有
  │             生成嵌套循环写入 self.body：
  │             for split_axis in range(offset, min(offset+XBLOCK, numel)):
  │               for tiling_axis in range(loops_tiling):
  │                 x = tl.arange(0, XBLOCK_SUB) + offset
  │                 mask = x < numel
  │                 splice(indexing_code)
  │                 splice(loads)        # tl.load(...)
  │                 splice(compute)      # tl.add / tl.sum / ...
  │                 splice(stores)       # tl.store(...)
  │               splice(post_loop_store) # tl.store(reduce_result, ...)
  │
  ├─⑤  NPUIndexTritonKernel.codegen_kernel()    ← ★ Triton 独有
  │       ① gen_common_triton_imports()          (triton / triton.language / ...)
  │       ② gen_numel_args() → x0_numel, X0BLOCK, X0BLOCK_SUB 等参数
  │       ③ add_autotune_args()                 → constexpr tiling 参数
  │       ④ @triton_heuristics.pointwise/reduction/persistent_reduction(
  │              size_hints=..., triton_meta=..., inductor_meta=...)
  │          @triton.jit
  │          def kernel_fn(in_ptr0, out_ptr0, x0_numel,
  │                        X0BLOCK: tl.constexpr, X0BLOCK_SUB: tl.constexpr):
  │       ⑤ codegen_static_numels()             → persistent reduction 静态常量
  │       ⑥ splice(self.body)                   → ④ 中 codegen_body() 已填好的循环代码
  │
  ├─⑥  NPUTritonScheduling.define_kernel()      ← ★ Triton 独有
  │       kernel_name = "triton_poi/red_<fused>_<suffix>"
  │       async_compile.triton(kernel_name, '''
  │           import triton
  │           import triton.language as tl
  │           @triton_heuristics.pointwise(...)
  │           @triton.jit
  │           def kernel_fn(...): ...
  │       ''', device_str='npu')
  │       → wrapper.define_kernel(kernel_name, compile_wrapper)  → PyCodeCache
  │
  └─⑦  NPUIndexTritonKernel.call_kernel()       ← ★ Triton 独有
          kernel_name.run(in_ptr0, out_ptr0, x0_numel,
                          X0BLOCK=..., grid=grid_fn,
                          stream=stream)
```

生成的 Triton kernel 示例（pointwise add，1D）：

```python
@triton_heuristics.pointwise(
    size_hints=[1024],
    triton_meta={'signature': {'in_ptr0': '*fp32', ...}, 'mix_mode': 'aiv'},
    inductor_meta={'split_axis': [0], 'tiling_axis': [0], ...},
)
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, x0_numel,
                            X0BLOCK: tl.constexpr, X0BLOCK_SUB: tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0 = tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK + x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + x0, x0_mask)
        tmp1 = tl.load(in_ptr1 + x0, x0_mask)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + x0, tmp2, x0_mask)
```

---

## TileLang vs Triton Pipeline 差异

| 维度 | NPU Triton (triton.py) | TileLang (tilelang.py) |
|---|---|---|
| **代码模型** | 标量循环 + tl.arange + mask | DMA（T.copy）+ 向量 intrinsic |
| **load 行为** | emit tl.load → self.loads buffer | 记录 op graph，不 emit 代码 |
| **store 行为** | emit tl.store → self.stores buffer | 记录 _output_vars，不 emit 代码 |
| **codegen_body** | 生成 Triton 嵌套循环结构 | NOP（仅 CSE 失效 + buffer 清理） |
| **codegen_kernel** | 读 self.body 生成 @triton.jit | 从 op graph 重建 @T.prim_func |
| **reduction** | accumulator + tl.sum | T.reduce(2D block, dims=1) |
| **定义注册** | async_compile.triton → PyCodeCache | factory fn + dict cache + tilelang.compile |
| **调用风格** | kernel.run(..., stream=stream) | kernel(tensors, xnumel, rnumel) |
| **shape cache** | 无（静态 kernel） | (xnumel, rnumel,...) key 动态 compile |
| **编译器链** | Triton → NPU IR → bishengir-compile | tilelang.compile → TVM → MLIR → bishengir-compile |

### 完全共享的部分

以下代码/逻辑在两个后端**完全相同**，TileLang 通过继承直接复用，无任何覆写：

| 阶段 | 共享代码位置 | 说明 |
|---|---|---|
| **select_tiling** | `NPUTritonScheduling.select_tiling()` + `candidate_tilings()` | tile shape 选择，含 split/reduction numel 计算 |
| **decide_codegen_dims_in_kernel** | `NPUTritonScheduling._mark_store_index_keys()` 等 5 个方法 | 轴分析全流程 |
| **SplitTiling** | `split_tiling.py` 全部 | split/tiling/no_loop 轴选择 |
| **ReductionAnalysis** | `kernel_analysis.py::ReductionAnalysis` | reduced_dim 计算 |
| **split_and_set_ranges** | `NPUIndexTritonKernel.split_and_set_ranges()` | node var_ranges → range tree 映射 |
| **transform_dims_in_indexing** | `ir.py::transform_dims_in_indexing()` | index 线性化、FloorDiv/Mod 拆分 |
| **rebuild_flattened_dims** | `ir.py::rebuild_flattened_dims()` | flat+2D 混合访问的冗余轴替换 |
| **codegen_node_schedule_with_kernel** | 上游 `TritonScheduling` | node 遍历、调用 load/store/reduction |
| **codegen_node_schedule 主流程** | `NPUTritonScheduling.codegen_node_schedule()` | 7 步主循环（①~③⑤共用，④⑥⑦各自覆写）|
| **call_kernel 签名** | `TileLangKernel.active_range_trees()` | numel 参数列表生成 |

### 两边各自覆写的部分

| 阶段 | NPU Triton | TileLang |
|---|---|---|
| `load()` | emit `tl.load` → `self.loads` | 记录 `_tl_inputs` / `_var_bufs` |
| `store()` | emit `tl.store` → `self.stores` | 记录 `_output_vars` |
| `reduction()` | emit `tl.sum/max` → `self.compute` | 记录 `_var_ops[var] = ("reduce",...)` |
| `store_reduction()` | emit → `self.post_loop_store` | 记录 `_is_reduction_output` |
| `codegen_body()` | 生成嵌套循环写入 `self.body` | NOP（CSE 失效 + buffer 清理） |
| `codegen_kernel()` | 读 `self.body` 输出 `@triton.jit` | 从 op graph 重建 `@T.prim_func` |
| `define_kernel()` | `async_compile.triton(...)` → PyCodeCache | factory fn + shape dict cache |
| `call_kernel()` | `kernel_name.run(..., stream=stream)` | `kernel_name(tensors, xnumel, rnumel)` |

---

## 已知限制

| 限制 | 说明 |
|---|---|
| Reduction 仅支持 persistent | `rnumel` 必须能放入 SRAM（非 persistent 抛 `NotImplementedError` → Triton fallback）|
| Reduction M 需整除 _ROW_BLOCK | `xnumel % 32 != 0` 时尾部行会越界（暂无 tail guard）|
| 非 persistent reduction | 未实现，fallback Triton |
| argmax / argmin | 未实现，fallback Triton |
| welford reduction | 未实现，fallback Triton |
| atomic_add | 未实现，fallback Triton |
| 间接 indexing | 未测试 |
| 动态 shape | 部分支持（工厂函数按 shape 缓存）|

---

## 文件结构

```
torch_npu/_inductor/
├── __init__.py                     后端注册（TORCHINDUCTOR_NPU_BACKEND=tilelang）
└── codegen/
    ├── tilelang.py                 ★ TileLang 后端主文件
    │   ├── TileLangKernel          kernel 代码生成（op graph → @T.prim_func）
    │   ├── TileLangScheduling      scheduling + define_kernel + factory wrapper
    │   ├── TileLangOverrides       inductor op → TileLang 表达式字符串映射
    │   ├── _BINARY_VEC_OPS         二元向量 op 映射表
    │   ├── _UNARY_VEC_OPS          一元向量 op 映射表
    │   ├── _REDUCE_OPS             reduction mode 映射表
    │   └── _REDUCTION_ROW_BLOCK    每 AI Core 处理行数（默认 32）
    ├── triton.py                   NPU Triton 后端（含 NPUIndexTritonKernel）
    ├── scheduling.py               NPUTritonScheduling + decide_codegen_dims_in_kernel
    ├── split_tiling.py             SplitTiling：select_split/tiling/no_loop_axis
    └── kernel_analysis.py          ReductionAnalysis：确定 reduced_dim

examples/tilelang/
└── demo_ops.py                     覆盖所有 op 的测试 demo
```

---

## 调试

`TileLangKernel.codegen_kernel()` 和 `define_kernel()` 会把生成的 `@T.prim_func` 源码打印到 stdout（`<<< src_code`）。如需关闭，删除 `tilelang.py` 中的 `print()` 调用。

TileLang 编译后会打印 TVM IR / NPUIR / final NPUIR，由 `tilelang.compile` 的 verbose 级别控制。



## `decide_codegen_dims_in_kernel` 全流程解析

**总目标**：在正式 codegen 之前，把 node schedule 的 index 表达式和 kernel 的 range tree 对齐，并给每个 axis 贴标签（split/tiling/no-loop），决定循环结构。

------

### 1. `_mark_store_index_keys`

**作用**：收集所有 `STORE` / `STORE_REDUCTION` 操作的 index key，存入 `kernel.store_index_keys`。

**为什么需要**：在后续步骤中，axis 替换（父轴替换为子轴组合）可能会改变 index 表达式。但 Store 侧的 index 承担"写回地址"的 unified anchor 角色，如果被替换掉，会导致多个节点的写回地址发生偏移。提前标记 store key 就是为了在后续替换时跳过这些 index，保护写回地址不被破坏。

**实现**：遍历每个 node，从 `_body.memory_usage[STORE/STORE_REDUCTION]` 拿到 index_name，再从 `_body.indexing_exprs` 里找对应的 key，加入 `store_index_keys` set。

------

### 2. `_transform_schedule_indexing`

**作用**：把每个 node 的抽象循环变量（`var_ranges`）映射到 kernel 的 range tree 符号上，生成具体的 `body.indexing` 字典，并做冗余轴替换。

**三个子步骤**：

#### 2a. `split_and_set_ranges(node.get_ranges())`

**作用**：node 的 `var_ranges` 是抽象的循环区间（如 `{z0:512, z1:4, z2:8}`），kernel 的 range tree 是按 `xnumel`/`rnumel` 组织的。这个函数将两者对齐：把 node 的多段 sizes 分配到 kernel 的 range tree 里，创建 `IterationRangesEntry` 子节点（`x0`, `x1`, `r0`...），并返回 index_vars（sympy symbols）。

**实现关键**：调用 `map_kernel_groups_to_node_sizes` → `set_ranges`。对于 `xnumel=1024, node_sizes=[32,4,8]`，创建：



```
x0: divisor=32, length=32  → x0 = xindex // 32
x1: divisor=8,  length=4   → x1 = (xindex // 8) % 4
x2: divisor=1,  length=8   → x2 = xindex % 8
```

#### 2b. `transform_dims_in_indexing(index_vars)`

**作用**：用 index_vars 替换 node `_body.indexing_exprs` 里的原始循环变量，生成以 kernel axis 符号表达的 `_body.indexing`。同时做两件事：

1.  `remove_zero_terms`：消除 index 表达式里恒为 0 的项
2.  `analyze_expression` + `split_expression`：把 flat index（如 `x0*32 + x1*8 + x2`）分析其中的 FloorDiv/ModularIndexing，把它们拆成独立项，便于后续 axis 替换识别
3.  `rebuild_flattened_dims`（NPU 独有）：检测 index 里出现的 FloorDiv/Mod 对（如 `xindex//32` 和 `xindex%32`），如果在同一个父轴上，就把父轴拆成两个子轴，替换掉 flat 表达式，消除昂贵的 div/mod 运算

#### 2c. `substituted_dims_in_indexing`

**作用**：把所有 `range_tree_nodes_substituted` 中记录的"可被子轴代替的父轴"从 index 表达式里替换掉。

**触发条件**（`additional_nodes_to_be_subs` 中判断）：当一个父轴的 length 等于所有兄弟子轴 length 的乘积时，该父轴可以被这些子轴的线性组合代替。例如 `x0.length=32`, siblings `{x1:4, x2:8}`, `4×8=32` → `x0` 被替换为 `x1*8 + x2`。

**Store 保护**：若某个父轴在 Store index 里作为 unified anchor（只有该一个 prefix 符号），跳过替换，保留父轴，防止写回地址错乱（`_should_skip_store_substitution` 判断）。

------

### 3. `_record_store_unified_indexing`

**作用**：把所有 Store 侧的 index 表达式（已经过步骤 2 变换）收集到 `kernel.store_unified_indexing`。

**为什么在步骤 2 之后**：此时 `_body.indexing` 已经用 kernel axis 符号表达，收集到的 unified indexing 才是最终的写回地址形式，后续步骤用它来决定哪些被替换轴需要保留。

------

### 4. `_remove_substituted_dims_from_kernel`

**作用**：遍历所有被替换的父轴（`range_tree_nodes_substituted`），把不再需要的父轴从 range tree 里删除，减少不必要的循环变量。

**保留判断（`_should_preserve_substituted_var`）**：如果该父轴在某个 Store index 里是 unified anchor（index 的 free_symbols 里 same-prefix 的只有它一个），则保留——因为删掉它会导致 Store 无法正确索引。

**效果**：减少 range tree 里的轴数，从而减少 codegen_body 生成的循环层数。

------

### 5. `_finalize_kernel_codegen_dims`

综合运行 SplitTiling 和 ReductionAnalysis，给每个 axis 打标签。

#### 5a. `SplitTiling.__init__` + `find_lowest_dimension`

**作用**：构建 `kernel.sorted_axis`（按 stride 排序的 axis 列表），找出"低维轴"（`low_dims`）—— 在所有 load/store index 表达式中 stride=1 的轴。

**实现**：从 node_schedule 里所有 LOAD 对应的 `body.indexing` 中提取 index 表达式，调用 `as_coefficients_dict()` 取每个 axis 符号的系数，系数=1 的是 stride=1 的低维轴，系数>1 的是高维轴。`low_dims = stride1_axes - high_dims`（同时出现在高维的不算低维）。

------

#### 5b. `select_split_axis`

**作用**：选出"分配给 AI Core 的轴"（split axes），用于 kernel launch 的 grid 维度。

**原则**：累积 split_axis 的 numel 乘积，直到 >= `num_vector_core`（AI Core 数量），最多 3 个。

**优先级**（`select_one_split_axis` 内）：

1.  优先选非规约、非低维轴（大尺度的高维轴，分配均匀）
2.  若不够再选低维轴
3.  跳过 cat 节点的最后轴（防止 cat 的特殊 index 问题）

**效果**：split_axis 对应 `for x0 in range(x0_offset, x0_offset+X0BLOCK):` 这种 grid 级循环，每个 AI Core 负责其中一个 block。

------

#### 5c. `select_tiling_axis`

**作用**：选出"在 SRAM 内 tile 的轴"（tiling axes），对应 `tl.arange` 向量化的维度。

**四条原则**（注释直接写在代码里）：

1.  load/store index 里 stride=1 的低维轴 → 必须成为 tiling 轴（连续访问，向量效率最高）
2.  规约轴（`prefix == 'r'`） → 必须成为 tiling 轴（规约运算必须在 SRAM 里做）
3.  多维规约时，只有规约轴可以被选为 tiling 轴
4.  tiling 覆盖 total numel 80% 才停止

**停止条件**（`can_stop`）：当有多个规约轴且所有规约轴都已是 tiling 轴，且不是 contiguous_reduction 时停止。

------

#### 5d. `select_no_loop_axis`

**作用**：在 tiling 轴中，找出不需要显式 for 循环的轴（一次性全部放进 SRAM 的轴）。

**原则**（注释写在代码里）：

1.  优先从 low_dims tiling 轴中选
2.  未超过阈值，再从其他 tiling 轴选
3.  所有轴预估空间 ≤ 4KB → 无需循环
4.  有动态 shape 的轴不做此优化

**实现**：从 low_dims 开始，累积 `numel × dtype_bytes`，不超过 4KB 则设 `is_no_loop_axis = True`。对 persistent_reduction 轴，先把所有 r 轴的 numel 乘入基础累积量（持久规约已整体放进 SRAM）。

**效果**：`is_no_loop_axis = True` 的 tiling 轴，在 `codegen_body` 里直接生成 `tl.arange(0, SIZE)`，不生成 `for loop_x in range(loops_x):` 循环。

------

#### 5e. `ReductionAnalysis`

**作用**：确定规约操作的 `reduced_dim`——即在 tiling tensor 里，规约轴处于第几维（从右往左数）。Triton 的 `tl.sum(val, dim)` 和 TileLang 的 `T.reduce(..., dims=1)` 都需要这个信息。

**实现**：

1.  调用 `find_reduction_node()` 找到 schedule 中的 `ir.Reduction` 节点
2.  调用 `select_golden_varlist()` 确定 tiling 轴的排列顺序（golden_var_list），这个顺序来自 load/store index 的 stride 排序
3.  `analyze_reduction_dim()`：从 `golden_var_list` 的末尾往前找第一个 `r` 前缀的 axis，其位置就是 `reduced_dim`

**特殊处理**：

-   多个规约轴（multi-reduction）：`contiguous_reduction` 为 True 时取连续规约轴的最后一个位置；False 时固定为 0
-   `is_higher_order_reduction`：若 `reduced_dim < len(tiling_axis) - 1`，说明规约轴不在最后一维（高阶规约），此时对 SIMT kernel 需要关闭 `persistent_reduction`

------

### 整体数据流总结



```
node.var_ranges
    │ split_and_set_ranges
    ▼
range_tree_nodes (x0, x1, r0...)
    │ transform_dims_in_indexing
    ▼
body.indexing (sympy expr in kernel symbols)
    │ rebuild_flattened_dims
    ▼
range_tree_nodes_substituted (可被替换的父轴)
    │ substituted_dims_in_indexing
    ▼
body.indexing (冗余父轴已被子轴替换)
    │ _remove_substituted_dims_from_kernel
    ▼
range_tree_nodes (精简后的 axis 集合)
    │ SplitTiling
    ▼
sorted_axis, low_dims
split_axis    → AI Core grid 维度
tiling_axis   → SRAM 内向量化维度
no_loop_axis  → 整块 arange，无 for 循环
    │ ReductionAnalysis
    ▼
reduced_dim   → tl.sum(val, dim) / T.reduce(..., dims=)
```
