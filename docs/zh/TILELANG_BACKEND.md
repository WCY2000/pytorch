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
│           └── TileLangKernel       (tilelang.py)  ← TileLang pointwise/reduction kernel
│
└── SIMDScheduling
    └── TritonScheduling
        └── NPUTritonScheduling      (scheduling.py) ← NPU Triton scheduler
            └── TileLangScheduling   (tilelang.py)   ← TileLang scheduler
                                        ├── codegen_template()  ← T.gemm matmul 路径
                                        ├── define_kernel_matmul()
                                        └── define_kernel()     ← pointwise/reduction 路径
```

`TileLangKernel` 继承 `NPUIndexTritonKernel` 的目的是复用其 range tree 基础设施和 axis 分析管道（`decide_codegen_dims_in_kernel`），同时通过完整覆写 `load/store/reduction/codegen_kernel` 来生成 TileLang 代码而非 Triton 代码。

`TileLangScheduling` 继承 `NPUTritonScheduling` 以复用完整的 scheduling 流程（包括 SplitTiling、ReductionAnalysis），覆写 `__init__`（强制 `self.kernel_type = TileLangKernel`）、`define_kernel`、`codegen_sync`，以及新增的 `codegen_template()`（T.gemm matmul 路径）和 `define_kernel_matmul()`。

### 两条 Codegen 路径

TileLang 后端有两条完全独立的 codegen 路径：

| 路径 | 适用 Op | 触发条件 | 调度节点类型 |
|---|---|---|---|
| **Pointwise / Reduction** | 逐元素、归约 | 所有 `ComputedBuffer` | `SchedulerNode` / `FusedSchedulerNode` |
| **T.gemm Matmul** | `aten.mm` / `aten.addmm` / `aten.bmm` | 行优先 fp16/int8 输入 | `TemplateSchedulerNode` |

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
  │    │   ├─ T.copy(gm[offset], local)           GM → L1
  │    │   ├─ T.vadd/vsub/vexp/...               向量 intrinsic
  │    │   └─ T.copy(local, gm[offset])           L1 → GM
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

## Matmul (T.gemm) Pipeline

`aten.mm` 走与 pointwise/reduction **完全不同**的路径，产出 `TemplateBuffer` 而非 `ComputedBuffer`。

```
torch.mm(A, B)  /  torch.nn.Linear(bias=False)
        │
        ▼ _register_npu_inductor_mm() 注册的 lowering
tuned_mm(mat1, mat2)   [kernel/mm.py]
        │
        ├─ is_row_major_striding(mat1) AND is_row_major_striding(mat2)?
        │       否（如 weight.T 列优先）→ fallback: aten_mm / CATLASS
        │
        ├─ use_tilelang_template() == True?
        │       TORCHINDUCTOR_NPU_BACKEND=tilelang AND dtype ∈ {fp16, int8}
        │
        └─ add_tilelang_gemm_choices(choices=[TileLangGemmCaller])
                │  override layout: FixedLayout(dtype=fp32)  ← 防 fp16 buffer 误解 fp32 数据
                │
                ▼ autotune_select_algorithm(choices=[TileLangGemmCaller])
                │       单 choice → choices[0].output_node()（不 benchmark）
                │
                ▼ TileLangGemmCaller.output_node()
                        TemplateBuffer(
                            layout = FixedLayout(fp32, size=[M,N], stride=[N,1]),
                            inputs = [mat1, mat2],
                            make_kernel_render = _TileLangGemmRender(params)
                        )
        │
        ▼ 调度阶段：scheduler 看到 is_template() == True
Scheduler.codegen() → TileLangScheduling.codegen_template(template_node, epilogue_nodes)
        │
        ├─ isinstance(render, _TileLangGemmRender)?
        │       否 → _triton_scheduling.codegen_template()
        │
        ├─ codegen_tilelang_mm_src(kernel_name, ...)   → @T.prim_func 源码字符串
        │
        ├─ define_kernel_matmul(src_code, ...)          → 注入 factory wrapper 到 header
        │
        ├─ n.mark_run() for n in [template_node, *epilogue_nodes]
        │
        └─ wrapper.writeline(
               "tilelang_mm_0(mat1, mat2, out, M, N, K)"
           )
```

### 行优先约束

`T.copy(B[k * _block_K, bx * _block_N], B_shared)` 假设 B 是行优先布局。  
`nn.Linear` 的 `weight.T` 是列优先（stride `[1, K]`），Ascend DMA 用列优先 strides 读取 tile 会产生错误结果。  
因此 TileLang 路径只在两个输入都是行优先时才启用，列优先输入 fallback 到 `aten_mm` / CATLASS。

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
define_kernel(src, sched, k, h)    Pointwise/Reduction factory wrapper（见 Pipeline ⑥）
codegen_sync()                     torch.npu.synchronize()
── T.gemm 专用（新增）──────────────────────────────────────────────
codegen_template(node, epilogue)   检测 _TileLangGemmRender → 生成 T.gemm 代码
                                   否则 → _triton_scheduling.codegen_template()
define_kernel_matmul(src, sched)   生成 factory(M,N,K) + shape dict cache + wrapper
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
        _in_ptr0_local  = T.alloc_shared((_ROW_BLOCK, _rnumel), 'float32')  # cbuf
        _out_ptr0_local = T.alloc_shared((_ROW_BLOCK, 1),       'float32')  # cbuf

        T.copy(in_ptr0[cid * _ROW_BLOCK, 0], _in_ptr0_local,
               size=[_ROW_BLOCK, _rnumel])               # GM → cbuf: nd2nz

        T.reduce(_in_ptr0_local, _out_ptr0_local,
                 dims=1, reduce_mode='sum', clear=True)   # 批量规约（32 行）

        T.copy(_out_ptr0_local, out_ptr0[cid * _ROW_BLOCK, 0],
               size=[_ROW_BLOCK, 1])                      # cbuf → GM: 128B DMA
```

### dtype 提升

Inductor 对所有 reduction 调用 `upcast_acc_dtype()` 将 fp16 → fp32 accumulation。
即使输入 tensor 是 fp16，TileLang reduction kernel 的 `in_ptr0` 也是 fp32（inductor 在 kernel 外插入了 cast）。Pointwise kernel 不受此影响，保持原始 dtype。

---

## Matmul Kernel 生成（T.gemm）

### 生成的 @T.prim_func

`codegen_tilelang_mm_src()` 生成以下结构（形状由工厂函数的 `_M/_N/_K` 变量注入）：

```python
import tilelang.language as T

_block_M = 128   # M 方向 tile 大小
_block_N = 128   # N 方向 tile 大小
_block_K = 64    # K 方向 tile 大小（流水线方向）

@T.prim_func
def tilelang_mm_0_prim_fn(
    A: T.Tensor((_M, _K), 'float16'),   # 输入 mat1
    B: T.Tensor((_K, _N), 'float16'),   # 输入 mat2
    C: T.Tensor((_M, _N), 'float32'),   # 输出（fp32 累加结果）
):
    # grid = M_tiles × N_tiles，每个 AIC 负责一个 (block_M × block_N) 子块
    with T.Kernel(T.ceildiv(_N, _block_N) * T.ceildiv(_M, _block_M),
                  is_npu=True) as (cid, _):
        by = cid // T.ceildiv(_N, _block_N)   # M tile 索引
        bx = cid % T.ceildiv(_N, _block_N)    # N tile 索引

        # L1 缓冲（shared.dyn，FFTS 管理）
        A_shared = T.alloc_shared((_block_M, _block_K), 'float16')
        B_shared = T.alloc_shared((_block_K, _block_N), 'float16')
        # 片上 cube 累加器（local.fragment）
        C_local  = T.alloc_fragment((_block_M, _block_N), 'float32')

        # K 方向双缓冲流水（隐藏 DMA 延迟）
        for k in T.Pipelined(T.ceildiv(_K, _block_K), num_stages=2):
            T.copy(A[by * _block_M, k * _block_K], A_shared)   # GM → L1
            T.copy(B[k * _block_K, bx * _block_N], B_shared)   # GM → L1
            T.gemm(A_shared, B_shared, C_local, initC=(k == 0)) # AIC cube 指令

        T.copy(C_local, C[by * _block_M, bx * _block_N])        # L1 → GM
```

### 分块访问映射

| 变量 | 含义 | 计算方式 |
|---|---|---|
| `cid` | AI Core block 编号（`blockIdx.x`） | TileLang runtime |
| `by` | C 矩阵 M 方向 tile 索引 | `cid // N_tiles` |
| `bx` | C 矩阵 N 方向 tile 索引 | `cid % N_tiles` |
| `A_shared` | A 的 L1 片段（`shared.dyn`） | `T.alloc_shared` |
| `B_shared` | B 的 L1 片段（`shared.dyn`） | `T.alloc_shared` |
| `C_local` | 累加器（`local.fragment`） | `T.alloc_fragment` |
| `initC=(k==0)` | k=0 时清零，否则累加 | 条件表达式 |

### define_kernel_matmul — 生成的 wrapper

```python
# 工厂函数：运行时接受 M/N/K，构造 prim_func
def _prim_factory_tilelang_mm_0(M, N, K):
    _M, _N, _K = M, N, K
    import tilelang.language as T
    @T.prim_func
    def tilelang_mm_0_prim_fn(A, B, C): ...   # 同上
    return tilelang_mm_0_prim_fn

# shape-keyed 缓存：相同 (M,N,K) 复用已编译 binary
_tilelang_mm_0_cache = {}

def tilelang_mm_0(A, B, C, M, N, K):
    _key = (int(M), int(N), int(K))
    if _key not in _tilelang_mm_0_cache:
        _tilelang_mm_0_cache[_key] = _tilelang_0.compile(
            _prim_factory_tilelang_mm_0(*_key), target='npuir'
        )
    _tilelang_mm_0_cache[_key](A, B, C)
```

output_code.py 中对应的调用：

```python
tilelang_mm_0(arg1_1, arg0_1, buf0, 256, 512, 128)
```

### Dtype 处理

T.gemm 在 Ascend AIC cube 上使用 fp16 输入、fp32 片上累加，因此输出 buffer 必须是 fp32：

| 阶段 | dtype | 说明 |
|---|---|---|
| A、B（输入） | fp16 | 原始用户数据 |
| C_local（累加器） | fp32 | 防精度损失 |
| C（输出 buffer） | fp32 | inductor 分配 FixedLayout(fp32) |

`add_tilelang_gemm_choices()` 将 inductor 默认的 fp16 output layout 替换为 `FixedLayout(dtype=fp32, stride=[N,1])`。**必须用 `FixedLayout` 而非 `FlexibleLayout`**：当 mm 后有融合 epilogue（relu/sigmoid 等）时，scheduler 在 `SchedulerNode._compute_attrs()` 调 `make_indexer()`，而 `FlexibleLayout.allow_indexing = False` 会触发断言。

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

---

## 已知限制

### Pointwise / Reduction

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

### Matmul (T.gemm)

| 限制 | 说明 |
|---|---|
| 仅支持行优先输入 | 列优先（如 `weight.T`）fallback 到 aten_mm / CATLASS |
| 输出 dtype 固定 fp32 | fp16 输入 → fp32 输出（比 aten.mm 的 fp16 输出精度更高但类型不同） |
| 仅支持 fp16 / int8 输入 | bfloat16、fp32 等 fallback |
| block_M/N/K 固定 | 128/128/64，不支持 autotuning |
| `nn.Linear(bias=True)` | `aten.addmm` 需单独注册（已注册但未实现 T.gemm bias 融合）|
| 无 benchmark 支持 | `TileLangGemmCaller.benchmark()` 返回 inf，不参与多候选竞争 |

---

## 文件结构

```
torch_npu/_inductor/
├── __init__.py                     后端注册 + mm/addmm/bmm lowering 注册
├── kernel/
│   └── mm.py                       tuned_mm() + is_row_major_striding()
│                                   TileLang 路径优先，列优先 fallback CATLASS/aten
└── codegen/
    ├── tilelang.py                 ★ TileLang 后端主文件
    │   ├── TileLangKernel          Pointwise/Reduction kernel（op graph → @T.prim_func）
    │   ├── TileLangScheduling      scheduling + define_kernel + codegen_template
    │   ├── TileLangOverrides       inductor op → TileLang 表达式字符串映射
    │   ├── _BINARY_VEC_OPS / _UNARY_VEC_OPS   向量 op 映射表
    │   ├── TileLangGemmCaller      ChoiceCaller（matmul 候选，不参与 benchmark）
    │   ├── _TileLangGemmRender     TemplateBuffer.make_kernel_render 标记类
    │   ├── add_tilelang_gemm_choices()  注册 T.gemm 为 mm 候选
    │   ├── codegen_tilelang_mm_src()    生成 @T.prim_func 源码
    │   └── define_kernel_matmul()       生成 factory(M,N,K) + shape cache wrapper
    ├── triton.py                   NPU Triton 后端（含 NPUIndexTritonKernel）
    ├── scheduling.py               NPUTritonScheduling + decide_codegen_dims_in_kernel
    ├── split_tiling.py             SplitTiling：select_split/tiling/no_loop_axis
    └── kernel_analysis.py          ReductionAnalysis：确定 reduced_dim

examples/tilelang/
├── demo_ops.py                     Pointwise / Reduction / Matmul 全量测试
└── demo_gemm.py                    T.gemm 独立测试（直接 compile + inductor 两种模式）
```

---

## 调试

### Pointwise / Reduction

`TileLangKernel.codegen_kernel()` 会在生成 `@T.prim_func` 时打印：

```
====== TileLang prim_func ======
<源码>
====== TileLang reduction prim_func ======
<源码>
```

### Matmul (T.gemm)

`codegen_tilelang_mm_src()` 和 `define_kernel_matmul()` 分别打印：

```
====== TileLang T.gemm prim_func ======
<@T.prim_func 源码>

====== TileLang T.gemm wrapper ======
<factory + cache + wrapper 源码>
```

如需关闭，删除 `tilelang.py` 中对应的 `print()` 调用。

TileLang 编译后会打印 TVM IR / NPUIR / final NPUIR，由 `tilelang.compile` 的 verbose 级别控制。
