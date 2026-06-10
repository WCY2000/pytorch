# TileLang Codegen — 调用链与 Triton 对比

## 激活方式

```bash
TORCHINDUCTOR_NPU_BACKEND=tilelang python your_script.py
# rm -rf /tmp/torchinductor_root/ && python demo.py
```

---

## 文件归属一览

| 文件 | 归属 | 角色 |
|------|------|------|
| `torch/_dynamo/eval_frame.py` | **torch** | FX Graph 捕获入口 |
| `torch/_inductor/compile_fx.py` | **torch** | inductor 总入口 `compile_fx()` |
| `torch/_inductor/graph.py` | **torch** | `InductorGraph.codegen()` |
| `torch/_inductor/scheduler.py` | **torch** | `Scheduler.codegen()` / `create_backend()` / 查注册表 |
| `torch/_inductor/loop_body.py` | **torch** | `LoopBody` / `InterpreterShim`，触发 load/store |
| `torch/_inductor/sizevars.py` | **torch** | sympy index 化简 |
| `torch/_inductor/codegen/common.py` | **torch** | 注册表 `device_codegens`、`CSEProxy`、`CSE`、`OpOverrides` |
| `torch/_inductor/codegen/simd.py` | **torch** | `SIMDKernel` / `SIMDScheduling` 基类 |
| `torch_npu/_inductor/__init__.py` | **torch_npu** | 启动时向注册表写入 `TileLangScheduling` |
| `torch_npu/_inductor/codegen/tilelang.py` | **torch_npu** | `TileLangKernel` / `TileLangScheduling` 核心实现 |
| `torch_npu/_inductor/codegen/scheduling.py` | **torch_npu** | `NPUTritonScheduling` + `decide_codegen_dims_in_kernel` |
| `torch_npu/_inductor/codegen/split_tiling.py` | **torch_npu** | `SplitTiling`：split/tiling/no_loop 轴选择 |
| `torch_npu/_inductor/codegen/kernel_analysis.py` | **torch_npu** | `ReductionAnalysis`：确定 reduced_dim |
| `torch_npu/_inductor/codegen/wrapper.py` | **torch_npu** | `NPUWrapperCodeGen`，生成 Python wrapper |
| `torch_npu/_inductor/utils.py` | **torch_npu** | `patch_has_triton()` 等运行时 patch |
| `torch_npu/_inductor/lowering.py` | **torch_npu** | NPU op lowering / fallback 注册 |

---

## 继承层次

```
上游 PyTorch Inductor
├── SIMDKernel
│   └── TritonKernel
│       └── NPUIndexTritonKernel   (triton.py)     ← NPU Triton kernel
│           └── TileLangKernel     (tilelang.py)   ← TileLang kernel ★
│
└── SIMDScheduling
    └── TritonScheduling
        └── NPUTritonScheduling    (scheduling.py)  ← NPU Triton scheduler
            └── TileLangScheduling (tilelang.py)    ← TileLang scheduler ★
```

> `TileLangKernel` 继承 `NPUIndexTritonKernel`（而非旧版的 `SIMDKernel`），从而复用完整的
> range tree 基础设施和 axis 分析管道（`decide_codegen_dims_in_kernel`）。
> `TileLangScheduling` 继承 `NPUTritonScheduling`（而非旧版的 `SIMDScheduling`），
> 同样复用完整的 scheduling 流程，仅覆写 `define_kernel`、`codegen_sync` 和 `__init__`。

---

## 完整调用链（含 torch / torch_npu 标注）

### 阶段 0 — 模块导入时注册后端

```
import torch_npu
  │
  ▼ [torch_npu] torch_npu/_inductor/__init__.py
      if TORCHINDUCTOR_NPU_BACKEND == 'tilelang':
        from .codegen.tilelang import TileLangScheduling
        register_backend_for_device('npu', TileLangScheduling,
                                    NPUWrapperCodeGen, CppWrapperNpu)
        patch_has_triton()   # 让 has_triton() 对 NPU 返回 True
        patch_is_gpu()
        ...
          │
          ▼ [torch] torch/_inductor/codegen/common.py
              device_codegens['npu'] = DeviceCodegen(TileLangScheduling, ...)
              # 全局注册表，供 Scheduler.create_backend() 查询
```

### 阶段 1 — torch.compile 触发编译

```
torch.compile(model)(inputs)
  │
  ▼ [torch] torch/_dynamo/eval_frame.py
      dynamo 追踪，捕获 FX Graph
  │
  ▼ [torch] torch/_inductor/compile_fx.py     ← inductor 入口
      compile_fx(gm, example_inputs)
        └─ _compile_fx_inner()
             └─ fx_codegen_and_compile()
  │
  ▼ [torch] torch/_inductor/graph.py
      InductorGraph.codegen()
        └─ self.scheduler.codegen()
  │
  ▼ [torch] torch/_inductor/scheduler.py
      Scheduler.codegen()
        for node in self.nodes:
          device = node.get_device()           # device.type = 'npu'
          self.get_backend(device).codegen_nodes(node)
              │
              └─ create_backend(device)        ← 首次调用时实例化
                   device_scheduling = get_scheduling_for_device('npu')
                   # → 从 device_codegens['npu'] 取出 TileLangScheduling
                   return device_scheduling(self)   # TileLangScheduling(scheduler)
  │
  ▼ [torch_npu] torch_npu/_inductor/codegen/tilelang.py  ← 分叉点
      TileLangScheduling.codegen_nodes(nodes)
      （继承自 [torch_npu] NPUTritonScheduling）
```

### 阶段 2 — Kernel 构造、Axis 分析与 load/store 收集

```
[torch_npu] NPUTritonScheduling.codegen_node_schedule(kernel_features, nodes)
  │         （TileLangScheduling 完全继承此方法）
  │
  ├─ select_tiling(numel, rnumel, nodes)
  │     output: tiling = {'x': sympy.Expr, ...}
  │
  ├─ [torch_npu] TileLangKernel(tiling, features=...)   ← kernel_type = TileLangKernel
  │     初始化（继承 NPUIndexTritonKernel.__init__）：
  │       self.range_trees         = [...]       # range tree 基础设施
  │       self.split_axis          = []
  │       self.tiling_axis         = []
  │       self.sorted_axis         = []
  │       self.cse                 = CSE(...)    # 覆写为普通 CSE
  │       self._tl_inputs          = {}
  │       self._tl_outputs         = {}
  │       self._tl_input_indices   = {}          # sympy index 记录
  │       self._tl_output_indices  = {}
  │       self._pending_op         = None
  │       self._var_ops            = {}
  │       self._var_bufs           = {}
  │       self._var_consts         = {}
  │       self._output_vars        = {}
  │       self._is_reduction_output = {}
  │
  ├─ [torch_npu] decide_codegen_dims_in_kernel(node_schedule, kernel)
  │   ★ TileLang 与 Triton 共用同一套轴分析管道
  │   ├─ _mark_store_index_keys()            记录 Store index key
  │   ├─ _transform_schedule_indexing()
  │   │   ├─ split_and_set_ranges()          var_ranges → range tree
  │   │   ├─ transform_dims_in_indexing()    线性化 index 表达式
  │   │   └─ substituted_dims_in_indexing()  冗余父轴替换
  │   ├─ _record_store_unified_indexing()    收集 Store index
  │   ├─ _remove_substituted_dims_from_kernel()
  │   └─ _finalize_kernel_codegen_dims()
  │       ├─ SplitTiling.select_split_tiling_axis()
  │       │   ├─ find_lowest_dimension()     stride=1 低维轴
  │       │   ├─ select_split_axis()         分配给 AI Core（≤3个）
  │       │   └─ select_tiling_axis()        SRAM 内 tile 的轴
  │       ├─ ReductionAnalysis()             确定 reduced_dim
  │       └─ select_no_loop_axis()           ≤4KB 的 tiling 轴 → 无 for 循环
  │
  └─ [torch] codegen_node_schedule_with_kernel(node_schedule, kernel)
        for node in node_schedule:
          node.codegen(index_vars)
              │
              ▼ [torch] scheduler.py: SchedulerNode.codegen(index_vars)
                  self._body(*index_vars)    ← _body 是 LoopBody 对象
                      │
                      ▼ [torch] loop_body.py: LoopBody.__call__()
                          InterpreterShim(fx_graph).run(V.get_ops_handler())
                              │
                              ▼ [torch] CSEProxy.load(name, index)
                                  self.kernel.load(name, index)
                                    │
                                    ▼ [torch_npu] TileLangKernel.load()
                                    ▼ [torch_npu] TileLangKernel.store()
                                    ▼ [torch_npu] TileLangKernel.reduction()
                          └─ [torch_npu] TileLangKernel.codegen_body()  ← NOP
                               CSE 失效 + buffer 清理，不写 Triton 循环代码
```

### 阶段 3 — load() / store() / reduction() 内部：op graph 追踪

```
[torch_npu] TileLangKernel.load(name, index):
  ① _pending_op = None                # 清空，防止污染 load var
  ② 记录 _tl_inputs[name] = (ptr_var, local_name, dtype)
  ③ 记录 _tl_input_indices[name] = index  (sympy Expr)
  ④ cse.generate(loads_buf, "_in_ptr0_local[_tl_i]")
       → create_cse_var("tmp0")
       → _var_bufs["tmp0"] = "_in_ptr0_local"
  output: CSEVariable "tmp0"

[torch_npu] TileLangOverrides.constant(2.0, float32):
  ① literal = "2.0"
  ② _set_pending("const", ["2.0"])
  ③ cse.generate(...) → create_cse_var("tmp1")
       → _var_consts["tmp1"] = "2.0"

[torch_npu] TileLangOverrides.mul(tmp0, tmp1):
  ① _set_pending("mul", [tmp0_var, tmp1_var])
  ② return "(tmp0 * tmp1)"
  ③ cse.generate(...) → create_cse_var("tmp2")
       → _var_ops["tmp2"] = ("mul", [tmp0_var, tmp1_var])

[torch_npu] TileLangKernel.store(name, index, tmp4_var):
  ① 记录 _tl_outputs[name] = (ptr_var, local_name, dtype)
  ② 记录 _tl_output_indices[name] = index  (sympy Expr)
  ③ _output_vars["_out_ptr0_local"] = (tmp4_var, dtype)

[torch_npu] TileLangKernel.reduction(dtype, src_dtype, reduction_type, value):
  ① rt = str(reduction_type).split(".")[-1].lower()  # e.g. "sum"
  ② result_var = cse.newvar(dtype=upcast_acc_dtype(src_dtype))  # fp16→fp32
  ③ _var_ops[result_var.name] = ("reduce", rt, str(value))
  output: CSEVariable result_var

[torch_npu] TileLangKernel.store_reduction(name, index, value):
  ① 记录 _tl_outputs[name] 和 _tl_output_indices[name]
  ② _output_vars[local_name] = (value, dtype)
  ③ _is_reduction_output[local_name] = True
```

### 阶段 4 — codegen_kernel()：生成 @T.prim_func

#### Pointwise（1-D / N-D 连续）

```
[torch_npu] TileLangKernel._codegen_pointwise_kernel():
  ① 读 split_axis / tiling_axis 确定 grid 和 T.copy offset
     1-D: grid = T.ceildiv(_xnumel, _XBLOCK), offset = cid * _XBLOCK
     2-D: grid = (M, T.ceildiv(N, _XBLOCK)), offset 由 _build_copy_offset() 计算
  ② T.alloc_shared(_XBLOCK) × 每个输入输出          # cbuf (UB)
  ③ T.copy(in_ptr[offset], local_buf)              # GM → cbuf
  ④ _build_vec_ops(result_var, out_loc, ops_list)
     递归遍历 op graph：
       var in _var_bufs   → local_buf_name
       var in _var_consts → 2.0 (float)
       var in _var_ops    → 递归 operands，追加 ops_list
     ops_list = [
       ("mul", ["_in_ptr0_local", 2.0],  "_tmp2_frag"),
       ("add", ["_tmp2_frag",     1.0],  "_out_ptr0_local"),
     ]
  ⑤ emit 中间 fragment 分配 + T.v* 调用：
       _tmp2_frag = T.alloc_shared((_XBLOCK,), 'float32')
       T.vmul(_in_ptr0_local, 2.0, _tmp2_frag)
       T.vadd(_tmp2_frag, 1.0, _out_ptr0_local)
  ⑥ T.copy(out_local, out_ptr[offset])              # cbuf → GM
  output: str — 完整 @T.prim_func 源码
```

#### Reduction（persistent）

```
[torch_npu] TileLangKernel._codegen_reduction_kernel():
  ① 2-D tensor 签名：input(_xnumel, _rnumel), output(_xnumel, 1)
  ② _ROW_BLOCK = 32（每个 AI Core 处理的行数，需能整除 xnumel）
  ③ T.alloc_shared((_ROW_BLOCK, _rnumel), dtype)     # 输入 block，cbuf
  ④ T.alloc_shared((_ROW_BLOCK, 1), dtype)           # 输出 block，cbuf
  ⑤ T.copy(in_ptr[cid*_ROW_BLOCK, 0], in_local,
           size=[_ROW_BLOCK, _rnumel])               # GM → cbuf
  ⑥ T.reduce(in_local, out_local,
             dims=1, reduce_mode='sum', clear=True)  # 批量规约（32行）
  ⑦ T.copy(out_local, out_ptr[cid*_ROW_BLOCK, 0],
           size=[_ROW_BLOCK, 1])                     # cbuf → GM（128B DMA）

  注意：输出 buffer 用 T.alloc_shared 而非 T.alloc_fragment，
        因为 fragment→gm 的 memref.copy 会触发不支持的 hivm.hir.store。
        共 _ROW_BLOCK×1×4B = 128B 满足 DMA 对齐要求。

  注意：所有 reduction 的 accumulator 由 inductor 的 upcast_acc_dtype()
        从 fp16 提升为 fp32，fp16 输入 kernel 内实际使用 fp32。
```

### 阶段 5 — define_kernel()：写入 wrapper

```
[torch_npu] TileLangScheduling.define_kernel(src_code, node_schedule, kernel, hash):
  直接写入 wrapper.header（绕过 wrapper.define_kernel）：

    import tilelang as _tilelang_0

    def _prim_factory_tilelang_fused_add_0(xnumel):
        _xnumel = xnumel
        # reduction kernel 会额外有 _rnumel = rnumel
        import tilelang.language as T
        _XBLOCK = 128
        @T.prim_func
        def tilelang_fused_add_0_prim_fn(...): ...
        return tilelang_fused_add_0_prim_fn

    _tilelang_fused_add_0_cache = {}

    def tilelang_fused_add_0(in_ptr0, out_ptr0, xnumel):
        _key = (int(xnumel),)               # 按 shape 懒编译
        if _key not in _tilelang_fused_add_0_cache:
            _tilelang_fused_add_0_cache[_key] = _tilelang_0.compile(
                _prim_factory_tilelang_fused_add_0(_key[0]), target='npuir'
            )
        _tilelang_fused_add_0_cache[_key](in_ptr0, out_ptr0)

[torch_npu] kernel.call_kernel(kernel_name):
  写入 wrapper call 段：
    tilelang_fused_add_0(arg0_1, buf0, 1024)
```

---

## Triton 完整调用链（对比）

```
[torch_npu] NPUCombinedScheduling.codegen_node(node)
  └─ [torch_npu] NPUTritonScheduling.codegen_node_schedule(...)
       │
       ├─ select_tiling(...)
       │
       ├─ [torch_npu] NPUIndexTritonKernel(tiling, ...)
       │
       ├─ [torch_npu] decide_codegen_dims_in_kernel(...)
       │    ★ TileLang 与 Triton 共用，见阶段 2
       │
       ├─ [torch] codegen_node_schedule_with_kernel(...)
       │    → load()    → emit tl.load(ptr + offset, mask=mask) → self.loads buffer
       │    → store()   → emit tl.store(ptr + offset, val, mask) → self.stores buffer
       │    → reduction()→ emit tl.sum(val, dim) → self.compute buffer
       │    → codegen_body() ★ Triton 独有：生成嵌套循环写入 self.body
       │         for split_axis in range(offset, min(offset+XBLOCK, numel)):
       │           for tiling_axis in range(loops_tiling):
       │             x = tl.arange(0, XBLOCK_SUB) + offset
       │             mask = x < numel
       │             splice(loads) / splice(compute) / splice(stores)
       │
       ├─ [torch_npu] NPUIndexTritonKernel.codegen_kernel()
       │    ① gen_common_triton_imports()
       │    ② @triton_heuristics.pointwise/reduction(size_hints=..., triton_meta=...)
       │    ③ @triton.jit
       │    ④ def triton_poi_fused_add_0(in_ptr0, ..., X0BLOCK: tl.constexpr,
       │                                  X0BLOCK_SUB: tl.constexpr):
       │    ⑤ splice(self.body)   ← codegen_body() 已生成好的循环代码
       │
       └─ [torch_npu] define_kernel(src_code, ..., traced_graph_hash)
            写入 wrapper.header：
              triton_poi_fused_add_0 = async_compile.triton(
                  'triton_poi_fused_add_0', '''...''', device_str='npu')

            call_kernel():
              stream0 = get_raw_stream(0)
              triton_poi_fused_add_0.run(arg0_1, buf0, 1024, stream=stream0)
```

---

## 关键差异对比

### 宏观架构

| | Triton | TileLang |
|--|--------|----------|
| Scheduling 基类 | `NPUTritonScheduling` [torch_npu] | `NPUTritonScheduling` [torch_npu]（相同）|
| Kernel 基类 | `NPUIndexTritonKernel` [torch_npu] | `NPUIndexTritonKernel` [torch_npu]（相同）|
| NPU 轴分析 | `decide_codegen_dims_in_kernel()` 7步 | **共用**（TileLang 继承，完全相同）|
| `codegen_body()` | 生成 Triton 嵌套循环 → `self.body` | **NOP**（CSE 失效 + buffer 清理）|
| `codegen_kernel()` | 读 `self.body` → `@triton.jit` | 从 op graph 重建 `@T.prim_func` |

### load() / store()

| | Triton | TileLang |
|--|--------|----------|
| `load()` 输出 | emit `tl.load(ptr + offset, mask=mask)` → `self.loads` buffer | 记录 `_tl_inputs` / `_var_bufs[var] = local_buf`；同时记录 sympy index |
| `store()` 输出 | emit `tl.store(ptr + offset, val, mask=mask)` → `self.stores` | 记录 `_tl_outputs` / `_output_vars[local] = (var, dtype)`；记录 sympy index |
| `reduction()` | emit tl.sum/max + accumulator | 记录 `_var_ops[result] = ("reduce", mode, src)` |
| `store_reduction()` | emit → `self.post_loop_store` | 记录 `_is_reduction_output = True` |

### codegen_kernel()

| | Triton | TileLang |
|--|--------|----------|
| 计算体 | 标量算术（寄存器支持标量写） | 向量 intrinsic：`T.vadd / T.vmul / T.vexp / ...` |
| 原因 | CUDA 寄存器支持标量 store | Ascend L1(cbuf) 不支持标量 `memref.store`，必须用向量指令 |
| Op 信息来源 | 表达式字符串，Triton 自行编译 | `_pending_op + create_cse_var()` 构建 op graph |
| 常量处理 | 直接内联 | `constant()` → `_var_consts[cse_var] = "2.0"` |
| 输入 buffer | 无（直接 tl.load） | `T.alloc_shared(XBLOCK, dtype)`（cbuf/UB）|
| 输出 buffer | 无（直接 tl.store） | `T.alloc_shared(XBLOCK, dtype)`（cbuf/UB）|
| 中间 buffer | 无（寄存器） | `T.alloc_shared` 按需分配 |
| Reduction buffer | accumulator（cbuf） | `T.alloc_shared(_ROW_BLOCK, 1)`（cbuf，128B 对齐）|

### define_kernel()

| | Triton | TileLang |
|--|--------|----------|
| 编译时机 | 导入时（`async_compile`，后台线程） | 第一次调用时（shape-keyed 懒编译）|
| 编译函数 | `async_compile.triton('name', src, device_str='npu')` | `tilelang.compile(prim_fn, target='npuir')` |
| 写入位置 | `wrapper.define_kernel()` | 直接写 `wrapper.header` |
| 参数签名 | 裸指针 `in_ptr0: *fp16` | 类型化张量 `T.Tensor((_xnumel,), 'float32')` |
| Shape cache | 无（静态 kernel）| `(xnumel, rnumel,...)` → dict 按 shape 懒编译 |

### call_kernel()

| | Triton | TileLang |
|--|--------|----------|
| 调用方式 | `name.run(*tensors, xnumel, stream=stream0)` | `name(*tensors, xnumel)` |

---

## Op Graph 追踪机制（TileLang 独有）

TileLang 需要把 inductor 的标量 CSE 表达式转换为向量 intrinsic，为此在 kernel 内维护一个轻量 op graph：

```
[torch_npu] TileLangOverrides.mul(tmp0, tmp1)
  │  ① _set_pending("mul", [tmp0_var, tmp1_var])
  │  ② return "(tmp0 * tmp1)"
  ▼
[torch] CSE.generate(compute_buf, "(tmp0 * tmp1)")
  │  cache miss → newvar("tmp2")
  ▼
[torch_npu] TileLangKernel.create_cse_var("tmp2", bounds, dtype)
  │  pending_op != None → _var_ops["tmp2"] = ("mul", [tmp0_var, tmp1_var])
  │  pending_op = None
  ▼
[torch_npu] codegen_kernel() 时调用 _build_vec_ops():
  tmp2 in _var_ops → op="mul", operands=[tmp0_var, tmp1_var]
    tmp0 in _var_bufs   → "_in_ptr0_local"    (input buffer)
    tmp1 in _var_consts → float("2.0") = 2.0  (constant)
  → emit: T.vmul(_in_ptr0_local, 2.0, _tmp2_frag)
```

三个字典的职责：

| 字典 | key | value | 来源 |
|------|-----|-------|------|
| `_var_bufs` | CSE var name | `"_in_ptr0_local"` | `[torch_npu] load()` |
| `_var_consts` | CSE var name | `"2.0"` | `[torch_npu] constant()` → `create_cse_var()` |
| `_var_ops` | CSE var name | `("mul", [operands])` | `[torch_npu] override → create_cse_var()` |
| `_is_reduction_output` | local_buf_name | `True` | `[torch_npu] store_reduction()` |

---

## 生成代码对比（`x * 2.0 + 1.0`）

**Triton 生成：**
```python
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask  = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, mask=xmask)
    tmp1 = tmp0 * 2.0
    tmp2 = tmp1 + 1.0
    tl.store(out_ptr0 + xindex, tmp2, mask=xmask)
```

**TileLang 生成：**
```python
@T.prim_func
def tilelang_fused_0_prim_fn(
    in_ptr0:  T.Tensor((_xnumel,), 'float32'),
    out_ptr0: T.Tensor((_xnumel,), 'float32'),
):
    with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):
        _in_ptr0_local  = T.alloc_shared((_XBLOCK,), 'float32')
        _out_ptr0_local = T.alloc_shared((_XBLOCK,), 'float32')

        T.copy(in_ptr0[cid * _XBLOCK], _in_ptr0_local)

        _tmp2_frag = T.alloc_shared((_XBLOCK,), 'float32')
        T.vmul(_in_ptr0_local, 2.0, _tmp2_frag)
        T.vadd(_tmp2_frag, 1.0, _out_ptr0_local)

        T.copy(_out_ptr0_local, out_ptr0[cid * _XBLOCK])
```

**TileLang Reduction 生成（`x.sum(dim=1, keepdim=True)` 在 shape [64, 128]）：**
```python
_ROW_BLOCK = 32

@T.prim_func
def tilelang_fused_sum_0_prim_fn(
    in_ptr0:  T.Tensor((_xnumel, _rnumel), 'float32'),  # 2-D: M × N
    out_ptr0: T.Tensor((_xnumel, 1),       'float32'),  # 2-D: M × 1
):
    with T.Kernel(T.ceildiv(_xnumel, _ROW_BLOCK), is_npu=True) as (cid, _):
        _in_ptr0_local  = T.alloc_shared((_ROW_BLOCK, 128), 'float32')
        _out_ptr0_local = T.alloc_shared((_ROW_BLOCK, 1),   'float32')

        T.copy(in_ptr0[cid * _ROW_BLOCK, 0], _in_ptr0_local,
               size=[_ROW_BLOCK, 128])

        T.reduce(_in_ptr0_local, _out_ptr0_local,
                 dims=1, reduce_mode='sum', clear=True)

        T.copy(_out_ptr0_local, out_ptr0[cid * _ROW_BLOCK, 0],
               size=[_ROW_BLOCK, 1])
```

---

## 当前支持情况

| 场景 | 是否支持 | 备注 |
|------|----------|------|
| 1-D / N-D **连续** pointwise | ✅ | inductor flatten 成 1-D xnumel，DMA 块拷贝 |
| 2-D / 3-D / 4-D pointwise | ✅ | 与 1-D 完全相同 kernel，只是 xnumel 更大 |
| fp32 pointwise | ✅ | 完整支持 |
| fp16 pointwise | ✅ | 完整支持 |
| int32 pointwise（vmul/vabs/vneg/vmin/vpow）| ✅ | 仅支持这几个 op |
| **persistent reduction**（sum/max/min/mean）| ✅ | xnumel 需整除 _ROW_BLOCK(=32)；rnumel 需放入 SRAM |
| fp16 reduction | ✅ | inductor 自动 upcast 到 fp32，kernel 内使用 fp32 |
| 非 persistent reduction（rnumel 过大）| ❌ → fallback Triton | `raise NotImplementedError` |
| argmax / argmin | ❌ → fallback Triton | 未实现 |
| welford reduction | ❌ → fallback Triton | 未实现 |
| 转置 `A.T + B` | ❌ 结果错误 | index 被忽略，stride 信息丢失 |
| strided / gather | ❌ 结果错误 | 同上 |
| `atomic_add` store | ❌ → fallback Triton | `raise NotImplementedError` |
| dtype 非 fp16/fp32（部分 op）| ❌ → fallback Triton | `_check_op_graph_dtype` 拒绝 |
