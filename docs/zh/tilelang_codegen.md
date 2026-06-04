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
| `torch_npu/_inductor/codegen/wrapper.py` | **torch_npu** | `NPUWrapperCodeGen`，生成 Python wrapper |
| `torch_npu/_inductor/utils.py` | **torch_npu** | `patch_has_triton()` 等运行时 patch |
| `torch_npu/_inductor/lowering.py` | **torch_npu** | NPU op lowering / fallback 注册 |

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
      （继承自 [torch] SIMDScheduling）
```

### 阶段 2 — Kernel 构造与 load/store 收集

```
[torch] SIMDScheduling.codegen_node_schedule(kernel_features, nodes)
  │
  ├─ select_tiling(numel, rnumel, nodes)
  │     output: tiling = {'x': sympy.Expr}
  │
  ├─ [torch_npu] TileLangKernel(tiling, features=...)   ← 构造 kernel 对象
  │     初始化：
  │       self.cse          = CSE(prefix, suffix)    # [torch] CSE
  │       self._tl_inputs   = {}
  │       self._tl_outputs  = {}
  │       self._pending_op  = None
  │       self._var_ops     = {}
  │       self._var_bufs    = {}
  │       self._var_consts  = {}
  │       self._output_vars = {}
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
                          # V.get_ops_handler() 返回包装了 kernel 的 CSEProxy
                              │
                              ▼ [torch] fx/interpreter.py
                                  run_node(node) → call_method(target, args)
                                  # target = "load" / "store" / "add" / ...
                                      │
                                      ▼ [torch] sizevars.py
                                          SizeVarAllocator.load(name, index)
                                            self._simplify(index)   ← 化简 sympy
                                            self._inner.load(name, index)
                                              │
                                              ▼ [torch] codegen/common.py
                                                  CSEProxy.load(name, index)
                                                    self.kernel.load(name, index)
                                                      │
                                                      ▼ [torch_npu] tilelang.py
                                                          TileLangKernel.load()
                                                          TileLangKernel.store()
```

### 阶段 3 — load() 内部：op graph 追踪

```
[torch_npu] TileLangKernel.load(name, index):
  ① _pending_op = None          # 清空，防止污染 load var
  ② cse.generate(loads_buf, "_in_ptr0_local[_tl_i]")
       → [torch_npu] create_cse_var("tmp0") 被回调
       → _var_bufs["tmp0"] = "_in_ptr0_local"
  ③ 注册 _tl_inputs[name] = (ptr_var, local_name, dtype)
  output: CSEVariable "tmp0"

[torch_npu] TileLangOverrides.constant(2.0, float32):
  ① literal = "2.0"
  ② _set_pending("const", ["2.0"])   # 标记常量
  ③ cse.generate(...) → create_cse_var("tmp1")
       → _var_consts["tmp1"] = "2.0"

[torch_npu] TileLangOverrides.mul(tmp0, tmp1):
  ① _set_pending("mul", [tmp0_var, tmp1_var])
  ② return "(tmp0 * tmp1)"
  ③ cse.generate(...) → create_cse_var("tmp2")
       → _var_ops["tmp2"] = ("mul", [tmp0_var, tmp1_var])

[torch_npu] TileLangKernel.store(name, index, tmp4_var):
  ① 注册 _tl_outputs[name] = (ptr_var, local_name, dtype)
  ② _output_vars["_out_ptr0_local"] = (tmp4_var, dtype)
```

### 阶段 4 — codegen_kernel()：生成 @T.prim_func

```
[torch_npu] TileLangKernel.codegen_kernel():
  ① 生成函数头：@T.prim_func def xxx_prim_fn(T.Tensor(...))
  ② T.alloc_shared  for each input   (L1/UB)
  ③ T.alloc_fragment for each output
  ④ T.copy(in_ptr[cid*XBLOCK], local_buf)   for each input
  ⑤ _build_vec_ops(result_var, out_loc, ops_list,
                   _var_bufs, _var_ops, _var_consts)
     递归遍历 op graph：
       var in _var_bufs   → 返回 local_buf_name
       var in _var_consts → 返回 float("2.0") = 2.0
       var in _var_ops    → 递归 operands，追加到 ops_list
     ops_list = [
       ("mul", ["_in_ptr0_local", 2.0], "_tmp2_frag"),
       ("add", ["_tmp2_frag",     1.0], "_out_ptr0_local"),
     ]
  ⑥ emit 每个 op：
       _tmp2_frag = T.alloc_fragment((_XBLOCK,), 'float32')
       T.vmul(_in_ptr0_local, 2.0, _tmp2_frag)
       T.vadd(_tmp2_frag, 1.0, _out_ptr0_local)
  ⑦ T.copy(out_local, out_ptr[cid*XBLOCK])  for each output
  output: str — 完整 @T.prim_func 源码
```

### 阶段 5 — define_kernel()：写入 wrapper

```
[torch_npu] TileLangScheduling.define_kernel(src_code, node_schedule, kernel):
  直接写入 wrapper.header（绕过 wrapper.define_kernel）：

    import tilelang as _tilelang_0

    def _prim_factory_tilelang_fused_add_0(xnumel):
        _xnumel = xnumel
        import tilelang.language as T
        _XBLOCK = 128
        @T.prim_func
        def tilelang_fused_add_0_prim_fn(...): ...
        return tilelang_fused_add_0_prim_fn

    _tilelang_fused_add_0_cache = {}

    def tilelang_fused_add_0(in_ptr0, out_ptr0, xnumel):
        _key = (int(xnumel),)
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
       ├─ [torch_npu] decide_codegen_dims_in_kernel(...)  ★ NPU 独有，TileLang 无
       │    Step1: split_and_set_ranges()
       │    Step2: transform_dims_in_indexing()
       │    Step3-5: 冗余 axis 替换与删除
       │    Step6: SplitTiling.select_split_tiling_axis()
       │    Step7: ReductionAnalysis()
       │
       ├─ [torch] codegen_node_schedule_with_kernel(...)
       │    → load()  → tl.load(ptr + offset, mask=mask)
       │    → store() → tl.store(ptr + offset, val, mask=mask)
       │
       ├─ [torch_npu] NPUIndexTritonKernel.codegen_kernel()
       │    output: @npu_triton_heuristics.pointwise(...)
       │            @triton.jit
       │            def triton_poi_fused_add_0(in_ptr0, ..., XBLOCK: tl.constexpr):
       │                x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
       │                tmp0 = tl.load(in_ptr0 + x0, mask=x0 < xnumel)
       │                tl.store(out_ptr0 + x0, tmp0 + tmp1, mask=...)
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
| Scheduling 基类 | `NPUTritonScheduling` [torch_npu] | `SIMDScheduling` [torch] |
| Kernel 基类 | `NPUIndexTritonKernel` [torch_npu] | `SIMDKernel` [torch] |
| NPU 轴分析 | `decide_codegen_dims_in_kernel()` 7步 [torch_npu] | 无，跳过 |
| `__init__.py` 注册开销 | ~30 个 patch | 6 个调用 |

### load() / store()

| | Triton | TileLang |
|--|--------|----------|
| `load()` 输出 | `tl.load(ptr + offset, mask=mask)` | 注册 `_tl_inputs`，记录 `_var_bufs[var] = local_buf` |
| `store()` 输出 | `tl.store(ptr + offset, val, mask=mask)` | 注册 `_tl_outputs`，记录 `_output_vars[local] = (var, dtype)` |
| index 表达式 | 完整计算（FloorDiv/Mod/...） | **忽略**（假设连续布局） |

### codegen_kernel()

| | Triton | TileLang |
|--|--------|----------|
| 计算体 | 标量算术（寄存器支持标量写） | 向量 intrinsic：`T.vadd / T.vmul / T.vexp / ...` |
| 原因 | CUDA 寄存器支持标量 store | Ascend L1(cbuf) 不支持标量 `memref.store`，必须用向量指令 |
| Op 信息来源 | 表达式字符串，Triton 自行编译 | `_pending_op + create_cse_var()` 构建 op graph |
| 常量处理 | 直接内联 | `constant()` → `_var_consts[cse_var] = "2.0"` |
| 中间缓冲区 | 无（寄存器） | `T.alloc_fragment` 按需分配 |

### define_kernel()

| | Triton | TileLang |
|--|--------|----------|
| 编译时机 | 导入时（`async_compile`，后台线程） | 第一次调用时（shape-keyed 懒编译） |
| 编译函数 | `async_compile.triton('name', src, device_str='npu')` | `tilelang.compile(prim_fn, target='npuir')` |
| 写入位置 | `wrapper.define_kernel()` | 直接写 `wrapper.header` |
| 参数签名 | 裸指针 `in_ptr0: *fp16` | 类型化张量 `T.Tensor((_xnumel,), 'float32')` |

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
        _out_ptr0_local = T.alloc_fragment((_XBLOCK,), 'float32')

        T.copy(in_ptr0[cid * _XBLOCK], _in_ptr0_local)

        _tmp2_frag = T.alloc_fragment((_XBLOCK,), 'float32')
        T.vmul(_in_ptr0_local, 2.0, _tmp2_frag)
        T.vadd(_tmp2_frag, 1.0, _out_ptr0_local)

        T.copy(_out_ptr0_local, out_ptr0[cid * _XBLOCK])
```

---

## 当前限制

| 场景 | 是否支持 | 原因 |
|------|----------|------|
| 1-D / N-D **连续** pointwise | ✅ | `load()` 忽略 index，直接用 DMA 块拷贝 |
| `reduction()` | ❌ → fallback Triton | `raise NotImplementedError` |
| 转置 `A.T + B` | ❌ 结果错误 | index 被忽略，stride 信息丢失 |
| strided / gather | ❌ 结果错误 | 同上 |
| `atomic_add` store | ❌ → fallback | `raise NotImplementedError` |
| dtype 非 fp16/fp32 | ❌ → fallback | `_assert_ub_dtype` 拒绝 |
