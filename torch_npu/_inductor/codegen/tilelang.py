"""
TileLang codegen backend for torch_npu inductor.

Generates TileLang (@T.prim_func) kernels targeting Ascend NPU via
`tilelang.compile(..., target='npuir')`.

Activated by: TORCHINDUCTOR_NPU_BACKEND=tilelang

Generated kernel structure (Developer mode, T.Parallel):

    @T.prim_func
    def <name>_prim_fn(
        in_ptr0: T.Tensor((_xnumel,), 'float16'),
        out_ptr0: T.Tensor((_xnumel,), 'float16'),
    ):
        with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):
            _in_ptr0_local  = T.alloc_shared((_XBLOCK,), 'float16')
            _out_ptr0_local = T.alloc_shared((_XBLOCK,), 'float16')
            T.copy(in_ptr0[cid * _XBLOCK], _in_ptr0_local)   # GM -> UB
            for _tl_i in T.Parallel(_XBLOCK):
                tmp0 = _in_ptr0_local[_tl_i]
                <compute…>
                _out_ptr0_local[_tl_i] = <result>
            T.copy(_out_ptr0_local, out_ptr0[cid * _XBLOCK]) # UB -> GM

Known limitations:
- Only float16 and float32 tensors supported (Ascend UB/alloc_shared restriction).
  Kernels with other dtypes raise NotImplementedError and fall back to Triton.
- reduction() raises NotImplementedError; those nodes fall back to Triton.
- Assumes flat contiguous layout; index expr ignored in load/store.
- No tail guard when xnumel % _XBLOCK != 0.
"""
from __future__ import annotations

from typing import Any, Optional

import sympy
import torch
from torch.utils._ordered_set import OrderedSet

from torch._inductor import config, ir
from torch._inductor.codegen.common import (
    BackendFeature,
    CSE,
    CSEVariable,
    IndentedBuffer,
    OpOverrides,
    SizeArg,
    TensorArg,
)
from torch._inductor.codegen.simd import (
    SIMDKernel,
    SIMDScheduling,
    IterationRangesRoot,
    IterationRangesEntry,
)
from torch._inductor.codegen.triton import (
    Placeholder,
    get_fused_kernel_name,
    get_kernel_metadata,
)
from torch._inductor.virtualized import ReductionType, StoreMode, V


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

# Ascend UB (T.alloc_shared / T.alloc_ub) only supports fp16 and fp32.
_TILELANG_UB_DTYPES: frozenset[torch.dtype] = frozenset({torch.float16, torch.float32})

_TORCH_TO_TL_TYPE: dict[torch.dtype, str] = {
    torch.float16:  "float16",
    torch.bfloat16: "bfloat16",
    torch.float32:  "float32",
    torch.float64:  "float64",
    torch.int8:     "int8",
    torch.int16:    "int16",
    torch.int32:    "int32",
    torch.int64:    "int64",
    torch.uint8:    "uint8",
    torch.bool:     "bool",
}


def tilelang_dtype(dtype: torch.dtype) -> str:
    return _TORCH_TO_TL_TYPE.get(dtype, "float32")


# ---------------------------------------------------------------------------
# CSE variable
# ---------------------------------------------------------------------------

class TileLangCSEVariable(CSEVariable):
    """CSE variable for scalar Python expressions inside a T.Parallel body."""
    pass


# ---------------------------------------------------------------------------
# Op overrides — scalar Python expressions for the T.Parallel body
# ---------------------------------------------------------------------------

class TileLangOverrides(OpOverrides):
    """
    Map inductor element-wise ops → scalar Python expressions valid inside
    a T.Parallel loop body.  TileLang's compiler lowers these to NPU vector
    instructions when the entire `for _tl_i in T.Parallel(N):` block is
    analysed.
    """

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype=None, use_compute_types=True):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"T.cast({x}, '{tilelang_dtype(dtype)}')"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        return (
            f"T.reinterpret_cast({x}, '{tilelang_dtype(src_dtype)}', "
            f"'{tilelang_dtype(dtype)}')"
        )

    @staticmethod
    def constant(value, dtype: torch.dtype):
        import torch._prims_common as prim
        py_type = prim.dtype_to_type(dtype)
        return repr(py_type(value))

    @staticmethod
    def abs(x):        return f"abs({x})"
    @staticmethod
    def neg(x):        return f"(-{x})"
    @staticmethod
    def exp(x):        return f"T.exp({x})"
    @staticmethod
    def exp2(x):       return f"_math.pow(2.0, {x})"
    @staticmethod
    def expm1(x):      return f"(T.exp({x}) - 1.0)"
    @staticmethod
    def log(x):        return f"_math.log({x})"
    @staticmethod
    def log2(x):       return f"_math.log2({x})"
    @staticmethod
    def log1p(x):      return f"_math.log1p({x})"
    @staticmethod
    def sqrt(x):       return f"_math.sqrt({x})"
    @staticmethod
    def rsqrt(x):      return f"(1.0 / _math.sqrt({x}))"
    @staticmethod
    def sin(x):        return f"_math.sin({x})"
    @staticmethod
    def cos(x):        return f"_math.cos({x})"
    @staticmethod
    def tan(x):        return f"_math.tan({x})"
    @staticmethod
    def tanh(x):       return f"T.sigmoid({x})"
    @staticmethod
    def asin(x):       return f"_math.asin({x})"
    @staticmethod
    def acos(x):       return f"_math.acos({x})"
    @staticmethod
    def atan(x):       return f"_math.atan({x})"
    @staticmethod
    def atan2(x, y):   return f"_math.atan2({x}, {y})"
    @staticmethod
    def sigmoid(x):    return f"T.sigmoid({x})"
    @staticmethod
    def relu(x):       return f"(({x}) if ({x}) > 0.0 else 0.0)"

    @staticmethod
    def minimum(a, b): return f"(({a}) if ({a}) < ({b}) else ({b}))"
    @staticmethod
    def maximum(a, b): return f"(({a}) if ({a}) > ({b}) else ({b}))"

    @staticmethod
    def where(cond, a, b):
        # T.Parallel supports T.if_then_else but not index-based conditions.
        # For constant conditions this will work; index-based will be caught
        # by the compiler.
        return f"(({a}) if ({cond}) else ({b}))"

    @staticmethod
    def add(a, b):      return f"({a} + {b})"
    @staticmethod
    def sub(a, b):      return f"({a} - {b})"
    @staticmethod
    def mul(a, b):      return f"({a} * {b})"
    @staticmethod
    def truediv(a, b):  return f"({a} / {b})"
    @staticmethod
    def floordiv(a, b): return f"({a} // {b})"
    @staticmethod
    def mod(a, b):      return f"({a} % {b})"
    @staticmethod
    def pow(a, b):      return f"_math.pow({a}, {b})"

    @staticmethod
    def logical_not(a):    return f"(not ({a}))"
    @staticmethod
    def logical_and(a, b): return f"(({a}) and ({b}))"
    @staticmethod
    def logical_or(a, b):  return f"(({a}) or ({b}))"
    @staticmethod
    def logical_xor(a, b): return f"(bool({a}) != bool({b}))"

    @staticmethod
    def bitwise_and(a, b): return f"(({a}) & ({b}))"
    @staticmethod
    def bitwise_or(a, b):  return f"(({a}) | ({b}))"
    @staticmethod
    def bitwise_xor(a, b): return f"(({a}) ^ ({b}))"
    @staticmethod
    def bitwise_not(a):    return f"(~({a}))"

    @staticmethod
    def sign(x): return f"(1 if ({x}) > 0 else (-1 if ({x}) < 0 else 0))"
    @staticmethod
    def floor(x): return f"_math.floor({x})"
    @staticmethod
    def ceil(x):  return f"_math.ceil({x})"
    @staticmethod
    def trunc(x): return f"_math.trunc({x})"
    @staticmethod
    def erf(x):   return f"_math.erf({x})"
    @staticmethod
    def erfc(x):  return f"_math.erfc({x})"

    @staticmethod
    def lt(a, b): return f"({a} < {b})"
    @staticmethod
    def le(a, b): return f"({a} <= {b})"
    @staticmethod
    def gt(a, b): return f"({a} > {b})"
    @staticmethod
    def ge(a, b): return f"({a} >= {b})"
    @staticmethod
    def eq(a, b): return f"({a} == {b})"
    @staticmethod
    def ne(a, b): return f"({a} != {b})"

    @staticmethod
    def rand(seed, offset):
        raise NotImplementedError("TileLang backend: rand() not supported")

    @staticmethod
    def randint64(seed, offset, low, high):
        raise NotImplementedError("TileLang backend: randint64() not supported")

    @staticmethod
    def load_seed(name, offset):
        raise NotImplementedError("TileLang backend: load_seed() not supported")

    @staticmethod
    def index_expr(expr, dtype):
        return str(V.kernel.rename_indexing(expr))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

_DEFAULT_XBLOCK = 128


def _check_ub_dtype(name: str, dtype: torch.dtype) -> None:
    """
    Raise NotImplementedError for dtypes not supported by Ascend UB memory.
    This lets the inductor scheduler fall back to the Triton backend for
    kernels that mix unsupported types.
    """
    if dtype not in _TILELANG_UB_DTYPES:
        raise NotImplementedError(
            f"TileLang backend: Ascend UB memory only supports float16/float32; "
            f"buffer '{name}' has dtype {dtype}. Falling back to Triton."
        )


class TileLangKernel(SIMDKernel):
    """
    Generates a TileLang @T.prim_func source for a fused set of pointwise nodes.

    Uses Developer mode (T.alloc_shared + T.Parallel scalar body) so that
    arbitrary fused element-wise expressions produced by inductor CSE are
    naturally lowered to NPU vector instructions by the TileLang compiler.

    T.Parallel constraint satisfied: every buffer access inside the loop body
    uses the unmodified loop variable ``_tl_i`` as the sole index.
    """

    overrides = TileLangOverrides  # type: ignore[assignment]
    # Use the Python-style sympy expression printer (no tl.* type suffixes).
    kexpr = SIMDKernel.sexpr  # type: ignore[assignment]

    def __init__(self, tiling: dict, **kwargs) -> None:
        super().__init__(tiling, **kwargs)
        # Re-create CSE without Triton-specific variable naming conventions.
        self.cse: CSE = CSE(self.newvar_prefix, self.suffix)
        # buffer name → (ptr_var, local_var, dtype) for T.copy pairs.
        self._tl_inputs:  dict[str, tuple[str, str, torch.dtype]] = {}
        self._tl_outputs: dict[str, tuple[str, str, torch.dtype]] = {}

    # ------------------------------------------------------------------
    # SIMDKernel abstract overrides
    # ------------------------------------------------------------------

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return tilelang_dtype(dtype)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        # T.Kernel + T.Parallel handle block/thread dispatch; suppress the
        # Triton-style xoffset / xindex / xmask range-tree preamble.
        pass

    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str:
        return "cid"

    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str:
        return f"T.arange(0, {entry.prefix.upper()}BLOCK)"

    def iteration_ranges_scalar_code(self, entry: IterationRangesRoot, value: Any) -> str:
        return repr(value)

    # ------------------------------------------------------------------
    # load / store / reduction
    # ------------------------------------------------------------------

    def load(self, name: str, index: sympy.Expr) -> TileLangCSEVariable:
        """
        Register *name* as needing a T.copy (GM → UB) and return a CSE
        variable that reads element ``_tl_i`` from the local shared buffer.

        The *index* sympy expression is intentionally ignored — flat 1-D
        contiguous layout is assumed for this initial implementation.
        """
        dtype = V.graph.get_dtype(name)
        _check_ub_dtype(name, dtype)

        var = self.args.input(name)
        local_name = f"_{var}_local"
        if name not in self._tl_inputs:
            self._tl_inputs[name] = (var, local_name, dtype)
        line = f"{local_name}[_tl_i]"
        return self.cse.generate(self.loads, line, dtype=dtype)

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: TileLangCSEVariable,
        mode: StoreMode = None,
    ) -> None:
        """
        Register *name* as needing a T.copy (UB → GM) and emit the element
        write into the local shared buffer.
        """
        if mode == StoreMode.ATOMIC_ADD:
            raise NotImplementedError(
                "TileLang backend: atomic_add store not yet supported"
            )
        dtype = V.graph.get_dtype(name)
        _check_ub_dtype(name, dtype)

        var = self.args.output(name)
        local_name = f"_{var}_local"
        if name not in self._tl_outputs:
            self._tl_outputs[name] = (var, local_name, dtype)
        self.stores.writeline(f"{local_name}[_tl_i] = {value}")

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: TileLangCSEVariable,
    ) -> TileLangCSEVariable:
        raise NotImplementedError(
            "TileLang backend: reductions not yet implemented; "
            "will fall back to Triton."
        )

    # ------------------------------------------------------------------
    # Kernel source generation
    # ------------------------------------------------------------------

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        """
        Return the @T.prim_func source string.

        ``TileLangScheduling.define_kernel()`` wraps this inside a per-shape
        factory function so that ``_xnumel`` is available as a closure variable
        when ``tilelang.compile()`` is invoked at runtime.

        T.copy syntax used (Developer mode, offset form):
            T.copy(global_buf[cid * _XBLOCK], local_buf)   # GM → UB
            T.copy(local_buf, global_buf[cid * _XBLOCK])   # UB → GM
        The copy size is inferred from the shape of the local buffer.
        """
        xblock = _DEFAULT_XBLOCK
        prim_fn_name = f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn"

        argdefs, _, signature, _ = self.args.python_argdefs()

        # Build T.Tensor argument annotations for the @T.prim_func signature.
        # SizeArg (numel) entries are omitted — they are captured as closures
        # from the factory function that wraps this block.
        prim_sig_parts: list[str] = []
        for argdef, sig in zip(argdefs, signature):
            if isinstance(sig, TensorArg):
                dtype_str = tilelang_dtype(sig.dtype)
                prim_sig_parts.append(
                    f"{argdef.name}: T.Tensor((_xnumel,), '{dtype_str}')"
                )

        code = IndentedBuffer()
        code.writeline("import tilelang.language as T")
        code.writeline("import math as _math")
        code.writeline("")
        code.writeline(f"_XBLOCK = {xblock}")
        code.writeline("")
        code.writeline("@T.prim_func")
        code.writeline(f"def {prim_fn_name}(")
        with code.indent():
            for i, part in enumerate(prim_sig_parts):
                comma = "," if i < len(prim_sig_parts) - 1 else ""
                code.writeline(f"{part}{comma}")
        code.writeline("):")

        with code.indent():
            # T.Kernel: one block dimension for NPU (is_npu=True required).
            code.writeline(
                "with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):"
            )
            with code.indent():
                # Allocate UB (T.alloc_shared = Developer mode) for each input.
                for _bname, (var, local_name, dtype) in self._tl_inputs.items():
                    code.writeline(
                        f"{local_name} = T.alloc_shared((_XBLOCK,), '{tilelang_dtype(dtype)}')"
                    )
                # Allocate UB for outputs that are not also inputs (in-place).
                input_locals = {loc for _, loc, _ in self._tl_inputs.values()}
                for _bname, (var, local_name, dtype) in self._tl_outputs.items():
                    if local_name not in input_locals:
                        code.writeline(
                            f"{local_name} = T.alloc_shared((_XBLOCK,), '{tilelang_dtype(dtype)}')"
                        )
                code.writeline("")

                # T.copy GM → UB for every input (offset form: size inferred
                # from local_name's shape (_XBLOCK,)).
                for _bname, (var, local_name, _) in self._tl_inputs.items():
                    code.writeline(
                        f"T.copy({var}[cid * _XBLOCK], {local_name})"
                    )
                code.writeline("")

                # T.Parallel compute body: scalar expressions from inductor CSE.
                # Constraint satisfied: every buffer subscript is the bare
                # loop variable ``_tl_i`` with no index transformation.
                code.writeline("for _tl_i in T.Parallel(_XBLOCK):")
                with code.indent():
                    loads_body   = self.loads.getvalue().strip()
                    compute_body = self.compute.getvalue().strip()
                    stores_body  = self.stores.getvalue().strip()
                    if loads_body:
                        code.splice(self.loads)
                    if compute_body:
                        code.splice(self.compute)
                    if stores_body:
                        code.splice(self.stores)
                    else:
                        code.writeline("pass")
                code.writeline("")

                # T.copy UB → GM for every output.
                for _bname, (var, local_name, _) in self._tl_outputs.items():
                    code.writeline(
                        f"T.copy({local_name}, {var}[cid * _XBLOCK])"
                    )

        return code.getvalue()

    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None) -> None:
        """
        Emit ``name(tensor_arg0, ..., xnumel)`` in the generated wrapper.
        The function *name* is the shape-keyed caching wrapper produced by
        ``TileLangScheduling.define_kernel()``.
        """
        wrapper = V.graph.wrapper_code
        _, call_args, signature, _ = self.args.python_argdefs()
        tensor_args = [
            a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)
        ]
        numel_args = [str(tree.numel) for tree in self.active_range_trees()]
        wrapper.writeline(f"{name}({', '.join(tensor_args + numel_args)})")

    def create_cse_var(self, *args, **kwargs) -> TileLangCSEVariable:
        return TileLangCSEVariable(*args, **kwargs)

    def should_use_persistent_reduction(self) -> bool:
        return False

    def should_use_cooperative_reduction(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class TileLangScheduling(SIMDScheduling):
    """
    Inductor scheduling backend that emits TileLang kernels for Ascend NPU.

    Register via::

        register_backend_for_device(
            "npu", TileLangScheduling, NPUWrapperCodeGen, CppWrapperNpu
        )

    Activate via environment variable::

        TORCHINDUCTOR_NPU_BACKEND=tilelang
    """

    kernel_type: type[Any] = TileLangKernel

    # Conservative initial feature set; extend as TileLang support matures.
    backend_features: OrderedSet[BackendFeature] = OrderedSet(
        [BackendFeature.INPLACE_BUFFERS]
    )

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return cls.backend_features

    def codegen_comment(self, node_schedule) -> None:
        wrapper = V.graph.wrapper_code
        origins, _ = get_kernel_metadata(node_schedule, wrapper)
        if origins:
            wrapper.writeline(origins)

    def codegen_sync(self) -> None:
        V.graph.wrapper_code.writeline("torch.npu.synchronize()")

    # ------------------------------------------------------------------
    # define_kernel
    # ------------------------------------------------------------------

    def define_kernel(
        self,
        src_code: str,
        node_schedule,
        kernel: TileLangKernel,
    ) -> str:
        """
        Splice a module-level caching wrapper into ``wrapper.header``.

        Generated pattern::

            import tilelang as _tilelang_<suffix>

            def _prim_factory_<name>(_xnumel):
                # src_code — defines <name>_prim_fn via @T.prim_func
                import tilelang.language as T
                import math as _math
                _XBLOCK = 128
                @T.prim_func
                def <name>_prim_fn(...):
                    with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):
                        ...
                return <name>_prim_fn

            _<name>_cache = {}

            def <name>(in_ptr0, ..., xnumel):
                _key = (int(xnumel),)
                if _key not in _<name>_cache:
                    _<name>_cache[_key] = _tilelang_<suffix>.compile(
                        _prim_factory_<name>(_key[0]), target='npuir')
                _<name>_cache[_key](in_ptr0, ...)
        """
        wrapper = V.graph.wrapper_code

        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        suffix = wrapper.next_kernel_suffix()
        kernel_name = "_".join(filter(None, ["tilelang", fused_name, suffix]))
        wrapper.src_to_kernel[src_code] = kernel_name

        # Substitute the KERNEL_NAME placeholder in the @T.prim_func definition.
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        # Collect call-site info from the kernel object.
        _, call_args, signature, _ = kernel.args.python_argdefs()
        tensor_call_args = [
            a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)
        ]
        active_trees  = kernel.active_range_trees()
        numel_arg_names = [f"{t.prefix}numel" for t in active_trees]
        outer_arg_list  = tensor_call_args + numel_arg_names

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment = f"{origins}\n{detailed_origins}".strip()

        # Resolve the tilelang package root so the generated output_code.py
        # can import it even if sys.path isn't pre-configured at inference time.
        tilelang_pkg_root: Optional[str] = None
        try:
            import tilelang as _tl_mod
            import os as _os
            tilelang_pkg_root = _os.path.dirname(_os.path.dirname(_tl_mod.__file__))
        except ImportError:
            pass

        import_alias = f"_tilelang_{suffix}"
        cache_var    = f"_{kernel_name}_cache"
        factory_fn   = f"_prim_factory_{kernel_name}"
        prim_fn_name = f"{kernel_name}_prim_fn"

        code = IndentedBuffer()
        code.writeline(f"\n# TileLang kernel — {metadata_comment}")

        # Ensure tilelang is importable in the generated output_code.py.
        if tilelang_pkg_root:
            code.writeline("import sys as _sys")
            code.writeline(
                f"if {tilelang_pkg_root!r} not in _sys.path: "
                f"_sys.path.insert(0, {tilelang_pkg_root!r})"
            )
        code.writeline(f"import tilelang as {import_alias}")
        code.writeline("")

        # Factory function: captures _xnumel at compile time for shape-keyed caching.
        if numel_arg_names:
            factory_params = ", ".join(f"_{n}" for n in numel_arg_names)
        else:
            factory_params = "_dummy=None"

        code.writeline(f"def {factory_fn}({factory_params}):")
        with code.indent():
            if numel_arg_names:
                # Expose xnumel as _xnumel (referenced in T.Tensor shape and
                # T.ceildiv call inside the @T.prim_func body).
                code.writeline(f"_xnumel = _{numel_arg_names[0]}")
            else:
                code.writeline("_xnumel = 1")
            code.splice(src_code)
            code.writeline(f"return {prim_fn_name}")

        code.writeline("")
        code.writeline(f"{cache_var} = {{}}")
        code.writeline("")

        # Public kernel function: compiles once per unique shape, then calls.
        code.writeline(f"def {kernel_name}({', '.join(outer_arg_list)}):")
        with code.indent():
            if numel_arg_names:
                key_parts = ", ".join(f"int({n})" for n in numel_arg_names)
                code.writeline(f"_key = ({key_parts},)")
            else:
                code.writeline("_key = ('static',)")
            code.writeline(f"if _key not in {cache_var}:")
            with code.indent():
                if numel_arg_names:
                    factory_call_args = ", ".join(
                        f"_key[{i}]" for i in range(len(numel_arg_names))
                    )
                else:
                    factory_call_args = ""
                code.writeline(f"{cache_var}[_key] = {import_alias}.compile(")
                with code.indent():
                    code.writeline(
                        f"{factory_fn}({factory_call_args}), target='npuir'"
                    )
                code.writeline(")")
            # Call with tensor args only — sizes are baked into the compiled fn.
            code.writeline(
                f"{cache_var}[_key]({', '.join(tensor_call_args)})"
            )

        # Inject into the module-level header (bypasses wrapper.define_kernel
        # which only supports the `name = expr` pattern).
        wrapper.header.splice(code.getvalue())
        return kernel_name
