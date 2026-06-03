"""
TileLang codegen backend for torch_npu inductor (Ascend NPU).

Generates TileLang @T.prim_func kernels compiled via
``tilelang.compile(..., target='npuir')``.

Activate with::

    TORCHINDUCTOR_NPU_BACKEND=tilelang

Generated kernel structure (Developer mode):

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
                ...
                _out_ptr0_local[_tl_i] = <result>
            T.copy(_out_ptr0_local, out_ptr0[cid * _XBLOCK]) # UB -> GM

Language constraints respected:
- T.alloc_shared / T.alloc_ub: Ascend only supports float16 and float32.
  Kernels with other dtypes raise NotImplementedError (fall back to Triton).
- T.Parallel: buffer subscripts inside the loop body must use the bare
  loop variable with no index transformation.  Our load/store always emit
  ``buf[_tl_i]``, satisfying this constraint.
- T.Kernel(is_npu=True) requires exactly one block dimension.
- T.copy(src[offset], dst) / T.copy(src, dst[offset]): offset-form; copy
  size is inferred from the destination buffer shape.

Known limitations:
- Only 1-D contiguous pointwise kernels; reduction() raises NotImplementedError.
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
    TensorArg,
)
from torch._inductor.codegen.simd import (
    SIMDKernel,
    SIMDScheduling,
    IterationRangesRoot,
    IterationRangesEntry,
)
from torch._inductor.codegen.triton import (
    get_fused_kernel_name,
    get_kernel_metadata,
)
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import ReductionType, StoreMode, V


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

# Ascend UB (alloc_shared / alloc_ub) only supports fp16 and fp32.
_TILELANG_UB_DTYPES: frozenset[torch.dtype] = frozenset({torch.float16, torch.float32})

_TORCH_TO_TL_DTYPE: dict[torch.dtype, str] = {
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
    return _TORCH_TO_TL_DTYPE.get(dtype, "float32")


def _assert_ub_dtype(name: str, dtype: torch.dtype) -> None:
    """
    Raise NotImplementedError for dtypes not allocatable on Ascend UB memory.
    The inductor scheduler catches this and falls back to the Triton backend.
    """
    if dtype not in _TILELANG_UB_DTYPES:
        raise NotImplementedError(
            f"TileLang backend: Ascend UB supports float16/float32 only; "
            f"buffer '{name}' has dtype {dtype}."
        )


# ---------------------------------------------------------------------------
# CSE variable
# ---------------------------------------------------------------------------

class TileLangCSEVariable(CSEVariable):
    """Scalar CSE variable for expressions inside a T.Parallel body."""
    pass


# ---------------------------------------------------------------------------
# Op overrides
# ---------------------------------------------------------------------------

class TileLangOverrides(OpOverrides):
    """
    Map inductor element-wise ops to scalar Python expressions valid inside
    a ``T.Parallel`` loop body.

    Inside T.Parallel the compiler treats the body as scalar code and lowers
    it to NPU vector instructions automatically.  T.exp / T.sigmoid map to
    native hardware instructions; arithmetic uses plain Python operators.
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
        return repr(prim.dtype_to_type(dtype)(value))

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
    def tanh(x):       return f"_math.tanh({x})"
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
    def where(cond, a, b): return f"(({a}) if ({cond}) else ({b}))"
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
    def sign(x):  return f"(1 if ({x}) > 0 else (-1 if ({x}) < 0 else 0))"
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


class TileLangKernel(SIMDKernel):
    """
    Generates a TileLang @T.prim_func body for a fused set of pointwise nodes.

    Uses **Developer mode** (T.alloc_shared + T.Parallel) so that arbitrary
    fused element-wise expressions produced by inductor CSE are naturally
    lowered to NPU vector instructions by the TileLang compiler.

    T.Parallel index constraint: every buffer access inside the loop body
    uses the bare loop variable ``_tl_i`` with no transformation, satisfying
    the "下标不能做变换" restriction documented in T.Parallel.
    """

    overrides = TileLangOverrides  # type: ignore[assignment]
    # Python-style sympy printer — no tl.* type suffixes needed for TileLang.
    kexpr = SIMDKernel.sexpr  # type: ignore[assignment]

    def __init__(self, tiling: dict, **kwargs) -> None:
        super().__init__(tiling, **kwargs)
        self.cse: CSE = CSE(self.newvar_prefix, self.suffix)
        # buffer name -> (ptr_var, local_var_name, dtype)
        self._tl_inputs:  dict[str, tuple[str, str, torch.dtype]] = {}
        self._tl_outputs: dict[str, tuple[str, str, torch.dtype]] = {}

    # ------------------------------------------------------------------
    # SIMDKernel abstract interface
    # ------------------------------------------------------------------

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return tilelang_dtype(dtype)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        # T.Kernel + T.Parallel handle dispatch; suppress Triton-style
        # xoffset/xindex/xmask range-tree preamble entirely.
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
        variable referencing element ``_tl_i`` of the local shared buffer.

        The sympy *index* is intentionally ignored — flat 1-D contiguous
        layout is assumed.  Strided / non-contiguous access requires
        index-aware UB mapping (future work).
        """
        dtype = V.graph.get_dtype(name)
        _assert_ub_dtype(name, dtype)
        var = self.args.input(name)
        local_name = f"_{var}_local"
        if name not in self._tl_inputs:
            self._tl_inputs[name] = (var, local_name, dtype)
        return self.cse.generate(self.loads, f"{local_name}[_tl_i]", dtype=dtype)

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
        if mode == "atomic_add":
            raise NotImplementedError(
                "TileLang backend: atomic_add store not yet supported"
            )
        dtype = V.graph.get_dtype(name)
        _assert_ub_dtype(name, dtype)
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
            "this node will fall back to Triton."
        )

    # ------------------------------------------------------------------
    # Source generation
    # ------------------------------------------------------------------

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        """
        Return the @T.prim_func source string.

        TileLangScheduling.define_kernel() wraps this inside a per-shape
        factory so that ``_xnumel`` is a closure variable at compile time.

        T.copy syntax (offset form — copy size inferred from dst shape):
            T.copy(global_buf[cid * _XBLOCK], local_buf)   # GM -> UB
            T.copy(local_buf, global_buf[cid * _XBLOCK])   # UB -> GM
        """
        xblock       = _DEFAULT_XBLOCK
        prim_fn_name = f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn"

        argdefs, _, signature, _ = self.args.python_argdefs()

        # @T.prim_func parameter list: T.Tensor annotations for tensor args;
        # numel SizeArgs are closure-captured from the factory function.
        prim_sig_parts: list[str] = []
        for argdef, sig in zip(argdefs, signature):
            if isinstance(sig, TensorArg):
                prim_sig_parts.append(
                    f"{argdef.name}: T.Tensor((_xnumel,), '{tilelang_dtype(sig.dtype)}')"
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
                code.writeline(f"{part}{',' if i < len(prim_sig_parts) - 1 else ''}")
        code.writeline("):")

        with code.indent():
            # T.Kernel: exactly one block dimension required for NPU.
            code.writeline(
                "with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):"
            )
            with code.indent():
                # Inputs: T.alloc_shared (UB/cbuf) — supports bulk GM→UB DMA.
                for _, (var, loc, dtype) in self._tl_inputs.items():
                    code.writeline(
                        f"{loc} = T.alloc_shared((_XBLOCK,), '{tilelang_dtype(dtype)}')"
                    )
                # Outputs: T.alloc_fragment (register file) — supports scalar
                # writes inside T.Parallel.  Ascend UB (cbuf) does NOT allow
                # scalar element stores (hivm.hir.store limitation).
                # After T.Parallel the fragment is copied directly to GM.
                input_locs = {loc for _, loc, _ in self._tl_inputs.values()}
                for _, (var, loc, dtype) in self._tl_outputs.items():
                    if loc not in input_locs:
                        code.writeline(
                            f"{loc} = T.alloc_fragment((_XBLOCK,), '{tilelang_dtype(dtype)}')"
                        )
                code.writeline("")

                # T.copy GM -> UB for every input.
                for _, (var, loc, __) in self._tl_inputs.items():
                    code.writeline(f"T.copy({var}[cid * _XBLOCK], {loc})")
                code.writeline("")

                # T.Parallel compute body.
                # Constraint: every buffer subscript is the bare loop variable
                # _tl_i with no index transformation.
                code.writeline("for _tl_i in T.Parallel(_XBLOCK):")
                with code.indent():
                    loads_s   = self.loads.getvalue().strip()
                    compute_s = self.compute.getvalue().strip()
                    stores_s  = self.stores.getvalue().strip()
                    if loads_s:
                        code.splice(self.loads)
                    if compute_s:
                        code.splice(self.compute)
                    if stores_s:
                        code.splice(self.stores)
                    else:
                        code.writeline("pass")
                code.writeline("")

                # T.copy fragment -> GM for every output.
                # (T.alloc_fragment → GM direct copy is supported on Ascend.)
                for _, (var, loc, __) in self._tl_outputs.items():
                    code.writeline(f"T.copy({loc}, {var}[cid * _XBLOCK])")

        return code.getvalue()

    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None) -> None:
        """Emit ``name(tensor_arg0, ..., xnumel)`` in the generated wrapper."""
        wrapper = V.graph.wrapper_code
        _, call_args, signature, _ = self.args.python_argdefs()
        tensor_args = [a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)]
        numel_args  = [str(tree.numel) for tree in self.active_range_trees()]
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

    Registered via ``register_backend_for_device`` in __init__.py when
    ``TORCHINDUCTOR_NPU_BACKEND=tilelang``.
    """

    kernel_type: type[Any] = TileLangKernel

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

    def define_kernel(
        self,
        src_code: str,
        node_schedule,
        kernel: TileLangKernel,
    ) -> str:
        """
        Splice a shape-keyed caching wrapper into ``wrapper.header``.

        Pattern emitted at module level::

            import tilelang as _tilelang_<N>

            def _prim_factory_<name>(xnumel):
                # @T.prim_func definition (src_code)
                return <name>_prim_fn

            _<name>_cache = {}

            def <name>(in_ptr0, ..., xnumel):
                _key = (int(xnumel),)
                if _key not in _<name>_cache:
                    _<name>_cache[_key] = _tilelang_<N>.compile(
                        _prim_factory_<name>(_key[0]), target='npuir')
                _<name>_cache[_key](in_ptr0, ...)
        """
        wrapper = V.graph.wrapper_code

        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names else ""
        )
        suffix      = wrapper.next_kernel_suffix()
        kernel_name = "_".join(filter(None, ["tilelang", fused_name, suffix]))
        wrapper.src_to_kernel[src_code] = kernel_name

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, call_args, signature, _ = kernel.args.python_argdefs()
        tensor_call_args = [a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)]
        active_trees     = kernel.active_range_trees()
        numel_arg_names  = [f"{t.prefix}numel" for t in active_trees]
        outer_arg_list   = tensor_call_args + numel_arg_names

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        meta_comment = f"{origins}\n{detailed}".strip()

        # Make tilelang importable in generated output_code.py even without
        # a pre-configured sys.path.
        tl_pkg_root: Optional[str] = None
        try:
            import tilelang as _tl
            import os as _os
            tl_pkg_root = _os.path.dirname(_os.path.dirname(_tl.__file__))
        except ImportError:
            pass

        import_alias = f"_tilelang_{suffix}"
        cache_var    = f"_{kernel_name}_cache"
        factory_fn   = f"_prim_factory_{kernel_name}"
        prim_fn_name = f"{kernel_name}_prim_fn"

        code = IndentedBuffer()
        code.writeline(f"\n# TileLang kernel — {meta_comment}")
        if tl_pkg_root:
            code.writeline("import sys as _sys")
            code.writeline(
                f"if {tl_pkg_root!r} not in _sys.path: "
                f"_sys.path.insert(0, {tl_pkg_root!r})"
            )
        code.writeline(f"import tilelang as {import_alias}")
        code.writeline("")

        # Factory: captures _xnumel as closure variable for the @T.prim_func.
        factory_params = (
            ", ".join(numel_arg_names) if numel_arg_names else "_dummy=None"
        )
        code.writeline(f"def {factory_fn}({factory_params}):")
        with code.indent():
            code.writeline(f"_xnumel = {numel_arg_names[0]}" if numel_arg_names else "_xnumel = 1")
            code.splice(src_code)
            code.writeline(f"return {prim_fn_name}")

        code.writeline("")
        code.writeline(f"{cache_var} = {{}}")
        code.writeline("")

        code.writeline(f"def {kernel_name}({', '.join(outer_arg_list)}):")
        with code.indent():
            if numel_arg_names:
                code.writeline(
                    f"_key = ({', '.join(f'int({n})' for n in numel_arg_names)},)"
                )
            else:
                code.writeline("_key = ('static',)")
            code.writeline(f"if _key not in {cache_var}:")
            with code.indent():
                factory_call = (
                    ", ".join(f"_key[{i}]" for i in range(len(numel_arg_names)))
                    if numel_arg_names else ""
                )
                code.writeline(f"{cache_var}[_key] = {import_alias}.compile(")
                with code.indent():
                    code.writeline(f"{factory_fn}({factory_call}), target='npuir'")
                code.writeline(")")
            code.writeline(f"{cache_var}[_key]({', '.join(tensor_call_args)})")

        wrapper.header.splice(code.getvalue())
        return kernel_name
