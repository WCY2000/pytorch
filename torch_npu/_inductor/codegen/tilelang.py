"""
TileLang codegen backend for torch_npu inductor (Ascend NPU).

Generates TileLang @T.prim_func kernels compiled via
``tilelang.compile(..., target='npuir')``.

Activate with::

    TORCHINDUCTOR_NPU_BACKEND=tilelang

Generated kernel structure:

    @T.prim_func
    def <name>_prim_fn(
        in_ptr0: T.Tensor((_xnumel,), 'float32'),
        out_ptr0: T.Tensor((_xnumel,), 'float32'),
    ):
        with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (cid, _):
            _in_ptr0_local  = T.alloc_shared((_XBLOCK,), 'float32')  # L1/UB
            _out_ptr0_local = T.alloc_shared((_XBLOCK,), 'float32') # fragment
            T.copy(in_ptr0[cid * _XBLOCK], _in_ptr0_local)  # GM -> L1
            T.vadd(_in_ptr0_local, _in_ptr1_local, _out_ptr0_local)  # vector op
            T.copy(_out_ptr0_local, out_ptr0[cid * _XBLOCK])  # fragment -> GM

Why vector ops instead of T.Parallel scalar loops:
- On Ascend NPU, scalar element-wise stores to L1 (shared.dyn / cbuf) are not
  supported by the BiShengHIR pipeline ('hivm.hir.store' only allows DMA copies).
- T.alloc_fragment on NPU also maps to shared.dyn (same L1 memory), so scalar
  stores to fragment buffers also fail.
- The correct approach is to use TileLang's NPU vector intrinsics (T.vadd, T.vexp,
  etc.) which lower to hardware vector instructions (hivm.hir.vadd, etc.).

Op graph tracking:
- TileLangCSE wraps inductor's CSE and records what op produced each temp var.
- TileLangOverrides.* sets _pending_op on the kernel before returning the expr.
- TileLangKernel.load() records which local buffer each CSE var reads from.
- TileLangKernel.store() records which CSE var holds each output's value.
- codegen_kernel() traverses this graph and emits T.v* calls.

Known limitations:
- Only 1-D contiguous pointwise kernels; reduction() raises NotImplementedError.
- No tail guard when xnumel % _XBLOCK != 0.
- Ops without a T.v* equivalent raise NotImplementedError (fallback to Triton).
"""
from __future__ import annotations

import re
from typing import Any, Optional

import sympy
import torch
from torch._inductor import config, ir
from torch._inductor.codegen.common import (
    CSE,
    CSEVariable,
    IndentedBuffer,
    OpOverrides,
    TensorArg,
)
from torch._inductor.codegen.simd import (
    SIMDKernel,
    IterationRangesRoot,
    IterationRangesEntry,
)
from torch._inductor.codegen.triton import (
    get_fused_kernel_name,
    get_kernel_metadata,
)
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import ReductionType, StoreMode, V

# NPU-specific imports — resolved at class definition time to avoid circulars
from .triton import NPUIndexTritonKernel
from .scheduling import NPUTritonScheduling


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NPU vector op mappings  (op_name → (tilelang_fn, supported_dtypes))
#
# Dtype support sourced from tilelang-mlir-ascend/docs/Tilelang.language/
# Only dtypes reachable through inductor (no uint16 / uint32 / float64 paths).
# ---------------------------------------------------------------------------

_FP      = frozenset({torch.float16, torch.float32})
_FP_INT  = frozenset({torch.float16, torch.float32,
                      torch.int16, torch.int32, torch.int64})

# Binary vector ops: T.vXXX(A, B, C)  where B may be a scalar
_BINARY_VEC_OPS: dict[str, tuple[str, frozenset]] = {
    "add":         ("vadd", _FP),
    "sub":         ("vsub", _FP),
    "mul":         ("vmul", _FP_INT),
    "truediv":     ("vdiv", frozenset({torch.float16, torch.float32, torch.int64})),
    "maximum":     ("vmax", _FP),
    "minimum":     ("vmin", frozenset({torch.float16, torch.float32, torch.bfloat16,
                                       torch.int16, torch.int32, torch.int64})),
    # vpow: int32 only per hardware docs (fp16/fp32 cause MLIR verification failure).
    # All T.v* ops including vadd/vmul/vpow lower to AIV instructions, so there is
    # no AIC/AIV mixing issue; the dtype constraint is the only restriction.
    "pow":         ("vpow", frozenset({torch.int32})),
    "bitwise_and": ("vand", frozenset({torch.int8, torch.int64,
                                       torch.float16, torch.float32, torch.bool})),
    "bitwise_or":  ("vor",  frozenset()),   # uint16 only — not reachable via inductor
    "bitwise_xor": ("vxor", frozenset()),   # same
}

# Unary vector ops: T.vXXX(A, B)
_UNARY_VEC_OPS: dict[str, tuple[str, frozenset]] = {
    "exp":     ("vexp",     _FP),
    "log":     ("vln",      _FP),
    "exp2":    ("vexp2",    _FP),
    "log2":    ("vlog2",    _FP),
    "relu":    ("vrelu",    _FP),
    "sigmoid": ("vsigmoid", _FP),
    "sqrt":    ("vsqrt",    _FP),
    "rsqrt":   ("vrsqrt",   _FP),
    "abs":     ("vabs",     frozenset({torch.float16, torch.float32,
                                       torch.uint8, torch.int32, torch.int64})),
    "cos":     ("vcos",     _FP),
    "sin":     ("vsin",     _FP),
    "erf":     ("verf",     _FP),
    "tanh":    ("vtanh",    _FP),
}

# Union of all dtypes supported by at least one op — used as early gate in load().
_ANY_SUPPORTED_DTYPE: frozenset[torch.dtype] = frozenset().union(
    *[s for _, s in _BINARY_VEC_OPS.values()],
    *[s for _, s in _UNARY_VEC_OPS.values()],
    {torch.float16, torch.float32},   # always include for neg (vmul fallback)
)


# ---------------------------------------------------------------------------
# Expression → op parser
# ---------------------------------------------------------------------------

def _parse_expr_op(expr: str) -> Optional[tuple[str, list[str]]]:
    """
    Parse a scalar expression string emitted by TileLangOverrides and return
    (op_name, [operand_str, ...]) or None if unrecognised.

    Since inductor's CSE assigns a fresh tmp var to each compound expression,
    both operands of a binary op are always simple identifiers or number
    literals at the time this is called.
    """
    s = expr.strip()

    # --- unary: (-x) ---
    m = re.match(r'^\(-(\w+)\)$', s)
    if m:
        return ("neg", [m.group(1)])

    # --- unary: abs(x) ---
    m = re.match(r'^abs\((\w+)\)$', s)
    if m:
        return ("abs", [m.group(1)])

    # --- unary: T.exp(x), T.sigmoid(x) ---
    m = re.match(r'^T\.(\w+)\((\w+)\)$', s)
    if m:
        fn = m.group(1)
        arg = m.group(2)
        if fn in ("exp", "sigmoid"):
            return (fn, [arg])

    # --- unary: _math.xxx(x) ---
    _math_unary = {
        "log": "log", "log2": "log2", "log1p": "log1p",
        "sqrt": "sqrt", "sin": "sin", "cos": "cos",
        "tan": "tan", "tanh": "tanh",
        "asin": "asin", "acos": "acos", "atan": "atan",
        "erf": "erf", "erfc": "erfc",
        "floor": "floor", "ceil": "ceil", "trunc": "trunc",
    }
    m = re.match(r'^_math\.(\w+)\((\w+)\)$', s)
    if m and m.group(1) in _math_unary:
        return (_math_unary[m.group(1)], [m.group(2)])

    # --- relu: ((x) if (x) > 0.0 else 0.0) ---
    m = re.match(r'^\(\((\w+)\) if \(\1\) > 0\.0 else 0\.0\)$', s)
    if m:
        return ("relu", [m.group(1)])

    # --- binary: _math.pow(a, b) and _math.atan2(a, b) ---
    m = re.match(r'^_math\.pow\((\w+),\s*(\S+)\)$', s)
    if m:
        return ("pow", [m.group(1), m.group(2)])
    m = re.match(r'^_math\.atan2\((\w+),\s*(\w+)\)$', s)
    if m:
        return ("atan2", [m.group(1), m.group(2)])

    # --- binary: (a OP b) ---
    _bin_patterns: list[tuple[str, str]] = [
        (r'^\((\w+) \+ (\S+)\)$',  "add"),
        (r'^\((\w+) - (\S+)\)$',   "sub"),
        (r'^\((\w+) \* (\S+)\)$',  "mul"),
        (r'^\((\w+) / (\S+)\)$',   "truediv"),
        (r'^\((\w+) // (\S+)\)$',  "floordiv"),
        (r'^\((\w+) % (\S+)\)$',   "mod"),
        (r'^\((\w+) & (\S+)\)$',   "bitwise_and"),
        (r'^\((\w+) \| (\S+)\)$',  "bitwise_or"),
        (r'^\((\w+) \^ (\S+)\)$',  "bitwise_xor"),
        (r'^\((\w+) < (\S+)\)$',   "lt"),
        (r'^\((\w+) <= (\S+)\)$',  "le"),
        (r'^\((\w+) > (\S+)\)$',   "gt"),
        (r'^\((\w+) >= (\S+)\)$',  "ge"),
        (r'^\((\w+) == (\S+)\)$',  "eq"),
        (r'^\((\w+) != (\S+)\)$',  "ne"),
    ]
    for pattern, op in _bin_patterns:
        m = re.match(pattern, s)
        if m:
            return (op, [m.group(1), m.group(2)])

    return None


# ---------------------------------------------------------------------------
# CSE variable
# ---------------------------------------------------------------------------

class TileLangCSEVariable(CSEVariable):
    pass


# ---------------------------------------------------------------------------
# Op overrides
# ---------------------------------------------------------------------------

def _set_pending(op: str, operands: list, dtype: Optional[torch.dtype] = None) -> None:
    """Set _pending_op on the current TileLangKernel (called from overrides)."""
    try:
        k = V.kernel
        if isinstance(k, TileLangKernel):
            k._pending_op = (op, operands, dtype)
    except AttributeError:
        pass


class TileLangOverrides(OpOverrides):
    """
    Maps inductor element-wise ops to scalar Python expressions (used for
    the string CSE) and simultaneously records the op type on the kernel
    for NPU vector code emission.
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
        literal = repr(prim.dtype_to_type(dtype)(value))
        _set_pending("const", [literal])
        return literal

    @staticmethod
    def abs(x):
        _set_pending("abs", [x])
        return f"abs({x})"

    @staticmethod
    def neg(x):
        _set_pending("neg", [x])
        return f"(-{x})"

    @staticmethod
    def exp(x):
        _set_pending("exp", [x])
        return f"T.exp({x})"

    @staticmethod
    def exp2(x):
        _set_pending("exp2", [x])
        return f"_math.pow(2.0, {x})"

    @staticmethod
    def expm1(x):
        # No single NPU intrinsic; will be caught as unsupported during emit
        return f"(T.exp({x}) - 1.0)"

    @staticmethod
    def log(x):
        _set_pending("log", [x])
        return f"_math.log({x})"

    @staticmethod
    def log2(x):
        _set_pending("log2", [x])
        return f"_math.log2({x})"

    @staticmethod
    def log1p(x):
        return f"_math.log1p({x})"

    @staticmethod
    def sqrt(x):
        _set_pending("sqrt", [x])
        return f"_math.sqrt({x})"

    @staticmethod
    def rsqrt(x):
        _set_pending("rsqrt", [x])
        return f"(1.0 / _math.sqrt({x}))"

    @staticmethod
    def sin(x):
        _set_pending("sin", [x])
        return f"_math.sin({x})"

    @staticmethod
    def cos(x):
        _set_pending("cos", [x])
        return f"_math.cos({x})"

    @staticmethod
    def tan(x):
        _set_pending("tan", [x])
        return f"_math.tan({x})"

    @staticmethod
    def tanh(x):
        _set_pending("tanh", [x])
        return f"_math.tanh({x})"

    @staticmethod
    def asin(x):
        _set_pending("asin", [x])
        return f"_math.asin({x})"

    @staticmethod
    def acos(x):
        _set_pending("acos", [x])
        return f"_math.acos({x})"

    @staticmethod
    def atan(x):
        _set_pending("atan", [x])
        return f"_math.atan({x})"

    @staticmethod
    def atan2(x, y):
        _set_pending("atan2", [x, y])
        return f"_math.atan2({x}, {y})"

    @staticmethod
    def sigmoid(x):
        _set_pending("sigmoid", [x])
        return f"T.sigmoid({x})"

    @staticmethod
    def relu(x):
        _set_pending("relu", [x])
        return f"(({x}) if ({x}) > 0.0 else 0.0)"

    @staticmethod
    def minimum(a, b):
        _set_pending("minimum", [a, b])
        return f"(({a}) if ({a}) < ({b}) else ({b}))"

    @staticmethod
    def maximum(a, b):
        _set_pending("maximum", [a, b])
        return f"(({a}) if ({a}) > ({b}) else ({b}))"

    @staticmethod
    def where(cond, a, b):
        return f"(({a}) if ({cond}) else ({b}))"

    @staticmethod
    def add(a, b):
        _set_pending("add", [a, b])
        return f"({a} + {b})"

    @staticmethod
    def sub(a, b):
        _set_pending("sub", [a, b])
        return f"({a} - {b})"

    @staticmethod
    def mul(a, b):
        _set_pending("mul", [a, b])
        return f"({a} * {b})"

    @staticmethod
    def truediv(a, b):
        _set_pending("truediv", [a, b])
        return f"({a} / {b})"

    @staticmethod
    def floordiv(a, b):
        _set_pending("floordiv", [a, b])
        return f"({a} // {b})"

    @staticmethod
    def mod(a, b):
        _set_pending("mod", [a, b])
        return f"({a} % {b})"

    @staticmethod
    def pow(a, b):
        _set_pending("pow", [a, b])
        return f"_math.pow({a}, {b})"

    @staticmethod
    def logical_not(a):    return f"(not ({a}))"
    @staticmethod
    def logical_and(a, b): return f"(({a}) and ({b}))"
    @staticmethod
    def logical_or(a, b):  return f"(({a}) or ({b}))"
    @staticmethod
    def logical_xor(a, b): return f"(bool({a}) != bool({b}))"

    @staticmethod
    def bitwise_and(a, b):
        _set_pending("bitwise_and", [a, b])
        return f"(({a}) & ({b}))"

    @staticmethod
    def bitwise_or(a, b):
        _set_pending("bitwise_or", [a, b])
        return f"(({a}) | ({b}))"

    @staticmethod
    def bitwise_xor(a, b):
        _set_pending("bitwise_xor", [a, b])
        return f"(({a}) ^ ({b}))"

    @staticmethod
    def bitwise_not(a):    return f"(~({a}))"

    @staticmethod
    def sign(x):  return f"(1 if ({x}) > 0 else (-1 if ({x}) < 0 else 0))"

    @staticmethod
    def floor(x):
        _set_pending("floor", [x])
        return f"_math.floor({x})"

    @staticmethod
    def ceil(x):
        _set_pending("ceil", [x])
        return f"_math.ceil({x})"

    @staticmethod
    def trunc(x):
        _set_pending("trunc", [x])
        return f"_math.trunc({x})"

    @staticmethod
    def erf(x):
        _set_pending("erf", [x])
        return f"_math.erf({x})"

    @staticmethod
    def erfc(x):
        _set_pending("erfc", [x])
        return f"_math.erfc({x})"

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


def _resolve_operand(operand) -> str:
    """Convert a CSEVariable or string operand to a string suitable for T.v* calls."""
    if isinstance(operand, (int, float)):
        return repr(operand)
    return str(operand)


def _build_vec_ops(
    var,
    target_buf: str,
    ops_list: list,
    var_bufs: dict,
    var_ops: dict,
    var_consts: Optional[dict] = None,
    _visited: Optional[set] = None,
) -> str:
    """
    Recursively traverse the op graph rooted at `var` and append
    (op_name, resolved_operands, out_buf) tuples to `ops_list` in
    topological (post) order.

    Returns the buffer name that will hold the result of `var`.
    For input-buffer vars this is the existing local buffer name.
    For computed vars this is `target_buf`.
    """
    if _visited is None:
        _visited = set()

    var_name = str(var)

    # Direct input buffer
    if var_name in var_bufs:
        return var_bufs[var_name]

    # Constant literal that went through CSE (e.g. tmp1 = 2.0)
    if var_consts and var_name in var_consts:
        try:
            return float(var_consts[var_name])
        except (ValueError, TypeError):
            return var_consts[var_name]

    # Computed var
    if var_name in var_ops:
        if var_name in _visited:
            return f"_{var_name}_frag"

        _visited.add(var_name)
        op_name, operands = var_ops[var_name]
        resolved = []
        for op in operands:
            op_str = str(op)
            if op_str in var_bufs:
                resolved.append(var_bufs[op_str])
            elif var_consts and op_str in var_consts:
                try:
                    resolved.append(float(var_consts[op_str]))
                except (ValueError, TypeError):
                    resolved.append(var_consts[op_str])
            elif op_str in var_ops:
                inter_buf = f"_{op_str}_frag"
                src = _build_vec_ops(op, inter_buf, ops_list, var_bufs, var_ops, var_consts, _visited)
                resolved.append(src)
            else:
                # Raw literal string (e.g. "2.0" passed directly without going through CSE)
                try:
                    resolved.append(float(op_str))
                except (ValueError, TypeError):
                    resolved.append(op_str)
        ops_list.append((op_name, resolved, target_buf))
        return target_buf

    # Fallback: treat as a literal / unknown symbol
    return var_name


# TileLang reduction op names (ReductionType string → T.* function name)
_REDUCE_OPS: dict[str, str] = {
    "sum":     "reduce_sum",
    "max":     "reduce_max",
    "min":     "reduce_min",
    "any":     "reduce_or",
    "prod":    "reduce_prod",
    "xor_sum": "reduce_xor",
    "argmax":  "reduce_max",   # index extraction handled separately
    "argmin":  "reduce_min",
}


class TileLangKernel(NPUIndexTritonKernel):
    """
    Generates a TileLang @T.prim_func body for fused pointwise and reduction
    nodes targeting Ascend NPU.

    Inherits NPUIndexTritonKernel so that NPUTritonScheduling can run
    decide_codegen_dims_in_kernel (SplitTiling / ReductionAnalysis) on this
    kernel before codegen, populating split_axis / tiling_axis / sorted_axis.

    codegen_kernel() then uses that axis metadata to emit correct multi-dim
    T.copy offsets and T.reduce calls instead of Triton tl.* primitives.
    """

    overrides = TileLangOverrides  # type: ignore[assignment]
    kexpr = SIMDKernel.sexpr       # type: ignore[assignment]

    def __init__(self, tiling: dict, **kwargs) -> None:
        super().__init__(tiling=tiling, **kwargs)
        # Override with plain CSE so create_cse_var returns TileLangCSEVariable.
        self.cse: CSE = CSE(self.newvar_prefix, self.suffix)

        # Buffer registry
        self._tl_inputs:  dict[str, tuple[str, str, torch.dtype]] = {}
        self._tl_outputs: dict[str, tuple[str, str, torch.dtype]] = {}
        # Sympy index expression recorded at first load/store of each buffer
        self._tl_input_indices:  dict[str, sympy.Expr] = {}
        self._tl_output_indices: dict[str, sympy.Expr] = {}

        # Op graph (built during load/store/overrides calls)
        self._pending_op: Optional[tuple] = None
        self._var_ops:    dict[str, tuple] = {}   # var_name → (op, operands)
        self._var_bufs:   dict[str, str] = {}     # var_name → local_buf_name
        self._var_consts: dict[str, str] = {}     # var_name → literal string (e.g. "2.0")
        self._var_dtypes: dict[str, torch.dtype] = {}
        self._output_vars: dict[str, tuple] = {}  # local_buf_name → (var, dtype)
        self._is_reduction_output: dict[str, bool] = {}  # local_buf_name → bool

    # ------------------------------------------------------------------
    # SIMDKernel / NPUIndexTritonKernel interface overrides
    # ------------------------------------------------------------------

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return tilelang_dtype(dtype)

    def should_use_persistent_reduction(self) -> bool:
        # Always use persistent reduction — non-persistent raises NotImplementedError
        # in _codegen_reduction_kernel so the caller falls back to Triton.
        return True

    def should_use_cooperative_reduction(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # load / store / reduction
    # ------------------------------------------------------------------

    def load(self, name: str, index: sympy.Expr) -> TileLangCSEVariable:
        dtype = V.graph.get_dtype(name)
        if dtype not in _ANY_SUPPORTED_DTYPE:
            raise NotImplementedError(
                f"TileLang backend: no T.v* op supports dtype {dtype} "
                f"(buffer '{name}'); falling back to Triton."
            )
        var = self.args.input(name)
        local_name = f"_{var}_local"
        if name not in self._tl_inputs:
            self._tl_inputs[name] = (var, local_name, dtype)
            self._tl_input_indices[name] = index
        # Clear stale _pending_op so load vars are never mis-attributed.
        self._pending_op = None
        cse_var = self.cse.generate(self.loads, f"{local_name}[_tl_i]", dtype=dtype)
        self._var_bufs[cse_var.name] = local_name
        self._var_dtypes[cse_var.name] = dtype
        return cse_var

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: TileLangCSEVariable,
        mode: StoreMode = None,
    ) -> None:
        if mode == "atomic_add":
            raise NotImplementedError(
                "TileLang backend: atomic_add store not yet supported"
            )
        dtype = V.graph.get_dtype(name)
        var = self.args.output(name)
        local_name = f"_{var}_local"
        if name not in self._tl_outputs:
            self._tl_outputs[name] = (var, local_name, dtype)
            self._tl_output_indices[name] = index
        self._check_op_graph_dtype(str(value), dtype)
        self._output_vars[local_name] = (value, dtype)
        self.stores.writeline(f"{local_name}[_tl_i] = {value}")

    def store_reduction(self, name: str, index: sympy.Expr, value: TileLangCSEVariable) -> None:
        dtype = V.graph.get_dtype(name)
        var = self.args.output(name)
        local_name = f"_{var}_local"
        if name not in self._tl_outputs:
            self._tl_outputs[name] = (var, local_name, dtype)
            self._tl_output_indices[name] = index
        self._output_vars[local_name] = (value, dtype)
        self._is_reduction_output[local_name] = True

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: TileLangCSEVariable,
    ) -> TileLangCSEVariable:
        if not self.inside_reduction:
            raise RuntimeError("reduction() called outside reduction context")
        if not self.persistent_reduction:
            raise NotImplementedError(
                "TileLang backend: non-persistent reduction not yet implemented; "
                "falling back to Triton."
            )
        result_var = self.cse.newvar(dtype=dtype)
        rt = str(reduction_type).split(".")[-1].lower()  # e.g. "ReductionType.SUM" → "sum"
        self._var_ops[result_var.name] = ("reduce", rt, str(value))
        self.outside_loop_vars.add(result_var)
        return result_var

    # ------------------------------------------------------------------
    # Index offset helpers
    # ------------------------------------------------------------------

    def _axis_syms(self):
        """Return (x_syms, r_syms) as dicts {sympy_symbol: IterationRangesEntry}."""
        x_syms: dict = {}
        r_syms: dict = {}
        for sym, node in self.range_tree_nodes.items():
            if node.name[0] == "r":
                r_syms[sym] = node
            else:
                x_syms[sym] = node
        return x_syms, r_syms

    def _build_copy_offset(
        self,
        index: sympy.Expr,
        x_syms: dict,
        r_syms: dict,
        grid_vars: dict,       # sym → sympy.Symbol  (the replacement grid variable)
        r_start: sympy.Expr,   # replacement for all r-syms (0 for persistent)
    ) -> str:
        """Substitute axis symbols in *index* with grid/block variables and stringify."""
        subs: dict = {r: r_start for r in r_syms}
        subs.update(grid_vars)
        result = index.subs(subs)
        return self.index_to_str(result)

    # ------------------------------------------------------------------
    # Source generation
    # ------------------------------------------------------------------

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        if self.inside_reduction:
            return self._codegen_reduction_kernel(name)
        return self._codegen_pointwise_kernel(name)

    def _prim_sig_parts(self, argdefs, signature):
        parts = []
        for argdef, sig in zip(argdefs, signature):
            if isinstance(sig, TensorArg):
                parts.append(
                    f"{argdef.name}: T.Tensor((_xnumel,), '{tilelang_dtype(sig.dtype)}')"
                )
        return parts

    def _emit_vec_ops_block(self, code, out_loc, result_var, dtype, already_allocated, block_size_str):
        """Traverse op graph and emit T.v* calls into *code*."""
        ops_list: list[tuple] = []
        _build_vec_ops(
            result_var, out_loc, ops_list,
            self._var_bufs, self._var_ops, self._var_consts,
        )
        if not ops_list:
            src = self._var_bufs.get(str(result_var), str(result_var))
            if src != out_loc:
                code.writeline(f"T.copy({src}, {out_loc})")
            return
        last_op, last_operands, _ = ops_list[-1]
        ops_list[-1] = (last_op, last_operands, out_loc)
        for op_name, operands, out_buf in ops_list:
            if out_buf not in already_allocated:
                code.writeline(
                    f"{out_buf} = T.alloc_shared(({block_size_str},), '{tilelang_dtype(dtype)}')"
                )
                already_allocated.add(out_buf)
            code.writeline(self._emit_vec_op(op_name, operands, out_buf))

    def _codegen_pointwise_kernel(self, name: Optional[str] = None) -> str:
        """
        Emit a TileLang pointwise kernel.

        Grid layout derived from axis metadata set by decide_codegen_dims_in_kernel:
          - split_axis[i]  → outer grid dim i  (each AI Core owns one "row")
          - tiling_axis[-1] (no_loop) → inner blocked dim, size _XBLOCK

        Flat 1-D fallback is used when axis metadata is unavailable (e.g. very
        first 1-D kernel before the NPUTritonScheduling path is activated).
        """
        xblock       = _DEFAULT_XBLOCK
        prim_fn_name = f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn"

        argdefs, _, signature, _ = self.args.python_argdefs()
        prim_sig_parts = self._prim_sig_parts(argdefs, signature)

        # ---- axis metadata (populated by decide_codegen_dims_in_kernel) ----
        split_axes   = getattr(self, "split_axis", [])
        tiling_axes  = getattr(self, "tiling_axis", [])

        # Innermost no-loop tiling axis → the vectorised XBLOCK dimension.
        inner_axis = next((ax for ax in reversed(tiling_axes) if ax.is_no_loop_axis), None)
        outer_loop_axes = [ax for ax in tiling_axes if not ax.is_no_loop_axis]

        # Build grid and the symbol→replacement mapping used for T.copy offsets.
        x_syms, r_syms = self._axis_syms()

        if split_axes and inner_axis and inner_axis is not split_axes[0]:
            # 2-D (or higher) layout:  split dims → row grid,  inner → col grid
            grid_var_syms: list[str] = []
            grid_parts:   list[str] = []
            sym_to_grid:  dict = {}

            for i, ax in enumerate(split_axes):
                gv = f"_gs{i}"
                grid_var_syms.append(gv)
                grid_parts.append(str(ax.length))          # one core per element
                sym_to_grid[ax.symbol()] = sympy.Symbol(gv)

            # Inner (no-loop) tiling axis → blocked grid dim
            inner_gv = "_gi"
            grid_var_syms.append(inner_gv)
            grid_parts.append(f"T.ceildiv({inner_axis.length}, _XBLOCK)")
            sym_to_grid[inner_axis.symbol()] = sympy.Symbol(inner_gv) * xblock

            grid_str = ", ".join(grid_parts)
            grid_vars_str = ", ".join(grid_var_syms)

            def copy_offset(name_):
                idx = self._tl_input_indices.get(name_) or self._tl_output_indices.get(name_)
                if idx is None:
                    # fallback: last split axis * inner_length + inner * xblock
                    return f"_gs0 * {inner_axis.length} + _gi * _XBLOCK"
                return self._build_copy_offset(idx, x_syms, r_syms, sym_to_grid, sympy.Integer(0))

        else:
            # 1-D layout (or single split-tiling axis): simple cid * _XBLOCK
            grid_str = "T.ceildiv(_xnumel, _XBLOCK)"
            grid_vars_str = "cid, _"

            def copy_offset(_name):  # type: ignore[misc]
                return "cid * _XBLOCK"

        # ---- assemble prim_func ----
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
            code.writeline(f"with T.Kernel({grid_str}, is_npu=True) as ({grid_vars_str}):")
            with code.indent():
                # allocate input buffers
                for nm, (var, loc, dtype) in self._tl_inputs.items():
                    code.writeline(f"{loc} = T.alloc_shared((_XBLOCK,), '{tilelang_dtype(dtype)}')")

                # allocate output buffers not already allocated as inputs
                input_locs = {loc for _, loc, _ in self._tl_inputs.values()}
                for nm, (var, loc, dtype) in self._tl_outputs.items():
                    if loc not in input_locs:
                        code.writeline(f"{loc} = T.alloc_shared((_XBLOCK,), '{tilelang_dtype(dtype)}')")
                code.writeline("")

                # GM → L1
                for nm, (var, loc, _) in self._tl_inputs.items():
                    off = copy_offset(nm)
                    code.writeline(f"T.copy({var}[{off}], {loc})")
                code.writeline("")

                # vector ops
                already_allocated: set = (
                    {loc for _, loc, _ in self._tl_inputs.values()}
                    | {loc for _, loc, _ in self._tl_outputs.values()}
                )
                for out_loc, (result_var, dtype) in self._output_vars.items():
                    self._emit_vec_ops_block(code, out_loc, result_var, dtype, already_allocated, "_XBLOCK")

                code.writeline("")

                # L1 → GM
                for nm, (var, loc, _) in self._tl_outputs.items():
                    off = copy_offset(nm)
                    code.writeline(f"T.copy({loc}, {var}[{off}])")

        src = code.getvalue()
        print("====== TileLang pointwise prim_func ======")
        print(src)
        return src

    def _codegen_reduction_kernel(self, name: Optional[str] = None) -> str:
        """
        Emit a TileLang persistent-reduction kernel.

        Layout:
          Grid  = xnumel  (one AI Core per output element)
          Input = T.copy(in_ptr[cid * _rnumel], _in_local)  size=rnumel
          Reduce= T.reduce_{op}(_in_local, _out_local, dim=0)
          Output= T.copy(_out_local, out_ptr[cid])

        Non-persistent reduction raises NotImplementedError (→ Triton fallback).
        """
        if not self.persistent_reduction:
            raise NotImplementedError(
                "TileLang backend: non-persistent reduction not implemented; "
                "falling back to Triton."
            )

        prim_fn_name = f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn"
        argdefs, _, signature, _ = self.args.python_argdefs()
        prim_sig_parts = self._prim_sig_parts(argdefs, signature)

        x_syms, r_syms = self._axis_syms()
        _cid  = sympy.Symbol("cid")
        _zero = sympy.Integer(0)

        # Determine r_numel as a code string
        r_nodes = list(r_syms.values())
        if r_nodes:
            simplified = V.graph.sizevars.simplify(r_nodes[0].length)
            r_numel_str = str(int(simplified)) if isinstance(simplified, (sympy.Integer, int)) else "_rnumel"
        else:
            r_numel_str = "1"

        # Substitution maps for input/output offset computation
        x_to_grid = {s: _cid for s in x_syms}
        r_to_zero = {s: _zero for s in r_syms}

        def in_offset(nm):
            idx = self._tl_input_indices.get(nm)
            if idx is None:
                return f"cid * {r_numel_str}"
            return self._build_copy_offset(idx, x_syms, r_syms, {**x_to_grid, **r_to_zero}, _zero)

        def out_offset(nm):
            idx = self._tl_output_indices.get(nm)
            if idx is None:
                return "cid"
            return self._build_copy_offset(idx, x_syms, r_syms, {**x_to_grid, **r_to_zero}, _zero)

        # ---- assemble prim_func ----
        code = IndentedBuffer()
        code.writeline("import tilelang.language as T")
        code.writeline("import math as _math")
        code.writeline("")
        code.writeline("@T.prim_func")
        code.writeline(f"def {prim_fn_name}(")
        with code.indent():
            for i, part in enumerate(prim_sig_parts):
                code.writeline(f"{part}{',' if i < len(prim_sig_parts) - 1 else ''}")
        code.writeline("):")

        with code.indent():
            code.writeline("with T.Kernel(_xnumel, is_npu=True) as (cid, _):")
            with code.indent():
                # allocate input (full r-dim per core)
                for nm, (var, loc, dtype) in self._tl_inputs.items():
                    code.writeline(f"{loc} = T.alloc_shared(({r_numel_str},), '{tilelang_dtype(dtype)}')")

                # allocate output (scalar per core)
                input_locs = {loc for _, loc, _ in self._tl_inputs.values()}
                for nm, (var, loc, dtype) in self._tl_outputs.items():
                    if loc not in input_locs:
                        code.writeline(f"{loc} = T.alloc_shared((1,), '{tilelang_dtype(dtype)}')")
                code.writeline("")

                # GM → L1: load full reduction slice
                for nm, (var, loc, _) in self._tl_inputs.items():
                    off = in_offset(nm)
                    code.writeline(f"T.copy({var}[{off}], {loc})")
                code.writeline("")

                # Reduction ops
                already_allocated: set = (
                    {loc for _, loc, _ in self._tl_inputs.values()}
                    | {loc for _, loc, _ in self._tl_outputs.values()}
                )
                for out_loc, (result_var, dtype) in self._output_vars.items():
                    is_red = self._is_reduction_output.get(out_loc, False)
                    if is_red and result_var.name in self._var_ops:
                        op_entry = self._var_ops[result_var.name]
                        if op_entry[0] == "reduce":
                            _, rt, in_var_str = op_entry
                            src_buf = self._var_bufs.get(in_var_str, f"_{in_var_str}_local")
                            tl_fn = _REDUCE_OPS.get(rt, "reduce_sum")
                            code.writeline(f"T.{tl_fn}({src_buf}, {out_loc}, dim=0)")
                    else:
                        # pointwise epilogue fused with reduction output
                        self._emit_vec_ops_block(code, out_loc, result_var, dtype, already_allocated, "1")
                code.writeline("")

                # L1 → GM: store scalar result
                for nm, (var, loc, _) in self._tl_outputs.items():
                    off = out_offset(nm)
                    code.writeline(f"T.copy({loc}, {var}[{off}])")

        src = code.getvalue()
        print("====== TileLang reduction prim_func ======")
        print(src)
        return src

    def _check_op_graph_dtype(
        self,
        var_name: str,
        dtype: torch.dtype,
        _visited: Optional[set] = None,
    ) -> None:
        """
        Walk the op graph from var_name and raise NotImplementedError if any
        op does not support `dtype`.  Called from store() so it executes
        inside _body(*index_vars) and triggers inductor's Triton fallback.
        """
        if _visited is None:
            _visited = set()
        if var_name in _visited:
            return
        _visited.add(var_name)

        if var_name not in self._var_ops:
            return  # input buffer or constant — no op to check

        op_name, operands = self._var_ops[var_name]

        if op_name in _BINARY_VEC_OPS:
            _, supported = _BINARY_VEC_OPS[op_name]
            if dtype not in supported:
                raise NotImplementedError(
                    f"TileLang NPU: op '{op_name}' (T.{_BINARY_VEC_OPS[op_name][0]}) "
                    f"does not support dtype {dtype}; supported: {supported}. "
                    f"Falling back to Triton."
                )
        elif op_name in _UNARY_VEC_OPS:
            _, supported = _UNARY_VEC_OPS[op_name]
            if dtype not in supported:
                raise NotImplementedError(
                    f"TileLang NPU: op '{op_name}' (T.{_UNARY_VEC_OPS[op_name][0]}) "
                    f"does not support dtype {dtype}; supported: {supported}. "
                    f"Falling back to Triton."
                )
        elif op_name == "neg":
            _, supported = _BINARY_VEC_OPS["mul"]
            if dtype not in supported:
                raise NotImplementedError(
                    f"TileLang NPU: neg (→vmul×-1) does not support dtype {dtype}. "
                    f"Falling back to Triton."
                )

        for op in operands:
            self._check_op_graph_dtype(str(op), dtype, _visited)

    @staticmethod
    def _emit_vec_op(op_name: str, operands: list, out_buf: str) -> str:
        """Return the T.v* call string for one vector operation."""
        if op_name in _BINARY_VEC_OPS:
            vec_fn, _ = _BINARY_VEC_OPS[op_name]
            a = _resolve_operand(operands[0])
            b = _resolve_operand(operands[1])
            return f"T.{vec_fn}({a}, {b}, {out_buf})"

        if op_name in _UNARY_VEC_OPS:
            vec_fn, _ = _UNARY_VEC_OPS[op_name]
            a = _resolve_operand(operands[0])
            return f"T.{vec_fn}({a}, {out_buf})"

        # neg: implement as vmul(x, -1.0, out)
        if op_name == "neg":
            a = _resolve_operand(operands[0])
            return f"T.vmul({a}, -1.0, {out_buf})"

        raise NotImplementedError(
            f"TileLang NPU backend: op '{op_name}' has no T.v* equivalent. "
            f"This kernel will fall back to Triton."
        )

    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None) -> None:
        wrapper = V.graph.wrapper_code
        _, call_args, signature, _ = self.args.python_argdefs()
        tensor_args = [a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)]
        numel_args  = [str(tree.numel) for tree in self.active_range_trees()]
        wrapper.writeline(f"{name}({', '.join(tensor_args + numel_args)})")

    def create_cse_var(self, name, bounds=None, dtype=None) -> TileLangCSEVariable:
        var = TileLangCSEVariable(name, bounds)
        if dtype is not None:
            self._var_dtypes[name] = dtype
        # Consume any pending op set by TileLangOverrides.*
        # _pending_op is cleared in load() before load-expr generate calls,
        # so only compute-expression vars pick it up here.
        if self._pending_op is not None:
            op_name, operands = self._pending_op[0], self._pending_op[1]
            if op_name == "const":
                self._var_consts[name] = operands[0]  # e.g. "tmp1" → "2.0"
            else:
                self._var_ops[name] = (op_name, operands)
            self._pending_op = None
        return var

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class TileLangScheduling(NPUTritonScheduling):
    """
    Inductor scheduling backend that emits TileLang kernels for Ascend NPU.

    Inherits NPUTritonScheduling so that the full decide_codegen_dims_in_kernel
    pipeline (SplitTiling / ReductionAnalysis) runs before codegen, populating
    axis metadata on TileLangKernel.  Only define_kernel and codegen_sync are
    overridden; everything else (codegen_node_schedule, codegen_comment, …) is
    inherited from NPUTritonScheduling / TritonScheduling / SIMDScheduling.

    Activated via ``TORCHINDUCTOR_NPU_BACKEND=tilelang``.
    """

    kernel_type: type[Any] = TileLangKernel

    def codegen_sync(self) -> None:
        V.graph.wrapper_code.writeline("torch.npu.synchronize()")

    def define_kernel(
        self,
        src_code: str,
        node_schedule,
        kernel: TileLangKernel,
        traced_graph_hash: Optional[str] = None,
    ) -> tuple[str, str]:
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
        cache_key = (src_code, traced_graph_hash)

        if cache_key in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[cache_key]
            return kernel_name, src_code

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names else ""
        )
        suffix      = wrapper.next_kernel_suffix()
        kernel_name = "_".join(filter(None, ["tilelang", fused_name, suffix]))
        wrapper.src_to_kernel[cache_key] = kernel_name

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, call_args, signature, _ = kernel.args.python_argdefs()
        tensor_call_args = [a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)]
        active_trees     = kernel.active_range_trees()
        numel_arg_names  = [f"{t.prefix}numel" for t in active_trees]
        outer_arg_list   = tensor_call_args + numel_arg_names

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        meta_comment = f"{origins}\n{detailed}".strip()

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

        factory_params = (
            ", ".join(numel_arg_names) if numel_arg_names else "_dummy=None"
        )
        code.writeline(f"def {factory_fn}({factory_params}):")
        with code.indent():
            # Expose every numel as _<prefix>numel inside the factory closure
            # so the prim_func body can reference _xnumel, _rnumel, etc.
            if numel_arg_names:
                for n in numel_arg_names:
                    prefix = n[0]   # "xnumel" → "x",  "rnumel" → "r"
                    code.writeline(f"_{prefix}numel = {n}")
            else:
                code.writeline("_xnumel = 1")
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
        return kernel_name, src_code
