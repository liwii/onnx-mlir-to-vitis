import numpy as np
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict
import typing

@dataclass
class F32:
    pass

@dataclass
class I64:
    pass

@dataclass
class I8:
    pass


def str_to_valtype(s):
    if s == "f32":
        return F32()
    elif s == "i64":
        return I64()
    elif s == "i8":
        return I8()

ValType = Union[F32, I64, I8]

@dataclass
class MemRef:
    size: List[int]
    base_type: ValType

@dataclass
class TensorType:
    size: List[int]
    base_type: ValType


SingleType = Union[ValType, MemRef]
TupleType = List[SingleType]
ObjType = Union[SingleType, TupleType]

@dataclass
class FuncType:
    args: ObjType
    rets: ObjType

AllType = Union[ObjType, FuncType]

@dataclass
class Attr:
    pass

@dataclass
class Op:
    pass

Block = List[Op]


@dataclass
class Func:
    args: List[Tuple[str, ObjType]]
    rets: ObjType
    attr: Dict[str, str]
    block: Block

@dataclass
class F32Const:
    val: np.float32

@dataclass
class I64Const:
    val: np.int64

Const = Union[F32Const, I64Const]

@dataclass
class Alloc:
    memref: MemRef

@dataclass
class Alloca:
    memref: MemRef

@dataclass
class GetRef:
    args: List[str]
    getref_type: FuncType

@dataclass
class Tensor:
    value: np.ndarray
    tensor_type: TensorType

@dataclass
class Global:
    name: str
    shape: List[int]
    value: Tensor
    global_type: FuncType

@dataclass
class Load:
    mem_var: str
    mem_idx: List[str]
    memref: MemRef

@dataclass
class Mulf:
    var1: str
    var2: str
    val_type: ValType

@dataclass
class Addf:
    var1: str
    var2: str
    val_type: ValType

@dataclass
class Divf:
    var1: str
    var2: str
    val_type: ValType

@dataclass
class Subf:
    var1: str
    var2: str
    val_type: ValType

@dataclass
class Exp:
    var: str
    val_type: ValType

@dataclass
class Gt:
    var1: str
    var2: str
    val_type: ValType

@dataclass
class Lt:
    var1: str
    var2: str
    val_type: ValType

@dataclass
class Select:
    var1: str
    var2: str
    var3: str
    val_type: ValType

ValExp = Union[Const, Alloc, GetRef, Global, Alloca, Mulf, Addf, Divf, Subf, Exp, Gt, Lt, Select]

@dataclass
class SubstOp:
    var_name: str
    exp: ValExp

@dataclass
class ForOp:
    arg_name: str
    loop_range: Tuple[int, int]
    block: Block

@dataclass
class StoreOp:
    val_var: str
    mem_var: str
    mem_idx: List[str]
    memref: MemRef

@dataclass
class DeallocOp:
    var:str
    memref: MemRef

@dataclass
class ReturnOp:
    var:str
    memref: MemRef



Op = Union[SubstOp, ForOp, StoreOp]








