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

ValExp = Union[Const, Alloc]

@dataclass
class SubstOp:
    var_name: str
    exp: ValExp

@dataclass
class GetRef:
    args: List[str]
    getref_type: FuncType

Op = Union[SubstOp]








