import sys
import struct
import ast
from classes import *

SEP_CHARS = ["(", ")", "[", "]", "<", ">", "\"", ",", "{", "}", ":"]
ARROW_MLIR = "->"
ARROW_TOKEN = "$"
INDEX_VARIABLES = ["i", "j", "k", "l", "m"]


def lex(content):
    content = content.replace(ARROW_MLIR, ARROW_TOKEN)
    for c in SEP_CHARS:
        content = content.replace(c, " " + c + " ")
    return content.split()

class ParserError(RuntimeError):
    def __init__(self, idx):
        self.idx = idx

def parse_module(tokens, idx):
    if tokens[idx] != "module":
        raise ParserError(idx)
    if tokens[idx + 1] != "{":
        raise ParserError(idx + 1)
    ast, newidx = parse_func(tokens, idx + 2)
    return ast, newidx
    #TODO: process krnl_entrypoint
    #if newidx != len(tokens) - 1 or tokens[newidx] != "}":
    #    raise ParserError(newidx)
    #return ast, newidx + 1

def parse_func(tokens, idx):
    if tokens[idx] != "func":
        raise ParserError(idx)
    if tokens[idx + 1] != "@main_graph":
        raise ParserError(idx + 1)
    arg_ast, newidx = parse_arg(tokens, idx + 2)
    if tokens[newidx] != "$":
        raise ParserError(newidx)
    ret_ast, newidx = parse_type(tokens, newidx + 1)
    attr_ast, newidx = parse_attr(tokens, newidx)
    block_ast, newidx = parse_block(tokens, newidx)
    return Func(arg_ast, ret_ast, attr_ast, block_ast), newidx

def parse_arg(tokens, idx):
    if tokens[idx] != "(":
        raise ParserError(idx)
    arg_vars_ast, newidx = parse_arg_var(tokens, idx + 1)
    if tokens[newidx] != ")":
        raise ParserError(newidx)
    return arg_vars_ast, newidx + 1

def parse_arg_var(tokens, idx):
    if tokens[idx][0] != "%":
        raise ParserError(idx)
    arg_name = tokens[idx][1:]
    if tokens[idx + 1] != ":":
        raise ParserError(idx + 1)
    arg_type_ast, newidx = parse_type(tokens, idx + 2)
    if tokens[newidx] == ",":
        new_args_ast, newidx = parse_arg_var(tokens, newidx + 1)
        return [(arg_name, arg_type_ast)] + new_args_ast, newidx
    else:
        return [(arg_name, arg_type_ast)], newidx


def parse_type(tokens, idx):
    if tokens[idx] == "memref":
        arg_ast, newidx = parse_memref(tokens, idx + 1)
    elif tokens[idx] == "tensor":
        arg_ast, newidx = parse_tensor_type(tokens, idx + 1)
    elif tokens[idx] == "i64":
        arg_ast, newidx = I64(), idx + 1
    elif tokens[idx] == "i8":
        arg_ast, newidx = I8(), idx + 1
    elif tokens[idx] == "f32":
        arg_ast, newidx = F32(), idx + 1
    elif tokens[idx] == "(":
        arg_ast, newidx = parse_tuple(tokens, idx)
    else:
        raise ParserError(idx)
    if tokens[newidx] == "$":
        ret_ast, newidx = parse_type(tokens, newidx + 1)
        return FuncType(arg_ast, ret_ast), newidx
    else:
        return arg_ast, newidx

def parse_memref(tokens, idx):
    if tokens[idx] != "<":
        raise ParserError(idx)
    idx += 1
    split_type = tokens[idx].split('x')
    shape = [int(s) for s in split_type[:-1]]
    val_type = str_to_valtype(split_type[-1])
    idx += 1
    if tokens[idx] != ">":
        raise ParseError(idx)
    return MemRef(shape, val_type), idx + 1

def parse_tensor_type(tokens, idx):
    if tokens[idx] != "<":
        raise ParserError(idx)
    idx += 1
    split_type = tokens[idx].split('x')
    shape = [int(s) for s in split_type[:-1]]
    val_type = str_to_valtype(split_type[-1])
    idx += 1
    if tokens[idx] != ">":
        raise ParseError(idx)
    return TensorType(shape, val_type), idx + 1


def parse_tuple(tokens, idx):
    if tokens[idx] != "(":
        raise ParserError(idx)
    tuple_result = []
    newidx = idx + 1
    while True:
        if tokens[newidx] == ")":
            break
        if tokens[newidx] == ",":
            newidx += 1
        type_ast, newidx = parse_type(tokens, newidx)
        tuple_result.append(type_ast)
    return tuple_result, newidx + 1

def parse_attr(tokens, idx):
    if tokens[idx] != "attributes":
        raise ParserError(idx)
    if tokens[idx + 1] != "{":
        raise ParserError(idx + 1)
    attr_ast, newidx = parse_single_attr(tokens, idx + 2)

    if tokens[newidx] != "}":
        raise ParserError(newidx)
    return attr_ast, newidx + 1

def parse_single_attr(tokens, idx):
    name = tokens[idx]
    if tokens[idx + 1] != "=":
        raise ParserError(idx + 1)
    idx = idx + 2
    str_open = False
    val = ""
    while (True):
        if str_open:
            if tokens[idx] == '"':
                str_open = False
        else:
            if tokens[idx] == "," or tokens[idx] == "}":
                break

        val += tokens[idx]
        idx += 1
    if tokens[idx] == "}":
        return {name:val}, idx
    else:
        next_attr_ast, newidx = parse_single_attr(tokens, idx + 1)
        next_attr_ast[name] = val
        return next_attr_ast, newidx

def parse_block(tokens, idx):
    if tokens[idx] != "{":
        raise ParserError(idx)
    op_ast, newidx = parse_op(tokens, idx + 1)

    #if tokens[newidx] != "}":
    #    raise ParserError(idx)

    return op_ast, newidx + 1

def parse_op(tokens, idx):
    if tokens[idx] == "}":
        return [], idx

    if tokens[idx][0] == "%":
        var_name = tokens[idx][1:]
        if tokens[idx + 1] != "=":
            raise ParserError(idx + 1)

        if tokens[idx + 2] == "constant":
            val_exp, newidx = parse_const(tokens, idx + 2)
        elif tokens[idx + 2] == "alloc":
            val_exp, newidx = parse_alloc(tokens, idx + 2)
        elif tokens[idx + 2] == "alloca":
            val_exp, newidx = parse_alloca(tokens, idx + 2)
        elif tokens[idx + 2] == "affine.load":
            val_exp, newidx = parse_load(tokens, idx + 2)
        elif tokens[idx + 2] == "mulf":
            val_exp, newidx = parse_mulf(tokens, idx + 2)
        elif tokens[idx + 2] == "addf":
            val_exp, newidx = parse_addf(tokens, idx + 2)
        elif tokens[idx + 2] == "divf":
            val_exp, newidx = parse_divf(tokens, idx + 2)
        elif tokens[idx + 2] == "subf":
            val_exp, newidx = parse_subf(tokens, idx + 2)
        elif tokens[idx + 2] == "exp":
            val_exp, newidx = parse_exp(tokens, idx + 2)
        elif tokens[idx + 2] == "select":
            val_exp, newidx = parse_select(tokens, idx + 2)
        elif tokens[idx + 2] == "cmpf" and tokens[idx + 3] == "ogt":
            val_exp, newidx = parse_gt(tokens, idx + 2)
        elif tokens[idx + 2] == "cmpf" and tokens[idx + 3] == "olt":
            val_exp, newidx = parse_lt(tokens, idx + 2)
        elif tokens[idx + 2] == '"':
            if tokens[idx + 3] == "krnl.getref":
                val_exp, newidx = parse_getref(tokens, idx + 2)
            elif tokens[idx + 3] == "krnl.global":
                val_exp, newidx = parse_global(tokens, idx + 2)
            else:
                return [], idx
        else:
            return [], idx
        op = SubstOp(var_name, val_exp)

    else:
        if tokens[idx] == "affine.for":
            op, newidx = parse_for(tokens, idx)
        elif tokens[idx] == "affine.store":
            op, newidx = parse_store(tokens, idx)
        elif tokens[idx] == "dealloc":
            op, newidx = parse_dealloc(tokens, idx)
        elif tokens[idx] == "return":
            op, newidx = parse_return(tokens, idx)
        else:
            return [], idx

    next_exp, newidx = parse_op(tokens, newidx)
    return [op] + next_exp, newidx

def parse_const(tokens, idx):
    if tokens[idx] != "constant":
        raise ParserError(idx)

    if tokens[idx + 2] != ":":
        raise ParserError(idx + 2)

    if tokens[idx + 3] == "i64":
        return I64Const(np.int64(tokens[idx + 1])), idx + 4
    elif tokens[idx + 3] == "f32":
        if tokens[idx + 1][0:2] == "0x":
            return F32Const(np.float32(struct.unpack('>f', struct.pack('>I', int(tokens[idx + 1], 16)))[0])), idx + 4
        else:
            return F32Const(np.float32(tokens[idx + 1])), idx + 4
    else:
        raise ParserError(idx + 2)

def parse_alloc(tokens, idx):
    if tokens[idx] != "alloc":
        raise ParserError(idx)
    if tokens[idx + 1] != "(":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != ")":
        raise ParserError(idx + 2)
    if tokens[idx + 3] != ":":
        raise ParserError(idx + 3)

    memref_type, newidx = parse_type(tokens, idx + 4)
    return Alloc(memref_type), newidx

def parse_getref(tokens, idx):
    if tokens[idx] != '"':
        raise ParserError(idx)
    if tokens[idx + 1] != "krnl.getref":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != '"':
        raise ParserError(idx + 2)
    if tokens[idx + 3] != '(':
        raise ParserError(idx + 3)
    idx = idx + 4
    args = []
    while tokens[idx] != ")":
        if tokens[idx] == ",":
            idx += 1
            continue
        args.append(tokens[idx][1:])
        idx += 1
    if tokens[idx + 1] != ":":
        raise ParserError(idx + 1)
    getref_type, newidx = parse_type(tokens, idx + 2)
    return GetRef(args, getref_type), newidx

def parse_global(tokens, idx):
    if tokens[idx] != '"':
        raise ParserError(idx)
    if tokens[idx + 1] != "krnl.global":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != '"':
        raise ParserError(idx + 2)
    if tokens[idx + 3] != '(':
        raise ParserError(idx + 3)
    if tokens[idx + 4] != ')':
        raise ParserError(idx + 4)
    if tokens[idx + 5] != '{':
        raise ParserError(idx + 5)

    name, newidx = parse_name(tokens, idx + 6)
    if tokens[newidx] != ",":
        raise ParserError(newidx)
    newidx += 1

    shape, newidx = parse_shape(tokens, newidx)
    if tokens[newidx] != ",":
        raise ParserError(newidx)
    newidx += 1

    value, newidx = parse_value(tokens, newidx, shape)
    if tokens[newidx] != ":":
        raise ParserError(newidx)
    newidx += 1

    val_type, newidx = parse_type(tokens, newidx)
    if tokens[newidx] != "}":
        raise ParserError(newidx)
    newidx += 1

    if tokens[newidx] != ":":
        raise ParserError(newidx)
    newidx += 1

    ref_type, newidx = parse_type(tokens, newidx)

    return Global(name, shape, value, ref_type), newidx

def parse_name(tokens, idx):
    if tokens[idx] != "name":
        raise ParserError(idx)
    if tokens[idx + 1] != "=":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != '"':
        raise ParserError(idx + 2)
    name = tokens[idx + 3]
    if tokens[idx + 4] != '"':
        raise ParserError(idx + 4)
    return name, idx + 5

def parse_shape(tokens, idx):
    if tokens[idx] != "shape":
        raise ParserError(idx)
    if tokens[idx + 1] != "=":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != "[":
        raise ParserError(idx + 2)
    ls, newidx = parse_int_list(tokens, idx + 3)
    if tokens[newidx] != "]":
        raise ParserError(newidx)
    return ls, newidx + 1

def parse_int_list(tokens, idx):
    hd = int(tokens[idx])
    if tokens[idx + 1] == ",":
        tl, newidx = parse_int_list(tokens, idx + 2)
        return [hd] + tl, newidx
    else:
        return [hd], idx + 1

def parse_value(tokens, idx, shape):
    if tokens[idx] != "value":
        raise ParserError(idx)
    if tokens[idx + 1] != "=":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != "dense":
        raise ParserError(idx + 2)
    if tokens[idx + 3] != "<":
        raise ParserError(idx + 3)
    nparray, newidx = parse_nparray(tokens, idx + 4, shape)
    if tokens[newidx] != ">":
        raise ParserError(newidx)
    return nparray, newidx + 1

def parse_nparray(tokens, idx, shape):
    if tokens[idx] == '"':
        return parse_buffer_nparray(tokens, idx, shape)
    if tokens[idx] != "[":
        raise ParserError(idx)
    idx += 1
    literal = "["
    paren_count = 1
    while paren_count > 0:
        literal += tokens[idx]
        if tokens[idx] == "[":
            paren_count += 1
        if tokens[idx] == "]":
            paren_count -= 1
        idx += 1
    return np.array(ast.literal_eval(literal), dtype=np.float32), idx

def parse_buffer_nparray(tokens, idx, shape):
    if tokens[idx] != '"':
        raise ParserError(idx)
    if len(shape) != 2:
        raise RuntimeError("Non-2d buffer array is not supported")
    buf_bytes = bytes.fromhex(tokens[idx + 1][2:])
    array = np.frombuffer(buf_bytes, dtype=np.float32).reshape(shape)
    if tokens[idx + 2] != '"':
        raise ParserError(idx)
    return array, idx + 3


def parse_for(tokens, idx):
    if tokens[idx] != "affine.for":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    arg_name = tokens[idx + 1][1:]
    if tokens[idx + 2] != "=":
        raise ParserError(idx + 2)
    loop_range = (int(tokens[idx + 3]), int(tokens[idx + 5]))
    if tokens[idx + 4] != "to":
        raise ParserError(idx + 1)
    block, newidx = parse_block(tokens, idx + 6)
    return ForOp(arg_name, loop_range, block),  newidx

def parse_alloca(tokens, idx):
    if tokens[idx] != "alloca":
        raise ParserError(idx)
    if tokens[idx + 1] != "(":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != ")":
        raise ParserError(idx + 2)
    if tokens[idx + 3] != ":":
        raise ParserError(idx + 3)

    memref_type, newidx = parse_type(tokens, idx + 4)
    return Alloca(memref_type), newidx

def parse_load(tokens, idx):
    if tokens[idx] != "affine.load":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    mem_var = tokens[idx + 1][1:]
    if tokens[idx + 2] != "[":
        raise ParserError(idx + 2)
    mem_idx, newidx = parse_var_list(tokens, idx + 3)
    if tokens[newidx] != "]":
        raise ParserError(newidx)
    if tokens[newidx + 1] != ":":
        raise ParserError(newidx + 1)
    memref, newidx = parse_type(tokens, newidx + 2)
    return Load(mem_var, mem_idx, memref),  newidx

def parse_var_list(tokens, idx):
    if tokens[idx] == "]":
        return [], idx
    hd = tokens[idx][1:]
    if tokens[idx + 1] == ",":
        tl, newidx = parse_var_list(tokens, idx + 2)
        return [hd] + tl, newidx
    else:
        return [hd], idx + 1

def parse_store(tokens, idx):
    if tokens[idx] != "affine.store":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    val_var = tokens[idx + 1][1:]
    if tokens[idx + 2] != ",":
        raise ParserError(idx + 2)
    if tokens[idx + 3][0] != "%":
        raise ParserError(idx + 1)
    mem_var = tokens[idx + 3][1:]
    if tokens[idx + 4][0] != "[":
        raise ParserError(idx + 4)
    mem_idx, newidx = parse_var_list(tokens, idx + 5)
    if tokens[newidx] != "]":
        raise ParserError(newidx)
    if tokens[newidx + 1] != ":":
        raise ParserError(newidx + 1)
    memref, newidx = parse_type(tokens, newidx + 2)
    return StoreOp(val_var, mem_var, mem_idx, memref), newidx

def parse_mulf(tokens, idx):
    if tokens[idx] != "mulf":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    var1 = tokens[idx + 1][1:]
    if tokens[idx + 2] != ",":
        raise ParserError(idx + 2)
    if tokens[idx + 3][0] != "%":
        raise ParserError(idx + 3)
    var2 = tokens[idx + 3][1:]
    if tokens[idx + 4] != ":":
        raise ParserError(idx + 4)
    val_type, newidx = parse_type(tokens, idx + 5)
    return Mulf(var1, var2, val_type), newidx

def parse_addf(tokens, idx):
    if tokens[idx] != "addf":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    var1 = tokens[idx + 1][1:]
    if tokens[idx + 2] != ",":
        raise ParserError(idx + 2)
    if tokens[idx + 3][0] != "%":
        raise ParserError(idx + 3)
    var2 = tokens[idx + 3][1:]
    if tokens[idx + 4] != ":":
        raise ParserError(idx + 4)
    val_type, newidx = parse_type(tokens, idx + 5)
    return Addf(var1, var2, val_type), newidx

def parse_divf(tokens, idx):
    if tokens[idx] != "divf":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    var1 = tokens[idx + 1][1:]
    if tokens[idx + 2] != ",":
        raise ParserError(idx + 2)
    if tokens[idx + 3][0] != "%":
        raise ParserError(idx + 3)
    var2 = tokens[idx + 3][1:]
    if tokens[idx + 4] != ":":
        raise ParserError(idx + 4)
    val_type, newidx = parse_type(tokens, idx + 5)
    return Divf(var1, var2, val_type), newidx

def parse_subf(tokens, idx):
    if tokens[idx] != "subf":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    var1 = tokens[idx + 1][1:]
    if tokens[idx + 2] != ",":
        raise ParserError(idx + 2)
    if tokens[idx + 3][0] != "%":
        raise ParserError(idx + 3)
    var2 = tokens[idx + 3][1:]
    if tokens[idx + 4] != ":":
        raise ParserError(idx + 4)
    val_type, newidx = parse_type(tokens, idx + 5)
    return Subf(var1, var2, val_type), newidx

def parse_exp(tokens, idx):
    if tokens[idx] != "exp":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    var = tokens[idx + 1][1:]
    if tokens[idx + 2] != ":":
        raise ParserError(idx + 2)
    val_type, newidx = parse_type(tokens, idx + 3)
    return Exp(var, val_type), newidx

def parse_gt(tokens, idx):
    if tokens[idx] != "cmpf":
        raise ParserError(idx)
    if tokens[idx + 1] != "ogt":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != ",":
        raise ParserError(idx + 2)
    if tokens[idx + 3][0] != "%":
        raise ParserError(idx + 3)
    var1 = tokens[idx + 3][1:]
    if tokens[idx + 4] != ",":
        raise ParserError(idx + 4)
    if tokens[idx + 5][0] != "%":
        raise ParserError(idx + 5)
    var2 = tokens[idx + 5][1:]
    if tokens[idx + 6] != ":":
        raise ParserError(idx + 6)
    val_type, newidx = parse_type(tokens, idx + 7)
    return Gt(var1, var2, val_type), newidx

def parse_lt(tokens, idx):
    if tokens[idx] != "cmpf":
        raise ParserError(idx)
    if tokens[idx + 1] != "olt":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != ",":
        raise ParserError(idx + 2)
    if tokens[idx + 3][0] != "%":
        raise ParserError(idx + 3)
    var1 = tokens[idx + 3][1:]
    if tokens[idx + 4] != ",":
        raise ParserError(idx + 4)
    if tokens[idx + 5][0] != "%":
        raise ParserError(idx + 5)
    var2 = tokens[idx + 5][1:]
    if tokens[idx + 6] != ":":
        raise ParserError(idx + 6)
    val_type, newidx = parse_type(tokens, idx + 7)
    return Lt(var1, var2, val_type), newidx

def parse_select(tokens, idx):
    if tokens[idx] != "select":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    var1 = tokens[idx + 1][1:]
    if tokens[idx + 2] != ",":
        raise ParserError(idx + 2)
    if tokens[idx + 3][0] != "%":
        raise ParserError(idx + 3)
    var2 = tokens[idx + 3][1:]
    if tokens[idx + 4] != ",":
        raise ParserError(idx + 4)
    if tokens[idx + 5][0] != "%":
        raise PasrerError(idx + 5)
    var3 = tokens[idx + 5][1:]
    if tokens[idx + 6] != ":":
        raise ParserError(idx + 6)
    val_type, newidx = parse_type(tokens, idx + 7)
    return Select(var1, var2, var3, val_type), newidx

def parse_dealloc(tokens, idx):
    if tokens[idx] != "dealloc":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    var = tokens[idx + 1][1:]
    if tokens[idx + 2] != ":":
        raise ParserError(idx + 2)
    memref, newidx = parse_type(tokens, idx + 3)
    return DeallocOp(var, memref), newidx

def parse_return(tokens, idx):
    if tokens[idx] != "return":
        raise ParserError(idx)
    if tokens[idx + 1][0] != "%":
        raise ParserError(idx + 1)
    var = tokens[idx + 1][1:]
    if tokens[idx + 2] != ":":
        raise ParserError(idx + 2)
    memref, newidx = parse_type(tokens, idx + 3)
    return ReturnOp(var, memref), newidx

memory_types = {}

def print_prelude():
    print("#include <ap_int.h>")
    print("#include <hls_stream.h>")
    print("#include <ap_axi_sdata.h>")
    print("#include <math.h>")
    print("using namespace hls;")
    print("typedef ap_axiu<32, 0, 0, 0> AP_AXIS;")
    print("typedef stream<AP_AXIS> AXI_STREAM;")
    print()

def shape_brackets(shape):
    ret = ""
    for i in shape:
        ret += "[" + str(i) + "]"
    return ret
def val_type_str(val_type):
    if isinstance(val_type, F32):
        return "float"
    if isinstance(val_type, I64):
        return "long"
    if isinstance(val_type, I8):
        return "char"
    return "int"

def memref_type(memref):
    return val_type_str(memref.base_type)

def print_memref_var(var_name, memref, tabs=""):
    brackets = shape_brackets(memref.size)
    type_name = memref_type(memref)
    print(f"{tabs}{type_name} {var_name}{brackets};")

def print_input(var_name, memref, tabs=""):
    print(f"{tabs}AP_AXIS element;")
    print()
    for i in range(len(memref.size)):
        var = INDEX_VARIABLES[i]
        linetabs = tabs + '\t' * i
        s = f"{linetabs}for (int {var} = 0; {var} < {memref.size[i]}; {var}++) {{"
        print(s)
    linetabs = tabs + '\t' * len(memref.size)
    print(f"{linetabs}element = in_strm.read();")
    print(f"{linetabs}my_converter.as_uint32 = element.data;")

    print(f"{linetabs}{var_name}{shape_brackets(INDEX_VARIABLES[:len(memref.size)])} = my_converter.as_floatingpoint;")

    for i in range(len(memref.size)):
        linetabs = tabs + '\t' * (len(memref.size) - 1 - i)
        print(f"{linetabs}}}")

def print_f32_const(var_name, exp, tabs):
    if exp.val == np.inf:
        exp_str = "INFINITY"
    elif exp.val == -np.inf:
        exp_str = "-INFINITY"
    else:
        exp_str = str(exp.val)
    print(f"{tabs}float {var_name} = {exp_str};")

def print_i64_const(var_name, exp, tabs):
    print(f"{tabs}long {var_name} = {exp.val};")

def print_getref(var_name, exp, tabs):
    memref = exp.getref_type.rets
    memory_types[var_name] = memref
    print(f"{tabs}{memref_type(memref)} {var_name}{shape_brackets(memref.size)};")

def stringify_numpy_array(ndarray):
    if isinstance(ndarray, np.ndarray):
        return "{" + ",".join([stringify_numpy_array(n) for n in ndarray]) + "}"
    else:
        return str(ndarray)

def print_global(var_name, exp, tabs):
    tensor_str = stringify_numpy_array(exp.value)
    memref = exp.global_type.rets
    memory_types[var_name] = memref
    print(f"{tabs}{memref_type(memref)} {var_name}{shape_brackets(memref.size)} = {tensor_str};")

def print_alloc(var_name, exp, tabs):
    memref = exp.memref
    memory_types[var_name] = memref
    print(f"{tabs}{memref_type(memref)} {var_name}{shape_brackets(memref.size)};")

def print_alloca(var_name, exp, tabs):
    memref = exp.memref
    memory_types[var_name] = memref
    print(f"{tabs}{memref_type(memref)} {var_name}{shape_brackets(memref.size)};")

def print_load(var_name, exp, tabs):
    memref = exp.memref
    mem_idx_vars = ["x" + v for v in exp.mem_idx]
    mem_var_name = "x" + exp.mem_var
    print(f"{tabs}{memref_type(memref)} {var_name} = {mem_var_name}{shape_brackets(mem_idx_vars)};")

def print_mulf(var_name, exp, tabs):
    var_name1 = "x" + exp.var1
    var_name2 = "x" + exp.var2
    print(f"{tabs}float {var_name} = {var_name1} * {var_name2};")

def print_addf(var_name, exp, tabs):
    var_name1 = "x" + exp.var1
    var_name2 = "x" + exp.var2
    print(f"{tabs}float {var_name} = {var_name1} + {var_name2};")

def print_divf(var_name, exp, tabs):
    var_name1 = "x" + exp.var1
    var_name2 = "x" + exp.var2
    print(f"{tabs}float {var_name} = {var_name1} / {var_name2};")

def print_subf(var_name, exp, tabs):
    var_name1 = "x" + exp.var1
    var_name2 = "x" + exp.var2
    print(f"{tabs}float {var_name} = {var_name1} - {var_name2};")

def print_subf(var_name, exp, tabs):
    var_name1 = "x" + exp.var1
    var_name2 = "x" + exp.var2
    print(f"{tabs}float {var_name} = {var_name1} - {var_name2};")

def print_exp(var_name, exp, tabs):
    arg_name = "x" + exp.var
    print(f"{tabs}float {var_name} = exp({arg_name});")

def print_gt(var_name, exp, tabs):
    var_name1 = "x" + exp.var1
    var_name2 = "x" + exp.var2
    print(f"{tabs}bool {var_name} = {var_name1} > {var_name2};")

def print_lt(var_name, exp, tabs):
    var_name1 = "x" + exp.var1
    var_name2 = "x" + exp.var2
    print(f"{tabs}bool {var_name} = {var_name1} < {var_name2};")

def print_select(var_name, exp, tabs):
    var_name1 = "x" + exp.var1
    var_name2 = "x" + exp.var2
    var_name3 = "x" + exp.var3
    print(f"{tabs}{val_type_str(exp.val_type)} {var_name} = {var_name1} ? {var_name2} : {var_name3};")

def print_subst(subst_op, tabs):
    var_name = "x" + subst_op.var_name
    exp = subst_op.exp
    if isinstance(exp, F32Const):
        print_f32_const(var_name, exp, tabs)
    elif isinstance(exp, I64Const):
        print_i64_const(var_name, exp, tabs)
    elif isinstance(exp, GetRef):
        print_getref(var_name, exp, tabs)
    elif isinstance(exp, Alloc):
        print_alloc(var_name, exp, tabs)
    elif isinstance(exp, Global):
        print_global(var_name, exp, tabs)
    elif isinstance(exp, Alloca):
        print_alloca(var_name, exp, tabs)
    elif isinstance(exp, Load):
        print_load(var_name, exp, tabs)
    elif isinstance(exp, Mulf):
        print_mulf(var_name, exp, tabs)
    elif isinstance(exp, Addf):
        print_addf(var_name, exp, tabs)
    elif isinstance(exp, Divf):
        print_divf(var_name, exp, tabs)
    elif isinstance(exp, Subf):
        print_subf(var_name, exp, tabs)
    elif isinstance(exp, Exp):
        print_exp(var_name, exp, tabs)
    elif isinstance(exp, Gt):
        print_gt(var_name, exp, tabs)
    elif isinstance(exp, Lt):
        print_lt(var_name, exp, tabs)
    elif isinstance(exp, Select):
        print_select(var_name, exp, tabs)

def print_for(op, tabs):
    arg_name = "x" + op.arg_name
    print(f"{tabs}for (int {arg_name} = {op.loop_range[0]}; {arg_name} < {op.loop_range[1]}; {arg_name}++) {{")
    print_block(op.block, tabs + "\t")
    print(f"{tabs}}}")

def print_store(op, tabs):
    val_var_name = "x" + op.val_var
    mem_var_name = "x" + op.mem_var
    mem_idx_vars = ["x" + v for v in op.mem_idx]
    print(f"{tabs}{mem_var_name}{shape_brackets(mem_idx_vars)} = {val_var_name};")

def loop_termination(memref_size):
    return "&&".join([INDEX_VARIABLES[i] + " == " + str(memref_size[i] - 1) for i in range(len(memref_size))])

def print_return(op, tabs):
    print(f"{tabs}AP_AXIS val;")
    print(f"{tabs}val.keep = element.keep;")
    print(f"{tabs}val.strb = element.strb;")
    print(f"{tabs}val.last = 0;")
    print()
    var_name = "x" + op.var
    memref = memory_types[var_name]

    for i in range(len(memref.size)):
        var = INDEX_VARIABLES[i]
        linetabs = tabs + '\t' * i
        s = f"{linetabs}for (int {var} = 0; {var} < {memref.size[i]}; {var}++) {{"
        print(s)
    linetabs = tabs + '\t' * len(memref.size)

    print(f"{linetabs}my_converter.as_floatingpoint = {var_name}{shape_brackets(INDEX_VARIABLES[:len(memref.size)])};")
    print(f"{linetabs}val.data = my_converter.as_uint32;")
    print(f"{linetabs}if({loop_termination(memref.size)}) val.last = 1;")
    print(f"{linetabs}out_strm << val;")

    for i in range(len(memref.size)):
        linetabs = tabs + '\t' * (len(memref.size) - 1 - i)
        print(f"{linetabs}}}")

    print(f"{tabs}return;")

def print_op(op, tabs):
    if isinstance(op, SubstOp):
        print_subst(op, tabs)
    elif isinstance(op, ForOp):
        print_for(op, tabs)
    elif isinstance(op, StoreOp):
        print_store(op, tabs)
    elif isinstance(op, ReturnOp):
        print_return(op, tabs)

def print_block(block, tabs=""):
    for op in block:
        print_op(op, tabs)

def print_vitis_module(func):
    print_prelude()
    print("void graph(AXI_STREAM &in_strm, AXI_STREAM &out_strm) {")
    print_memref_var("x" + func.args[0][0], func.args[0][1], tabs="\t")
    print("\t#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS")
    print("\t#pragma HLS INTERFACE axis port=in_strm")
    print("\t#pragma HLS INTERFACE axis port=out_strm")
    print("\tunion")
    print("\t{")
    print("\t    unsigned int as_uint32;")
    print("\t    float as_floatingpoint;")
    print("\t} my_converter;")
    print_input("x" + func.args[0][0], func.args[0][1], tabs="\t")
    print_block(func.block, tabs="\t")
    print("}")



filename = sys.argv[1]
with open(filename) as f: content = f.read()
tokens = lex(content)
module_ast, idx = parse_module(tokens, 0)
print_vitis_module(module_ast)



