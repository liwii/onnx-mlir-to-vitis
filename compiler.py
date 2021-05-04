import sys
import struct
import ast
from classes import *

SEP_CHARS = ["(", ")", "[", "]", "<", ">", "\"", ",", "{", "}", ":"]
ARROW_MLIR = "->"
ARROW_TOKEN = "$"


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
        arg_ast, newidx == F32(), idx + 1
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
        elif tokens[idx + 2] == '"':
            if tokens[idx + 3] == "krnl.getref":
                val_exp, newidx = parse_getref(tokens, idx + 2)
            elif tokens[idx + 3] == "krnl.global":
                val_exp, newidx = parse_global(tokens, idx + 2)
            else:
                return [], idx
        else:
            return [], idx

        next_exp, newidx = parse_op(tokens, newidx)
        return [SubstOp(var_name, val_exp)] + next_exp, newidx
    else:
        return [], idx

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

    value, newidx = parse_value(tokens, newidx)
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
    ls, newidx = parse_list(tokens, idx + 3)
    if tokens[newidx] != "]":
        raise ParserError(newidx)
    return ls, newidx + 1

def parse_list(tokens, idx):
    hd = int(tokens[idx])
    if tokens[idx + 1] == ",":
        tl, newidx = parse_list(tokens, idx + 2)
        return [hd] + tl, newidx
    else:
        return [hd], idx + 1

def parse_value(tokens, idx):
    if tokens[idx] != "value":
        raise ParserError(idx)
    if tokens[idx + 1] != "=":
        raise ParserError(idx + 1)
    if tokens[idx + 2] != "dense":
        raise ParserError(idx + 2)
    if tokens[idx + 3] != "<":
        raise ParserError(idx + 3)
    nparray, newidx = parse_nparray(tokens, idx + 4)
    if tokens[newidx] != ">":
        raise ParserError(newidx)
    return nparray, newidx + 1

def parse_nparray(tokens, idx):
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








filename = sys.argv[1]
with open(filename) as f: content = f.read()
tokens = lex(content)
module_ast, idx = parse_module(tokens, 0)

breakpoint()



