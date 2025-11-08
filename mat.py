from xdsl.ir import MLContext, ModuleOp, Region, Block
from xdsl.dialects.builtin import RankedTensorType, f32, FunctionType
from xdsl.dialects import func, linalg, tensor

# Static example sizes (change as needed)
M, K, N = 128, 64, 256

# Types
A_t = RankedTensorType.get([M, K], f32)
B_t = RankedTensorType.get([K, N], f32)
C_t = RankedTensorType.get([M, N], f32)

# Module + function
mod = ModuleOp([])
fn_type = FunctionType.from_lists([A_t, B_t], [C_t])
fn = func.FuncOp("matmul", fn_type, visibility="private")
mod.body.block.add_op(fn)

# Create entry block with arguments A,B
entry = Block(arg_types=[A_t, B_t])
fn.body = Region(entry)
A, B = entry.args

# Destination tensor for C
init = tensor.EmptyOp([M, N], f32)

# linalg.matmul in destination style
mm = linalg.MatmulOp(inputs=[A, B], outputs=[init], result=[C_t])

# Return the single result
func.ReturnOp([mm.result])

print(mod)
