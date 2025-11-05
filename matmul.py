from xdsl.dialects.gpu import GPU
from xdsl.dialects.tensor import Tensor
from xdsl.dialects.builtin import Builtin, ModuleOp, FunctionType, f32
from xdsl.dialects.func import Func, FuncOp, Return
from xdsl.dialects.memref import MemRef, MemRefType
from xdsl.dialects.arith import Arith
from xdsl.dialects.linalg import Linalg, Matmul
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.printer import Printer
from io import StringIO

# First example: Parse existing MLIR
mlir_src = r"""
builtin.module {
  func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                  outs(%C : memref<?x?xf32>)
    func.return
  }
}
"""

print("Parsing existing MLIR:")
print("=" * 60)
ctx = Context()
ctx.load_dialect(Builtin)
ctx.load_dialect(Func)
ctx.load_dialect(MemRef)
ctx.load_dialect(Arith)
ctx.load_dialect(Linalg)

mod = Parser(ctx, mlir_src).parse_module()
output = StringIO()
printer = Printer(stream=output)
printer.print(mod)
print(output.getvalue())
print()


def create_matmul_module():
    """Create an MLIR module with matrix multiplication using xDSL builder"""
    ctx = MLContext()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Linalg)
    ctx.load_dialect(Tensor)
    ctx.load_dialect(GPU)
    ctx.load_dialect(MemRef)
    
    # Define types for our matrices
    # Matrix A: 1024x512, Matrix B: 512x256, Result C: 1024x256
    m, n, k = 1024, 256, 512
    
    # Use memref types for GPU compatibility
    a_type = MemRefType(f32, [m, k])
    b_type = MemRefType(f32, [k, n])
    c_type = MemRefType(f32, [m, n])
    
    # Create function signature
    func_type = FunctionType.from_lists([a_type, b_type, c_type], [])
    
    # Create the main function
    @Builder.implicit_region
    def func_body(args: tuple) -> None:
        a, b, c = args
        
        # Perform matrix multiplication: C = A @ B
        # linalg.matmul performs: C[i,j] += sum_k(A[i,k] * B[k,j])
        Matmul.get(a, b, c)
        
        Return()
    
    matmul_func = FuncOp(
        "matmul",
        func_type,
        func_body
    )
    
    # Create the module
    module = ModuleOp([matmul_func])
    
    return module


def main():
    print("Generated MLIR for GPU MatMul:")
    print("=" * 60)
    
    # Create the module
    module = create_matmul_module()
    
    # Print the MLIR
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(module)
    print(output.getvalue())
    
    print("\n" + "=" * 60)
    print("\nTo compile and run on GPU:")
    print("1. Save this output to 'matmul.mlir'")
    print("2. Lower to GPU using MLIR compiler:")
    print("   mlir-opt matmul.mlir \\")
    print("     --linalg-tile='tile-sizes=32,32,32' \\")
    print("     --linalg-bufferize \\")
    print("     --convert-linalg-to-parallel-loops \\")
    print("     --gpu-map-parallel-loops \\")
    print("     --convert-parallel-loops-to-gpu \\")
    print("     --gpu-kernel-outlining \\")
    print("     --lower-affine \\")
    print("     --convert-scf-to-cf \\")
    print("     --gpu-to-llvm \\")
    print("     -o matmul_gpu.mlir")
    print("\n3. For NVIDIA GPUs, add:")
    print("   --convert-gpu-to-nvvm \\")
    print("   --gpu-to-cubin")
    print("\nNote: This generates linalg.matmul which MLIR can compile to GPU kernels")


if __name__ == "__main__":
    main()
