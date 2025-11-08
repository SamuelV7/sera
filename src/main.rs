use melior::dialect::DialectRegistry;
use melior::ir::attribute::*;
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::*;
use melior::ir::*;
use melior::utility::register_all_dialects;
use melior::Context;

fn main() {
    // Setup MLIR context with all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Build the module
    let module = build_matmul_module(&context);

    println!("=== Generated MLIR (Linalg) ===");
    println!("{}", module.as_operation());

    // Lower to GPU
    lower_to_gpu(&context, &module);

    println!("\n=== After GPU Lowering ===");
    println!("{}", module.as_operation())
}

fn build_matmul_module(context: &Context) -> Module<'_> {
    let location = Location::unknown(context);
    let module = Module::new(location);

    // Types
    let f32_type = Type::float32(context);
    let index_type = Type::index(context);

    // Create 1024x1024 matrix type: memref<1024x1024xf32>
    let matrix_type = Type::parse(context, "memref<1024x1024xf32>").unwrap();

    // Function type: (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()
    let func_type = FunctionType::new(context, &[matrix_type, matrix_type, matrix_type], &[]);

    // Create function
    let region = Region::new();
    let block = Block::new(&[
        (matrix_type, location),
        (matrix_type, location),
        (matrix_type, location),
    ]);

    let a = block.argument(0).unwrap().into();
    let b = block.argument(1).unwrap().into();
    let c = block.argument(2).unwrap().into();

    // Build linalg.matmul operation
    // C = A * B where A, B, C are 1024x1024
    let matmul_op = OperationBuilder::new("linalg.matmul", location)
        .add_operands(&[a, b, c])
        .build()
        .unwrap();

    block.append_operation(matmul_op);

    // Return
    let return_op = OperationBuilder::new("func.return", location)
        .build()
        .unwrap();
    block.append_operation(return_op);

    region.append_block(block);

    // Create the function operation
    let func_op = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "matmul_1024").into(),
            ),
            (
                Identifier::new(context, "function_type"),
                TypeAttribute::new(func_type.into()).into(),
            ),
        ])
        .add_regions([region])
        .build()
        .unwrap();

    module.body().append_operation(func_op);

    module
}

fn lower_to_gpu(context: &Context, _module: &Module) {
    // Create pass manager for the module
    use melior::pass::PassManager;

    let _pass_manager = PassManager::new(context);

    // Add passes for lowering:
    // Note: melior might not expose all passes directly
    // You may need to use the C API or write custom passes

    // For now, let's show the structure:
    // 1. Convert linalg to loops (scf)
    // 2. Map loops to GPU
    // 3. Lower GPU dialect to NVVM
    // 4. Convert NVVM to LLVM/PTX

    // This is the typical pipeline:
    // pass_manager.add_pass(pass::conversion::create_convert_linalg_to_loops());
    // pass_manager.add_pass(pass::conversion::create_gpu_kernel_outlining());
    // pass_manager.add_pass(pass::conversion::create_convert_scf_to_gpu());
    // pass_manager.add_pass(pass::conversion::create_convert_gpu_to_nvvm());

    println!("\nNote: Full GPU lowering pipeline requires additional pass configuration");
    println!("The linalg.matmul operation is ready for GPU lowering with passes like:");
    println!("  1. -convert-linalg-to-loops");
    println!("  2. -gpu-map-parallel-loops");
    println!("  3. -convert-parallel-loops-to-gpu");
    println!("  4. -gpu-kernel-outlining");
    println!("  5. -convert-gpu-to-nvvm");
}
