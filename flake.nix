{
  description = "Rust + MLIR (working for mlir-sys v0.5.0 expecting LLVM 20)";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/ef959e7a610ebe8a134cf361b5ffafdce0f7bf5e";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        rust = pkgs.rust-bin.stable.latest.default;
        llvmPkgs = pkgs.llvmPackages_20;
        mlir = llvmPkgs.mlir;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            rust
            pkgs.rust-analyzer
            pkgs.cmake
            pkgs.fish
            pkgs.rustfmt
            llvmPkgs.llvm.dev
            llvmPkgs.clang
            llvmPkgs.libclang
            llvmPkgs.lld
            mlir
            pkgs.libxml2
            pkgs.zlib
            llvmPkgs.llvm.lib
            llvmPkgs.libclang.lib
          ];
          
          MLIR_SYS_200_PREFIX = "${llvmPkgs.llvm.dev}";
          TABLEGEN_200_PREFIX = "${llvmPkgs.llvm.dev}";
          LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";
          
          hardeningDisable = [ "fortify" ];
          
          shellHook = ''
            export LD_LIBRARY_PATH=${llvmPkgs.llvm.lib}/lib:${llvmPkgs.libclang.lib}/lib:${mlir}/lib:$LD_LIBRARY_PATH
            
            # Link every single MLIR library explicitly
            MLIR_LIBS=$(find ${mlir}/lib -name "*.a" -exec basename {} \; | sed 's/^lib//;s/\.a$//' | tr '\n' ',' | sed 's/,$//')
            export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-L ${mlir}/lib -C link-args=-Wl,--start-group $(find ${mlir}/lib -name '*.a' -printf '-l%f ' | sed 's/-llib/-l/g;s/\.a//g') -Wl,--end-group"
            
            echo "llvm-config → $MLIR_SYS_200_PREFIX/bin/llvm-config"
            $MLIR_SYS_200_PREFIX/bin/llvm-config --version
          '';
        };
      }
    );
}
