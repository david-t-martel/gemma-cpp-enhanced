@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cargo clean
cargo build --manifest-path C:\codedev\llm\rag-redis\Cargo.toml --features "python-bindings,redis-backend"