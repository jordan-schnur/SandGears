//! Build script: compiles the GLSL shaders in `shaders/` into SPIR-V.
//!
//! The compiled files are written to Cargo's `OUT_DIR` and embedded into the binary
//! by `src/render/pipeline.rs` with `include_bytes!`. Using `naga` (pure Rust) means
//! players don't need the Vulkan SDK's `glslc` just to build the game.

use std::path::Path;
use std::{env, fs};

use naga::back::spv;
use naga::front::glsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::ShaderStage;

const SHADERS: &[(&str, ShaderStage)] = &[
    ("world.vert", ShaderStage::Vertex),
    ("world.frag", ShaderStage::Fragment),
];

fn main() {
    let out_dir = env::var("OUT_DIR").expect("cargo always sets OUT_DIR");

    for (file_name, stage) in SHADERS {
        let source_path = Path::new("shaders").join(file_name);
        println!("cargo:rerun-if-changed={}", source_path.display());

        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|e| panic!("could not read {}: {e}", source_path.display()));

        let spirv = compile_glsl(&source, *stage)
            .unwrap_or_else(|e| panic!("could not compile {}:\n{e}", source_path.display()));

        let bytes: Vec<u8> = spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
        let out_path = Path::new(&out_dir).join(format!("{file_name}.spv"));
        fs::write(&out_path, bytes).expect("could not write compiled shader");
    }
}

fn compile_glsl(source: &str, stage: ShaderStage) -> Result<Vec<u32>, String> {
    let module = glsl::Frontend::default()
        .parse(&glsl::Options::from(stage), source)
        .map_err(|e| e.emit_to_string(source))?;

    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| e.emit_to_string(source))?;

    let options = spv::Options {
        // Our shaders are written for Vulkan's coordinate system already; don't let
        // naga flip Y the way it would for wgpu-style shaders.
        flags: spv::WriterFlags::empty(),
        ..Default::default()
    };
    let pipeline_options = spv::PipelineOptions {
        shader_stage: stage,
        entry_point: "main".into(),
    };

    spv::write_vec(&module, &info, &options, Some(&pipeline_options)).map_err(|e| e.to_string())
}
