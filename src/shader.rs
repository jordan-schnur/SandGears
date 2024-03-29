use glium::Display;
use glium::glutin::surface::{ResizeableSurface, SurfaceTypeTrait};
use std::fs;

pub struct Program {
    pub gl_program: glium::Program,
}

impl Program {
    pub fn new<T: SurfaceTypeTrait + ResizeableSurface + 'static>(
        display: &Display<T>,
        vertex_shader_path: &str,
        fragment_shader_path: &str,
    ) -> Self {
        let vertex_shader_src = fs::read_to_string(vertex_shader_path).unwrap();
        let fragment_shader_src = fs::read_to_string(fragment_shader_path).unwrap();

        let shader =
            glium::Program::from_source(display, &vertex_shader_src, &fragment_shader_src, None)
                .unwrap();

        Program { gl_program: shader }
    }
}