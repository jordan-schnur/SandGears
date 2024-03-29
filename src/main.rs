mod shader;

use std::time::Instant;
use glium::Surface;
use winit::event_loop;

#[macro_use]
extern crate glium;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
implement_vertex!(Vertex, position);

fn main() {
    let event_loop = event_loop::EventLoopBuilder::new().build().expect("event loop building");
    let (window, display) = glium::backend::glutin::SimpleWindowBuilder::new().build(&event_loop);

    let mut target = display.draw();
    target.clear_color(0.0, 0.0, 1.0, 1.0);


    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [ 0.0,  0.5] };
    let vertex3 = Vertex { position: [ 0.5, -0.25] };
    let shape = vec![vertex1, vertex2, vertex3];
    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let program = shader::Program::new(&display, "./shaders/shader.vert", "./shaders/shader.frag");

    let mut t: f32 = 0.0;
    let mut last_frame_time = Instant::now();
    let mut frame_count = 0;
    let _ = event_loop.run(move |event, window_target| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => window_target.exit(),
                winit::event::WindowEvent::Resized(window_size) => {
                    display.resize(window_size.into());
                },
                winit::event::WindowEvent::RedrawRequested => {
                    // Calculate FPS
                    frame_count += 1;
                    let elapsed = last_frame_time.elapsed();
                    if elapsed.as_secs() >= 1 {
                        let fps = frame_count as f32 / elapsed.as_secs_f32();
                        println!("FPS: {:.2}", fps);
                        frame_count = 0;
                        last_frame_time = Instant::now();
                    }

                    // We update `t`
                    t += 0.02;
                    // We use the sine of t as an offset, this way we get a nice smooth animation
                    let x_off = t.sin() * 0.5;

                    let mut target = display.draw();
                    target.clear_color(0.0, 0.0, 1.0, 1.0);
                    let uniforms = uniform! { x_off: x_off };
                    target.draw(&vertex_buffer, &indices, &program.gl_program, &uniforms,
                                &Default::default()).unwrap();
                    target.finish().unwrap();
                },
                _ => (),
            },
            winit::event::Event::AboutToWait => {
                window.request_redraw();
            },
            _ => (),
        };
    });
}