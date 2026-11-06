// SPDX-License-Identifier: Apache-2.0

//! SandGears – a falling-sand sandbox.
//!
//! This file only wires things together: it opens the window, forwards input to the
//! [`Game`], and asks the [`Renderer`] to draw. The interesting parts live in:
//!
//! * `game.rs`   – controls, the brush, pausing, ticking the simulation.
//! * `toolbar.rs` – the clickable material buttons above the world.
//! * `camera.rs` – zoom and pan.
//! * `sim/`      – the world grid, the materials, and the physics rules.
//! * `render/`   – Vulkan code that draws a grid of colours to the window.

mod camera;
mod game;
mod render;
mod sim;
mod toolbar;
mod viewport;

use std::time::{Duration, Instant};

use anyhow::Result;
use log::error;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, KeyboardInput, MouseScrollDelta, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Icon, WindowBuilder};

use game::{Game, KeyResult};
use render::Renderer;
use sim::World;
use viewport::Viewport;

/// Size of the world in cells.
const WORLD_WIDTH: u32 = 512;
const WORLD_HEIGHT: u32 = 384;
/// Each cell is this many (logical) pixels wide in the initial window.
const PIXELS_PER_CELL: u32 = 2;

/// How often the window title (our stand-in HUD) is refreshed.
const TITLE_REFRESH: Duration = Duration::from_millis(250);

fn main() -> Result<()> {
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "sandgears=info");
    }
    pretty_env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("SandGears")
        .with_inner_size(LogicalSize::new(
            WORLD_WIDTH * PIXELS_PER_CELL,
            (WORLD_HEIGHT + toolbar::HEIGHT as u32) * PIXELS_PER_CELL,
        ))
        .with_window_icon(load_icon())
        .build(&event_loop)?;

    // The renderer draws a grid of colours; ours is the toolbar stacked on the world view.
    let display_size = (WORLD_WIDTH, WORLD_HEIGHT + toolbar::HEIGHT as u32);
    let mut renderer = Some(Renderer::new(&window, display_size)?);
    let mut game = Game::new(World::new(WORLD_WIDTH as usize, WORLD_HEIGHT as usize));
    let mut fps = FpsCounter::new();
    let mut minimized = false;

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),

                WindowEvent::Resized(size) => {
                    minimized = size.width == 0 || size.height == 0;
                    if let Some(renderer) = renderer.as_mut() {
                        renderer.window_resized();
                    }
                }

                WindowEvent::CursorMoved { position, .. } => {
                    game.set_cursor(Some(cursor_to_cell(
                        position,
                        window.inner_size(),
                        display_size,
                    )));
                }
                WindowEvent::CursorLeft { .. } => game.set_cursor(None),

                WindowEvent::MouseInput { state, button, .. } => {
                    game.mouse_button(button, state == ElementState::Pressed);
                }

                // A mouse wheel zooms. A trackpad's two-finger scroll pans and its pinch
                // zooms, matching how macOS apps usually treat the two.
                WindowEvent::MouseWheel { delta, .. } => match delta {
                    MouseScrollDelta::LineDelta(_, y) => game.zoom(1.1_f32.powf(y)),
                    MouseScrollDelta::PixelDelta(d) => {
                        let viewport = Viewport::fit(
                            (window.inner_size().width, window.inner_size().height),
                            display_size,
                        );
                        let cells_per_pixel = display_size.0 as f32 / viewport.width;
                        game.pan((-d.x as f32 * cells_per_pixel, -d.y as f32 * cells_per_pixel));
                    }
                },
                WindowEvent::TouchpadMagnify { delta, .. } => game.zoom(1.0 + delta as f32),

                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                } => {
                    if game.key_pressed(key) == KeyResult::Quit {
                        control_flow.set_exit();
                    }
                }

                _ => {}
            },

            // All input for this frame has been handled: update, then draw.
            Event::MainEventsCleared => {
                let frame_time = fps.tick();
                game.update(frame_time);

                if let (Some(renderer), false) = (renderer.as_mut(), minimized) {
                    if let Err(e) = renderer.render(&window, |pixels| game.draw(pixels)) {
                        error!("Rendering failed: {e:?}");
                        control_flow.set_exit();
                    }
                }

                if let Some(fps) = fps.refreshed_value() {
                    window.set_title(&game.status_line(fps));
                }
            }

            // Release GPU resources before the window goes away.
            Event::LoopDestroyed => drop(renderer.take()),

            _ => {}
        }
    });
}

fn cursor_to_cell(
    position: PhysicalPosition<f64>,
    window_size: PhysicalSize<u32>,
    world_size: (u32, u32),
) -> (i32, i32) {
    Viewport::fit((window_size.width, window_size.height), world_size)
        .window_to_world(position.x, position.y, world_size)
}

/// The icon is compiled into the binary so the game works from any directory.
fn load_icon() -> Option<Icon> {
    let bytes = include_bytes!("../resources/App/Icon.png");
    let image = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (width, height) = image.dimensions();
    Icon::from_rgba(image.into_raw(), width, height).ok()
}

/// Measures time between frames and averages frames-per-second for the title bar.
struct FpsCounter {
    last_frame: Instant,
    last_refresh: Instant,
    frames_since_refresh: u32,
}

impl FpsCounter {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            last_frame: now,
            last_refresh: now,
            frames_since_refresh: 0,
        }
    }

    /// Call once per frame. Returns the time since the previous call.
    fn tick(&mut self) -> Duration {
        let now = Instant::now();
        let elapsed = now - self.last_frame;
        self.last_frame = now;
        self.frames_since_refresh += 1;
        elapsed
    }

    /// Returns the average FPS every [`TITLE_REFRESH`], otherwise `None`.
    fn refreshed_value(&mut self) -> Option<f32> {
        let elapsed = self.last_refresh.elapsed();
        if elapsed < TITLE_REFRESH {
            return None;
        }
        let fps = self.frames_since_refresh as f32 / elapsed.as_secs_f32();
        self.last_refresh = Instant::now();
        self.frames_since_refresh = 0;
        Some(fps)
    }
}
