// SPDX-License-Identifier: Apache-2.0

//! # Storm Cloud
//!
//! Storm Cloud is a small 2D particle engine that hides all the Vulkan and
//! windowing boilerplate behind a compact, **safe** interface. The goal is
//! that you can write game logic without ever thinking about device creation,
//! swapchains, command buffers or `unsafe`.
//!
//! To build a game on Storm Cloud you only need three things:
//!
//! * implement the [`Game`] trait — its [`update`](Game::update) method is
//!   called once per frame,
//! * read user input from the [`Input`] snapshot you are handed, and
//! * push the particles you want drawn into the [`Frame`].
//!
//! Hand your game to [`Engine::run`] and Storm Cloud takes care of creating
//! the window, initialising Vulkan, running the main loop and uploading your
//! particles to the GPU. No `unsafe` is required (or possible) in game code —
//! it is all confined to the [`vulkan`] submodule.

mod vulkan;

use std::collections::HashSet;

use anyhow::Result;
use image::io::Reader as ImageReader;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Icon, WindowBuilder};

use vulkan::Renderer;

/// The maximum number of particles that may be drawn in a single frame.
///
/// This is the capacity reserved on the GPU; [`Frame::push`] silently drops
/// any particles beyond this limit so the simulation can never overrun the
/// storage buffers.
pub use vulkan::MAX_PARTICLES;

/// Keyboard keys reported to game code.
///
/// `Escape` and the window close button are handled by the engine itself
/// (they quit the application) and are intentionally not exposed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyCode {
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Space,
}

impl KeyCode {
    fn from_winit(key: VirtualKeyCode) -> Option<Self> {
        Some(match key {
            VirtualKeyCode::Key1 => KeyCode::Key1,
            VirtualKeyCode::Key2 => KeyCode::Key2,
            VirtualKeyCode::Key3 => KeyCode::Key3,
            VirtualKeyCode::Key4 => KeyCode::Key4,
            VirtualKeyCode::Key5 => KeyCode::Key5,
            VirtualKeyCode::Space => KeyCode::Space,
            _ => return None,
        })
    }
}

/// A snapshot of user input for the current frame.
#[derive(Clone, Debug, Default)]
pub struct Input {
    /// Cursor position in window pixels, `(x, y)`.
    pub mouse_position: (f32, f32),
    /// Whether the left mouse button is currently held down.
    pub mouse_down: bool,
    /// Keys that transitioned to "pressed" during this frame.
    keys_pressed: HashSet<KeyCode>,
}

impl Input {
    /// Returns `true` if `key` was pressed during this frame.
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }
}

/// The set of particles to draw this frame.
///
/// Game code fills this in [`Game::update`] via [`push`](Frame::push); the
/// engine uploads it to the GPU and renders one quad per particle. The frame
/// is cleared automatically before each call to [`Game::update`].
#[derive(Clone, Debug, Default)]
pub struct Frame {
    positions: Vec<f32>,
    kinds: Vec<u32>,
}

impl Frame {
    /// Removes all queued particles.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.kinds.clear();
    }

    /// Queues a particle at `(x, y)` drawn with palette index `kind`.
    ///
    /// Particles beyond [`MAX_PARTICLES`] are dropped.
    pub fn push(&mut self, x: f32, y: f32, kind: u32) {
        if self.kinds.len() >= MAX_PARTICLES {
            return;
        }
        self.positions.push(x);
        self.positions.push(y);
        self.kinds.push(kind);
    }

    /// Number of particles queued this frame.
    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    /// Returns `true` when no particles are queued.
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// Interleaved `x, y` positions, two floats per particle.
    pub(crate) fn positions(&self) -> &[f32] {
        &self.positions
    }

    /// One palette index per particle.
    pub(crate) fn kinds(&self) -> &[u32] {
        &self.kinds
    }
}

/// Game logic driven by the engine.
///
/// Implementors are pure, safe Rust and never touch the renderer directly.
/// Each frame the engine clears `frame`, then calls [`update`](Game::update)
/// with the latest [`Input`] for the game to advance its state and emit the
/// particles that should be drawn.
pub trait Game {
    fn update(&mut self, input: &Input, frame: &mut Frame);
}

/// Window and renderer configuration.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Window title.
    pub title: String,
    /// Window width in pixels.
    pub width: u32,
    /// Window height in pixels.
    pub height: u32,
    /// Optional path to a PNG to use as the window icon.
    pub icon_path: Option<String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            title: "Game".to_string(),
            width: 1024,
            height: 768,
            icon_path: None,
        }
    }
}

/// Drives a [`Game`]: owns the window, the Vulkan renderer and the main loop.
pub struct Engine;

impl Engine {
    /// Opens a window and runs `game` until the user quits.
    ///
    /// This call does not return: it hands control to the OS event loop and
    /// exits the process when the window is closed.
    pub fn run<G: Game + 'static>(config: EngineConfig, mut game: G) -> Result<()> {
        pretty_env_logger::init();

        let event_loop = EventLoop::new();

        let mut builder = WindowBuilder::new()
            .with_title(config.title.clone())
            .with_inner_size(LogicalSize::new(config.width, config.height));

        if let Some(icon) = config.icon_path.as_deref().and_then(load_icon) {
            builder = builder.with_window_icon(Some(icon));
        }

        let window = builder.build(&event_loop)?;

        let mut renderer = unsafe { Renderer::create(&window)? };
        let mut input = Input::default();
        let mut frame = Frame::default();
        let mut destroying = false;
        let mut minimized = false;

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                // Advance the game and render a frame.
                Event::MainEventsCleared if !destroying && !minimized => {
                    frame.clear();
                    game.update(&input, &mut frame);
                    input.keys_pressed.clear();
                    unsafe { renderer.render(&window, &frame) }.unwrap();
                }
                // Track window resizing / minimising.
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    if size.width == 0 || size.height == 0 {
                        minimized = true;
                    } else {
                        minimized = false;
                        renderer.resized = true;
                    }
                }
                // Mouse movement.
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { position, .. },
                    ..
                } => {
                    input.mouse_position = (position.x as f32, position.y as f32);
                }
                // Left mouse button.
                Event::WindowEvent {
                    event:
                        WindowEvent::MouseInput {
                            state,
                            button: MouseButton::Left,
                            ..
                        },
                    ..
                } => {
                    input.mouse_down = state == ElementState::Pressed;
                }
                // Keyboard. Escape quits; everything else is forwarded to the game.
                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            input: key_input, ..
                        },
                    ..
                } => {
                    if key_input.state != ElementState::Pressed {
                        return;
                    }

                    if let Some(keycode) = key_input.virtual_keycode {
                        if keycode == VirtualKeyCode::Escape {
                            destroying = true;
                            *control_flow = ControlFlow::Exit;
                            unsafe {
                                renderer.destroy();
                            }
                            return;
                        }

                        if let Some(key) = KeyCode::from_winit(keycode) {
                            input.keys_pressed.insert(key);
                        }
                    }
                }
                // Window close button.
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    destroying = true;
                    *control_flow = ControlFlow::Exit;
                    unsafe {
                        renderer.destroy();
                    }
                }
                _ => {}
            }
        });
    }
}

/// Loads a PNG from `path` as a window icon, returning `None` on any failure.
fn load_icon(path: &str) -> Option<Icon> {
    let image = ImageReader::open(path).ok()?.decode().ok()?.to_rgba8();
    let (width, height) = (image.width(), image.height());
    Icon::from_rgba(image.into_raw(), width, height).ok()
}
