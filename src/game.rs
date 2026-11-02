//! The game: owns the world and turns player input into actions.
//!
//! # Controls
//!
//! | Input                    | Action                               |
//! |--------------------------|--------------------------------------|
//! | Left mouse (hold)        | Paint the selected element           |
//! | Right mouse (hold)       | Erase                                |
//! | `1`–`5`                  | Select Sand / Water / Gas / Metal / Eraser |
//! | Mouse wheel, `[` and `]` | Shrink / grow the brush              |
//! | `Space`                  | Pause / resume                       |
//! | `.`                      | Advance one tick (while paused)      |
//! | `C`                      | Clear the world                      |
//! | `Esc`                    | Quit                                 |

use std::time::Duration;

use rand::rngs::ThreadRng;
use winit::event::{MouseButton, VirtualKeyCode};

use crate::sim::{physics, Element, World};

/// Simulation ticks per second. Fixed, so sand falls at the same speed on every
/// machine no matter how fast it renders.
const TICKS_PER_SECOND: f32 = 60.0;
/// If a frame takes very long (e.g. the window was dragged) don't try to catch up
/// with hundreds of ticks at once.
const MAX_TICKS_PER_FRAME: u32 = 4;

const MIN_BRUSH_RADIUS: i32 = 0;
const MAX_BRUSH_RADIUS: i32 = 40;
const DEFAULT_BRUSH_RADIUS: i32 = 4;

/// What the main loop should do after a key press.
#[derive(Debug, PartialEq, Eq)]
pub enum KeyResult {
    Continue,
    Quit,
}

pub struct Game {
    world: World,
    rng: ThreadRng,

    selected: Element,
    brush_radius: i32,
    paused: bool,
    step_once: bool,
    tick_timer: f32,

    /// Cell under the mouse, or `None` if the mouse is outside the window.
    cursor: Option<(i32, i32)>,
    painting: bool,
    erasing: bool,
    /// Where the brush was last frame, so fast strokes can be joined up.
    last_brush_position: Option<(i32, i32)>,
}

impl Game {
    pub fn new(world: World) -> Self {
        Self {
            world,
            rng: rand::thread_rng(),
            selected: Element::Sand,
            brush_radius: DEFAULT_BRUSH_RADIUS,
            paused: false,
            step_once: false,
            tick_timer: 0.0,
            cursor: None,
            painting: false,
            erasing: false,
            last_brush_position: None,
        }
    }

    // ---------------------------------------------------------------------------------
    // Input
    // ---------------------------------------------------------------------------------

    pub fn key_pressed(&mut self, key: VirtualKeyCode) -> KeyResult {
        use VirtualKeyCode as Key;

        match key {
            Key::Escape => return KeyResult::Quit,
            Key::Key1 => self.selected = Element::Sand,
            Key::Key2 => self.selected = Element::Water,
            Key::Key3 => self.selected = Element::Gas,
            Key::Key4 => self.selected = Element::Metal,
            Key::Key5 => self.selected = Element::Empty,
            Key::LBracket | Key::Minus => self.resize_brush(-1),
            Key::RBracket | Key::Equals => self.resize_brush(1),
            Key::Space => self.paused = !self.paused,
            Key::Period => self.step_once = true,
            Key::C => self.world.clear(),
            _ => {}
        }
        KeyResult::Continue
    }

    pub fn mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.painting = pressed,
            MouseButton::Right => self.erasing = pressed,
            _ => {}
        }
    }

    /// `steps` is positive when scrolling up (grow) and negative when scrolling down.
    pub fn scroll(&mut self, steps: i32) {
        self.resize_brush(steps);
    }

    pub fn set_cursor(&mut self, cell: Option<(i32, i32)>) {
        self.cursor = cell;
    }

    fn resize_brush(&mut self, by: i32) {
        self.brush_radius = (self.brush_radius + by).clamp(MIN_BRUSH_RADIUS, MAX_BRUSH_RADIUS);
    }

    // ---------------------------------------------------------------------------------
    // Update
    // ---------------------------------------------------------------------------------

    /// Called once per rendered frame with the time since the previous frame.
    pub fn update(&mut self, frame_time: Duration) {
        self.apply_brush();

        if self.paused {
            self.tick_timer = 0.0;
            if std::mem::take(&mut self.step_once) {
                physics::step(&mut self.world, &mut self.rng);
            }
            return;
        }

        let tick_length = 1.0 / TICKS_PER_SECOND;
        self.tick_timer += frame_time.as_secs_f32();

        let mut ticks = 0;
        while self.tick_timer >= tick_length && ticks < MAX_TICKS_PER_FRAME {
            physics::step(&mut self.world, &mut self.rng);
            self.tick_timer -= tick_length;
            ticks += 1;
        }
        if ticks == MAX_TICKS_PER_FRAME {
            self.tick_timer = 0.0; // fell behind; drop the backlog
        }
    }

    fn apply_brush(&mut self) {
        let element = if self.erasing {
            Element::Empty
        } else if self.painting {
            self.selected
        } else {
            self.last_brush_position = None;
            return;
        };

        let Some(cursor) = self.cursor else {
            self.last_brush_position = None;
            return;
        };

        let from = self.last_brush_position.unwrap_or(cursor);
        self.world
            .paint_line(from, cursor, self.brush_radius, element, &mut self.rng);
        self.last_brush_position = Some(cursor);
    }

    // ---------------------------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------------------------

    /// Writes the colour of every cell into `pixels` (one `0xAABBGGRR` per cell, row by
    /// row), with an outline of the brush drawn on top.
    pub fn draw(&self, pixels: &mut [u32]) {
        self.world.write_colors(pixels);
        if let Some(cursor) = self.cursor {
            self.draw_brush_outline(pixels, cursor);
        }
    }

    fn draw_brush_outline(&self, pixels: &mut [u32], (cx, cy): (i32, i32)) {
        let r = self.brush_radius;
        let inner = (r as f32 - 0.5).max(0.0).powi(2);
        let outer = (r as f32 + 0.5).powi(2);

        for y in (cy - r - 1)..=(cy + r + 1) {
            for x in (cx - r - 1)..=(cx + r + 1) {
                let distance = ((x - cx).pow(2) + (y - cy).pow(2)) as f32;
                let on_edge = distance <= outer && (distance >= inner || r == 0);
                if on_edge && self.world.in_bounds(x, y) {
                    let index = y as usize * self.world.width() + x as usize;
                    pixels[index] = lighten(pixels[index]);
                }
            }
        }
    }

    /// One-line summary shown in the window title (there's no in-game text yet).
    pub fn status_line(&self, fps: f32) -> String {
        format!(
            "SandGears  |  {}  |  brush {}  |  {} particles  |  {:.0} fps{}",
            self.selected.name(),
            self.brush_radius,
            self.world.particle_count(),
            fps,
            if self.paused {
                "  |  PAUSED (space)"
            } else {
                ""
            },
        )
    }
}

/// Blends a packed colour halfway towards white.
fn lighten(color: u32) -> u32 {
    let [r, g, b, a] = color.to_le_bytes();
    let towards_white = |c: u8| c / 2 + 128;
    u32::from_le_bytes([towards_white(r), towards_white(g), towards_white(b), a])
}
