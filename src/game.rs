//! The game: owns the world and turns player input into actions.
//!
//! # Controls
//!
//! | Input                    | Action                               |
//! |--------------------------|--------------------------------------|
//! | Left mouse (hold)        | Paint the selected element           |
//! | Left click on toolbar    | Select an element                    |
//! | Right mouse (hold)       | Erase                                |
//! | `1`–`5`                  | Select Sand / Water / Gas / Metal / Eraser |
//! | `[` and `]`              | Shrink / grow the brush              |
//! | Mouse wheel, pinch       | Zoom in / out at the mouse           |
//! | Middle drag, two-finger scroll | Pan while zoomed in            |
//! | `0`                      | Reset zoom                           |
//! | `Space`                  | Pause / resume                       |
//! | `.`                      | Advance one tick (while paused)      |
//! | `C`                      | Clear the world                      |
//! | `Esc`                    | Quit                                 |

use std::time::Duration;

use rand::rngs::ThreadRng;
use winit::event::{MouseButton, VirtualKeyCode};

use crate::camera::Camera;
use crate::sim::{physics, Element, World};
use crate::toolbar;

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

    /// Display cell under the mouse (toolbar rows first, then the world view), or
    /// `None` if the mouse is outside the window.
    mouse: Option<(i32, i32)>,
    camera: Camera,
    painting: bool,
    erasing: bool,
    panning: bool,
    /// Scratch copy of the world's colours with the brush outline on top, which the
    /// camera then samples from. Kept between frames to avoid reallocating.
    world_pixels: Vec<u32>,
    /// Where the brush was last frame, so fast strokes can be joined up.
    last_brush_position: Option<(i32, i32)>,
}

impl Game {
    pub fn new(world: World) -> Self {
        let size = (world.width(), world.height());
        Self {
            world,
            rng: rand::thread_rng(),
            selected: Element::Sand,
            brush_radius: DEFAULT_BRUSH_RADIUS,
            paused: false,
            step_once: false,
            tick_timer: 0.0,
            mouse: None,
            camera: Camera::new(size),
            painting: false,
            erasing: false,
            panning: false,
            world_pixels: vec![0; size.0 * size.1],
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
            Key::Key0 => self.camera.reset(),
            _ => {}
        }
        KeyResult::Continue
    }

    pub fn mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => {
                // A click on the toolbar only selects; it never starts a stroke.
                let on_button = self.mouse.and_then(toolbar::button_at);
                match (pressed, on_button) {
                    (true, Some(element)) => self.selected = element,
                    _ => self.painting = pressed && !self.over_toolbar(),
                }
            }
            MouseButton::Right => self.erasing = pressed,
            MouseButton::Middle => self.panning = pressed,
            _ => {}
        }
    }

    /// Zooms by `factor` around the mouse (or the top-left corner if it's off-screen).
    pub fn zoom(&mut self, factor: f32) {
        let anchor = self.mouse.map_or((0.0, 0.0), view_position);
        self.camera.zoom_at(anchor, factor);
    }

    /// Moves the view by a distance in display cells.
    pub fn pan(&mut self, delta: (f32, f32)) {
        self.camera.pan(delta);
    }

    /// `cell` is a display cell: the toolbar's rows come first, then the world view.
    pub fn set_cursor(&mut self, cell: Option<(i32, i32)>) {
        if let (true, Some(old), Some(new)) = (self.panning, self.mouse, cell) {
            // Dragging moves the world with the mouse, so the view moves the other way.
            self.camera
                .pan(((old.0 - new.0) as f32, (old.1 - new.1) as f32));
        }
        self.mouse = cell;
    }

    fn over_toolbar(&self) -> bool {
        matches!(self.mouse, Some((_, y)) if y < toolbar::HEIGHT as i32)
    }

    /// The world cell under the mouse, or `None` over the toolbar or outside the window.
    fn world_cursor(&self) -> Option<(i32, i32)> {
        if self.over_toolbar() {
            return None;
        }
        let (x, y) = self.camera.view_to_world(view_position(self.mouse?));
        Some((x.floor() as i32, y.floor() as i32))
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

        let Some(cursor) = self.world_cursor() else {
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

    /// Fills the display grid (one `0xAABBGGRR` per cell, row by row): the toolbar on
    /// top, and below it the part of the world the camera shows, with an outline of the
    /// brush drawn on top.
    pub fn draw(&mut self, pixels: &mut [u32]) {
        let (width, height) = (self.world.width(), self.world.height());

        let mut world_pixels = std::mem::take(&mut self.world_pixels);
        self.world.write_colors(&mut world_pixels);
        if let Some(cursor) = self.world_cursor() {
            self.draw_brush_outline(&mut world_pixels, cursor);
        }

        // Sample from each display cell's centre (nearest neighbour), so pixels stay crisp
        // at every zoom level. At 1x this copies the world unchanged.
        let view = &mut pixels[toolbar::HEIGHT * width..];
        for y in 0..height {
            let world_y = self.camera.view_to_world((0.0, y as f32 + 0.5)).1 as usize;
            let world_row = world_y.min(height - 1) * width;
            for x in 0..width {
                let world_x = self.camera.view_to_world((x as f32 + 0.5, 0.0)).0 as usize;
                view[y * width + x] = world_pixels[world_row + world_x.min(width - 1)];
            }
        }
        self.world_pixels = world_pixels;

        toolbar::draw(pixels, width, self.selected);
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
            "SandGears  |  {}  |  brush {}  |  zoom {:.1}x  |  {} particles  |  {:.0} fps{}",
            self.selected.name(),
            self.brush_radius,
            self.camera.zoom(),
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

/// Converts a display cell to a position in the world view, at the cell's centre.
fn view_position((x, y): (i32, i32)) -> (f32, f32) {
    (x as f32 + 0.5, (y - toolbar::HEIGHT as i32) as f32 + 0.5)
}

/// Blends a packed colour halfway towards white.
fn lighten(color: u32) -> u32 {
    let [r, g, b, a] = color.to_le_bytes();
    let towards_white = |c: u8| c / 2 + 128;
    u32::from_le_bytes([towards_white(r), towards_white(g), towards_white(b), a])
}
