// SPDX-License-Identifier: Apache-2.0

//! The SandGears game logic.
//!
//! This module is plain, **safe** Rust. It knows nothing about Vulkan,
//! windows or the GPU: it only reads [`Input`] and pushes the particles it
//! wants drawn into a [`Frame`]. All the unsafe rendering machinery lives
//! behind the [`crate::engine`] interface.

use nalgebra_glm as glm;
use rand::Rng;

use crate::stormcloud::{Frame, Game, Input, KeyCode};

/// Width of the simulation grid (also the window width in pixels).
pub const WIDTH: usize = 1024;
/// Height of the simulation grid (also the window height in pixels).
pub const HEIGHT: usize = 768;

/// The kinds of material a cell can hold.
///
/// The discriminants double as palette indices in the vertex shader, so the
/// order here must match the `COLORS` array in `shaders/shader.vert`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ParticleType {
    Water,
    Fire,
    Lava,
    Gas,
    Snow,
    Sand,
    #[default]
    Air,
    Metal,
}

impl ParticleType {
    /// Density in kg/m³. Heavier particles sink through lighter ones.
    fn density(&self) -> f32 {
        match self {
            ParticleType::Water => 1000.0,
            ParticleType::Fire => 300.0,
            ParticleType::Lava => 3100.0,
            ParticleType::Gas => 1.2,
            ParticleType::Snow => 200.0,
            ParticleType::Sand => 1600.0,
            ParticleType::Air => 1.2,
            ParticleType::Metal => 10000.0,
        }
    }

    /// Temperature in °C at which the particle melts.
    fn melting_temperature(&self) -> f32 {
        match self {
            ParticleType::Water => 0.0,
            ParticleType::Fire => 1200.0,
            ParticleType::Lava => 700.0,
            ParticleType::Gas => -100.0,
            ParticleType::Snow => -5.0,
            ParticleType::Sand => 1400.0,
            ParticleType::Air => -1.0,
            ParticleType::Metal => 3200.0,
        }
    }
}

/// A single cell of the simulation grid.
#[derive(Clone, Debug, Default)]
struct Particle {
    velocity: glm::Vec2,
    p_type: ParticleType,
    air_pressure: f32,
}

impl Particle {
    fn new(p_type: ParticleType) -> Self {
        Particle {
            velocity: glm::vec2(0.0, 0.0),
            p_type,
            air_pressure: 0.0,
        }
    }
}

/// The SandGears falling-sand simulation.
pub struct SandGears {
    screen: Vec<Vec<Particle>>,
    current_type: ParticleType,
    paused: bool,
}

impl SandGears {
    /// Creates an empty (all-air) world.
    pub fn new() -> Self {
        let mut screen = Vec::with_capacity(WIDTH);
        for _ in 0..WIDTH {
            screen.push(vec![Particle::new(ParticleType::Air); HEIGHT]);
        }

        Self {
            screen,
            current_type: ParticleType::Sand,
            paused: false,
        }
    }

    /// Returns whether a particle of `p_type` may move into cell `(x, y)`.
    fn can_move(&self, p_type: &ParticleType, x: usize, y: usize) -> bool {
        if x < 1 || x > WIDTH - 2 || y < 1 || y > HEIGHT - 2 || y >= 699 {
            return false;
        }

        let test_cell = &self.screen[x][y].p_type;

        if test_cell == &ParticleType::Air {
            return true;
        }

        if p_type.density() > test_cell.density() {
            return true;
        }

        false
    }

    /// Paints the currently selected material under the cursor while the mouse
    /// button is held down.
    fn add_particles(&mut self, input: &Input) {
        const CURSOR_SIZE: usize = 5;

        if input.mouse_down {
            let x: usize = input.mouse_position.0.floor() as usize;
            let y: usize = input.mouse_position.1.floor() as usize;
            let p_type = self.current_type;

            for px in 0..CURSOR_SIZE {
                for py in 0..CURSOR_SIZE {
                    if self.can_move(&p_type, px + x, py + x) {
                        self.screen[px + x][py + y] = Particle {
                            velocity: glm::vec2(0.0, 0.0),
                            p_type: self.current_type,
                            air_pressure: 0.0,
                        };
                    }
                }
            }
        }
    }

    /// Emits every non-air cell without advancing the simulation. Used while
    /// the world is paused so the contents stay visible.
    fn emit_all(&self, frame: &mut Frame) {
        for (x, column) in self.screen.iter().enumerate() {
            for (y, particle) in column.iter().enumerate() {
                if particle.p_type != ParticleType::Air {
                    frame.push(x as f32, y as f32, particle.p_type as u32);
                }
            }
        }
    }

    /// Advances the simulation by one step and emits the resulting particles.
    fn simulate(&mut self, frame: &mut Frame) {
        let mut updates: Vec<(usize, usize, usize, usize, f32)> = Vec::new();
        let mut rng = rand::thread_rng();

        for x in 0..self.screen.len() {
            for y in 0..self.screen[x].len() {
                match self.screen[x][y].p_type {
                    ParticleType::Air => continue,
                    ParticleType::Sand => {
                        let mut particle = self.screen[x][y].clone();
                        if y == 699 {
                            frame.push(x as f32, y as f32, particle.p_type as u32);

                            particle.velocity.y = 0.0;
                            self.screen[x][y] = particle;
                            continue;
                        }

                        let p_type = &ParticleType::Sand;
                        let mut new_x = x;
                        let mut new_y = if y as f32 >= 700.0 { 699 } else { y };

                        if self.can_move(p_type, new_x, new_y + 1) {
                            new_y += 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if self.can_move(p_type, new_x - 1, new_y + 1) {
                            new_y += 1;
                            new_x -= 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if self.can_move(p_type, new_x + 1, new_y + 1) {
                            new_y += 1;
                            new_x += 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        }

                        frame.push(new_x as f32, new_y as f32, particle.p_type as u32);
                    }
                    ParticleType::Water => {
                        let mut particle = self.screen[x][y].clone();
                        if y == 699 {
                            frame.push(x as f32, y as f32, particle.p_type as u32);

                            if particle.velocity.y != 0.0 {
                                particle.velocity.y = 0.0;
                                self.screen[x][y] = particle;
                            }

                            continue;
                        }

                        let p_type = &ParticleType::Water;
                        let mut new_x = x;
                        let mut new_y = if y as f32 >= 700.0 { 699 } else { y };

                        if self.can_move(p_type, new_x, new_y + 1) {
                            new_y += 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if self.can_move(p_type, new_x - 1, new_y + 1) {
                            new_y += 1;
                            new_x -= 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if self.can_move(p_type, new_x + 1, new_y + 1) {
                            new_y += 1;
                            new_x += 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        }

                        let random_direction = rng.gen_range(-1..2);

                        if random_direction == -1 && self.can_move(p_type, new_x - 1, new_y) {
                            new_x -= 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if random_direction == 1 && self.can_move(p_type, new_x + 1, new_y) {
                            new_x += 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        }

                        frame.push(new_x as f32, new_y as f32, particle.p_type as u32);
                    }
                    ParticleType::Gas => {
                        let particle = self.screen[x][y].clone();

                        let p_type = &ParticleType::Gas;
                        let mut new_x = x;
                        let mut new_y = y;

                        if self.can_move(p_type, new_x, new_y - 1) {
                            new_y -= 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if self.can_move(p_type, new_x - 1, new_y - 1) {
                            new_y -= 1;
                            new_x -= 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if self.can_move(p_type, new_x + 1, new_y - 1) {
                            new_y -= 1;
                            new_x += 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if self.can_move(p_type, new_x - 1, new_y) {
                            new_x -= 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        } else if self.can_move(p_type, new_x + 1, new_y) {
                            new_x += 1;
                            updates.push((x, y, new_x, new_y, 0.0));
                        }

                        frame.push(new_x as f32, new_y as f32, particle.p_type as u32);
                    }
                    ParticleType::Metal => {
                        frame.push(x as f32, y as f32, ParticleType::Metal as u32);
                    }
                    _ => continue,
                }
            }
        }

        for (old_x, old_y, new_x, new_y, updated_velocity) in updates {
            if old_x == new_x && old_y == new_y {
                continue;
            }

            let mut particle = self.screen[old_x][old_y].clone();
            let new_particle = self.screen[new_x][new_y].clone();
            particle.velocity.y = updated_velocity;
            self.screen[old_x][old_y] = new_particle;
            self.screen[new_x][new_y] = particle;
        }
    }
}

impl Default for SandGears {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for SandGears {
    fn update(&mut self, input: &Input, frame: &mut Frame) {
        // Material selection / pause toggling.
        if input.key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if let Some(p_type) = selected_type(input) {
            self.current_type = p_type;
        }

        self.add_particles(input);

        if self.paused {
            self.emit_all(frame);
        } else {
            self.simulate(frame);
        }
    }
}

/// Maps the number keys to the material they select, if any was pressed.
fn selected_type(input: &Input) -> Option<ParticleType> {
    if input.key_pressed(KeyCode::Key1) {
        Some(ParticleType::Water)
    } else if input.key_pressed(KeyCode::Key2) {
        Some(ParticleType::Sand)
    } else if input.key_pressed(KeyCode::Key3) {
        Some(ParticleType::Gas)
    } else if input.key_pressed(KeyCode::Key4) {
        Some(ParticleType::Metal)
    } else if input.key_pressed(KeyCode::Key5) {
        Some(ParticleType::Air)
    } else {
        None
    }
}
