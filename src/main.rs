// SPDX-License-Identifier: Apache-2.0

#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use std::collections::HashSet;
use std::ffi::CStr;
use std::fs::File;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use log::*;
use nalgebra_glm as glm;
use rand::Rng;
use thiserror::Error;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;
use vulkanalia::Version;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
/// The name of the validation layers.
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// The required device extensions.
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
/// The Vulkan SDK version that started requiring the portability subset extension for macOS.
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

/// The maximum number of frames that can be processed concurrently.
const MAX_FRAMES_IN_FLIGHT: usize = 2;

const MAX_PARTICLE_COUNT: usize = 100000;

const WIDTH: usize = 1024;
const HEIGHT: usize = 768;

lazy_static! {
    #[rustfmt::skip]
    static ref VERTICES: Vec<Vertex> = vec![
        Vertex::new(glm::vec3(0.0, 0.0, 0.0),glm::vec3(1.0, 0.0, 0.0),glm::vec2(1.0, 0.0)),
        Vertex::new(glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0), glm::vec2(0.0, 0.0)),
        Vertex::new(glm::vec3(1.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0), glm::vec2(0.0, 1.0)),
        Vertex::new(glm::vec3(0.0, 1.0, 0.0), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 1.0)),
        //
        Vertex::new(glm::vec3(100.0, 100.0, 0.0), glm::vec3(1.0, 0.0, 0.0), glm::vec2(1.0, 0.0)),
        Vertex::new(glm::vec3(500.0, 100.0, 0.0), glm::vec3(0.0, 1.0, 0.0), glm::vec2(0.0, 0.0)),
        Vertex::new(glm::vec3(500.0, 500.0, 0.0), glm::vec3(0.0, 0.0, 1.0), glm::vec2(0.0, 1.0)),
        Vertex::new(glm::vec3(100.0, 500.0, 0.0), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 1.0)),
    ];
}

#[rustfmt::skip]
const INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0,
];

const SIZE_OF_PARTICLE_POSITION: u64 = (std::mem::size_of::<f64>() * 2 * MAX_PARTICLE_COUNT) as u64;
const SIZE_OF_PARTICLE_COLOR: u64 = (std::mem::size_of::<u32>() * 1 * MAX_PARTICLE_COUNT) as u64;

#[rustfmt::skip]
fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App

    let mut app = unsafe { App::create(&window)? };
    let mut destroying = false;
    let mut minimized = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::MainEventsCleared if !destroying && !minimized => unsafe { app.render(&window) }.unwrap(),
            // Mark the window as having been resized.
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { device_id: _, position, .. }, .. } => {
                app.mouse_position = glm::vec2(position.x as f32, position.y as f32);
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. }, .. } => {
                app.is_mouse_button_down = true;
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left, .. }, .. } => {
                app.is_mouse_button_down = false;
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if let Some(keycode) = input.virtual_keycode {
                    use winit::event::VirtualKeyCode;
                    
                    if input.state != ElementState::Pressed {
                        return;
                    }
                    
                    if VirtualKeyCode::Escape == keycode {
                        destroying = true;
                        *control_flow = ControlFlow::Exit;
                        unsafe { app.destroy(); }
                        
                        return;
                    }
                    
                    if VirtualKeyCode::Space == keycode {
                        app.paused = !app.paused;
                    }

                    let particle_type = match keycode {
                        VirtualKeyCode::Key1 => ParticleType::Water,
                        VirtualKeyCode::Key2 => ParticleType::Sand,
                        VirtualKeyCode::Key3 => ParticleType::Gas,
                        VirtualKeyCode::Key4 => ParticleType::Metal,
                        VirtualKeyCode::Key5 => ParticleType::Air,
                        _ => return,
                    };
                    
                    info!("Selected {:?}", particle_type);

                    app.current_type = particle_type;
                }
            }
            // Destroy our Vulkan app.
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                destroying = true;
                *control_flow = ControlFlow::Exit;
                unsafe { app.destroy(); }
            },
            _ => {}
        }
    });
}

/// Our Vulkan app.
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    start: Instant,
    mouse_position: glm::Vec2,
    is_mouse_button_down: bool,
    screen: Vec<Vec<Particle>>,
    last_frame: f32,
    fps: f32,
    frames_since_last_second: usize,
    num_active_particles: usize,
    active_particle_positions: Vec<f32>,
    active_particle_types: Vec<u32>,
    current_type: ParticleType,
    paused: bool,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        create_depth_objects(&instance, &device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_storage_buffer(&device, &instance, &mut data)?;
        create_vertex_buffer(&instance, &device, &mut data)?;
        create_index_buffer(&instance, &device, &mut data)?;
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        let mut screen = vec![];

        for x in 0..WIDTH {
            screen.push(vec![]);
            for y in 0..HEIGHT {
                screen[x].push(Particle::new(ParticleType::Air));
            }
        }

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            mouse_position: glm::vec2(0.0, 0.0),
            is_mouse_button_down: false,
            screen,
            last_frame: 0 as f32,
            fps: 0 as f32,
            frames_since_last_second: 0,
            num_active_particles: 0,
            active_particle_positions: vec![],
            active_particle_types: vec![],
            current_type: ParticleType::Sand,
            paused: false,
        })
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device
                .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        self.update_command_buffer(image_index)?;
        self.update_uniform_buffer(image_index)?;
        self.add_particles();

        if !self.paused {
            self.simulate()?;
        }

        self.update_storage_buffers(image_index)?;

        self.calc_fps();

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[in_flight_fence])?;

        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    fn calc_fps(&mut self) {
        let now = self.start.elapsed().as_secs_f32();

        if (now - self.last_frame) >= 1.0 {
            self.fps = self.frames_since_last_second as f32;
            self.frames_since_last_second = 0;
            self.last_frame = now;

            info!("FPS: {}", self.fps);
            info!("Particles: {}", self.num_active_particles);
        } else {
            self.frames_since_last_second += 1;
        }
    }

    unsafe fn is_cell_filled(&self, x: usize, y: usize) -> bool {
        if x < 1 || x > WIDTH - 1 || y < 1 || y > HEIGHT {
            return true;
        }

        if self.screen[x][y].p_type == ParticleType::Air {
            return false;
        }

        if y >= 699 {
            return true;
        }

        return true;
    }

    unsafe fn can_move(&self, p_type: &ParticleType, x: usize, y: usize) -> bool {
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

        return false;
    }

    unsafe fn add_particles(&mut self) {
        const CURSOR_SIZE: usize = 5;

        if self.is_mouse_button_down {
            let x: usize = self.mouse_position.x.floor() as usize;
            let y: usize = self.mouse_position.y.floor() as usize;
            let p_type = self.current_type.clone();

            for px in 0..CURSOR_SIZE {
                for py in 0..CURSOR_SIZE {
                    if self.can_move(&p_type, px + x, py + x) {
                        let new_particle = Particle {
                            velocity: glm::vec2(0.0, 0.0),
                            p_type: self.current_type.clone(),
                            air_pressure: 0.0,
                        };

                        self.screen[px + x][py + y] = new_particle;
                    }
                }
            }
        }
    }

    unsafe fn simulate(&mut self) -> Result<()> {
        self.active_particle_positions.clear();
        self.active_particle_types.clear();
        let time = self.start.elapsed().as_secs_f32();

        const GRAVITY: f32 = 0.01;
        let mut num_active_particles = 0;

        let mut updates: Vec<
            (usize, usize, 
             usize, usize,
             f32)> = Vec::new();
        let mut rng = rand::thread_rng();

        for x in 0..self.screen.len() {
            for y           in 0..self.screen[x].len() {
                match self.screen[x][y].p_type {
                                        ParticleType::Air => continue,
                                    ParticleType::Sand => {
                        let mut particle = self.screen[x][y].clone();
                        num_active_particles += 1;
                        if y == 699 {
                            self.active_particle_positions.push(x as f32);
                            self.active_particle_positions.push(y as f32);
                            self.active_particle_types
                                .push(particle.p_type.clone() as u32);

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

                        self.active_particle_positions.push(new_x as f32);
                        self.active_particle_positions.push(new_y as f32);
                        self.active_particle_types.push(particle.p_type as u32);
                    }
                    ParticleType::Water => {
                        let mut particle = self.screen[x][y].clone();
                        num_active_particles += 1;
                        if y == 699 {
                            self.active_particle_positions.push(x as f32);
                            self.active_particle_positions.push(y as f32);
                            self.active_particle_types
                                .push(particle.p_type.clone() as u32);

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

                        self.active_particle_positions.push(new_x as f32);
                        self.active_particle_positions.push(new_y as f32);
                        self.active_particle_types.push(particle.p_type as u32);
                    }
                    ParticleType::Gas => {
                        let particle = self.screen[x][y].clone();
                        num_active_particles += 1;

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

                        self.active_particle_positions.push(new_x as f32);
                        self.active_particle_positions.push(new_y as f32);
                        self.active_particle_types.push(particle.p_type as u32);
                    }
                    ParticleType::Metal => {
                        num_active_particles += 1;
                        self.active_particle_positions.push(x as f32);
                        self.active_particle_positions.push(y as f32);
                        self.active_particle_types.push(ParticleType::Metal as u32);
                    }
                    _ => {
                        continue;
                    }
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

        self.num_active_particles = num_active_particles;
        Ok(())
    }

    unsafe fn update_storage_buffers(&mut self, image_index: usize) -> Result<()> {
        if self.num_active_particles == 0 {
            return Ok(());
        }

        let time = self.start.elapsed().as_secs_f32();

        let data_ptr = self.device.map_memory(
            self.data.particle_position_storage_buffer_memory,
            0,
            (self.num_active_particles * std::mem::size_of::<f32>() * 2) as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        let (prefix, aligned, suffix) = self.active_particle_positions.align_to::<f32>();

        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(anyhow!("Particles are not properly aligned."));
        }

        std::ptr::copy_nonoverlapping(
            aligned.as_ptr(),
            data_ptr.cast(),
            self.num_active_particles * 2,
        );

        self.device
            .unmap_memory(self.data.particle_position_storage_buffer_memory);

        let data_ptr_2 = self.device.map_memory(
            self.data.particle_color_storage_buffer_memory,
            0,
            (self.num_active_particles * std::mem::size_of::<u32>()) as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        let (prefix, aligned2, suffix) = self.active_particle_types.align_to::<u32>();

        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(anyhow!("Particles are not properly aligned."));
        }

        std::ptr::copy_nonoverlapping(
            aligned2.as_ptr(),
            data_ptr_2.cast(),
            self.num_active_particles,
        );

        self.device
            .unmap_memory(self.data.particle_color_storage_buffer_memory);

        Ok(())
    }

    /// Updates the uniform buffer object for our Vulkan app.
    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        // MVP

        let time = self.start.elapsed().as_secs_f32();

        let model = glm::identity();

        let view = glm::translate(&glm::identity(), &glm::vec3(0.0, 0.0, -1.0));

        let proj = glm::ortho_rh_zo(
            0.0,
            self.data.swapchain_extent.width as f32,
            0.0,
            self.data.swapchain_extent.height as f32,
            0.001, // This sets the depth range from [0, 1] to [-1, 1].
            10.0,
        );

        let ubo = UniformBufferObject { model, view, proj };
        // Copy

        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.device
            .unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        let command_buffer = self.data.command_buffers[image_index];

        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;

        let memory_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        self.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::DependencyFlags::empty(),
            &[memory_barrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[] as &[vk::ImageMemoryBarrier],
        );

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.device
            .cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE);
        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline,
        );
        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.vertex_buffer], &[0]);
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.data.index_buffer,
            0,
            vk::IndexType::UINT16,
        );
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );
        self.device.cmd_draw_indexed(
            command_buffer,
            INDICES.len() as u32,
            self.num_active_particles as u32,
            0,
            0,
            0,
        );
        self.device.cmd_end_render_pass(command_buffer);

        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    /// Recreates the swapchain for our Vulkan app.
    #[rustfmt::skip]
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;
        self.data.images_in_flight.resize(self.data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    /// Destroys our Vulkan app.
    #[rustfmt::skip]
    unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.device.free_memory(self.data.particle_position_storage_buffer_memory, None);
        self.device.destroy_buffer(self.data.particle_position_storage_buffer, None);
        self.device.free_memory(self.data.particle_color_storage_buffer_memory, None);
        self.device.destroy_buffer(self.data.particle_color_storage_buffer, None);
        self.data.in_flight_fences.iter().for_each(|f| self.device.destroy_fence(*f, None));
        self.data.render_finished_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.image_available_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    /// Destroys the parts of our Vulkan app related to the swapchain.
    #[rustfmt::skip]
    unsafe fn destroy_swapchain(&mut self) {
        self.device.free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data.uniform_buffers_memory.iter().for_each(|m| self.device.free_memory(*m, None));
        self.data.uniform_buffers.iter().for_each(|b| self.device.destroy_buffer(*b, None));
        self.device.destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    // Debug
    messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Framebuffers
    framebuffers: Vec<vk::Framebuffer>,
    // Command Pool
    command_pool: vk::CommandPool,
    // Depth
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    // Buffers
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_particle_buffers: Vec<vk::Buffer>,
    uniform_particle_buffers_memory: Vec<vk::DeviceMemory>,
    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    // Command Buffers
    command_buffers: Vec<vk::CommandBuffer>,
    // Sync Objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
    particle_position_storage_buffer: vk::Buffer,
    particle_position_storage_buffer_memory: vk::DeviceMemory,
    particle_color_storage_buffer: vk::Buffer,
    particle_color_storage_buffer_memory: vk::DeviceMemory,
    particles: Vec<Particle>,
}

struct InstanceBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
    descriptor: vk::DescriptorBufferInfo,
}

//================================================
// Instance
//================================================

unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // Application Info

    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial (Rust)\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Layers

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    if VALIDATION_ENABLED
        && !available_layers.contains(&vk::ExtensionName::from_bytes(b"VK_LAYER_LUNARG_api_dump"))
    {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // layers.push(vk::ExtensionName::from_bytes(b"VK_LAYER_LUNARG_api_dump").as_ptr());
    // VK_LAYER_KHRONOS_profiles
    // layers.push(vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_profiles").as_ptr());
    // layers.push(vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_synchronization2").as_ptr());
    // Extensions

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Create

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    // Messenger

    if VALIDATION_ENABLED {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

//================================================
// Physical Device
//================================================

#[derive(Debug, Error)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);

unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name, error
            );
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
    }

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}

//================================================
// Logical Device
//================================================

unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    // Queue Create Infos

    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Layers

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    // Extensions

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    // Features

    let features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

    // Create

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    // Queues

    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    Ok(device)
}

//================================================
// Swapchain
//================================================

unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Image

    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    // Create

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&info, None)?;

    // Images

    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;

    Ok(())
}

fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::max_value() {
        capabilities.current_extent
    } else {
        let size = window.inner_size();
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                size.width,
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                size.height,
            ))
            .build()
    }
}

unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            create_image_view(
                device,
                *i,
                data.swapchain_format,
                vk::ImageAspectFlags::COLOR,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

//================================================
// Pipeline
//================================================

unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Attachments

    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, data)?)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Subpasses

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_stencil_attachment_ref);

    // Dependencies

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    // Create

    let attachments = &[color_attachment, depth_stencil_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.render_pass = device.create_render_pass(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let storage_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let storage_color_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(3)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let bindings = &[
        ubo_binding,
        sampler_binding,
        storage_binding,
        storage_color_binding,
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}

unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // Stages

    let vert = include_bytes!("../shaders/vert.spv");
    let frag = include_bytes!("../shaders/frag.spv");

    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    // Vertex Input State

    let binding_descriptions = &[Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    // Input Assembly State

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Viewport State

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    // Rasterization State

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(true);

    // Multisample State

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);

    // Depth Stencil State

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // Color Blend State

    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // Layout

    let set_layouts = &[data.descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(set_layouts);

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // Create

    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    data.pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0;

    // Cleanup

    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Vec::<u8>::from(bytecode);
    let (prefix, code, suffix) = bytecode.align_to::<u32>();
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(anyhow!("Shader bytecode is not properly aligned."));
    }

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.len())
        .code(code);

    Ok(device.create_shader_module(&info, None)?)
}

//================================================
// Framebuffers
//================================================

unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[*i, data.depth_image_view];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

//================================================
// Command Pool
//================================================

unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(indices.graphics)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    data.command_pool = device.create_command_pool(&info, None)?;

    Ok(())
}

//================================================
// Depth Objects
//================================================

unsafe fn create_depth_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Image + Image Memory

    let format = get_depth_format(instance, data)?;

    let (depth_image, depth_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.depth_image = depth_image;
    data.depth_image_memory = depth_image_memory;

    // Image View

    data.depth_image_view = create_image_view(
        device,
        data.depth_image,
        format,
        vk::ImageAspectFlags::DEPTH,
    )?;

    Ok(())
}

unsafe fn get_depth_format(instance: &Instance, data: &AppData) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        data,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

unsafe fn get_supported_format(
    instance: &Instance,
    data: &AppData,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f| {
            let properties =
                instance.get_physical_device_format_properties(data.physical_device, *f);
            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find supported format!"))
}

//================================================
// Texture
//================================================

unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    file: String,
    image_handle: vk::Image,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    // Load

    let image = File::open(file.as_str())?;

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;

    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;

    let size = reader.info().raw_bytes() as u64;
    let (width, height) = reader.info().size();

    // Create (staging)

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy (staging)

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

    device.unmap_memory(staging_buffer_memory);

    // Create (image)

    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Transition + Copy (image)

    transition_image_layout(
        device,
        data,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )?;

    copy_buffer_to_image(device, data, staging_buffer, texture_image, width, height)?;

    transition_image_layout(
        device,
        data,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;

    // Cleanup

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((texture_image, texture_image_memory))
}

unsafe fn create_texture_image_view(
    device: &Device,
    data: &mut AppData,
    image: vk::Image,
) -> Result<vk::ImageView> {
    Ok(create_image_view(
        device,
        image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
    )?)
}

unsafe fn create_texture_sampler(device: &Device, data: &mut AppData) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR);

    Ok(device.create_sampler(&info, None)?)
}

//================================================
// Buffers
//================================================

unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Create (staging)

    let size = (size_of::<Vertex>() * VERTICES.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy (staging)

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(VERTICES.as_ptr(), memory.cast(), VERTICES.len());

    device.unmap_memory(staging_buffer_memory);

    // Create (vertex)

    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    // Copy (vertex)

    copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

    // Cleanup

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Create (staging)

    let size = (size_of::<u16>() * INDICES.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy (staging)

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(INDICES.as_ptr(), memory.cast(), INDICES.len());

    device.unmap_memory(staging_buffer_memory);

    // Create (index)

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    // Copy (index)

    copy_buffer(device, data, staging_buffer, index_buffer, size)?;

    // Cleanup

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}

//================================================
// Descriptors
//================================================

unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let particle_position_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let particle_color_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let pool_sizes = &[
        ubo_size,
        sampler_size,
        particle_position_size,
        particle_color_size,
    ];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    // Allocate

    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    // Update

    for i in 0..data.swapchain_images.len() {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);

        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);

        let particle_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.particle_position_storage_buffer)
            .offset(0)
            .range(SIZE_OF_PARTICLE_POSITION);

        let storage_info = &[particle_buffer_info];
        let storage_buffer_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(storage_info);

        let particle_color_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.particle_color_storage_buffer)
            .offset(0)
            .range(SIZE_OF_PARTICLE_COLOR);

        let storage_color_info = &[particle_color_buffer_info];
        let storage_color_buffer_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(storage_color_info);

        device.update_descriptor_sets(
            &[
                ubo_write,
                storage_buffer_descriptor_write,
                storage_color_buffer_descriptor_write,
            ],
            &[] as &[vk::CopyDescriptorSet],
        );
    }

    Ok(())
}

//================================================
// Command Buffers
//================================================

unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    // Allocate

    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    Ok(())
}

//================================================
// Sync Objects
//================================================

unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);

        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    data.images_in_flight = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

//================================================
// Structs
//================================================

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                data.surface,
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
struct ParticleBuffer {
    particle_positions: Vec<glm::Vec2>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    model: glm::Mat4,
    view: glm::Mat4,
    proj: glm::Mat4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: glm::Vec3,
    color: glm::Vec3,
    tex_coord: glm::Vec2,
}

impl Vertex {
    fn new(pos: glm::Vec3, color: glm::Vec3, tex_coord: glm::Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<glm::Vec3>() as u32)
            .build();
        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<glm::Vec3>() + size_of::<glm::Vec3>()) as u32)
            .build();
        [pos, color, tex_coord]
    }
}

//================================================
// Shared (Buffers)
//================================================

unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    // Buffer

    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    // Memory

    let requirements = device.get_buffer_memory_requirements(buffer);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

unsafe fn copy_buffer(
    device: &Device,
    data: &AppData,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

//================================================
// Shared (Images)
//================================================

unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    // Image

    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::_1);

    let image = device.create_image(&info, None)?;

    // Memory

    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    Ok(device.create_image_view(&info, None)?)
}

unsafe fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => return Err(anyhow!("Unsupported image layout transition!")),
        };

    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

//================================================
// Shared (Other)
//================================================

unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &AppData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(data.physical_device);
    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

unsafe fn begin_single_time_commands(device: &Device, data: &AppData) -> Result<vk::CommandBuffer> {
    // Allocate

    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    // Begin

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

#[repr(u32)]
#[derive(Clone, Debug, Default, PartialEq)]
enum ParticleType {
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

#[repr(C)]
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

unsafe fn create_storage_buffer(
    device: &Device,
    instance: &Instance,
    data: &mut AppData,
) -> Result<()> {
    data.particles = vec![];

    let buffer_info = vk::BufferCreateInfo::builder()
        .size(SIZE_OF_PARTICLE_POSITION)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    data.particle_position_storage_buffer = device.create_buffer(&buffer_info, None)?;

    let mem_requirements =
        device.get_buffer_memory_requirements(data.particle_position_storage_buffer);
    let memory_type = find_memory_type(
        instance,
        data,
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    data.particle_position_storage_buffer_memory = device.allocate_memory(&alloc_info, None)?;

    device.bind_buffer_memory(
        data.particle_position_storage_buffer,
        data.particle_position_storage_buffer_memory,
        0,
    )?;

    let buffer_info = vk::BufferCreateInfo::builder()
        .size(SIZE_OF_PARTICLE_COLOR)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    data.particle_color_storage_buffer = device.create_buffer(&buffer_info, None)?;

    let mem_requirements =
        device.get_buffer_memory_requirements(data.particle_color_storage_buffer);
    let memory_type = find_memory_type(
        instance,
        data,
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    data.particle_color_storage_buffer_memory = device.allocate_memory(&alloc_info, None)?;

    device.bind_buffer_memory(
        data.particle_color_storage_buffer,
        data.particle_color_storage_buffer_memory,
        0,
    )?;

    Ok(())
}

unsafe fn find_memory_type(
    instance: &Instance,
    data: &AppData,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, anyhow::Error> {
    let memory_properties = instance.get_physical_device_memory_properties(data.physical_device);

    for (index, memory_type) in memory_properties.memory_types.iter().enumerate() {
        if (type_filter & (1 << index)) != 0 && memory_type.property_flags.contains(properties) {
            return Ok(index as u32);
        }
    }

    Err(anyhow!("Failed to find memory type!"))
}

unsafe fn end_single_time_commands(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    // End

    device.end_command_buffer(command_buffer)?;

    // Submit

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    // Cleanup

    device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}
