// SPDX-License-Identifier: Apache-2.0

//! The Storm Cloud Vulkan renderer.
//!
//! This is the one part of the engine where `unsafe` lives. It owns every
//! Vulkan handle, sets the graphics pipeline up, and each frame uploads the
//! particles produced by game code (see [`super::Frame`]) to the GPU and draws
//! them as instanced quads. Nothing outside this module needs to touch Vulkan.
//!
//! The implementation is split across focused submodules — [`instance`],
//! [`device`], [`swapchain`], [`pipeline`], [`attachments`], [`buffers`],
//! [`descriptors`], [`commands`] and [`resources`] — but the only entry point
//! is [`Renderer`]. Every Vulkan handle lives in [`AppData`], which the
//! per-stage `create_*` helpers fill in.

mod attachments;
mod buffers;
mod commands;
mod descriptors;
mod device;
mod instance;
mod pipeline;
mod resources;
mod swapchain;

use std::mem::{size_of, size_of_val};
use std::os::raw::c_char;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use anyhow::{anyhow, Result};
use log::info;
use nalgebra_glm as glm;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
use vulkanalia::window as vk_window;
use vulkanalia::Version;
use winit::window::Window;

use super::Frame;

use attachments::{create_depth_objects, create_framebuffers};
use buffers::{
    create_index_buffer, create_storage_buffer, create_uniform_buffers, create_vertex_buffer,
    INDICES,
};
use commands::{create_command_buffers, create_command_pool, create_sync_objects};
use descriptors::{create_descriptor_pool, create_descriptor_sets};
use device::{create_logical_device, pick_physical_device};
use instance::create_instance;
use pipeline::{
    create_descriptor_set_layout, create_pipeline, create_render_pass, UniformBufferObject,
};
use swapchain::{create_swapchain, create_swapchain_image_views};

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

/// The maximum number of particles the renderer reserves GPU storage for.
pub const MAX_PARTICLES: usize = 100000;

/// The instance layers to enable, depending on whether validation is on.
fn validation_layers() -> Vec<*const c_char> {
    if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    }
}

/// The Storm Cloud renderer: owns all Vulkan state and drives the frame loop.
#[derive(Clone, Debug)]
pub struct Renderer {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    /// Set by the engine when the window has been resized so the swapchain is
    /// recreated on the next frame.
    pub resized: bool,
    start: Instant,
    last_frame: f32,
    fps: f32,
    frames_since_last_second: usize,
}

impl Renderer {
    /// Creates the renderer for `window`, bringing up the full Vulkan stack.
    pub unsafe fn create(window: &Window) -> Result<Self> {
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

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            last_frame: 0.0,
            fps: 0.0,
            frames_since_last_second: 0,
        })
    }

    /// Renders one frame, drawing the particles in `frame`.
    pub unsafe fn render(&mut self, window: &Window, frame: &Frame) -> Result<()> {
        let num_active = frame.len();

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

        self.update_command_buffer(image_index, num_active)?;
        self.update_uniform_buffer(image_index)?;
        self.update_storage_buffers(frame)?;

        self.calc_fps(num_active);

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

    fn calc_fps(&mut self, num_active: usize) {
        let now = self.start.elapsed().as_secs_f32();

        if (now - self.last_frame) >= 1.0 {
            self.fps = self.frames_since_last_second as f32;
            self.frames_since_last_second = 0;
            self.last_frame = now;

            info!("FPS: {}", self.fps);
            info!("Particles: {}", num_active);
        } else {
            self.frames_since_last_second += 1;
        }
    }

    /// Maps `memory`, copies `items` into it and unmaps. A no-op when empty.
    unsafe fn upload_slice<T>(&self, memory: vk::DeviceMemory, items: &[T]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        let size = size_of_val(items) as u64;
        let ptr = self
            .device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
        memcpy(items.as_ptr(), ptr.cast(), items.len());
        self.device.unmap_memory(memory);

        Ok(())
    }

    unsafe fn update_storage_buffers(&mut self, frame: &Frame) -> Result<()> {
        self.upload_slice(
            self.data.particle_position_storage_buffer_memory,
            frame.positions(),
        )?;
        self.upload_slice(
            self.data.particle_color_storage_buffer_memory,
            frame.kinds(),
        )?;
        Ok(())
    }

    /// Updates the uniform buffer object (the orthographic MVP matrices).
    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
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

    unsafe fn update_command_buffer(
        &mut self,
        image_index: usize,
        num_active: usize,
    ) -> Result<()> {
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
            num_active as u32,
            0,
            0,
            0,
        );
        self.device.cmd_end_render_pass(command_buffer);

        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    /// Recreates the swapchain (and everything that depends on it) after a resize.
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

    /// Tears down every Vulkan resource. Call exactly once before exiting.
    #[rustfmt::skip]
    pub unsafe fn destroy(&mut self) {
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

    /// Destroys the swapchain-dependent resources.
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

/// The Vulkan handles and associated properties used by the renderer. The
/// per-stage `create_*` helpers populate the fields; the `Renderer` reads them
/// each frame and frees them on teardown.
#[derive(Clone, Debug, Default)]
pub(super) struct AppData {
    // Debug
    pub(super) messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    pub(super) surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) graphics_queue: vk::Queue,
    pub(super) present_queue: vk::Queue,
    // Swapchain
    pub(super) swapchain_format: vk::Format,
    pub(super) swapchain_extent: vk::Extent2D,
    pub(super) swapchain: vk::SwapchainKHR,
    pub(super) swapchain_images: Vec<vk::Image>,
    pub(super) swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    pub(super) render_pass: vk::RenderPass,
    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) pipeline_layout: vk::PipelineLayout,
    pub(super) pipeline: vk::Pipeline,
    // Framebuffers
    pub(super) framebuffers: Vec<vk::Framebuffer>,
    // Command Pool
    pub(super) command_pool: vk::CommandPool,
    // Depth
    pub(super) depth_image: vk::Image,
    pub(super) depth_image_memory: vk::DeviceMemory,
    pub(super) depth_image_view: vk::ImageView,
    // Buffers
    pub(super) vertex_buffer: vk::Buffer,
    pub(super) vertex_buffer_memory: vk::DeviceMemory,
    pub(super) index_buffer: vk::Buffer,
    pub(super) index_buffer_memory: vk::DeviceMemory,
    pub(super) uniform_buffers: Vec<vk::Buffer>,
    pub(super) uniform_buffers_memory: Vec<vk::DeviceMemory>,
    // Descriptors
    pub(super) descriptor_pool: vk::DescriptorPool,
    pub(super) descriptor_sets: Vec<vk::DescriptorSet>,
    // Command Buffers
    pub(super) command_buffers: Vec<vk::CommandBuffer>,
    // Sync Objects
    pub(super) image_available_semaphores: Vec<vk::Semaphore>,
    pub(super) render_finished_semaphores: Vec<vk::Semaphore>,
    pub(super) in_flight_fences: Vec<vk::Fence>,
    pub(super) images_in_flight: Vec<vk::Fence>,
    // Particle storage buffers
    pub(super) particle_position_storage_buffer: vk::Buffer,
    pub(super) particle_position_storage_buffer_memory: vk::DeviceMemory,
    pub(super) particle_color_storage_buffer: vk::Buffer,
    pub(super) particle_color_storage_buffer_memory: vk::DeviceMemory,
}
