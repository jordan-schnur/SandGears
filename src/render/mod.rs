//! Draws the world to the window with Vulkan.
//!
//! The renderer knows nothing about sand or water: each frame it's handed a grid of
//! colours (one per world cell) and draws it, scaled to fit the window.
//!
//! * [`instance`]  – Vulkan instance and validation-layer logging.
//! * [`device`]    – choosing and opening the GPU.
//! * [`swapchain`] – the window's images; rebuilt on resize.
//! * [`pipeline`]  – render pass, shaders and pipeline state.
//! * [`frame`]     – per-frame command buffer, sync objects and colour buffer.

mod device;
mod frame;
mod instance;
mod pipeline;
mod swapchain;

use anyhow::{anyhow, Context, Result};
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::window as vk_window;
use winit::window::Window;

use crate::viewport::Viewport;
use device::{QueueFamilies, SurfaceSupport};
use frame::{Frame, FrameSetup, FRAMES_IN_FLIGHT};
use instance::VulkanInstance;
use pipeline::{Pipeline, PushConstants};
use swapchain::{Swapchain, SwapchainTarget};

pub struct Renderer {
    // Kept alive because the instance's function pointers come from it.
    _entry: Entry,
    instance: VulkanInstance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    queue_families: QueueFamilies,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    surface_format: vk::SurfaceFormatKHR,
    render_pass: vk::RenderPass,
    pipeline: Pipeline,
    swapchain: Swapchain,

    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    frames: Vec<Frame>,
    current_frame: usize,

    world_size: (u32, u32),
    window_resized: bool,
}

impl Renderer {
    /// Sets up Vulkan for `window`, ready to draw a world of `world_size` cells.
    pub fn new(window: &Window, world_size: (u32, u32)) -> Result<Self> {
        unsafe { Self::create(window, world_size) }
    }

    unsafe fn create(window: &Window, world_size: (u32, u32)) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)
            .context("Could not load the Vulkan library. Is a Vulkan driver installed?")?;
        let entry = Entry::new(loader).map_err(|e| anyhow!("{e}"))?;

        let instance = VulkanInstance::create(window, &entry)?;
        let vk_instance = &instance.instance;
        let surface = vk_window::create_surface(vk_instance, &window, &window)?;

        let (physical_device, queue_families) = device::pick_physical_device(vk_instance, surface)?;
        let (device, graphics_queue, present_queue) =
            device::create_logical_device(&entry, vk_instance, physical_device, queue_families)?;

        let support = SurfaceSupport::query(vk_instance, surface, physical_device)?;
        let surface_format = swapchain::choose_surface_format(&support.formats);
        let render_pass = pipeline::create_render_pass(&device, surface_format.format)?;
        let pipeline = Pipeline::create(&device, render_pass)?;

        let swapchain = Swapchain::create(&SwapchainTarget {
            window,
            instance: vk_instance,
            device: &device,
            surface,
            physical_device,
            queue_families,
            surface_format,
            render_pass,
        })?;

        let command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder()
                .queue_family_index(queue_families.graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let descriptor_pool = frame::create_descriptor_pool(&device)?;

        let frame_setup = FrameSetup {
            instance: vk_instance,
            device: &device,
            physical_device,
            command_pool,
            descriptor_pool,
            descriptor_set_layout: pipeline.descriptor_set_layout,
            cell_count: (world_size.0 * world_size.1) as usize,
        };
        let frames = (0..FRAMES_IN_FLIGHT)
            .map(|_| Frame::create(&frame_setup))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            _entry: entry,
            instance,
            surface,
            physical_device,
            queue_families,
            device,
            graphics_queue,
            present_queue,
            surface_format,
            render_pass,
            pipeline,
            swapchain,
            command_pool,
            descriptor_pool,
            frames,
            current_frame: 0,
            world_size,
            window_resized: false,
        })
    }

    /// Call when the window changes size; the swapchain is rebuilt before the next frame.
    pub fn window_resized(&mut self) {
        self.window_resized = true;
    }

    /// Draws one frame. `fill_cells` is given the colour buffer (one `0xAABBGGRR` per
    /// world cell, row by row) and must fill all of it.
    pub fn render(&mut self, window: &Window, fill_cells: impl FnOnce(&mut [u32])) -> Result<()> {
        unsafe { self.draw_frame(window, fill_cells) }
    }

    unsafe fn draw_frame(
        &mut self,
        window: &Window,
        fill_cells: impl FnOnce(&mut [u32]),
    ) -> Result<()> {
        if self.window_resized {
            self.window_resized = false;
            return self.recreate_swapchain(window);
        }

        let frame = &mut self.frames[self.current_frame];
        self.device
            .wait_for_fences(&[frame.in_flight], true, u64::MAX)?;

        let image_index = match self.device.acquire_next_image_khr(
            self.swapchain.handle,
            u64::MAX,
            frame.image_available,
            vk::Fence::null(),
        ) {
            Ok((index, _)) => index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        // The GPU is done with this frame's resources, so they're safe to reuse.
        self.device.reset_fences(&[frame.in_flight])?;
        fill_cells(frame.cell_colors_mut());

        let frame = &self.frames[self.current_frame];
        self.record_commands(frame, image_index)?;

        let wait_semaphores = &[frame.image_available];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[frame.command_buffer];
        let signal_semaphores = &[self.swapchain.render_finished[image_index]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);
        self.device
            .queue_submit(self.graphics_queue, &[submit_info], frame.in_flight)?;

        let swapchains = &[self.swapchain.handle];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);
        let result = self
            .device
            .queue_present_khr(self.present_queue, &present_info);

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;

        match result {
            Ok(vk::SuccessCode::SUBOPTIMAL_KHR) | Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(window)
            }
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    unsafe fn record_commands(&self, frame: &Frame, image_index: usize) -> Result<()> {
        let device = &self.device;
        let command_buffer = frame.command_buffer;
        let extent = self.swapchain.extent;

        device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let clear_values = &[pipeline::clear_value()];
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain.framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .clear_values(clear_values);
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );

        // Draw the world into a centred rectangle that keeps its aspect ratio.
        let view = Viewport::fit((extent.width, extent.height), self.world_size);
        let viewport = vk::Viewport::builder()
            .x(view.x)
            .y(view.y)
            .width(view.width)
            .height(view.height)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: view.x as i32,
                y: view.y as i32,
            },
            extent: vk::Extent2D {
                width: view.width.ceil() as u32,
                height: view.height.ceil() as u32,
            },
        };
        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        device.cmd_set_scissor(command_buffer, 0, &[scissor]);

        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.handle,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.layout,
            0,
            &[frame.descriptor_set],
            &[],
        );
        let push_constants = PushConstants {
            world_size: [self.world_size.0, self.world_size.1],
        };
        device.cmd_push_constants(
            command_buffer,
            self.pipeline.layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            &push_constants.as_bytes(),
        );
        device.cmd_draw(command_buffer, 3, 1, 0, 0);

        device.cmd_end_render_pass(command_buffer);
        device.end_command_buffer(command_buffer)?;
        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(()); // minimised; try again once the window is visible
        }

        self.device.device_wait_idle()?;
        self.swapchain.destroy(&self.device);
        self.swapchain = Swapchain::create(&SwapchainTarget {
            window,
            instance: &self.instance.instance,
            device: &self.device,
            surface: self.surface,
            physical_device: self.physical_device,
            queue_families: self.queue_families,
            surface_format: self.surface_format,
            render_pass: self.render_pass,
        })?;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            // Can't do anything useful with an error while shutting down.
            let _ = self.device.device_wait_idle();

            for frame in &self.frames {
                frame.destroy(&self.device);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain.destroy(&self.device);
            self.pipeline.destroy(&self.device);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_device(None);
            self.instance
                .instance
                .destroy_surface_khr(self.surface, None);
            self.instance.destroy();
        }
    }
}
