//! The swapchain: the queue of images we draw into and hand to the window to display.
//! It has to be rebuilt whenever the window changes size.

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;
use winit::window::Window;

use super::device::{QueueFamilies, SurfaceSupport};

pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    /// Signalled when rendering into image `i` is finished and it may be presented.
    /// One per image (not per frame) because the presentation engine holds on to it.
    pub render_finished: Vec<vk::Semaphore>,
}

/// Everything needed to (re)build the swapchain.
pub struct SwapchainTarget<'a> {
    pub window: &'a Window,
    pub instance: &'a Instance,
    pub device: &'a Device,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub queue_families: QueueFamilies,
    pub surface_format: vk::SurfaceFormatKHR,
    pub render_pass: vk::RenderPass,
}

impl Swapchain {
    pub unsafe fn create(target: &SwapchainTarget) -> Result<Self> {
        let device = target.device;
        let support =
            SurfaceSupport::query(target.instance, target.surface, target.physical_device)?;
        let capabilities = support.capabilities;

        let extent = choose_extent(target.window, capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count != 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }

        let families = target.queue_families;
        let (sharing_mode, family_indices) = if families.graphics != families.present {
            (
                vk::SharingMode::CONCURRENT,
                vec![families.graphics, families.present],
            )
        } else {
            (vk::SharingMode::EXCLUSIVE, vec![])
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(target.surface)
            .min_image_count(image_count)
            .image_format(target.surface_format.format)
            .image_color_space(target.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&family_indices)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            // FIFO is "vsync on" and is the one mode every GPU must support.
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let handle = device.create_swapchain_khr(&info, None)?;
        let images = device.get_swapchain_images_khr(handle)?;

        let mut swapchain = Self {
            handle,
            extent,
            image_views: Vec::new(),
            framebuffers: Vec::new(),
            render_finished: Vec::new(),
        };

        for image in images {
            let view = create_image_view(device, image, target.surface_format.format)?;
            swapchain.image_views.push(view);

            let attachments = &[view];
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(target.render_pass)
                .attachments(attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            swapchain
                .framebuffers
                .push(device.create_framebuffer(&framebuffer_info, None)?);

            swapchain
                .render_finished
                .push(device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?);
        }

        Ok(swapchain)
    }

    pub unsafe fn destroy(&self, device: &Device) {
        for &semaphore in &self.render_finished {
            device.destroy_semaphore(semaphore, None);
        }
        for &framebuffer in &self.framebuffers {
            device.destroy_framebuffer(framebuffer, None);
        }
        for &view in &self.image_views {
            device.destroy_image_view(view, None);
        }
        device.destroy_swapchain_khr(self.handle, None);
    }
}

/// Prefers a plain 8-bit UNORM format so the colours we upload appear exactly as written.
pub fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    let preferred = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM];
    formats
        .iter()
        .copied()
        .find(|f| {
            preferred.contains(&f.format) && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(formats[0])
}

fn choose_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }

    let size = window.inner_size();
    let (min, max) = (capabilities.min_image_extent, capabilities.max_image_extent);
    vk::Extent2D {
        width: size.width.clamp(min.width, max.width),
        height: size.height.clamp(min.height, max.height),
    }
}

unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
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
