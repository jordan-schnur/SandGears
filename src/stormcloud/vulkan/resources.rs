// SPDX-License-Identifier: Apache-2.0

//! Low-level Vulkan resource helpers shared across the renderer: image
//! creation, image views and memory-type selection.

use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;

use super::AppData;

/// Creates a 2D image and backs it with freshly allocated device memory.
pub(super) unsafe fn create_image(
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
        .memory_type_index(find_memory_type(
            instance,
            data,
            requirements.memory_type_bits,
            properties,
        )?);

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

/// Creates a 2D image view over `image`.
pub(super) unsafe fn create_image_view(
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

/// Finds the index of a memory type that satisfies `type_filter` (a bitmask of
/// allowed indices, e.g. from `vk::MemoryRequirements::memory_type_bits`) and
/// has all of `properties`.
pub(super) unsafe fn find_memory_type(
    instance: &Instance,
    data: &AppData,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(data.physical_device);
    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (type_filter & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}
