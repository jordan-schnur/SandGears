// SPDX-License-Identifier: Apache-2.0

//! GPU buffers: the static quad vertex/index buffers, the per-frame uniform
//! buffers and the particle storage buffers, plus the generic buffer helpers
//! they are built from.

use std::mem::{size_of, size_of_val};
use std::ptr::copy_nonoverlapping as memcpy;

use anyhow::Result;
use lazy_static::lazy_static;
use nalgebra_glm as glm;
use vulkanalia::prelude::v1_0::*;

use super::commands::{begin_single_time_commands, end_single_time_commands};
use super::pipeline::{UniformBufferObject, Vertex};
use super::resources::find_memory_type;
use super::{AppData, MAX_PARTICLES};

/// Storage reserved for particle positions: two `f64` slots per particle.
pub(super) const SIZE_OF_PARTICLE_POSITION: u64 = (size_of::<f64>() * 2 * MAX_PARTICLES) as u64;
/// Storage reserved for particle colours: one `u32` palette index per particle.
pub(super) const SIZE_OF_PARTICLE_COLOR: u64 = (size_of::<u32>() * MAX_PARTICLES) as u64;

lazy_static! {
    #[rustfmt::skip]
    static ref VERTICES: Vec<Vertex> = vec![
        Vertex::new(glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 0.0, 0.0), glm::vec2(1.0, 0.0)),
        Vertex::new(glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0), glm::vec2(0.0, 0.0)),
        Vertex::new(glm::vec3(1.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0), glm::vec2(0.0, 1.0)),
        Vertex::new(glm::vec3(0.0, 1.0, 0.0), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 1.0)),
    ];
}

#[rustfmt::skip]
pub(super) const INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0,
];

pub(super) unsafe fn create_vertex_buffer(
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

pub(super) unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Create (staging)

    let size = size_of_val(INDICES) as u64;

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

pub(super) unsafe fn create_uniform_buffers(
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

/// Allocates the two host-visible storage buffers the particle data is streamed
/// into each frame (positions and colours).
pub(super) unsafe fn create_storage_buffer(
    device: &Device,
    instance: &Instance,
    data: &mut AppData,
) -> Result<()> {
    let host_visible =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let (position_buffer, position_memory) = create_buffer(
        instance,
        device,
        data,
        SIZE_OF_PARTICLE_POSITION,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        host_visible,
    )?;
    data.particle_position_storage_buffer = position_buffer;
    data.particle_position_storage_buffer_memory = position_memory;

    let (color_buffer, color_memory) = create_buffer(
        instance,
        device,
        data,
        SIZE_OF_PARTICLE_COLOR,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        host_visible,
    )?;
    data.particle_color_storage_buffer = color_buffer;
    data.particle_color_storage_buffer_memory = color_memory;

    Ok(())
}

/// Creates a buffer of `size` bytes and backs it with freshly allocated memory
/// of the requested `properties`.
pub(super) unsafe fn create_buffer(
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
        .memory_type_index(find_memory_type(
            instance,
            data,
            requirements.memory_type_bits,
            properties,
        )?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

/// Copies `size` bytes from `source` to `destination` via a one-time command.
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
