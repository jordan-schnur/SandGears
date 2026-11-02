//! Per-frame resources.
//!
//! We let the CPU work on frame N+1 while the GPU is still drawing frame N. To do that
//! safely, everything the two might touch at the same time is duplicated per frame:
//! the command buffer, the sync objects, and the buffer of cell colours.

use std::slice;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::device::find_memory_type;

/// How many frames the CPU may get ahead of the GPU.
pub const FRAMES_IN_FLIGHT: usize = 2;

pub struct Frame {
    pub command_buffer: vk::CommandBuffer,
    /// Signalled when the swapchain image we acquired is ready to be drawn into.
    pub image_available: vk::Semaphore,
    /// Signalled when the GPU has finished this frame; the CPU waits on it before reuse.
    pub in_flight: vk::Fence,
    pub descriptor_set: vk::DescriptorSet,

    cell_buffer: vk::Buffer,
    cell_memory: vk::DeviceMemory,
    /// The cell buffer stays mapped for its whole life so we can write to it directly.
    cell_colors: *mut u32,
    cell_count: usize,
}

pub struct FrameSetup<'a> {
    pub instance: &'a Instance,
    pub device: &'a Device,
    pub physical_device: vk::PhysicalDevice,
    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub cell_count: usize,
}

impl Frame {
    pub unsafe fn create(setup: &FrameSetup) -> Result<Self> {
        let device = setup.device;

        let command_buffer = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(setup.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];

        let image_available = device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?;
        // Start signalled so the very first wait doesn't block forever.
        let in_flight = device.create_fence(
            &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )?;

        // A buffer the CPU can write and the GPU can read, holding one u32 colour per cell.
        let size = (setup.cell_count * std::mem::size_of::<u32>()) as vk::DeviceSize;
        let cell_buffer = device.create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let requirements = device.get_buffer_memory_requirements(cell_buffer);
        let memory_type = find_memory_type(
            setup.instance,
            setup.physical_device,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let cell_memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )?;
        device.bind_buffer_memory(cell_buffer, cell_memory, 0)?;
        let cell_colors = device
            .map_memory(cell_memory, 0, size, vk::MemoryMapFlags::empty())?
            .cast::<u32>();

        // Point the shader's `Cells` binding at this frame's buffer.
        let set_layouts = &[setup.descriptor_set_layout];
        let descriptor_set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(setup.descriptor_pool)
                .set_layouts(set_layouts),
        )?[0];
        let buffer_info = &[vk::DescriptorBufferInfo::builder()
            .buffer(cell_buffer)
            .offset(0)
            .range(size)];
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(buffer_info);
        device.update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);

        Ok(Self {
            command_buffer,
            image_available,
            in_flight,
            descriptor_set,
            cell_buffer,
            cell_memory,
            cell_colors,
            cell_count: setup.cell_count,
        })
    }

    /// The cell colours the GPU will read for this frame.
    ///
    /// Only write to this after waiting on [`Frame::in_flight`], otherwise the GPU
    /// might still be reading the previous contents.
    pub fn cell_colors_mut(&mut self) -> &mut [u32] {
        // SAFETY: the pointer comes from mapping exactly `cell_count` u32s, stays mapped
        // until `destroy`, and `&mut self` guarantees we hand out only one slice at a time.
        unsafe { slice::from_raw_parts_mut(self.cell_colors, self.cell_count) }
    }

    /// Command buffers and descriptor sets are freed with their pools.
    pub unsafe fn destroy(&self, device: &Device) {
        device.unmap_memory(self.cell_memory);
        device.destroy_buffer(self.cell_buffer, None);
        device.free_memory(self.cell_memory, None);
        device.destroy_fence(self.in_flight, None);
        device.destroy_semaphore(self.image_available, None);
    }
}

pub unsafe fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = &[vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(FRAMES_IN_FLIGHT as u32)];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(FRAMES_IN_FLIGHT as u32);
    Ok(device.create_descriptor_pool(&info, None)?)
}
