// SPDX-License-Identifier: Apache-2.0

//! The descriptor pool and the per-swapchain-image descriptor sets that bind
//! the uniform buffer and the two particle storage buffers to the shaders.

use std::mem::size_of;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::buffers::{SIZE_OF_PARTICLE_COLOR, SIZE_OF_PARTICLE_POSITION};
use super::pipeline::UniformBufferObject;
use super::AppData;

pub(super) unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    let count = data.swapchain_images.len() as u32;

    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(count);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(count);

    let particle_position_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(count);

    let particle_color_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(count);

    let pool_sizes = &[
        ubo_size,
        sampler_size,
        particle_position_size,
        particle_color_size,
    ];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(count);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

pub(super) unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
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
