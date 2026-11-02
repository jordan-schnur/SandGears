//! The render pass and graphics pipeline.
//!
//! There is exactly one draw call per frame: a single triangle covering the world's
//! viewport. The fragment shader (`shaders/world.frag`) looks up each pixel's cell in a
//! storage buffer of colours. So the whole "what does the world look like" question
//! is answered on the CPU, and the GPU side stays tiny.

use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;

const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/world.vert.spv"));
const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/world.frag.spv"));

/// Data pushed straight into the fragment shader with each draw call.
/// Must match the `push_constant` block in `shaders/world.frag`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PushConstants {
    pub world_size: [u32; 2],
}

impl PushConstants {
    pub fn as_bytes(&self) -> [u8; 8] {
        let [w, h] = self.world_size;
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&w.to_ne_bytes());
        bytes[4..].copy_from_slice(&h.to_ne_bytes());
        bytes
    }
}

/// Background colour behind the world (visible as bars when the window's aspect
/// ratio doesn't match the world's).
const CLEAR_COLOR: [f32; 4] = [0.03, 0.03, 0.04, 1.0];

pub fn clear_value() -> vk::ClearValue {
    vk::ClearValue {
        color: vk::ClearColorValue {
            float32: CLEAR_COLOR,
        },
    }
}

pub struct Pipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
}

impl Pipeline {
    pub unsafe fn create(device: &Device, render_pass: vk::RenderPass) -> Result<Self> {
        // Binding 0: the storage buffer of cell colours, read by the fragment shader.
        let cell_buffer_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = &[cell_buffer_binding];
        let descriptor_set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings),
            None,
        )?;

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);
        let set_layouts = &[descriptor_set_layout];
        let push_constant_ranges = &[push_constant_range];
        let layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(set_layouts)
                .push_constant_ranges(push_constant_ranges),
            None,
        )?;

        let vertex_module = create_shader_module(device, VERTEX_SHADER)?;
        let fragment_module = create_shader_module(device, FRAGMENT_SHADER)?;
        let stages = &[
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(b"main\0"),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_module)
                .name(b"main\0"),
        ];

        // No vertex buffers: the vertex shader generates the triangle from gl_VertexIndex.
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // Viewport and scissor are set every frame (see `Renderer::record_commands`),
        // so the pipeline doesn't need rebuilding when the window is resized.
        let viewports = &[vk::Viewport::default()];
        let scissors = &[vk::Rect2D::default()];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);
        let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::_1);

        let blend_attachments = &[vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)];
        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(blend_attachments);

        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .dynamic_state(&dynamic_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let result = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);

        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);

        Ok(Self {
            descriptor_set_layout,
            layout,
            handle: result?.0,
        })
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.handle, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
    }
}

/// A render pass with a single colour attachment: clear it, draw, present.
pub unsafe fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments);

    // Wait for the swapchain image to be released before writing colour to it.
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    Ok(device.create_render_pass(&info, None)?)
}

unsafe fn create_shader_module(device: &Device, spirv_bytes: &[u8]) -> Result<vk::ShaderModule> {
    if spirv_bytes.len() % 4 != 0 {
        return Err(anyhow!("SPIR-V shader size is not a multiple of 4 bytes."));
    }
    // Copy into u32 words; `include_bytes!` data isn't guaranteed to be 4-byte aligned.
    let words: Vec<u32> = spirv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(spirv_bytes.len())
        .code(&words);

    Ok(device.create_shader_module(&info, None)?)
}
