//! Picking a GPU ("physical device") and opening it ("logical device").

use std::collections::HashSet;

use anyhow::{anyhow, Result};
use log::{info, warn};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtension;

use super::instance::PORTABILITY_MACOS_VERSION;

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

/// The queue families we submit work to. Often both are the same family.
#[derive(Clone, Copy, Debug)]
pub struct QueueFamilies {
    pub graphics: u32,
    pub present: u32,
}

impl QueueFamilies {
    pub unsafe fn find(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for index in 0..properties.len() as u32 {
            if instance.get_physical_device_surface_support_khr(physical_device, index, surface)? {
                present = Some(index);
                break;
            }
        }

        match (graphics, present) {
            (Some(graphics), Some(present)) => Ok(Self { graphics, present }),
            _ => Err(anyhow!("missing graphics or present queue family")),
        }
    }
}

/// What the surface supports on a given GPU (image counts, formats, vsync modes).
pub struct SurfaceSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceSupport {
    pub unsafe fn query(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, surface)?,
            formats: instance.get_physical_device_surface_formats_khr(physical_device, surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, surface)?,
        })
    }
}

/// Picks the first GPU that can draw to our window.
pub unsafe fn pick_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, QueueFamilies)> {
    for physical_device in instance.enumerate_physical_devices()? {
        let name = instance
            .get_physical_device_properties(physical_device)
            .device_name;

        match check_physical_device(instance, surface, physical_device) {
            Ok(queue_families) => {
                info!("Using GPU `{name}`.");
                return Ok((physical_device, queue_families));
            }
            Err(reason) => warn!("Skipping GPU `{name}`: {reason}"),
        }
    }

    Err(anyhow!(
        "No GPU with Vulkan support for this window was found."
    ))
}

unsafe fn check_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<QueueFamilies> {
    let queue_families = QueueFamilies::find(instance, surface, physical_device)?;

    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if !DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        return Err(anyhow!("missing required device extensions"));
    }

    let support = SurfaceSupport::query(instance, surface, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!("insufficient swapchain support"));
    }

    Ok(queue_families)
}

/// Opens the GPU and returns it with its graphics and present queues.
pub unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_families: QueueFamilies,
) -> Result<(Device, vk::Queue, vk::Queue)> {
    let unique_families: HashSet<u32> = [queue_families.graphics, queue_families.present].into();

    let queue_priorities = &[1.0];
    let queue_infos = unique_families
        .iter()
        .map(|&family| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(family)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder();
    // Device layers are deprecated, but vulkanalia 0.18 reads this list even when it's
    // empty, so it must be set (to an empty slice) rather than left as a null pointer.
    let no_layers: &[*const std::os::raw::c_char] = &[];
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(no_layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(physical_device, &info, None)?;
    let graphics_queue = device.get_device_queue(queue_families.graphics, 0);
    let present_queue = device.get_device_queue(queue_families.present, 0);

    Ok((device, graphics_queue, present_queue))
}

/// Finds a memory type that satisfies both the resource's requirements and the
/// properties we want (e.g. "CPU can write to it").
pub unsafe fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    requirements: vk::MemoryRequirements,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(physical_device);
    (0..memory.memory_type_count)
        .find(|&i| {
            let allowed = requirements.memory_type_bits & (1 << i) != 0;
            allowed
                && memory.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find a suitable GPU memory type."))
}
