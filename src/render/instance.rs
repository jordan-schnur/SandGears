//! Creating the Vulkan instance, and (in debug builds) the validation layer that
//! reports API misuse to the log.

use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;

use anyhow::Result;
use log::{debug, error, info, trace, warn};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::window as vk_window;
use vulkanalia::Version;
use winit::window::Window;

const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// The Vulkan SDK version that started requiring the portability extensions on macOS.
pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

/// The instance plus the debug messenger, if validation is on.
pub struct VulkanInstance {
    pub instance: Instance,
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl VulkanInstance {
    pub unsafe fn create(window: &Window, entry: &Entry) -> Result<Self> {
        let application_info = vk::ApplicationInfo::builder()
            .application_name(b"SandGears\0")
            .application_version(vk::make_version(0, 1, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));

        let validation = should_enable_validation(entry)?;
        let layers = if validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let mut extensions = vk_window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

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

        if validation {
            extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        }

        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .user_callback(Some(debug_callback));

        let mut info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(flags);

        if validation {
            info = info.push_next(&mut debug_info);
        }

        let instance = entry.create_instance(&info, None)?;

        let debug_messenger = if validation {
            Some(instance.create_debug_utils_messenger_ext(&debug_info, None)?)
        } else {
            None
        };

        Ok(Self {
            instance,
            debug_messenger,
        })
    }

    pub unsafe fn destroy(&self) {
        if let Some(messenger) = self.debug_messenger {
            self.instance
                .destroy_debug_utils_messenger_ext(messenger, None);
        }
        self.instance.destroy_instance(None);
    }
}

/// Validation is on in debug builds when the layer is installed (it ships with the
/// Vulkan SDK). If it's missing we just warn – the game still runs fine without it.
unsafe fn should_enable_validation(entry: &Entry) -> Result<bool> {
    if !cfg!(debug_assertions) {
        return Ok(false);
    }

    let available = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if available.contains(&VALIDATION_LAYER) {
        Ok(true)
    } else {
        warn!("Vulkan validation layer not found; install the Vulkan SDK to enable it.");
        Ok(false)
    }
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
