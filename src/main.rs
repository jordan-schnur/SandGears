// SPDX-License-Identifier: Apache-2.0

#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

//! SandGears — an interactive falling-sand sandbox built on the Storm Cloud
//! engine.
//!
//! The entire entry point is now just "configure a window and run the game":
//! all of the Vulkan and windowing setup lives in [`stormcloud`], and the
//! simulation itself lives in [`game`] as plain, safe Rust.

mod game;
mod stormcloud;

use anyhow::Result;

use game::SandGears;
use stormcloud::{Engine, EngineConfig};

fn main() -> Result<()> {
    let config = EngineConfig {
        title: "SandGears".to_string(),
        width: game::WIDTH as u32,
        height: game::HEIGHT as u32,
        icon_path: Some("./resources/App/Icon.png".to_string()),
    };

    Engine::run(config, SandGears::new())
}
