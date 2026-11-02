//! The falling-sand simulation.
//!
//! Nothing in here knows about windows, input or the GPU, which keeps it easy to read
//! and easy to unit test.
//!
//! * [`element`] – the materials (sand, water, ...) and their properties.
//! * [`world`]   – the grid of cells and helpers for painting into it.
//! * [`physics`] – the rules that move cells around each tick.

pub mod element;
pub mod physics;
pub mod world;

pub use element::Element;
pub use world::World;
