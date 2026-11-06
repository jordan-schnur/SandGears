//! Zoom and pan: which part of the world is shown in the world area of the display.
//!
//! The world area is exactly as many cells as the world, so at 1x the whole world is
//! visible and each display cell shows one world cell. Zooming in shows a smaller part
//! of the world, with each world cell covering `zoom` display cells.

pub const MIN_ZOOM: f32 = 1.0;
pub const MAX_ZOOM: f32 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    world_size: (f32, f32),
    zoom: f32,
    /// World position shown at the top-left corner of the world area.
    offset: (f32, f32),
}

impl Camera {
    pub fn new(world_size: (usize, usize)) -> Self {
        Self {
            world_size: (world_size.0 as f32, world_size.1 as f32),
            zoom: MIN_ZOOM,
            offset: (0.0, 0.0),
        }
    }

    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// Converts a position in the world area (display cells) to a world position.
    pub fn view_to_world(&self, (x, y): (f32, f32)) -> (f32, f32) {
        (self.offset.0 + x / self.zoom, self.offset.1 + y / self.zoom)
    }

    /// Scales by `factor`, keeping the world point under `anchor` (display cells) fixed.
    pub fn zoom_at(&mut self, anchor: (f32, f32), factor: f32) {
        let fixed = self.view_to_world(anchor);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.offset = (
            fixed.0 - anchor.0 / self.zoom,
            fixed.1 - anchor.1 / self.zoom,
        );
        self.clamp();
    }

    /// Moves the view by a distance in display cells.
    pub fn pan(&mut self, (dx, dy): (f32, f32)) {
        self.offset.0 += dx / self.zoom;
        self.offset.1 += dy / self.zoom;
        self.clamp();
    }

    pub fn reset(&mut self) {
        self.zoom = MIN_ZOOM;
        self.offset = (0.0, 0.0);
    }

    /// Keeps the visible region inside the world so there's never empty space in view.
    fn clamp(&mut self) {
        let max_x = self.world_size.0 - self.world_size.0 / self.zoom;
        let max_y = self.world_size.1 - self.world_size.1 / self.zoom;
        self.offset.0 = self.offset.0.clamp(0.0, max_x);
        self.offset.1 = self.offset.1.clamp(0.0, max_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_showing_the_whole_world() {
        let camera = Camera::new((512, 384));
        assert_eq!(camera.view_to_world((0.0, 0.0)), (0.0, 0.0));
        assert_eq!(camera.view_to_world((511.0, 383.0)), (511.0, 383.0));
    }

    #[test]
    fn zoom_keeps_the_anchor_fixed() {
        let mut camera = Camera::new((512, 384));
        let before = camera.view_to_world((100.0, 200.0));
        camera.zoom_at((100.0, 200.0), 2.0);
        assert_eq!(camera.zoom(), 2.0);
        assert_eq!(camera.view_to_world((100.0, 200.0)), before);
    }

    #[test]
    fn zoom_and_pan_stay_inside_the_world() {
        let mut camera = Camera::new((512, 384));
        camera.zoom_at((0.0, 0.0), 100.0);
        assert_eq!(camera.zoom(), MAX_ZOOM);

        camera.pan((1.0e6, 1.0e6));
        let bottom_right = camera.view_to_world((512.0, 384.0));
        assert_eq!(bottom_right, (512.0, 384.0));

        camera.zoom_at((0.0, 0.0), 0.001);
        assert_eq!(camera.zoom(), MIN_ZOOM);
        assert_eq!(camera.view_to_world((0.0, 0.0)), (0.0, 0.0));
    }
}
