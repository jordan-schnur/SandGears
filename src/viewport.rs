//! Where the world is drawn inside the window.
//!
//! The world keeps its aspect ratio: it's scaled to fit the window and centred, with
//! dark bars filling any leftover space. Both the renderer (to draw) and the input
//! code (to work out which cell is under the mouse) use this, so they always agree.

/// A rectangle in window pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Viewport {
    /// The largest rectangle with the world's aspect ratio that fits in the window.
    pub fn fit(window_size: (u32, u32), world_size: (u32, u32)) -> Self {
        let (window_w, window_h) = (window_size.0 as f32, window_size.1 as f32);
        let (world_w, world_h) = (world_size.0 as f32, world_size.1 as f32);

        let scale = (window_w / world_w).min(window_h / world_h);
        let (width, height) = (world_w * scale, world_h * scale);

        Self {
            x: ((window_w - width) / 2.0).floor(),
            y: ((window_h - height) / 2.0).floor(),
            width,
            height,
        }
    }

    /// Converts a window pixel position to world cell coordinates. The result may be
    /// outside the world if the point is in the letterbox bars.
    pub fn window_to_world(&self, x: f64, y: f64, world_size: (u32, u32)) -> (i32, i32) {
        let cell_x = (x as f32 - self.x) / self.width * world_size.0 as f32;
        let cell_y = (y as f32 - self.y) / self.height * world_size.1 as f32;
        (cell_x.floor() as i32, cell_y.floor() as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_wide_window_with_side_bars() {
        let viewport = Viewport::fit((2000, 768), (512, 384));
        assert_eq!(viewport.height, 768.0);
        assert_eq!(viewport.width, 1024.0);
        assert_eq!(viewport.x, 488.0);
        assert_eq!(viewport.y, 0.0);
    }

    #[test]
    fn maps_window_pixels_to_cells() {
        let viewport = Viewport::fit((1024, 768), (512, 384));
        assert_eq!(viewport.window_to_world(0.0, 0.0, (512, 384)), (0, 0));
        assert_eq!(
            viewport.window_to_world(1023.0, 767.0, (512, 384)),
            (511, 383)
        );
    }
}
