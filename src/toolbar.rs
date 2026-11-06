//! The material toolbar: a strip of clickable buttons above the world.
//!
//! It's drawn straight into the display's colour grid, the same way the world is, so
//! the renderer doesn't need to know it exists. Labels use a tiny built-in pixel font
//! since there's no text rendering.

use crate::sim::Element;

/// Rows of the display taken up by the toolbar.
pub const HEIGHT: usize = 20;

/// Buttons in display order, matching the `1`–`5` keys.
pub const BUTTONS: [Element; 5] = [
    Element::Sand,
    Element::Water,
    Element::Gas,
    Element::Metal,
    Element::Empty,
];

const BUTTON_WIDTH: usize = 52;
const BUTTON_HEIGHT: usize = 14;
const BUTTON_TOP: usize = (HEIGHT - BUTTON_HEIGHT) / 2;
const BUTTON_GAP: usize = 4;
/// Each font pixel is drawn as a GLYPH_SCALE x GLYPH_SCALE square of cells.
const GLYPH_SCALE: usize = 2;
const GLYPH_ADVANCE: usize = 4 * GLYPH_SCALE;

const BACKGROUND: [u8; 3] = [32, 32, 38];
const SELECTED_FRAME: [u8; 3] = [255, 255, 255];
/// Empty space is drawn nearly black, which would vanish against the toolbar.
const ERASER_FILL: [u8; 3] = [72, 72, 82];
const DARK_TEXT: [u8; 3] = [20, 20, 20];
const LIGHT_TEXT: [u8; 3] = [235, 235, 235];

fn button_left(index: usize) -> usize {
    BUTTON_GAP + index * (BUTTON_WIDTH + BUTTON_GAP)
}

/// The button under a display cell, if any.
pub fn button_at((x, y): (i32, i32)) -> Option<Element> {
    if x < 0 || y < BUTTON_TOP as i32 || y >= (BUTTON_TOP + BUTTON_HEIGHT) as i32 {
        return None;
    }

    let x = x as usize;
    BUTTONS
        .iter()
        .enumerate()
        .find(|(i, _)| x >= button_left(*i) && x < button_left(*i) + BUTTON_WIDTH)
        .map(|(_, element)| *element)
}

/// Draws the toolbar into the top [`HEIGHT`] rows of `pixels`, a grid `width` cells wide.
pub fn draw(pixels: &mut [u32], width: usize, selected: Element) {
    let mut canvas = Canvas { pixels, width };
    canvas.fill_rect(0, 0, width, HEIGHT, BACKGROUND);

    for (i, &element) in BUTTONS.iter().enumerate() {
        let left = button_left(i);

        // The frame is drawn first and the button over it, leaving a 1-cell border.
        if element == selected {
            canvas.fill_rect(
                left - 1,
                BUTTON_TOP - 1,
                BUTTON_WIDTH + 2,
                BUTTON_HEIGHT + 2,
                SELECTED_FRAME,
            );
        }

        let fill = if element.is_empty() {
            ERASER_FILL
        } else {
            element.color()
        };
        canvas.fill_rect(left, BUTTON_TOP, BUTTON_WIDTH, BUTTON_HEIGHT, fill);

        let label = element.name().to_uppercase();
        let text_width = label.len() * GLYPH_ADVANCE - GLYPH_SCALE;
        let text_left = left + (BUTTON_WIDTH - text_width) / 2;
        let text_top = BUTTON_TOP + (BUTTON_HEIGHT - 5 * GLYPH_SCALE) / 2;
        let text_color = if luminance(fill) > 110.0 {
            DARK_TEXT
        } else {
            LIGHT_TEXT
        };

        for (n, ch) in label.chars().enumerate() {
            canvas.draw_glyph(text_left + n * GLYPH_ADVANCE, text_top, ch, text_color);
        }
    }
}

struct Canvas<'a> {
    pixels: &'a mut [u32],
    width: usize,
}

impl Canvas<'_> {
    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, rgb: [u8; 3]) {
        let color = u32::from_le_bytes([rgb[0], rgb[1], rgb[2], 255]);
        for row in y..(y + h).min(HEIGHT) {
            for col in x..(x + w).min(self.width) {
                self.pixels[row * self.width + col] = color;
            }
        }
    }

    fn draw_glyph(&mut self, x: usize, y: usize, ch: char, rgb: [u8; 3]) {
        for (row, bits) in glyph(ch).iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    self.fill_rect(
                        x + col * GLYPH_SCALE,
                        y + row * GLYPH_SCALE,
                        GLYPH_SCALE,
                        GLYPH_SCALE,
                        rgb,
                    );
                }
            }
        }
    }
}

/// Perceived brightness (0–255), used to pick label colours that stay readable.
fn luminance([r, g, b]: [u8; 3]) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// 3x5 bitmap glyphs, one row per entry, most significant of the three bits on the left.
/// Only the letters the button labels use; anything else draws as a blank.
fn glyph(c: char) -> [u8; 5] {
    match c {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        _ => [0; 5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_button_under_a_cell() {
        let middle_y = (BUTTON_TOP + BUTTON_HEIGHT / 2) as i32;
        assert_eq!(
            button_at((button_left(0) as i32, middle_y)),
            Some(Element::Sand)
        );
        assert_eq!(
            button_at(((button_left(4) + BUTTON_WIDTH - 1) as i32, middle_y)),
            Some(Element::Empty)
        );
        // The gaps between buttons and the strip above them select nothing.
        assert_eq!(button_at(((button_left(1) - 1) as i32, middle_y)), None);
        assert_eq!(button_at((button_left(0) as i32, 0)), None);
    }

    #[test]
    fn every_label_fits_its_button() {
        for element in BUTTONS {
            let label = element.name().to_uppercase();
            assert!(label.len() * GLYPH_ADVANCE - GLYPH_SCALE <= BUTTON_WIDTH);
            for ch in label.chars() {
                assert_ne!(glyph(ch), [0; 5], "no glyph for {ch:?}");
            }
        }
    }

    #[test]
    fn buttons_fit_in_the_world_width() {
        assert!(button_left(BUTTONS.len()) <= crate::WORLD_WIDTH as usize);
    }
}
