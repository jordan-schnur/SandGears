//! The world: a fixed-size grid of cells, plus helpers to paint into it.

use rand::Rng;

use super::Element;

/// One square of the world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cell {
    pub element: Element,
    /// Final colour of this cell packed as `0xAABBGGRR` – exactly what the GPU reads.
    pub color: u32,
    /// Set once the cell has moved during the current tick, so a grain that falls
    /// into a row we haven't visited yet doesn't get to move twice.
    pub updated: bool,
}

impl Cell {
    pub fn new(element: Element, rng: &mut impl Rng) -> Self {
        Self {
            element,
            color: random_shade(element, rng),
            updated: false,
        }
    }

    pub fn empty() -> Self {
        Self {
            element: Element::Empty,
            color: pack_rgb(Element::Empty.color()),
            updated: false,
        }
    }
}

/// A `width` × `height` grid of cells. `(0, 0)` is the top-left corner and `y` grows downwards.
pub struct World {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl World {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell::empty(); width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    fn index(&self, x: i32, y: i32) -> usize {
        debug_assert!(self.in_bounds(x, y));
        y as usize * self.width + x as usize
    }

    pub fn cell(&self, x: i32, y: i32) -> &Cell {
        &self.cells[self.index(x, y)]
    }

    pub fn cell_mut(&mut self, x: i32, y: i32) -> &mut Cell {
        let index = self.index(x, y);
        &mut self.cells[index]
    }

    /// The element at `(x, y)`, or `None` if that's outside the world.
    pub fn element_at(&self, x: i32, y: i32) -> Option<Element> {
        self.in_bounds(x, y).then(|| self.cell(x, y).element)
    }

    pub fn set(&mut self, x: i32, y: i32, cell: Cell) {
        *self.cell_mut(x, y) = cell;
    }

    pub fn swap(&mut self, (ax, ay): (i32, i32), (bx, by): (i32, i32)) {
        let a = self.index(ax, ay);
        let b = self.index(bx, by);
        self.cells.swap(a, b);
    }

    pub fn clear(&mut self) {
        self.cells.fill(Cell::empty());
    }

    pub fn cells_mut(&mut self) -> &mut [Cell] {
        &mut self.cells
    }

    /// Number of non-empty cells.
    pub fn particle_count(&self) -> usize {
        self.cells.iter().filter(|c| !c.element.is_empty()).count()
    }

    /// Copies every cell's colour into `out`, row by row, ready to upload to the GPU.
    pub fn write_colors(&self, out: &mut [u32]) {
        for (out, cell) in out.iter_mut().zip(&self.cells) {
            *out = cell.color;
        }
    }

    /// Fills a disc of radius `radius` centred on `(cx, cy)` with `element`.
    ///
    /// Painting [`Element::Empty`] erases everything. Painting anything else only
    /// fills empty cells, so you can pour sand onto a metal wall without wiping it out.
    pub fn paint_circle(
        &mut self,
        cx: i32,
        cy: i32,
        radius: i32,
        element: Element,
        rng: &mut impl Rng,
    ) {
        for y in (cy - radius)..=(cy + radius) {
            for x in (cx - radius)..=(cx + radius) {
                let (dx, dy) = (x - cx, y - cy);
                if dx * dx + dy * dy > radius * radius || !self.in_bounds(x, y) {
                    continue;
                }

                let erasing = element.is_empty();
                if erasing || self.cell(x, y).element.is_empty() {
                    self.set(x, y, Cell::new(element, rng));
                }
            }
        }
    }

    /// Paints discs along the straight line between two points, so a quick mouse
    /// stroke leaves an unbroken trail instead of a row of dots.
    pub fn paint_line(
        &mut self,
        from: (i32, i32),
        to: (i32, i32),
        radius: i32,
        element: Element,
        rng: &mut impl Rng,
    ) {
        let (dx, dy) = (to.0 - from.0, to.1 - from.1);
        let steps = dx.abs().max(dy.abs()).max(1);
        for step in 0..=steps {
            let x = from.0 + dx * step / steps;
            let y = from.1 + dy * step / steps;
            self.paint_circle(x, y, radius, element, rng);
        }
    }
}

fn pack_rgb([r, g, b]: [u8; 3]) -> u32 {
    u32::from_le_bytes([r, g, b, 255])
}

/// The element's base colour, randomly lightened or darkened a little.
fn random_shade(element: Element, rng: &mut impl Rng) -> u32 {
    let variation = element.color_variation() as i16;
    let offset = if variation == 0 {
        0
    } else {
        rng.gen_range(-variation..=variation)
    };
    let shade = |channel: u8| (channel as i16 + offset).clamp(0, 255) as u8;
    let [r, g, b] = element.color();
    pack_rgb([shade(r), shade(g), shade(b)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn painting_does_not_overwrite_other_elements_but_erasing_does() {
        let mut rng = rand::thread_rng();
        let mut world = World::new(10, 10);

        world.paint_circle(5, 5, 0, Element::Metal, &mut rng);
        world.paint_circle(5, 5, 0, Element::Sand, &mut rng);
        assert_eq!(world.element_at(5, 5), Some(Element::Metal));

        world.paint_circle(5, 5, 0, Element::Empty, &mut rng);
        assert_eq!(world.element_at(5, 5), Some(Element::Empty));
    }

    #[test]
    fn painting_near_the_edge_is_clipped() {
        let mut rng = rand::thread_rng();
        let mut world = World::new(10, 10);
        world.paint_circle(0, 0, 3, Element::Sand, &mut rng);
        assert!(world.particle_count() > 0);
    }

    #[test]
    fn paint_line_leaves_no_gaps() {
        let mut rng = rand::thread_rng();
        let mut world = World::new(20, 5);
        world.paint_line((0, 2), (19, 2), 0, Element::Metal, &mut rng);
        assert_eq!(world.particle_count(), 20);
    }
}
