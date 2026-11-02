//! The rules that move cells around. Call [`step`] once per simulation tick.
//!
//! The approach is the classic "falling sand" cellular automaton:
//! every tick we visit each cell from the bottom row to the top row and let it try a
//! short list of moves (straight down, then diagonally down, then sideways ...).
//! The first move that's possible wins. Gases use the same rules flipped upside down.

use rand::Rng;

use super::element::Behavior;
use super::world::World;
use super::Element;

/// How many cells a liquid can flow sideways in one tick. Higher = levels out faster.
const LIQUID_SPREAD: i32 = 4;
/// How many cells a gas can drift sideways in one tick.
const GAS_SPREAD: i32 = 2;

const DOWN: i32 = 1;
const UP: i32 = -1;

/// Advances the world by one tick.
pub fn step(world: &mut World, rng: &mut impl Rng) {
    for cell in world.cells_mut() {
        cell.updated = false;
    }

    // Alternate the horizontal scan direction randomly so piles don't lean one way.
    let left_to_right = rng.gen::<bool>();
    let width = world.width() as i32;

    for y in (0..world.height() as i32).rev() {
        for i in 0..width {
            let x = if left_to_right { i } else { width - 1 - i };
            update_cell(world, x, y, rng);
        }
    }
}

fn update_cell(world: &mut World, x: i32, y: i32, rng: &mut impl Rng) {
    let cell = *world.cell(x, y);
    if cell.updated {
        return;
    }

    match cell.element.behavior() {
        Behavior::Static => {}
        Behavior::Powder => {
            fall(world, x, y, DOWN, rng);
        }
        Behavior::Liquid => {
            if !fall(world, x, y, DOWN, rng) {
                flow_sideways(world, x, y, LIQUID_SPREAD, rng);
            }
        }
        Behavior::Gas => {
            if !fall(world, x, y, UP, rng) {
                flow_sideways(world, x, y, GAS_SPREAD, rng);
            }
        }
    }
}

/// Tries to move one cell in direction `dy` (`DOWN` or `UP`): straight first, then
/// diagonally (in a random order). Returns `true` if the cell moved.
fn fall(world: &mut World, x: i32, y: i32, dy: i32, rng: &mut impl Rng) -> bool {
    let mover = world.cell(x, y).element;

    let side = if rng.gen::<bool>() { 1 } else { -1 };
    let targets = [(x, y + dy), (x + side, y + dy), (x - side, y + dy)];

    for (tx, ty) in targets {
        if can_displace(world, mover, tx, ty, dy) {
            move_cell(world, (x, y), (tx, ty));
            return true;
        }
    }
    false
}

/// Slides a fluid up to `max_distance` cells left or right through empty space.
/// Returns `true` if the cell moved.
fn flow_sideways(world: &mut World, x: i32, y: i32, max_distance: i32, rng: &mut impl Rng) -> bool {
    let first = if rng.gen::<bool>() { 1 } else { -1 };

    for dx in [first, -first] {
        let mut distance = 0;
        while distance < max_distance && is_empty(world, x + dx * (distance + 1), y) {
            distance += 1;
        }
        if distance > 0 {
            move_cell(world, (x, y), (x + dx * distance, y));
            return true;
        }
    }
    false
}

/// Can `mover` move into the cell at `(x, y)` while travelling vertically in direction `dy`?
///
/// * Anything can move into empty space.
/// * Heavier things sink through lighter fluids (sand through water, water through gas).
/// * Lighter things rise through heavier fluids (gas bubbles up through water).
/// * Nothing passes through solids/powders or the edge of the world.
fn can_displace(world: &World, mover: Element, x: i32, y: i32, dy: i32) -> bool {
    let Some(target) = world.element_at(x, y) else {
        return false; // edge of the world
    };

    if target.is_empty() {
        return true;
    }

    if !target.is_fluid() || target == mover || world.cell(x, y).updated {
        return false;
    }

    if dy == DOWN {
        mover.density() > target.density()
    } else {
        mover.density() < target.density()
    }
}

fn is_empty(world: &World, x: i32, y: i32) -> bool {
    world.element_at(x, y) == Some(Element::Empty)
}

/// Swaps two cells and marks both as done for this tick.
fn move_cell(world: &mut World, from: (i32, i32), to: (i32, i32)) {
    world.swap(from, to);
    world.cell_mut(from.0, from.1).updated = true;
    world.cell_mut(to.0, to.1).updated = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::world::Cell;

    fn place(world: &mut World, x: i32, y: i32, element: Element) {
        world.set(x, y, Cell::new(element, &mut rand::thread_rng()));
    }

    fn run(world: &mut World, ticks: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..ticks {
            step(world, &mut rng);
        }
    }

    fn count(world: &World, element: Element) -> usize {
        let mut n = 0;
        for y in 0..world.height() as i32 {
            for x in 0..world.width() as i32 {
                if world.element_at(x, y) == Some(element) {
                    n += 1;
                }
            }
        }
        n
    }

    #[test]
    fn sand_falls_one_cell_per_tick_and_rests_on_the_floor() {
        let mut world = World::new(5, 10);
        place(&mut world, 2, 0, Element::Sand);

        run(&mut world, 1);
        assert_eq!(world.element_at(2, 1), Some(Element::Sand));

        run(&mut world, 20);
        assert_eq!(world.element_at(2, 9), Some(Element::Sand));
    }

    #[test]
    fn sand_forms_a_pile_instead_of_a_tower() {
        let mut world = World::new(21, 20);
        for _ in 0..30 {
            place(&mut world, 10, 0, Element::Sand);
            run(&mut world, 3);
        }
        run(&mut world, 50);

        let tower_height = (0..20)
            .filter(|&y| world.element_at(10, y) == Some(Element::Sand))
            .count();
        assert!(
            tower_height < 30,
            "sand should spread out, column height was {tower_height}"
        );
        assert_eq!(count(&world, Element::Sand), 30);
    }

    #[test]
    fn sand_sinks_through_water() {
        let mut world = World::new(1, 3);
        place(&mut world, 0, 0, Element::Sand);
        place(&mut world, 0, 1, Element::Water);
        place(&mut world, 0, 2, Element::Water);

        run(&mut world, 5);
        assert_eq!(world.element_at(0, 2), Some(Element::Sand));
    }

    #[test]
    fn water_levels_out() {
        let mut world = World::new(20, 10);
        for y in 0..10 {
            place(&mut world, 0, y, Element::Water);
        }
        run(&mut world, 200);

        // 10 water cells in a 20-wide box should end up as a single flat layer.
        let bottom_row = (0..20).filter(|&x| world.element_at(x, 9) == Some(Element::Water));
        assert_eq!(bottom_row.count(), 10);
        assert_eq!(count(&world, Element::Water), 10);
    }

    #[test]
    fn gas_rises_through_water() {
        let mut world = World::new(1, 5);
        for y in 0..4 {
            place(&mut world, 0, y, Element::Water);
        }
        place(&mut world, 0, 4, Element::Gas);

        run(&mut world, 10);
        assert_eq!(world.element_at(0, 0), Some(Element::Gas));
    }

    #[test]
    fn metal_never_moves_and_holds_things_up() {
        let mut world = World::new(5, 10);
        for x in 0..5 {
            place(&mut world, x, 5, Element::Metal);
        }
        place(&mut world, 2, 0, Element::Sand);

        run(&mut world, 50);
        assert_eq!(count(&world, Element::Metal), 5);
        assert_eq!(world.element_at(2, 4), Some(Element::Sand));
    }

    #[test]
    fn particles_are_never_created_or_destroyed() {
        let mut rng = rand::thread_rng();
        let mut world = World::new(40, 30);
        let elements = [Element::Sand, Element::Water, Element::Gas, Element::Metal];
        for y in 0..30 {
            for x in 0..40 {
                if rng.gen_bool(0.5) {
                    place(&mut world, x, y, elements[rng.gen_range(0..elements.len())]);
                }
            }
        }
        let before: Vec<usize> = elements.iter().map(|&e| count(&world, e)).collect();
        run(&mut world, 100);
        let after: Vec<usize> = elements.iter().map(|&e| count(&world, e)).collect();
        assert_eq!(before, after);
    }
}
