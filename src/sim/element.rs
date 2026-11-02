//! Every material that can occupy a cell, and the properties that drive its behaviour.
//!
//! To add a new material:
//! 1. add a variant to [`Element`],
//! 2. fill in its row in each `match` below (the compiler will point them all out),
//! 3. give it a key in `src/game.rs` if players should be able to paint it.

/// A material that can occupy one cell of the world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Element {
    /// Nothing – background air.
    #[default]
    Empty,
    Sand,
    Water,
    Gas,
    Metal,
}

/// How an element moves. The actual movement rules live in `physics.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Behavior {
    /// Never moves (empty space, walls).
    Static,
    /// Falls straight down or slides diagonally, forming piles.
    Powder,
    /// Falls like a powder, and also flows sideways to find its level.
    Liquid,
    /// Like a liquid, but upside down: rises and spreads along ceilings.
    Gas,
}

impl Element {
    pub fn name(self) -> &'static str {
        match self {
            Element::Empty => "Eraser",
            Element::Sand => "Sand",
            Element::Water => "Water",
            Element::Gas => "Gas",
            Element::Metal => "Metal",
        }
    }

    pub fn behavior(self) -> Behavior {
        match self {
            Element::Empty | Element::Metal => Behavior::Static,
            Element::Sand => Behavior::Powder,
            Element::Water => Behavior::Liquid,
            Element::Gas => Behavior::Gas,
        }
    }

    /// Density in kg/m³. Heavier elements sink through lighter liquids and gases.
    pub fn density(self) -> f32 {
        match self {
            Element::Empty => 1.2,
            Element::Sand => 1600.0,
            Element::Water => 1000.0,
            Element::Gas => 0.6,
            Element::Metal => 7800.0,
        }
    }

    /// Base colour as `[red, green, blue]`. Each grain gets a slight random variation.
    pub fn color(self) -> [u8; 3] {
        match self {
            Element::Empty => [18, 18, 24],
            Element::Sand => [222, 184, 112],
            Element::Water => [40, 110, 230],
            Element::Gas => [190, 200, 120],
            Element::Metal => [150, 150, 160],
        }
    }

    /// How much each grain's brightness may differ from the base colour (0 = none).
    pub fn color_variation(self) -> u8 {
        match self {
            Element::Empty => 0,
            Element::Sand => 24,
            Element::Water => 10,
            Element::Gas => 20,
            Element::Metal => 8,
        }
    }

    /// Liquids and gases can be pushed aside by heavier things falling into them.
    pub fn is_fluid(self) -> bool {
        matches!(self.behavior(), Behavior::Liquid | Behavior::Gas)
    }

    pub fn is_empty(self) -> bool {
        self == Element::Empty
    }
}
