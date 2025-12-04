use crate::core::Dim;

pub enum Idx {
    Coord(Vec<Dim>),
    At(usize),
    Item
}

impl From<&Idx> for Idx {
    /// Clones an index reference into an owned index.
    fn from(value: &Idx) -> Self {
        match value {
            Idx::Coord(coords) => Idx::Coord(coords.clone()),
            Idx::At(i) => Idx::At(*i),
            Idx::Item => Idx::Item,
        }
    }
}

// Vec<Dim> implementation
impl From<Vec<Dim>> for Idx {
    fn from(value: Vec<Dim>) -> Self {
        Idx::Coord(value)
    }
}

// Array slice implementation
impl From<&[Dim]> for Idx {
    fn from(value: &[Dim]) -> Self {
        Idx::Coord(value.to_vec())
    }
}

// Fixed-size array implementations
impl<const N: usize> From<[Dim; N]> for Idx {
    fn from(value: [Dim; N]) -> Self {
        Idx::Coord(value.to_vec())
    }
}

// Reference to fixed-size array
impl<const N: usize> From<&[Dim; N]> for Idx {
    fn from(value: &[Dim; N]) -> Self {
        Idx::Coord(value.to_vec())
    }
}

// Single usize (1D)
impl From<usize> for Idx {
    fn from(value: usize) -> Self {
        Idx::Coord(vec![value])
    }
}

// Tuple implementations up to length 6
impl From<(usize,)> for Idx {
    fn from(value: (usize,)) -> Self {
        Idx::Coord(vec![value.0])
    }
}

impl From<(usize, usize)> for Idx {
    fn from(value: (usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1])
    }
}

impl From<(usize, usize, usize)> for Idx {
    fn from(value: (usize, usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1, value.2])
    }
}

impl From<(usize, usize, usize, usize)> for Idx {
    fn from(value: (usize, usize, usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1, value.2, value.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Idx {
    fn from(value: (usize, usize, usize, usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1, value.2, value.3, value.4])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Idx {
    fn from(value: (usize, usize, usize, usize, usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1, value.2, value.3, value.4, value.5])
    }
}

