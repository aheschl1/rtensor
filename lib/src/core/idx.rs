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

impl<V> From<V> for Idx 
where V: AsRef<[Dim]> 
{
    /// Converts any type that can be referenced as a slice of coordinates into a multi-dimensional index.
    fn from(value: V) -> Self {
        Idx::Coord(value.as_ref().to_vec())
    }
}
