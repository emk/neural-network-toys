//! Support for limited reshaping of `ndarray` arrays.
use ndarray::{
    ArrayBase, ArrayViewMut, DataMut, Dimension, ErrorKind, ShapeArg, ShapeError,
    StrideShape,
};

/// Helper trait for reshaping `ndarray` arrays mutably. See [issue #390][] for
/// more details on upstream support for this.
///
/// [issue #390]: https://github.com/rust-ndarray/ndarray/issues/390
pub trait TryToShapeMut<'a, Elem> {
    /// Attempt to reshape a mutable `ndarray` array. This is basically
    /// the same as `ArrayBase::to_shape` but for mutable arrays. It will fail
    /// if the array is not contiguous in memory.
    fn try_to_shape_mut<Shape>(
        &'a mut self,
        new_shape: Shape,
    ) -> Result<ArrayViewMut<'a, Elem, Shape::Dim>, ShapeError>
    where
        Shape: ShapeArg + Into<StrideShape<Shape::Dim>>;
}

impl<'a, Elem, A, D> TryToShapeMut<'a, Elem> for ArrayBase<A, D>
where
    A: DataMut<Elem = Elem>,
    D: Dimension,
{
    fn try_to_shape_mut<Shape>(
        &'a mut self,
        new_shape: Shape,
    ) -> Result<ArrayViewMut<'a, Elem, Shape::Dim>, ShapeError>
    where
        Shape: ShapeArg + Into<StrideShape<Shape::Dim>>,
    {
        let slice_mut = self
            .as_slice_mut()
            .ok_or_else(|| ShapeError::from_kind(ErrorKind::IncompatibleLayout))?;
        ArrayViewMut::from_shape(new_shape.into(), slice_mut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, arr3};

    #[test]
    fn test_1d() {
        let mut a = arr1(&[1, 2, 3, 4, 5, 6]);
        let b = a.try_to_shape_mut((3, 2)).unwrap();
        assert_eq!(b, arr2(&[[1, 2], [3, 4], [5, 6]]));
    }

    #[test]
    fn test_2d() {
        let mut a = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let b = a.try_to_shape_mut((3, 2)).unwrap();
        assert_eq!(b, arr2(&[[1, 2], [3, 4], [5, 6]]));
    }

    #[test]
    fn test_3d() {
        let mut a = arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let b = a.try_to_shape_mut((4, 2)).unwrap();
        assert_eq!(b, arr2(&[[1, 2], [3, 4], [5, 6], [7, 8]]));
    }

    #[test]
    fn test_3d_fail() {
        let mut a = arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let b = a.try_to_shape_mut((4, 3));
        assert!(b.is_err());
    }

    #[test]
    fn reshape_view_mut() {
        let mut a = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let mut v = a.view_mut();
        let b = v.try_to_shape_mut((3, 2)).unwrap();
        assert_eq!(b, arr2(&[[1, 2], [3, 4], [5, 6]]));
    }
}
