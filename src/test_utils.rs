//! Test utilities for the crate.

use ndarray::{Array, Array1, Array2, Array4};
use serde::Deserialize;

pub fn deserialize_array1<'de, D, T>(
    deserializer: D,
) -> ::std::result::Result<Array1<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Clone,
{
    let v: Vec<T> = Deserialize::deserialize(deserializer)?;
    ::std::result::Result::Ok(Array::from_vec(v))
}

pub fn deserialize_array2<'de, D, T>(deserializer: D) -> Result<Array2<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Clone,
{
    let v: Vec<Vec<T>> = Deserialize::deserialize(deserializer)?;
    Array2::from_shape_vec((v.len(), v[0].len()), v.concat())
        .map_err(serde::de::Error::custom)
}

pub fn deserialize_array4<'de, D, T>(deserializer: D) -> Result<Array4<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Clone,
{
    let v: Vec<Vec<Vec<Vec<T>>>> = Deserialize::deserialize(deserializer)?;
    let mut flattened = Vec::new();
    for a in &v {
        for b in a {
            for c in b {
                for d in c {
                    flattened.push(d.to_owned());
                }
            }
        }
    }

    Array4::from_shape_vec(
        (v.len(), v[0].len(), v[0][0].len(), v[0][0][0].len()),
        flattened,
    )
    .map_err(serde::de::Error::custom)
}
