//! CSV format support.

use std::path::Path;

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;

use crate::training::TrainAndTestData;

/// Read our training and test data from a CSV file.
///
/// The file is in a format similar to the Iris dataset:
///
/// ```csv
/// sepal_length,sepal_width,petal_length,petal_width,iris_virginica,iris_versicolor,iris_setosa
/// 5.1,3.5,1.4,0.2,0,0,1
/// 4.9,3.0,1.4,0.2,0,0,1
/// ```
///
/// ...where the first `input_cols` columns are the input data, and the
/// remainder are the target data in one-hot format.
///
/// We return our data as `(inputs, targets)`, where the arrays have shape
/// `[examples, features]`.
pub fn read_csv_data(
    path: &Path,
    input_cols: usize,
) -> Result<(Array2<f32>, Array2<f32>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for result in rdr.records() {
        let record = result?;
        inputs.push(
            record
                .iter()
                .take(input_cols)
                .map(|s| s.parse::<f32>())
                .collect::<Result<Vec<_>, _>>()?,
        );
        targets.push(
            record
                .iter()
                .skip(input_cols)
                .map(|s| s.parse::<f32>())
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok((
        Array2::from_shape_vec(
            (inputs.len(), input_cols),
            inputs.into_iter().flatten().collect(),
        )?,
        Array2::from_shape_vec(
            (targets.len(), targets[0].len()),
            targets.into_iter().flatten().collect(),
        )?,
    ))
}

/// Split the data into training and test sets. The parameters have the shape
/// `[examples, features]`.
pub fn split_data(
    inputs: Array2<f32>,
    targets: Array2<f32>,
    split: f32,
) -> TrainAndTestData {
    // To shuffle, we need to convert `inputs` and `targets` to
    // `Vec<(Array1<f32>, Array1<f32>)>`, shuffle in place, split it, then convert back to
    // a pair of `Array2<f32>` values.

    // First we convert to a `Vec<(Array1<f32>, Array1<f32>)>`.
    let mut data = inputs
        .outer_iter()
        .zip(targets.outer_iter())
        .map(|(input, target)| (input.to_owned(), target.to_owned()))
        .collect::<Vec<_>>();

    // Then we shuffle in place.
    data.shuffle(&mut rand::thread_rng());

    // Convert our `Vec<(Array1<f32>, Array1<f32>)>` back to a pair of `Vec<Array1<f32>>` values.
    let inputs = data
        .iter()
        .cloned()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let targets = data
        .into_iter()
        .map(|(_, target)| target)
        .collect::<Vec<_>>();

    // Then we split and convert back to `Array2<f32>` using a helper function.
    let split_index = (inputs.len() as f32 * split) as usize;
    let array2_from_slice_array1 = |data: &[Array1<f32>]| -> Array2<f32> {
        Array2::from_shape_vec(
            (data.len(), data[0].len()),
            data.into_iter().cloned().flatten().collect(),
        )
        .expect("invalid shape")
    };
    let train_inputs = array2_from_slice_array1(&inputs[..split_index]);
    let train_targets = array2_from_slice_array1(&targets[..split_index]);
    let test_inputs = array2_from_slice_array1(&inputs[split_index..]);
    let test_targets = array2_from_slice_array1(&targets[split_index..]);

    // Finally, we split and pack into a `TrainAndTestData` struct.
    TrainAndTestData {
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    }
}
