/// This is a combination of the tutorial from
/// https://visualstudiomagazine.com/articles/2014/08/01/batch-training.aspx and
/// a lot of code generated by Copilot, which is apparently _very_ enthusiastic
/// about writing large stretches of neural network code at a go.
///
/// I ended up needing to check everything Copilot wrote. It did an unbelievable
/// amount of work, but it also made plenty of mistakes. And the entire point of
/// this exercise was for me to internalize the details that lives down below
/// Tensorflow and PyTorch.
///
/// Thanks to the careful test suites on the `Layer` implementations, this
/// code actually works. I'm particularly happy about the numerical gradient
/// checking, which is a great way to verify that the backpropagation code
/// is vaguely trustworthy.
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand};
use csv;
use log::debug;
use mnist::MnistBuilder;
use ndarray::{Array1, Array2, ArrayView1};
use plotters::{
    prelude::{ChartBuilder, IntoDrawingArea, PathElement},
    series::LineSeries,
    style::{Color, IntoFont, BLACK, BLUE, RED, WHITE},
};
use plotters_svg::SVGBackend;
use rand::seq::SliceRandom;

mod history;
mod initialization;
mod layers;

use crate::layers::Network;
use crate::{
    history::{EpochStats, TrainingHistory},
    layers::ActivationFunction,
};

#[derive(Debug, Parser)]
#[structopt(about = "Neural network experiments")]
struct Opt {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train a neural network on the iris dataset.
    Iris(IrisOpt),

    /// Train a neural network on the MNIST dataset.
    Mnist(MnistOpt),
}

#[derive(Debug, Parser)]
struct TrainOpt {
    // Parameters are number of epochs, the training/test split, the learning rate, and size of input (4), hidden (7) and output (3) layers.
    /// Number of epochs.
    #[arg(long = "epochs", default_value = "2000")]
    epochs: usize,

    /// Learning rate.
    #[arg(long = "learning-rate", default_value = "0.01")]
    learning_rate: f32,

    /// What fraction of our hidden layers should be dropped out while training?
    /// This is a good way to reduce overfitting.
    #[arg(long = "dropout", default_value = "0.5")]
    dropout: f32,

    /// Which activation function should we use? Does not apply to the output
    /// layer, which always uses softmax.
    #[arg(long = "activation", default_value = "tanh", value_names = ["leaky_relu", "tanh"])]
    activation: String,

    /// Leaky ReLU alpha.
    #[arg(long = "relu-leak", default_value = "0.01")]
    relu_leak: f32,

    /// How long should we wait before stopping training? If we see this many
    /// models in a row that are worse than our best model, we'll stop.
    #[arg(long = "patience", default_value = "5")]
    patience: usize,

    /// Path to save plot of training and test loss.
    #[arg(long = "plot")]
    plot: Option<PathBuf>,
}

impl TrainOpt {
    fn activation(&self) -> ActivationFunction {
        match self.activation.as_str() {
            "leaky_relu" => ActivationFunction::LeakyReLU(self.relu_leak),
            "tanh" => ActivationFunction::Tanh,
            _ => panic!("Unknown activation function"),
        }
    }
}

#[derive(Debug, Parser)]
struct IrisOpt {
    /// Our training options.
    #[structopt(flatten)]
    train: TrainOpt,

    /// Training/test split.
    #[arg(long = "split", default_value = "0.8")]
    split: f32,

    /// Hidden layer size. This needs to be adjusted if you change the dropout.
    #[arg(long = "hidden-layer-width", default_value = "14")]
    hidden_layer_width: usize,
}

#[derive(Debug, Parser)]
struct MnistOpt {
    /// Our training options.
    #[structopt(flatten)]
    train: TrainOpt,

    /// Hidden layer size. This needs to be adjusted if you change the dropout.
    #[arg(long = "hidden-layer-width", default_value = "128")]
    hidden_layer_width: usize,

    /// Hidden layer count.
    #[arg(long = "hidden-layers", default_value = "2")]
    hidden_layers: usize,
}

/// Our main entry point.
fn main() -> Result<()> {
    env_logger::init();
    let opt = Opt::parse();

    match opt.cmd {
        Command::Iris(opt) => train_iris(opt)?,
        Command::Mnist(opt) => train_mnist(opt)?,
    }
    Ok(())
}

/// Read the iris data from a CSV file and train a neural network on it.
fn train_iris(opt: IrisOpt) -> Result<()> {
    let feature_count = 4;
    let class_count = 3;

    let (inputs, targets) = read_csv_data(Path::new("data/iris.csv"), feature_count)?;
    let train_and_test_data = split_data(inputs, targets, opt.split);
    eprintln!(
        "Training data: {} examples, test data: {} examples",
        train_and_test_data.train_inputs.nrows(),
        train_and_test_data.test_inputs.nrows()
    );

    let mut network = Network::new(feature_count);
    let activation = opt.train.activation();
    network.add_fully_connected_layer(opt.hidden_layer_width, activation);
    network.add_dropout_layer(1.0 - opt.train.dropout);
    network.add_fully_connected_layer(class_count, ActivationFunction::Softmax);

    train(opt.train, &mut network, &train_and_test_data)
}

/// Read the MNIST data from a file using the `minist` crate and train a neural
/// network on it.
fn train_mnist(opt: MnistOpt) -> Result<()> {
    let mnist = MnistBuilder::new()
        // If you omit the trailing "/" here, the downloader breaks.
        .base_path("data/mnist/")
        .label_format_one_hot()
        .training_set_length(50000)
        .validation_set_length(10000)
        .test_set_length(10000)
        .download_and_extract()
        .finalize();

    // Convert a giant array of u8s into a 2D array of f32s, each with dimension
    // `[examples, features]`.
    let array2_f32_from_vec_u8 = |v: Vec<u8>, cols: usize| {
        let v = v.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let rows = v.len() / cols;
        assert!(rows * cols == v.len());
        Array2::from_shape_vec((rows, cols), v).expect("invalid shape")
    };

    // Convert the MNIST data into normalized `[examples, features]` arrays.
    let img_size = 28 * 28;
    let num_digits = 10;
    let train_and_test_data = TrainAndTestData {
        train_inputs: array2_f32_from_vec_u8(mnist.trn_img, img_size) / 255.0,
        train_targets: array2_f32_from_vec_u8(mnist.trn_lbl, num_digits),
        test_inputs: array2_f32_from_vec_u8(mnist.tst_img, img_size) / 255.0,
        test_targets: array2_f32_from_vec_u8(mnist.tst_lbl, num_digits),
    };

    let mut network = Network::new(img_size);
    let activation = opt.train.activation();
    network.add_fully_connected_layer(opt.hidden_layer_width, activation);
    for _ in 1..opt.hidden_layers {
        network.add_fully_connected_layer(opt.hidden_layer_width, activation);
        network.add_dropout_layer(1.0 - opt.train.dropout);
    }
    network.add_fully_connected_layer(num_digits, ActivationFunction::Softmax);

    train(opt.train, &mut network, &train_and_test_data)
}

/// The data we'll be using for training and testing.
///
/// All of the data arrays are stored as `[examples, features]` arrays.
struct TrainAndTestData {
    train_inputs: Array2<f32>,
    train_targets: Array2<f32>,
    test_inputs: Array2<f32>,
    test_targets: Array2<f32>,
}

/// First attempt at a training function, originally written largely by Copilot
/// but revised heavily since.
fn train(
    opt: TrainOpt,
    network: &mut Network,
    training_data: &TrainAndTestData,
) -> Result<()> {
    let mut history = TrainingHistory::new();
    let mut rng = rand::thread_rng();
    'epochs: for epoch in 0..opt.epochs {
        debug!("Model: {:#?}", network);
        let mut train_loss = 0.0;
        let mut train_correct = 0;

        // Shuffle our indices so we can train on the data in a random order.
        let train_count = training_data.train_inputs.nrows();
        let mut shuffled_indices = (0..train_count).collect::<Vec<_>>();
        shuffled_indices.shuffle(&mut rng);

        // Train on each example.
        for i in shuffled_indices {
            let input = training_data.train_inputs.row(i);
            let target = training_data.train_targets.row(i);

            network.update(&input, &target, opt.learning_rate);
            let output = network.forward(&input);
            let loss = network.loss(&output.view(), &target);
            if !loss.is_finite() {
                eprintln!("Loss is not finite: {}", loss);
                break 'epochs;
            }
            train_loss += loss;
            if predicted_class_index(&output.view()) == predicted_class_index(&target)
            {
                train_correct += 1;
            }
        }
        train_loss /= train_count as f32;

        let mut test_loss = 0.0;
        let mut test_correct = 0;
        let test_count = training_data.test_inputs.nrows();
        for i in 0..test_count {
            let input = training_data.test_inputs.row(i);
            let target = training_data.test_targets.row(i);

            let output = network.forward(&input);
            let loss = network.loss(&output.view(), &target);
            test_loss += loss;
            if predicted_class_index(&output.view()) == predicted_class_index(&target)
            {
                test_correct += 1;
            }
        }
        test_loss /= test_count as f32;

        let train_accuracy = train_correct as f32 / train_count as f32;
        let test_accuracy = test_correct as f32 / test_count as f32;
        history.add_epoch(
            EpochStats {
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
            },
            &network,
        );

        // Report loss & accuracy as a percentage.
        eprintln!(
            "Epoch {}: train loss = {:.4} ({:.2}%), test loss = {:.4} ({:.2}%)",
            epoch,
            train_loss,
            100.0 * train_accuracy,
            test_loss,
            100.0 * test_accuracy,
        );

        // Stop training if our history shows we're not improving.
        if history.should_stop(opt.patience) {
            eprintln!("Training stopped early due to lack of improvement.");
            break;
        }
    }

    // Report the best epoch and its stats.
    let (best_epoch, stats, _network) = history.best_epoch()?;
    eprintln!(
        "Best epoch: {}: train loss = {:.4} ({:.2}%), test loss = {:.4} ({:.2}%)",
        best_epoch,
        stats.train_loss,
        100.0 * stats.train_accuracy,
        stats.test_loss,
        100.0 * stats.test_accuracy,
    );

    // Optionally plot the training and test loss.
    if let Some(plot_path) = opt.plot {
        let epoch_training_losses = history
            .epochs()
            .iter()
            .map(|e| e.train_loss)
            .collect::<Vec<_>>();
        let epoch_test_losses = history
            .epochs()
            .iter()
            .map(|e| e.test_loss)
            .collect::<Vec<_>>();
        plot_loss(&plot_path, &epoch_training_losses, &epoch_test_losses)?;
    }

    // TODO: Reimplement `Network::save`.

    Ok(())
}

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
fn read_csv_data(
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
fn split_data(
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

/// Convert a 1-hot encoded array to a class index.
fn predicted_class_index(output: &ArrayView1<f32>) -> usize {
    output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

/// Use `plotters` to plot the training and test losses for each epoch as an
/// SVG and save it to `path`. Almost entirely written by Copilot.
fn plot_loss(path: &Path, training_losses: &[f32], test_losses: &[f32]) -> Result<()> {
    let root = SVGBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0f32..training_losses.len() as f32, 0.0f32..1.0f32)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            training_losses
                .iter()
                .enumerate()
                .map(|(x, y)| (x as f32, *y)),
            &RED,
        ))?
        .label("Training")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            test_losses.iter().enumerate().map(|(x, y)| (x as f32, *y)),
            &BLUE,
        ))?
        .label("Test")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    Ok(())
}
