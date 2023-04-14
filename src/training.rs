///! Our core training code.
use std::{
    cmp::min,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::Parser;
use log::{debug, trace};
use ndarray::{s, Array2, ArrayView1};
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::{
    history::{EpochStats, TrainingHistory},
    layers::ActivationFunction,
    network::Network,
    plot::plot_loss,
};

#[derive(Debug, Clone, Parser, Serialize)]
pub struct TrainOpt {
    /// Path to a JSONL file which will be used to log training history.
    #[arg(long = "history", default_value = "history.jsonl")]
    pub history_path: PathBuf,

    // Parameters are number of epochs, the training/test split, the learning rate, and size of input (4), hidden (7) and output (3) layers.
    /// Number of epochs.
    #[arg(long = "epochs", default_value = "2000")]
    pub epochs: usize,

    /// Batch size.
    #[arg(long = "batch-size", default_value = "32")]
    pub batch_size: usize,

    /// Learning rate.
    #[arg(long = "learning-rate", default_value = "0.01")]
    pub learning_rate: f32,

    /// What fraction of our hidden layers should be dropped out while training?
    /// This is a good way to reduce overfitting.
    #[arg(long = "dropout", default_value = "0.5")]
    pub dropout: f32,

    /// Which activation function should we use? Does not apply to the output
    /// layer, which always uses softmax.
    #[arg(long = "activation", default_value = "tanh", value_parser = ["leaky_relu", "tanh"])]
    pub activation: String,

    /// Leaky ReLU alpha.
    #[arg(long = "relu-leak", default_value = "0.01")]
    pub relu_leak: f32,

    /// How long should we wait before stopping training? If we see this many
    /// models in a row that are worse than our best model, we'll stop.
    #[arg(long = "patience", default_value = "5")]
    pub patience: usize,

    /// Path to save plot of training and test loss.
    #[arg(long = "plot")]
    pub plot: Option<PathBuf>,
}

impl TrainOpt {
    pub fn activation(&self) -> ActivationFunction {
        match self.activation.as_str() {
            "leaky_relu" => ActivationFunction::LeakyReLU(self.relu_leak),
            "tanh" => ActivationFunction::Tanh,
            _ => panic!("Unknown activation function"),
        }
    }
}

/// The data we'll be using for training and testing.
///
/// All of the data arrays are stored as `[examples, features]` arrays.
pub struct TrainAndTestData {
    pub train_inputs: Array2<f32>,
    pub train_targets: Array2<f32>,
    pub test_inputs: Array2<f32>,
    pub test_targets: Array2<f32>,
}

/// First attempt at a training function, originally written largely by Copilot
/// but revised heavily since.
pub fn train(
    dataset_name: &str,
    opt: TrainOpt,
    network: &mut Network,
    training_data: &TrainAndTestData,
) -> Result<()> {
    let mut history = TrainingHistory::new(dataset_name, opt.clone(), network);
    let mut rng = rand::thread_rng();
    'epochs: for epoch in 0..opt.epochs {
        debug!("Model: {:#?}", network);
        let mut train_loss = 0.0;
        let mut train_correct = 0;

        // Compute how many batches we'll have.
        let train_count = training_data.train_inputs.nrows();
        let batch_count = (train_count + opt.batch_size - 1) / opt.batch_size;

        // Shuffle our indices so we can train on the data in a random order.
        let train_count = training_data.train_inputs.nrows();
        let mut shuffled_batch_indices = (0..batch_count).collect::<Vec<_>>();
        shuffled_batch_indices.shuffle(&mut rng);

        // Train on each example.
        for i in shuffled_batch_indices {
            trace!("Epoch {} batch {}/{}", epoch, i, batch_count);

            // Find the start and end of this batch.
            let batch_start = i * opt.batch_size;
            let batch_end = min(batch_start + opt.batch_size, train_count);
            let inputs = training_data
                .train_inputs
                .slice(s![batch_start..batch_end, ..]);
            let targets = training_data
                .train_targets
                .slice(s![batch_start..batch_end, ..]);

            // Compute the gradients for our batch.
            network.compute_gradients(&inputs, &targets);

            // Now update the parameters.
            network.update_parameters(opt.learning_rate);

            // Compute the loss for our batch.
            let outputs = network.forward(&inputs);
            let loss = network.loss(&outputs.view(), &targets).sum();
            if !loss.is_finite() {
                eprintln!("Loss is not finite: {}", loss);
                history.set_training_failure(format!("Loss is not finite: {}", loss));
                break 'epochs;
            }
            train_loss += loss;

            for (output, target) in outputs.outer_iter().zip(targets.outer_iter()) {
                if predicted_class_index(&output) == predicted_class_index(&target) {
                    train_correct += 1;
                }
            }
        }
        train_loss /= train_count as f32;

        let mut test_loss = 0.0;
        let mut test_correct = 0;
        let test_count = training_data.test_inputs.nrows();
        for i in 0..test_count {
            let inputs = training_data.test_inputs.slice(s![i..i + 1, ..]);
            let targets = training_data.test_targets.slice(s![i..i + 1, ..]);

            let outputs = network.forward(&inputs);
            let loss = network
                .loss(&outputs.view(), &targets)
                .mean()
                .expect("empty batch");
            test_loss += loss;

            for (output, target) in outputs.outer_iter().zip(targets.outer_iter()) {
                if predicted_class_index(&output) == predicted_class_index(&target) {
                    test_correct += 1;
                }
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
            network,
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

    // Append history to a JSONL file.
    append_history_as_jsonl(&opt.history_path, &history).with_context(|| {
        format!("failed to save history to {}", opt.history_path.display())
    })?;

    // TODO: Reimplement `Network::save`.

    Ok(())
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

fn append_history_as_jsonl(
    history_path: &Path,
    history: &TrainingHistory,
) -> Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(history_path)?;
    serde_json::to_writer(&mut file, history)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}
