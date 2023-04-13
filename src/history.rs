//! Training history for our neural network.

use anyhow::{anyhow, Result};
use serde::Serialize;

use crate::{
    network::{Network, NetworkMetadata},
    training::TrainOpt,
};

/// The results of a single epoch of training.
#[derive(Debug, Clone, Serialize)]
pub struct EpochStats {
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub test_loss: f32,
    pub test_accuracy: f32,
}

/// The history of a neural network's training.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingHistory {
    /// The name of the dataset used for training.
    dataset_name: String,

    /// The options used for this training run.
    opt: TrainOpt,

    /// Metadata about the network, including the layers.
    network: NetworkMetadata,

    /// History of each epoch.
    epochs: Vec<EpochStats>,

    /// The best test accuracy seen during training.
    best_test_accuracy: f32,

    /// The epoch with the best test accuracy.
    best_epoch: usize,

    /// The model with the best test accuracy.
    #[serde(skip)]
    best_model: Option<Network>,

    /// If training failed, why?
    #[serde(skip_serializing_if = "Option::is_none")]
    training_failure: Option<String>,
}

impl TrainingHistory {
    /// Create a new training history.
    pub fn new(dataset_name: &str, opt: TrainOpt, network: &Network) -> Self {
        TrainingHistory {
            dataset_name: dataset_name.to_string(),
            opt,
            network: network.metadata(),
            epochs: Vec::new(),
            best_test_accuracy: 0.0,
            best_epoch: 0,
            best_model: None,
            training_failure: None,
        }
    }

    /// If we failed to train, why?
    pub fn set_training_failure<S: Into<String>>(&mut self, reason: S) {
        self.training_failure = Some(reason.into());
    }

    /// Get information about each epoch.
    pub fn epochs(&self) -> &[EpochStats] {
        &self.epochs
    }

    /// The best epoch seen during training.
    pub fn best_epoch(&self) -> Result<(usize, &EpochStats, &Network)> {
        if self.epochs.is_empty() || self.best_model.is_none() {
            return Err(anyhow!("No epochs have been added to the history yet."));
        }
        Ok((
            self.best_epoch,
            &self.epochs[self.best_epoch],
            self.best_model.as_ref().unwrap(),
        ))
    }

    /// Add a new epoch to the history.
    pub fn add_epoch(&mut self, epoch_history: EpochStats, model: &Network) {
        let test_accuracy = epoch_history.test_accuracy;
        self.epochs.push(epoch_history);
        if test_accuracy > self.best_test_accuracy {
            self.best_test_accuracy = test_accuracy;
            self.best_epoch = self.epochs.len() - 1;
            self.best_model = Some(model.clone());
        }
    }

    /// Should we stop training? If all of the last `patience` epochs have been
    /// worse than our best accuracy, then we should stop.
    pub fn should_stop(&self, patience: usize) -> bool {
        if self.epochs.len() < patience {
            return false;
        }

        let last_n = &self.epochs[self.epochs.len() - patience..];
        last_n
            .iter()
            .all(|epoch| epoch.test_accuracy < self.best_test_accuracy)
    }
}
