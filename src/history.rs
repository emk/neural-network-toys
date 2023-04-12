//! Training history for our neural network.

use anyhow::{anyhow, Result};

use crate::layers::Network;

/// The results of a single epoch of training.
pub struct EpochStats {
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub test_loss: f32,
    pub test_accuracy: f32,
}

/// The history of a neural network's training.
pub struct TrainingHistory {
    epochs: Vec<EpochStats>,
    best_test_accuracy: f32,
    best_epoch: usize,
    best_model: Option<Network>,
}

impl TrainingHistory {
    /// Create a new training history.
    pub fn new() -> Self {
        TrainingHistory {
            epochs: Vec::new(),
            best_test_accuracy: 0.0,
            best_epoch: 0,
            best_model: None,
        }
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
