//! A terminal user interface for our program.

use std::io;

use anyhow::Result;
use tui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Spans},
    widgets::{Axis, Block, Borders, Dataset, Paragraph},
    Terminal,
};

use crate::history::{EpochStats, TrainingHistory};

type Frame<'a> = tui::Frame<'a, CrosstermBackend<io::Stdout>>;

/// A terminal user interface for our program. We implement this using the `tui`
/// crate.
///
/// We want to display useful information about the training process in a
/// visually interesting and succinct fashion. This includes:
///
/// - A few lines containing the most important information about the training
///   run.
/// - A graph of the training and test loss over time.
/// - A scrolling log showing the output of each epoch.  Every epoch which is
///   better than the previous best should be highlighted.
pub struct Ui {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl Ui {
    /// Create a new user interface.
    pub fn new() -> Result<Self> {
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    /// Finish drawing the user interface.
    pub fn close(&mut self) -> Result<()> {
        self.terminal.show_cursor()?;
        self.terminal.clear()?;
        self.terminal.flush()?;
        Ok(())
    }

    /// Draw the user interface.
    pub fn draw(&mut self, history: &TrainingHistory) -> Result<()> {
        self.terminal.clear()?;
        self.terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(2),
                    // This should take up 2/5ths of the screen.
                    Constraint::Ratio(2, 5),
                    Constraint::Min(1),
                ])
                .split(f.size());

            Self::draw_header(f, chunks[0], history);
            Self::draw_loss_graph(f, chunks[1], history);
            Self::draw_log(f, chunks[2], history);
        })?;

        Ok(())
    }

    /// Draw the header.
    fn draw_header(f: &mut Frame, area: Rect, history: &TrainingHistory) {
        let style = Style::default().fg(Color::Yellow);
        let mut text = vec![Spans::from(vec![Span::styled(
            format!(
                "{} {} layers={}",
                history.dataset_name(),
                history.optimizer_metadata().optimizer_type,
                history.network_metadata().layers.len(),
            ),
            style,
        )])];

        if let Some((epoch, stats, _)) = history.best_epoch() {
            text.push(Spans::from(vec![Span::styled(
                // Show loss and accuracy for training and test data.
                format!("Best epoch: {}/{} {}", epoch, history.epochs().len(), stats),
                style,
            )]))
        }

        let paragraph = Paragraph::new(text);
        f.render_widget(paragraph, area);
    }

    /// Draw the loss graph.
    fn draw_loss_graph(f: &mut Frame, area: Rect, history: &TrainingHistory) {
        // The maximum number of data points we can display.
        let max_data_points = area.width as usize;
        let offset = history.epochs().len().saturating_sub(max_data_points);

        // Our data points.
        let prep_data = |history: &TrainingHistory, f: fn(&EpochStats) -> f32| {
            let data = history
                .epochs()
                .iter()
                .map(|stats| f(stats))
                .enumerate()
                .map(|(i, loss)| (i as f64, loss as f64))
                .collect::<Vec<_>>();
            if data.len() > max_data_points {
                data[offset..].to_owned()
            } else {
                data
            }
        };
        let training_losses = prep_data(history, |stats| stats.train_loss);
        let test_losses = prep_data(history, |stats| stats.test_loss);

        let datasets = vec![
            Dataset::default()
                .name("training")
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::LightBlue))
                .data(&training_losses),
            Dataset::default()
                .name("test")
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::Red))
                .data(&test_losses),
        ];

        let block = Block::default().borders(Borders::ALL).title("Loss");
        let chart = tui::widgets::Chart::new(datasets)
            .block(block)
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([offset as f64, (offset + max_data_points) as f64]),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 1.0]),
            );
        f.render_widget(chart, area);
    }

    /// Draw the log, scrolled to show the most recent epochs.
    fn draw_log(f: &mut Frame, area: Rect, history: &TrainingHistory) {
        let style = Style::default();
        let mut text = vec![];
        for (epoch, stats) in history.epochs().iter().enumerate() {
            let is_best =
                epoch == history.best_epoch().map(|(epoch, _, _)| epoch).unwrap_or(0);
            let style = if is_best {
                style.add_modifier(Modifier::BOLD).fg(Color::Green)
            } else {
                style
            };

            text.push(Spans::from(vec![Span::styled(
                format!("Epoch {} {}", epoch, stats),
                style,
            )]))
        }

        let line_count = text.len();
        let block = Block::default().borders(Borders::ALL).title("Log");
        let paragraph = Paragraph::new(text).block(block);
        let scrolled =
            paragraph.scroll(((line_count as u16).saturating_sub(area.height - 2), 0));
        f.render_widget(scrolled, area);
    }
}
