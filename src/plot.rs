///! Plotting functions.
use std::path::Path;

use anyhow::Result;
use plotters::{
    prelude::{ChartBuilder, IntoDrawingArea, PathElement},
    series::LineSeries,
    style::{Color, IntoFont, BLACK, BLUE, RED, WHITE},
};
use plotters_svg::SVGBackend;

/// Use `plotters` to plot the training and test losses for each epoch as an
/// SVG and save it to `path`. Almost entirely written by Copilot.
pub fn plot_loss(
    path: &Path,
    training_losses: &[f32],
    test_losses: &[f32],
) -> Result<()> {
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
