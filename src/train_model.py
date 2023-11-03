
import os
import pprint
import torch

from typing import List

from src import datasets, run_logging, models, metrics
from src import model_visualizations as mv
from src.trainer import Trainer
from src.default_settings import DefaultSettings


def train_model(settings: DefaultSettings):
    # For printing and logging:
    line_formatter = run_logging.LineFormatter(column_width=7, seperator=" |")
    run_logger = run_logging.RunLogger(log_folder=settings.log_folder)
    run_logger.write_to_file("settings.txt", pprint.pformat(settings.as_dict))
    run_logger.write_to_file("progress.txt",
                             f"Run id: {run_logger.run_id}\n"
                             f"Logging folder: {run_logger.log_dir}\n"
                             f"About to start training!\n")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader:
    train_loader, val_loader = datasets.get_image_data_loaders(
        dataset_name=settings.dataset_name,
        data_dir=settings.data_dir,
        batch_size=settings.batch_size,
        use_test_set=settings.use_test_set,
        val_proportion=0.1,
        shuffle_in_train_loader=True,
        num_workers=4,
    )

    # Validation image batch, for certain visualizations:
    val_images, val_labels = next(iter(val_loader))
    validation_image_batch = val_images.to(device)

    # Model:
    model = models.get_model(
        settings.model_name,
        activation_name=settings.activation_name,
        conv_name=settings.conv_name,
        **settings.model_kwargs
    )
    model.to(device)
    run_logger.write_to_file("model.txt", models.util.get_summary(model))

    # Loss and optimizer
    loss_cls = getattr(metrics, settings.loss_name)
    loss_function = loss_cls(**settings.loss_kwargs)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.,  # Learning rate is determined by the scheduler (below).
        weight_decay=settings.weight_decay,
        momentum=settings.momentum,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=10 ** settings.log_lr,
        epochs=settings.nrof_epochs,
        steps_per_epoch=1
    )

    # Metrics:
    training_metrics: List[metrics.Metric] = [
        loss_function,
        metrics.Accuracy(),
        metrics.BatchVariance(),
        metrics.Margin(),
        metrics.CRAFromScores(1 * 36 / 255),
        metrics.CRAFromScores(2 * 36 / 255),
        metrics.CRAFromScores(3 * 36 / 255),
        metrics.CRAFromScores(1.),
    ]

    # Visualizations:
    mv.Visualization.log_dir = run_logger.log_dir
    save_activation_variances = mv.SaveActivationVariances(
        model,
        validation_image_batch,
    )
    visualizations = [save_activation_variances]

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        training_metrics,
        device,
    )

    # Timing:
    timer = run_logging.Timer()
    timer.start()

    for epoch in range(1, settings.nrof_epochs + 1):
        train_metrics = trainer.train_epoch()
        scheduler.step()
        val_metrics = trainer.evaluate()

        for visualize in visualizations:
            visualize(epoch)

        epoch_metrics = {
            "Epoch": epoch,
            # "AvgEpochTimeSeconds": timer.seconds_elapsed / epoch,
            **train_metrics,
            **val_metrics,
        }

        outline = line_formatter.create_line(epoch_metrics)
        run_logger.log_progress(outline)
        run_logger.write_to_file("results.txt", str(epoch_metrics) + "\n", "a")

    time_in_hours = timer.seconds_elapsed / 3600
    run_logger.write_to_file("progress.txt",
                             f"Done in {time_in_hours}h.\n",
                             "a")


if __name__ == "__main__":
    chosen_settings = DefaultSettings()
    chosen_settings.data_dir = os.path.join("..", "data")
    print(f"\nSettings:")
    pprint.pprint(chosen_settings.as_dict)
    print()

    train_model(chosen_settings)
