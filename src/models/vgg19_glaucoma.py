"""
VGG19 Glaucoma Classification Model.

A pretrained VGG19 model fine-tuned for binary glaucoma classification
(Glaucoma vs No Glaucoma) on fundus images from the Eye-AI catalog.

The model uses transfer learning with ImageNet pretrained weights and
replaces the final classifier layer for binary classification.

Data Loading:
The model uses DerivaML's restructure_assets() method to reorganize downloaded
images into a directory structure that torchvision's ImageFolder can consume:

    data_dir/
        training/
            No Glaucoma/
            Suspected Glaucoma/
        testing/
            No Glaucoma/
            Suspected Glaucoma/

Note: Images labeled "Unknown" are filtered out for binary classification.

Test Evaluation:
After training, the model evaluates on the test set and records per-image
classification predictions to the DerivaML catalog using the Image_Diagnosis
feature. This enables tracking of model predictions and comparison with ground
truth labels.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from deriva_ml import DerivaML, MLAsset, ExecAssetType
from deriva_ml.execution import Execution


# Binary classification: map diagnosis to binary label
# "No Glaucoma" -> 0, "Suspected Glaucoma" -> 1
GLAUCOMA_CLASSES = ["No Glaucoma", "Suspected Glaucoma"]
EXCLUDED_CLASSES = ["Unknown"]


def build_filename_to_rid_map(ml_instance: DerivaML) -> dict[str, str]:
    """Build a mapping from filenames to RIDs from the Image table.

    Queries all Image records in the catalog to create a lookup table
    that maps image filenames to their catalog RIDs.

    Args:
        ml_instance: DerivaML instance for catalog access.

    Returns:
        Dictionary mapping filename (without path) to RID.
    """
    # Query the Image table directly using the path builder
    # since Image may not have an Asset_Type association in all catalogs
    table_path = ml_instance.domain_path.tables["Image"]
    images = list(table_path.attributes(table_path.RID, table_path.Filename).fetch())
    return {img["Filename"]: img["RID"] for img in images if "Filename" in img and "RID" in img}


def record_test_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    filename_to_rid: dict[str, str],
    execution: Execution,
    ml_instance: DerivaML,
    device: torch.device,
) -> int:
    """Record per-image classification predictions to the DerivaML catalog.

    Runs inference on the test set and records each prediction as an
    Image_Diagnosis feature value, linking the predicted diagnosis
    to the image RID.

    Additionally saves a CSV file with probability distributions for
    ROC curve analysis.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data (from ImageFolder).
        class_names: List of class names in index order.
        filename_to_rid: Mapping from image filename to RID.
        execution: DerivaML execution context.
        ml_instance: DerivaML instance for catalog access.
        device: PyTorch device for inference.

    Returns:
        Number of predictions recorded.
    """
    model.eval()

    # Get the feature record class for Image_Diagnosis
    ImageDiagnosis = ml_instance.feature_record_class("Image", "Image_Diagnosis")

    # Collect all predictions
    feature_records = []

    # Collect data for CSV output (probability distributions)
    csv_rows = []

    # Get the underlying dataset to access file paths
    test_dataset = test_loader.dataset
    # Handle Subset wrapper if present
    if isinstance(test_dataset, Subset):
        base_dataset = test_dataset.dataset
        indices = test_dataset.indices
    else:
        base_dataset = test_dataset
        indices = list(range(len(test_dataset)))

    with torch.no_grad():
        sample_idx = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Compute softmax probabilities
            probabilities = F.softmax(outputs, dim=1)

            # Get predicted class and confidence
            confidences, predicted = probabilities.max(1)

            # Process each sample in the batch
            for i in range(inputs.size(0)):
                # Get the file path for this sample
                actual_idx = indices[sample_idx]
                img_path, _ = base_dataset.samples[actual_idx]
                filename = Path(img_path).name

                # Get probability distribution for this sample
                probs = probabilities[i].cpu().numpy()

                # Look up the RID
                rid = filename_to_rid.get(filename)
                if rid:
                    predicted_class = class_names[predicted[i].item()]

                    # Record to catalog
                    feature_records.append(
                        ImageDiagnosis(
                            Image=rid,
                            Diagnosis_Image=predicted_class,
                        )
                    )

                    # Build CSV row with probability distribution
                    csv_row = {
                        "Image_RID": rid,
                        "Filename": filename,
                        "Predicted_Class": predicted_class,
                        "Confidence": confidences[i].item(),
                        "True_Label": class_names[labels[i].item()],
                    }
                    # Add probability for each class
                    for j, class_name in enumerate(class_names):
                        csv_row[f"prob_{class_name.replace(' ', '_')}"] = probs[j]
                    csv_rows.append(csv_row)

                sample_idx += 1

    # Bulk add all predictions to the execution
    if feature_records:
        execution.add_features(feature_records)
        print(f"  Recorded {len(feature_records)} classification predictions")
    else:
        print("  WARNING: No predictions could be recorded (no RID matches)")

    # Save probability distributions to CSV
    if csv_rows:
        csv_file = execution.asset_file_path(
            MLAsset.execution_asset, "prediction_probabilities.csv", ExecAssetType.output_file
        )
        fieldnames = ["Image_RID", "Filename", "Predicted_Class", "Confidence", "True_Label"] + [
            f"prob_{c.replace(' ', '_')}" for c in class_names
        ]
        with csv_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Saved probability distributions to: {csv_file}")

    return len(feature_records)


class VGG19Classifier(nn.Module):
    """VGG19 pretrained model adapted for binary glaucoma classification.

    Uses ImageNet pretrained weights and replaces the classifier head
    for binary classification (No Glaucoma vs Suspected Glaucoma).

    Args:
        num_classes: Number of output classes (default: 2 for binary).
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_features: Whether to freeze the convolutional feature extractor.
        dropout_rate: Dropout probability in the classifier.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_features: bool = False,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        # Load pretrained VGG19
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg19 = models.vgg19(weights=weights)

        # Optionally freeze feature extractor
        if freeze_features:
            for param in self.vgg19.features.parameters():
                param.requires_grad = False

        # Replace classifier for binary classification
        # VGG19 classifier: Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, 1000)
        in_features = self.vgg19.classifier[0].in_features  # 25088
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg19(x)


def filter_unknown_class(dataset: ImageFolder) -> Subset:
    """Filter out samples with 'Unknown' class label.

    Args:
        dataset: ImageFolder dataset with class labels.

    Returns:
        Subset containing only samples from valid classes.
    """
    valid_indices = []
    for idx, (_, label) in enumerate(dataset.samples):
        class_name = dataset.classes[label]
        if class_name not in EXCLUDED_CLASSES:
            valid_indices.append(idx)

    print(f"  Filtered {len(dataset) - len(valid_indices)} 'Unknown' samples")
    return Subset(dataset, valid_indices)


def load_glaucoma_data_from_execution(
    execution: Execution,
    batch_size: int,
    image_size: int = 224,
) -> tuple[DataLoader | None, DataLoader | None, list[str], Path]:
    """Load glaucoma data from DerivaML execution datasets.

    Uses DerivaML's restructure_assets() method to organize downloaded images
    into the directory structure expected by torchvision's ImageFolder.
    Images are grouped by the Image_Diagnosis feature value.

        data_dir/
            training/          # From dataset with type "Training"
                No Glaucoma/
                Suspected Glaucoma/
            testing/           # From dataset with type "Testing"
                No Glaucoma/
                Suspected Glaucoma/

    Args:
        execution: DerivaML execution containing downloaded datasets.
        batch_size: Batch size for DataLoader.
        image_size: Target image size (default: 224 for VGG19).

    Returns:
        Tuple of (train_loader, test_loader, class_names, data_dir).
        - train_loader: DataLoader for training data (or None)
        - test_loader: DataLoader for test data (or None)
        - class_names: List of class names in index order
        - data_dir: The restructured directory path
    """
    # VGG19 expects 224x224 images normalized with ImageNet stats
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],   # ImageNet stds
        )
    ])

    # Create output directory for restructured data
    data_dir = execution.working_dir / "glaucoma_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Restructure assets from each dataset
    def type_selector(types: list[str]) -> str:
        """Select dataset type for directory structure."""
        type_lower = [t.lower() for t in types]
        if "training" in type_lower:
            return "training"
        elif "testing" in type_lower or "test" in type_lower:
            return "testing"
        elif types:
            return types[0].lower()
        return "unknown"

    # Process each dataset in the execution
    for dataset in execution.datasets:
        # Restructure images by dataset type and diagnosis label
        # This creates: data_dir/<dataset_type>/<diagnosis>/image.png
        dataset.restructure_assets(
            asset_table="Image",
            output_dir=data_dir,
            group_by=["Image_Diagnosis.Diagnosis_Image"],  # Use Diagnosis_Image column from Image_Diagnosis feature
            use_symlinks=True,
            type_selector=type_selector,
            enforce_vocabulary=False,  # Allow multiple values, use first found
        )

    # Create DataLoaders
    train_loader = None
    test_loader = None
    class_names: list[str] = GLAUCOMA_CLASSES.copy()

    # Find training and testing directories
    def find_data_dir(base_dir: Path, target_name: str) -> Path | None:
        """Find a directory with the given name, searching recursively."""
        direct_path = base_dir / target_name
        if direct_path.exists() and any(direct_path.iterdir()):
            return direct_path
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                nested_path = subdir / target_name
                if nested_path.exists() and any(nested_path.iterdir()):
                    return nested_path
        return None

    train_dir = find_data_dir(data_dir, "training")
    if train_dir:
        train_dataset = ImageFolder(train_dir, transform=transform)
        # Filter out Unknown class
        train_subset = filter_unknown_class(train_dataset)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        print(f"  Training classes: {train_dataset.classes}")
        print(f"  Training samples (after filtering): {len(train_subset)}")

    test_dir = find_data_dir(data_dir, "testing")
    if test_dir:
        test_dataset = ImageFolder(test_dir, transform=transform)
        # Filter out Unknown class
        test_subset = filter_unknown_class(test_dataset)
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        print(f"  Testing classes: {test_dataset.classes}")
        print(f"  Testing samples (after filtering): {len(test_subset)}")

    return train_loader, test_loader, class_names, data_dir


def vgg19_glaucoma(
    # Model architecture parameters
    pretrained: bool = True,
    freeze_features: bool = False,
    dropout_rate: float = 0.5,
    # Training parameters
    learning_rate: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    # Image parameters
    image_size: int = 224,
    # Checkpoint settings
    save_optimizer_state: bool = False,
    # Test-only mode
    test_only: bool = False,
    weights_filename: str = "vgg19_glaucoma_weights.pt",
    # DerivaML integration
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
    """Train or evaluate VGG19 for binary glaucoma classification.

    This function integrates with DerivaML to:
    - Load data from execution datasets using restructure_assets()
    - Track training progress
    - Save model weights as execution assets
    - Record per-image predictions to the catalog

    The function expects datasets containing Image assets with Image_Diagnosis
    feature values. Images are reorganized into a directory structure by dataset
    type (training/testing) and diagnosis label, then loaded using ImageFolder.

    Binary Classification:
        The model classifies fundus images as either "No Glaucoma" or
        "Suspected Glaucoma". Images labeled "Unknown" are filtered out.

    Test-only mode:
        When test_only=True, the model loads pre-trained weights from an
        execution asset and runs evaluation on the test set without training.

    Args:
        pretrained: Use ImageNet pretrained weights (default: True).
        freeze_features: Freeze the convolutional layers (default: False).
        dropout_rate: Dropout probability in classifier (default: 0.5).
        learning_rate: Optimizer learning rate (default: 1e-4).
        epochs: Number of training epochs (default: 10).
        batch_size: Training batch size (default: 32).
        weight_decay: L2 regularization weight decay (default: 1e-4).
        image_size: Input image size (default: 224).
        save_optimizer_state: If True, save full checkpoint with optimizer state
            (larger file, needed for resuming training). If False, save only model
            weights (smaller file, sufficient for inference). Default: False.
        test_only: Skip training and only run evaluation (default: False).
        weights_filename: Filename for weights file (default: vgg19_glaucoma_weights.pt).
        ml_instance: DerivaML instance for catalog access.
        execution: DerivaML execution context with datasets and assets.
    """
    mode = "Test-only" if test_only else "Training"
    print(f"VGG19 Glaucoma Classification ({mode})")
    print(f"  Host: {ml_instance.host_name}, Catalog: {ml_instance.catalog_id}")
    print(f"  Architecture: VGG19, pretrained={pretrained}, freeze_features={freeze_features}")
    if not test_only:
        print(f"  Training: lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # Create model
    model = VGG19Classifier(
        num_classes=2,
        pretrained=pretrained,
        freeze_features=freeze_features,
        dropout_rate=dropout_rate,
    ).to(device)

    # Load data from execution datasets
    print("\nLoading and restructuring data from execution datasets...")
    train_loader, test_loader, class_names, data_dir = load_glaucoma_data_from_execution(
        execution, batch_size, image_size
    )
    print(f"  Data directory: {data_dir}")
    print(f"  Classes: {class_names}")

    # Build filename -> RID mapping for recording predictions
    filename_to_rid = build_filename_to_rid_map(ml_instance)
    print(f"  Loaded {len(filename_to_rid)} filename-to-RID mappings")

    # Test-only mode: load weights and run evaluation
    if test_only:
        if test_loader is None:
            print("ERROR: No test data found in execution datasets.")
            print("  Test-only mode requires a dataset with type 'Testing'.")
            return

        print(f"  Test batches: {len(test_loader)}")

        # Find weights file in execution assets
        weights_path = None
        all_assets = []
        for table_name, paths in execution.asset_paths.items():
            for asset_path in paths:
                all_assets.append(asset_path)
                if asset_path.name == weights_filename:
                    weights_path = asset_path
                    break
            if weights_path:
                break

        if weights_path is None:
            print(f"ERROR: Weights file '{weights_filename}' not found in execution assets.")
            print(f"  Available assets: {[p.name for p in all_assets]}")
            return

        print(f"\nLoading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # Load model config from checkpoint if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  Checkpoint config: {config}")
            model = VGG19Classifier(
                num_classes=2,
                pretrained=False,  # Don't load ImageNet weights
                freeze_features=config.get('freeze_features', freeze_features),
                dropout_rate=config.get('dropout_rate', dropout_rate),
            ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Weights loaded successfully")

        # Run evaluation
        _evaluate_and_save(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            filename_to_rid=filename_to_rid,
            execution=execution,
            ml_instance=ml_instance,
            device=device,
            weights_filename=weights_filename,
        )
        return

    # Training mode: check for training data
    if train_loader is None:
        print("WARNING: No training data found in execution datasets.")
        status_file = execution.asset_file_path(
            MLAsset.execution_asset, "training_status.txt", ExecAssetType.output_file
        )
        with status_file.open("w") as f:
            f.write("No training data available in execution datasets.\n")
        return

    print(f"  Training batches: {len(train_loader)}")
    if test_loader:
        print(f"  Test batches: {len(test_loader)}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    print("\nTraining...")
    training_log = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'learning_rate': scheduler.get_last_lr()[0],
        }

        # Evaluate on test set if available
        if test_loader:
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()

            test_acc = 100.0 * test_correct / test_total
            test_loss = test_loss / len(test_loader)
            log_entry['test_loss'] = test_loss
            log_entry['test_acc'] = test_acc

            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%, "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%")
        else:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%")

        training_log.append(log_entry)

    # Save model weights
    print("\nSaving model...")
    weights_file = execution.asset_file_path(
        MLAsset.execution_asset, "vgg19_glaucoma_weights.pt", ExecAssetType.model_file
    )

    # Build checkpoint dict - optionally include optimizer state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'pretrained': pretrained,
            'freeze_features': freeze_features,
            'dropout_rate': dropout_rate,
            'image_size': image_size,
        },
        'training_log': training_log,
        'class_names': class_names,
    }
    if save_optimizer_state:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        print("  Including optimizer state (full checkpoint for resuming training)")
    else:
        print("  Saving weights only (smaller file, sufficient for inference)")

    torch.save(checkpoint, weights_file)
    print(f"  Saved weights to: {weights_file}")

    # Save training log as text
    log_file = execution.asset_file_path(
        MLAsset.execution_asset, "training_log.txt", ExecAssetType.output_file
    )
    with log_file.open("w") as f:
        f.write("VGG19 Glaucoma Classification Training Log\n")
        f.write("=" * 50 + "\n\n")
        f.write("Architecture:\n")
        f.write(f"  pretrained: {pretrained}\n")
        f.write(f"  freeze_features: {freeze_features}\n")
        f.write(f"  dropout_rate: {dropout_rate}\n")
        f.write(f"  image_size: {image_size}\n\n")
        f.write("Training Parameters:\n")
        f.write(f"  learning_rate: {learning_rate}\n")
        f.write(f"  epochs: {epochs}\n")
        f.write(f"  batch_size: {batch_size}\n")
        f.write(f"  weight_decay: {weight_decay}\n\n")
        f.write("Training Progress:\n")
        for entry in training_log:
            line = f"  Epoch {entry['epoch']}: train_loss={entry['train_loss']:.4f}, train_acc={entry['train_acc']:.2f}%"
            if 'test_acc' in entry:
                line += f", test_acc={entry['test_acc']:.2f}%"
            f.write(line + "\n")
    print(f"  Saved log to: {log_file}")

    # Record test predictions to catalog if test data is available
    if test_loader and filename_to_rid:
        print("\nRecording test predictions to catalog...")
        record_test_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            filename_to_rid=filename_to_rid,
            execution=execution,
            ml_instance=ml_instance,
            device=device,
        )

    print("\nTraining complete!")


def _evaluate_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    filename_to_rid: dict[str, str],
    execution: Execution,
    ml_instance: DerivaML,
    device: torch.device,
    weights_filename: str,
) -> None:
    """Evaluate model and save results (used in test-only mode)."""
    print("\nEvaluating on test set...")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_correct = 0
    test_total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * test_correct / test_total
    test_loss = test_loss / len(test_loader)
    print(f"  Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

    # Record predictions to catalog
    if filename_to_rid:
        print("\nRecording test predictions to catalog...")
        record_test_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            filename_to_rid=filename_to_rid,
            execution=execution,
            ml_instance=ml_instance,
            device=device,
        )

    # Save evaluation results
    results_file = execution.asset_file_path(
        MLAsset.execution_asset, "evaluation_results.txt", ExecAssetType.output_file
    )
    with results_file.open("w") as f:
        f.write("VGG19 Glaucoma Classification Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Weights file: {weights_filename}\n")
        f.write(f"Test samples: {test_total}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.2f}%\n")
    print(f"  Saved results to: {results_file}")

    print("\nEvaluation complete!")
