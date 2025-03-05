import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import the classifier class from our main module
from space_object_classifier import SpaceObjectClassifier


def train_model(args):
    """
    Train a space object classifier model

    Args:
        args: Command-line arguments
    """
    print("=" * 50)
    print(f"Training space object classifier with {args.model_type} model")
    print("=" * 50)

    # Create output directory for results
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the classifier
    classifier = SpaceObjectClassifier(
        data_dir=args.data_dir,
        img_height=args.img_size,
        img_width=args.img_size,
        batch_size=args.batch_size
    )

    # Load and prepare data
    train_generator, valid_generator = classifier.load_and_prepare_data(
        validation_split=args.validation_split
    )

    # Build the model
    if args.model_type == 'custom':
        model = classifier.build_custom_cnn()
    elif args.model_type == 'resnet50':
        model = classifier.build_transfer_learning_model('resnet50')
    elif args.model_type == 'mobilenet':
        model = classifier.build_transfer_learning_model('mobilenet')
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Define callbacks
    callbacks = []

    # Early stopping
    if args.early_stopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

    # Model checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'model_{epoch:02d}.h5')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
    callbacks.append(checkpoint)

    # TensorBoard logging
    log_dir = os.path.join(args.output_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)

    # Learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # Train the model
    print("\nStarting training...")
    history = classifier.model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)

    # Evaluate the model
    print("\nEvaluating model...")
    loss, accuracy = classifier.model.evaluate(valid_generator)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    # Generate evaluation metrics
    print("\nGenerating evaluation metrics...")

    # Get true labels
    y_true = valid_generator.classes

    # Get predictions
    steps = np.ceil(valid_generator.samples / valid_generator.batch_size).astype(int)
    predictions = classifier.model.predict(valid_generator, steps=steps)
    y_pred = np.argmax(predictions, axis=1)

    # Classification report
    class_names = list(valid_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(args.output_dir, 'classification_report.csv'))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300)

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=300)

    # Save the final model
    final_model_path = os.path.join(args.output_dir, 'final_model.h5')
    classifier.model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    # Save model summary
    with open(os.path.join(args.output_dir, 'model_summary.txt'), 'w') as f:
        classifier.model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Generate a sample visualization of predictions
    visualize_predictions(classifier, valid_generator, class_names, args.output_dir)

    print("\nTraining complete! Results saved to:", args.output_dir)


def visualize_predictions(classifier, data_generator, class_names, output_dir, num_samples=10):
    """
    Generate a visualization of model predictions on sample images

    Args:
        classifier: Trained SpaceObjectClassifier instance
        data_generator: Data generator for the validation set
        class_names: List of class names
        output_dir: Directory to save the visualization
        num_samples: Number of samples to visualize
    """
    print(f"\nGenerating prediction visualization for {num_samples} samples...")

    # Get a batch of validation data
    x_batch, y_batch = next(data_generator)

    # Make predictions
    predictions = classifier.model.predict(x_batch)

    # Create the visualization
    num_samples = min(num_samples, len(x_batch))
    fig = plt.figure(figsize=(15, num_samples * 3))

    for i in range(num_samples):
        # Display the image
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(x_batch[i])
        plt.title(f"True: {class_names[np.argmax(y_batch[i])]}")
        plt.axis('off')

        # Display the prediction probabilities
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.barh(range(len(class_names)), predictions[i])
        plt.yticks(range(len(class_names)), class_names)
        plt.title(f"Predicted: {class_names[np.argmax(predictions[i])]}")
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'), dpi=300)
    plt.close()


def parse_arguments():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train Space Object Classifier')

    # Dataset and model parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['custom', 'resnet50', 'mobilenet'],
                        help='Type of model to train')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for training (width and height)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Train the model
    train_model(args)