import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class SpaceObjectClassifier:
    def __init__(self, data_dir, img_height=224, img_width=224, batch_size=32):
        """
        Initialize the Space Object Classifier

        Args:
            data_dir (str): Directory containing the dataset
            img_height (int): Target image height
            img_width (int): Target image width
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None

    def load_and_prepare_data(self, validation_split=0.2):
        """
        Load and prepare the dataset for training

        Args:
            validation_split (float): Fraction of data to use for validation

        Returns:
            tuple: Training and validation data generators
        """
        print("Loading and preparing data...")

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Only rescaling for validation
        valid_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=validation_split
        )

        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        # Load validation data
        self.valid_generator = valid_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)

        print(f"Found {self.num_classes} classes: {self.class_names}")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.valid_generator.samples}")

        return self.train_generator, self.valid_generator

    def build_custom_cnn(self):
        """
        Build a custom CNN model for space object classification
        """
        print("Building custom CNN model...")

        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        self.model = model
        return model

    def build_transfer_learning_model(self, base_model='resnet50'):
        """
        Build a transfer learning model for space object classification

        Args:
            base_model (str): Base model to use ('resnet50' or 'mobilenet')
        """
        print(f"Building transfer learning model with {base_model}...")

        # Choose the base model
        if base_model.lower() == 'resnet50':
            base = ResNet50(weights='imagenet', include_top=False, input_shape=(self.img_height, self.img_width, 3))
        elif base_model.lower() == 'mobilenet':
            base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.img_height, self.img_width, 3))
        else:
            raise ValueError("Base model must be 'resnet50' or 'mobilenet'")

        # Freeze the base model layers
        base.trainable = False

        # Build the model
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        self.model = model
        return model

    def train(self, epochs=20, early_stopping=True):
        """
        Train the model

        Args:
            epochs (int): Number of epochs to train
            early_stopping (bool): Whether to use early stopping

        Returns:
            History object containing training metrics
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_custom_cnn() or build_transfer_learning_model() first.")

        callbacks = []
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ))

        print(f"Training model for {epochs} epochs...")

        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=epochs,
            callbacks=callbacks
        )

        return self.history

    def evaluate(self):
        """
        Evaluate the model on the validation set

        Returns:
            tuple: (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("Evaluating model...")

        loss, accuracy = self.model.evaluate(self.valid_generator)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        return loss, accuracy

    def plot_training_history(self):
        """
        Plot the training history
        """
        if self.history is None:
            raise ValueError("Model not trained yet. Call train() first.")

        plt.figure(figsize=(12, 5))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    def predict_and_visualize(self, num_samples=5):
        """
        Make predictions on a few samples and visualize the results

        Args:
            num_samples (int): Number of samples to visualize
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print(f"Visualizing predictions for {num_samples} samples...")

        # Get a batch of validation data
        x_batch, y_batch = next(self.valid_generator)

        # Make predictions
        predictions = self.model.predict(x_batch)

        # Plot the results
        plt.figure(figsize=(15, num_samples * 3))

        for i in range(min(num_samples, len(x_batch))):
            plt.subplot(num_samples, 2, 2 * i + 1)
            plt.imshow(x_batch[i])
            plt.title(f"True: {self.class_names[np.argmax(y_batch[i])]}")
            plt.axis('off')

            plt.subplot(num_samples, 2, 2 * i + 2)
            plt.bar(self.class_names, predictions[i])
            plt.title(f"Predicted: {self.class_names[np.argmax(predictions[i])]}")
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('prediction_visualization.png')
        plt.show()

    def generate_confusion_matrix(self):
        """
        Generate and plot a confusion matrix for the validation set
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("Generating confusion matrix...")

        # Get the true labels
        y_true = self.valid_generator.classes

        # Get the predicted labels
        steps = np.ceil(self.valid_generator.samples / self.valid_generator.batch_size).astype(int)
        predictions = self.model.predict(self.valid_generator, steps=steps)
        y_pred = np.argmax(predictions, axis=1)

        # Generate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

    def save_model(self, filepath='space_object_classifier_model.h5'):
        """
        Save the trained model

        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='space_object_classifier_model.h5'):
        """
        Load a saved model

        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def create_sample_dataset_structure():
    """
    Create a sample dataset structure explanation
    """
    print("Sample Dataset Structure:")
    print("""
    data_dir/
    ├── satellites/
    │   ├── satellite_001.jpg
    │   ├── satellite_002.jpg
    │   └── ...
    ├── debris/
    │   ├── debris_001.jpg
    │   ├── debris_002.jpg
    │   └── ...
    ├── planets/
    │   ├── planet_001.jpg
    │   ├── planet_002.jpg
    │   └── ...
    └── stars/
        ├── star_001.jpg
        ├── star_002.jpg
        └── ...
    """)


def main():
    """
    Main function to run the space object classifier
    """
    # Display sample dataset structure
    create_sample_dataset_structure()

    # Initialize the classifier (replace 'path_to_dataset' with your dataset path)
    data_dir = 'path_to_dataset'  # Change this to your dataset directory

    # Example usage instructions
    print("\nExample Usage:")
    print("1. Initialize the classifier")
    print("   classifier = SpaceObjectClassifier(data_dir='path_to_dataset')")
    print("2. Load and prepare data")
    print("   classifier.load_and_prepare_data()")
    print("3. Build a model (choose one)")
    print("   classifier.build_custom_cnn() or classifier.build_transfer_learning_model('resnet50')")
    print("4. Train the model")
    print("   classifier.train(epochs=20)")
    print("5. Evaluate and visualize")
    print("   classifier.evaluate()")
    print("   classifier.plot_training_history()")
    print("   classifier.predict_and_visualize()")
    print("   classifier.generate_confusion_matrix()")
    print("6. Save the model")
    print("   classifier.save_model()")

    # Uncomment the following code to run the full pipeline
    """
    # Create classifier instance
    classifier = SpaceObjectClassifier(data_dir)

    # Load and prepare data
    classifier.load_and_prepare_data()

    # Build transfer learning model
    classifier.build_transfer_learning_model('resnet50')
    # Alternatively, use a custom CNN
    # classifier.build_custom_cnn()

    # Train the model
    classifier.train(epochs=20)

    # Evaluate the model
    classifier.evaluate()

    # Visualize training history
    classifier.plot_training_history()

    # Make predictions and visualize
    classifier.predict_and_visualize()

    # Generate confusion matrix
    classifier.generate_confusion_matrix()

    # Save the model
    classifier.save_model()
    """


if __name__ == "__main__":
    main()