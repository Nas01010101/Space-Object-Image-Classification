import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


def load_model(model_path):
    """
    Load a trained model

    Args:
        model_path (str): Path to the saved model

    Returns:
        tf.keras.Model: Loaded model
    """
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_and_preprocess_image(image_path, target_size=(224, 224), preprocess_func=None):
    """
    Load and preprocess an image for prediction

    Args:
        image_path (str): Path to the image
        target_size (tuple): Target image size (height, width)
        preprocess_func: Preprocessing function to apply

    Returns:
        numpy.ndarray: Preprocessed image
    """
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        if preprocess_func:
            img_array = preprocess_func(img_array)
        else:
            # Default preprocessing: rescale to [0, 1]
            img_array = img_array / 255.0

        return img_array, img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None


def predict_image(model, image_path, class_names, target_size=(224, 224), model_type='custom'):
    """
    Make a prediction for a single image

    Args:
        model (tf.keras.Model): Trained model
        image_path (str): Path to the image
        class_names (list): List of class names
        target_size (tuple): Target image size
        model_type (str): Type of model ('custom', 'resnet50', or 'mobilenet')

    Returns:
        tuple: (predicted_class, confidence, preprocessed_image)
    """
    # Choose the appropriate preprocessing function
    if model_type == 'resnet50':
        preprocess_func = resnet_preprocess
    elif model_type == 'mobilenet':
        preprocess_func = mobilenet_preprocess
    else:
        preprocess_func = None

    # Load and preprocess the image
    img_array, original_img = load_and_preprocess_image(
        image_path, target_size=target_size, preprocess_func=preprocess_func
    )

    if img_array is None:
        return None, None, None

    # Make the prediction
    predictions = model.predict(img_array)

    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]

    return class_names[predicted_class_idx], confidence, original_img


def predict_batch(model, image_paths, class_names, target_size=(224, 224), model_type='custom'):
    """
    Make predictions for a batch of images

    Args:
        model (tf.keras.Model): Trained model
        image_paths (list): List of image paths
        class_names (list): List of class names
        target_size (tuple): Target image size
        model_type (str): Type of model ('custom', 'resnet50', or 'mobilenet')

    Returns:
        list: List of (image_path, predicted_class, confidence) tuples
    """
    results = []

    for image_path in image_paths:
        predicted_class, confidence, _ = predict_image(
            model, image_path, class_names, target_size, model_type
        )

        if predicted_class is not None:
            results.append((image_path, predicted_class, confidence))

    return results


def visualize_prediction(image_path, predicted_class, confidence, class_names, original_img=None):
    """
    Visualize a prediction

    Args:
        image_path (str): Path to the image
        predicted_class (str): Predicted class name
        confidence (float): Prediction confidence
        class_names (list): List of class names
        original_img (PIL.Image): Original image (if already loaded)
    """
    if original_img is None:
        try:
            original_img = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return

    plt.figure(figsize=(10, 5))

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
    plt.axis('off')

    # Display the prediction probabilities
    plt.subplot(1, 2, 2)
    plt.barh(range(len(class_names)), [0] * len(class_names))
    plt.yticks(range(len(class_names)), class_names)
    plt.xlim(0, 1)
    plt.title('Prediction Confidence')
    plt.xlabel('Confidence')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


def main(args):
    """
    Main function to run predictions

    Args:
        args: Command-line arguments
    """
    # Load the model
    model = load_model(args.model_path)
    if model is None:
        return

    # Load class names
    if args.class_names_file:
        try:
            with open(args.class_names_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading class names: {e}")
            print("Using default class names...")
            class_names = ['satellite', 'debris', 'planet', 'star']
    else:
        class_names = ['satellite', 'debris', 'planet', 'star']

    print(f"Using class names: {class_names}")

    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Get image paths
    if os.path.isdir(args.input_path):
        image_paths = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_paths.extend(glob.glob(os.path.join(args.input_path, f'*.{ext}')))
            image_paths.extend(glob.glob(os.path.join(args.input_path, f'*.{ext.upper()}')))
    else:
        image_paths = [args.input_path]

    print(f"Found {len(image_paths)} images")

    # Process all images in batch mode
    if args.batch_mode:
        results = predict_batch(
            model, image_paths, class_names,
            target_size=(args.img_size, args.img_size),
            model_type=args.model_type
        )

        # Save results to a file
        if args.output_dir:
            output_file = os.path.join(args.output_dir, 'predictions.txt')
            with open(output_file, 'w') as f:
                f.write("Image Path,Predicted Class,Confidence\n")
                for image_path, predicted_class, confidence in results:
                    f.write(f"{image_path},{predicted_class},{confidence:.4f}\n")

            print(f"Batch predictions saved to {output_file}")

        # Print summary
        print("\nPrediction Summary:")
        for image_path, predicted_class, confidence in results:
            print(f"{os.path.basename(image_path)}: {predicted_class} ({confidence:.4f})")

    # Process each image individually
    else:
        for image_path in image_paths:
            predicted_class, confidence, original_img = predict_image(
                model, image_path, class_names,
                target_size=(args.img_size, args.img_size),
                model_type=args.model_type
            )

            if predicted_class is None:
                continue

            print(f"Image: {os.path.basename(image_path)}")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            print("-" * 50)

            # Save visualizations
            if args.output_dir:
                plt.figure(figsize=(10, 5))

                # Display the image
                plt.subplot(1, 2, 1)
                plt.imshow(original_img)
                plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
                plt.axis('off')

                # Display the prediction probabilities
                plt.subplot(1, 2, 2)
                plt.barh(range(len(class_names)), [0] * len(class_names))
                plt.yticks(range(len(class_names)), class_names)
                plt.xlim(0, 1)
                plt.title('Prediction Confidence')
                plt.xlabel('Confidence')
                plt.gca().invert_yaxis()

                plt.tight_layout()

                # Save the visualization
                output_file = os.path.join(
                    args.output_dir,
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_prediction.png"
                )
                plt.savefig(output_file, dpi=300)
                plt.close()

                print(f"Visualization saved to {output_file}")

            # Show the visualization
            if args.visualize:
                visualize_prediction(image_path, predicted_class, confidence, class_names, original_img)


def parse_arguments():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Space Object Classifier Prediction')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input image or directory of images')

    # Optional arguments
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output visualizations')
    parser.add_argument('--class_names_file', type=str, default=None,
                        help='File containing class names (one per line)')
    parser.add_argument('--model_type', type=str, default='custom',
                        choices=['custom', 'resnet50', 'mobilenet'],
                        help='Type of model used for training')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for prediction (width and height)')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process all images in batch mode')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization for each prediction')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)