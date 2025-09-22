

import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        """Initialize ResNet50 model for feature extraction"""
        print("Loading ResNet50 model...")
        # Load pre-trained ResNet50 without top classification layer
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("ResNet50 model loaded successfully!")
    
    def extract_features_from_image(self, image_path):
        """
        Extract features from a single image using ResNet50
        Args:
            image_path: Path to image file
        Returns:
            Feature vector (numpy array)
        """
        try:
            # Load and preprocess image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = self.model.predict(img_array, verbose=0)
            return features.flatten()
        
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return None
    
    def extract_features_from_frame_array(self, frame_array):
        """
        Extract features from a numpy frame array
        Args:
            frame_array: Numpy array of shape (224, 224, 3)
        Returns:
            Feature vector (numpy array)
        """
        try:
            # Ensure correct shape and preprocessing
            if len(frame_array.shape) == 3:
                frame_array = np.expand_dims(frame_array, axis=0)
            
            frame_array = preprocess_input(frame_array.astype(np.float32))
            
            # Extract features
            features = self.model.predict(frame_array, verbose=0)
            return features.flatten()
        
        except Exception as e:
            print(f"Error extracting features from frame array: {str(e)}")
            return None
    
    def extract_features_from_frames(self, frame_paths):
        """
        Extract features from multiple frames
        Args:
            frame_paths: List of frame file paths
        Returns:
            Dictionary mapping frame paths to feature vectors
        """
        features_dict = {}
        total_frames = len(frame_paths)
        
        print(f"Extracting features from {total_frames} frames...")
        
        for i, frame_path in enumerate(frame_paths):
            if i % 50 == 0:
                print(f"  Processing frame {i+1}/{total_frames}")
            
            features = self.extract_features_from_image(frame_path)
            if features is not None:
                features_dict[frame_path] = features
        
        print(f"Feature extraction completed. Processed {len(features_dict)} frames.")
        return features_dict
    
    def extract_features_from_query_image(self, query_image_path):
        """
        Extract features from query image
        Args:
            query_image_path: Path to query image
        Returns:
            Feature vector
        """
        print(f"Extracting features from query image: {query_image_path}")
        return self.extract_features_from_image(query_image_path)


class LSTMVideoSummarizer:
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=1):
        """
        Initialize LSTM model for video summarization
        Args:
            input_dim: Dimension of input features (ResNet50 output)
            hidden_dim: Hidden dimension of LSTM
            output_dim: Output dimension (1 for importance score)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build the LSTM model architecture"""
        print("Building LSTM model...")
        
        # Input layer
        inputs = Input(shape=(None, self.input_dim), name='video_features')
        
        # Bidirectional LSTM layers
        lstm_out = Bidirectional(LSTM(self.hidden_dim, return_sequences=True, dropout=0.3))(inputs)
        lstm_out = Dropout(0.5)(lstm_out)
        
        # Second LSTM layer
        lstm_out = Bidirectional(LSTM(self.hidden_dim//2, return_sequences=True, dropout=0.3))(lstm_out)
        lstm_out = Dropout(0.5)(lstm_out)
        
        # Output layer
        outputs = Dense(self.output_dim, activation='sigmoid', name='importance_scores')(lstm_out)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        print("Model architecture:")
        self.model.summary()
    
    def prepare_training_sequences(self, training_data, max_sequence_length=None):
        """
        Prepare sequences for training
        Args:
            training_data: List of training samples
            max_sequence_length: Maximum sequence length (None for no limit)
        Returns:
            X, y: Training features and labels
        """
        X, y = [], []
        
        for sample in training_data:
            features = sample['features']
            gt_scores = sample['gt_scores']
            
            # Ensure same length
            min_len = min(len(features), len(gt_scores))
            features = features[:min_len]
            gt_scores = gt_scores[:min_len]
            
            # Truncate if too long
            if max_sequence_length and min_len > max_sequence_length:
                features = features[:max_sequence_length]
                gt_scores = gt_scores[:max_sequence_length]
            
            X.append(features)
            y.append(gt_scores.reshape(-1, 1))  # Reshape for model
        
        return X, y
    
    def pad_sequences(self, sequences, max_length=None):
        """Pad sequences to same length"""
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                # Pad with zeros
                pad_length = max_length - len(seq)
                if len(seq.shape) == 2:
                    padding = np.zeros((pad_length, seq.shape[1]))
                else:
                    padding = np.zeros((pad_length, 1))
                seq = np.vstack([seq, padding])
            padded.append(seq)
        
        return np.array(padded)
    
    def train(self, training_data, validation_split=0.2, epochs=50, batch_size=8):
        """
        Train the LSTM model
        Args:
            training_data: Training dataset
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Batch size
        """
        print("Preparing training data...")
        
        # Prepare sequences
        X, y = self.prepare_training_sequences(training_data, max_sequence_length=500)
        
        # Pad sequences
        X_padded = self.pad_sequences(X)
        y_padded = self.pad_sequences(y)
        
        print(f"Training data shape: {X_padded.shape}")
        print(f"Training labels shape: {y_padded.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_padded, y_padded, test_size=validation_split, random_state=42
        )
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_lstm_model.h5', save_best_only=True, monitor='val_loss'
            )
        ]
        
        # Train model
        print("Starting training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def predict_importance_scores(self, features):
        """
        Predict importance scores for video features
        Args:
            features: Video features array
        Returns:
            Importance scores
        """
        # Ensure correct shape
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        predictions = self.model.predict(features)
        return predictions.squeeze()
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
