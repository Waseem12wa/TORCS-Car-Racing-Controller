#!/usr/bin/env python
'''
Neural Network training script for TORCS driver
This script trains a neural network model that will be used by the driver.py module
to control a car in the TORCS racing simulator.
'''

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import argparse

# Configure the argument parser
parser = argparse.ArgumentParser(description='Train a neural network for TORCS racing.')
parser.add_argument('--data', action='store', dest='data_file', default='datareducednew.csv',
                    help='Data file to train on (default: datareducednew.csv)')
parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=50,
                    help='Number of training epochs (default: 50)')
parser.add_argument('--batch', action='store', type=int, dest='batch_size', default=32,
                    help='Batch size for training (default: 32)')
parser.add_argument('--test', action='store', type=float, dest='test_size', default=0.3,
                    help='Proportion of data to use for testing (default: 0.3)')
parser.add_argument('--viz', action='store_true', dest='visualize', default=False,
                    help='Visualize training history')
parser.add_argument('--save-sklearn', action='store_true', dest='save_sklearn', default=False,
                    help='Also save scikit-learn models (default: False)')

def load_data(file_path):
    """Load and preprocess the training data"""
    print(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
        data = data.dropna()
        print(f"Data loaded successfully. Shape: {data.shape}")
        
        # Extract features and targets
        X = data.iloc[:, :-2].values  # All columns except the last two
        y = data.iloc[:, -2:].values  # Last two columns (acceleration and steering)
        
        print(f"Features shape: {X.shape}, Targets shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def create_model(input_dim, output_dim=2):
    """Create the neural network model architecture"""
    print("Creating neural network model...")
    model = Sequential([
        # Input layer
        Dense(64, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        
        # Hidden layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        
        # Output layer (linear activation for regression)
        Dense(output_dim, activation='linear')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    print("Model created and compiled")
    print(model.summary())
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, visualize=False):
    """Train the neural network model"""
    print("Training model...")
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
        ModelCheckpoint('model_checkpoint.keras', monitor='val_loss', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks
    )
    
    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {score[0]}")
    print(f"Test MAE: {score[1]}")
    
    # Visualize training history
    if visualize:
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation mean absolute error
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    return model, history

def save_sklearn_models(X_train, X_test):
    """Create and save scikit-learn models as fallback"""
    print("Creating scikit-learn models as fallback...")
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")
    
    # Create and fit a simple PCA model (for dimensionality reduction)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(10, X_train.shape[1]))
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Save the PCA model
    joblib.dump(pca, 'pca.pkl')
    print("PCA model saved as 'pca.pkl'")
    
    # Create and fit a simple linear model as fallback
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    linear_model.fit(X_train_pca, y_train)
    
    # Save the linear model
    joblib.dump(linear_model, 'model.pkl')
    print("Linear model saved as 'model.pkl'")
    
    # Evaluate the linear model
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    y_pred = linear_model.predict(X_test_pca)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Linear model MSE: {mse}")
    print(f"Linear model MAE: {mae}")

def main():
    """Main function to train and save the model"""
    args = parser.parse_args()
    
    # Load and preprocess data
    X, y = load_data(args.data_file)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    # Create and train the model
    model = create_model(input_dim=X_train.shape[1])
    model, history = train_model(
        model, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        visualize=args.visualize
    )
    
    # Save the model
    print("Saving model as 'model.keras'...")
    model.save('model.keras')
    print("Model saved successfully!")
    
    # Create scikit-learn models as fallback if requested
    if args.save_sklearn:
        save_sklearn_models(X_train, X_test)
    
    print("Training completed!")
    print("\nTo use this model with TORCS:")
    print("1. Make sure 'model.keras' is in the same directory as driver.py")
    print("2. Run pyclient.py to connect to the TORCS server")
    print("3. The driver will automatically use this neural network model")

if __name__ == "__main__":
    # Enable memory growth for TensorFlow (helps prevent OOM errors)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    main()