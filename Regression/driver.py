import msgParser
import carControl
import carState
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class Driver(object):

    def __init__(self, stage, track):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        self.control = carControl.CarControl()
        self.state = carState.CarState()
        
        # Try to load the neural network model first
        try:
            if os.path.exists('model.keras'):
                print("Loading Keras model...")
                self.model = load_model('model.keras')
                self.model_type = 'keras'
                print("Neural network model loaded successfully")
            else:
                # Fall back to scikit-learn model if available
                print("No Keras model found, trying to load scikit-learn model...")
                self.scaler = joblib.load('scaler.pkl')
                self.pca = joblib.load('pca.pkl')
                self.model = joblib.load('model.pkl')
                self.model_type = 'sklearn'
                print("scikit-learn model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load any existing model: {e}")
            print("Creating default model...")
            # Create default model if no trained model is found
            self.model = LinearRegression()
            self.model_type = 'default'
            print("Default linear model created")

        self.forward = False
        self.first = True
        self.io = 0
        
        # Store track information
        self.stage = stage
        self.track = track
        print(f"Driver initialized for track: {track}, stage: {stage}")

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]

        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5

        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)

        # Parse message into a dictionary of features
        parsemessage = self.parser.parse2(msg)

        # Remove unnecessary features
        features_to_remove = [f'opponents{i}' for i in range(36)] + \
                              ['focus0', 'focus1', 'focus2', 'focus3', 'focus4', 
                               'lastLapTime', 'racePos', 'damage', 'fuel']
        
        for feature in features_to_remove:
            if feature in parsemessage:
                parsemessage.pop(feature)

        # Convert to numpy array
        data = list(parsemessage.values())
        array = np.array(data)
        array = array.reshape(1, -1)

        # Only make predictions every few steps to reduce computational load
        if self.io == 5:
            try:
                if self.model_type == 'keras':
                    # Direct prediction with neural network
                    output = self.model.predict(array, verbose=0)
                elif self.model_type == 'sklearn':
                    # Preprocess with scaler and PCA before prediction
                    scaled_data = self.scaler.transform(array)
                    reduced_data = self.pca.transform(scaled_data)
                    output = self.model.predict(reduced_data)
                else:
                    # Default model with simple prediction
                    output = [[0.5, 0.0]]  # Default acceleration and steering
                
                print("Prediction output:", output)
                
                # Set control values
                self.control.setAccel(max(0, min(output[0][0], 1)))
                self.control.setSteer(max(-1, min(output[0][1], 1)))
                
                self.io = 0
            except Exception as e:
                print(f"Prediction error: {e}")
                # Fallback to default values
                self.control.setAccel(0.5)
                self.control.setSteer(0)

        self.io += 1

        # Handle gear shifting
        self.gear()

        return self.control.toMsg()

    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()

        # Upshift if RPM is too high
        if rpm > 7000:
            gear += 1

        # Downshift if RPM is too low and we're moving forward
        if rpm < 4000 and self.forward == False and gear > 1:
            gear -= 1

        self.control.setGear(gear)

    def onShutDown(self):
        """Called when the client is shutting down"""
        print("Driver shutting down")
        # Save any data or perform cleanup if needed
        pass

    def onRestart(self):
        """Called when the client is restarting"""
        print("Driver restarting")
        self.forward = False
        self.first = True
        self.io = 0