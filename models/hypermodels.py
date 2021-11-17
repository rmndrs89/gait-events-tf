from kerastuner import HyperModel
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tcn import TCN
from tensorflow.keras.optimizers import Adam
from .losses import MyWeightedBinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

class BaseTCNHyperModel(HyperModel):
    def __init__(self, input_shape, n_classes, weights=None):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.weights = weights
    
    def build(self, hp):

        # Define the layers
        input_layer = Input(shape=(None, self.input_shape[-1]), name='input_layer')
        tcn_layer = TCN(
            nb_filters=16,
            kernel_size=3,
            nb_stacks=1,
            dilations=(1, 2, 4, 8, 16, 32),
            padding='causal', 
            return_sequences=True,
            use_skip_connections=True,
            use_batch_norm=True
        )(input_layer)

        # Define an output for each class
        output_layer = []
        for i in range(self.n_classes):
            output_layer.append(Dense(units=1, activation='sigmoid', name=f'output_layer_{i+1}')(tcn_layer))
        
        # Instantiate the model
        model = Model(inputs=input_layer, outputs=output_layer, name='tcn_model')

        # Define losses and metrics
        losses, metrics = {}, {}
        for i in range(self.n_classes):
            if (self.weights == None) or (len(self.weights)==0):
                losses[f'output_layer_{i+1}'] = MyWeightedBinaryCrossentropy(weights=0.01)
            else:
                losses[f'output_layer_{i+1}'] = MyWeightedBinaryCrossentropy(weights=self.weights[f'output_layer_{i+1}'])
            metrics[f'output_layer_{i+1}'] = BinaryAccuracy()

        # Compile the model
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-3)),
            loss=losses,
            metrics=metrics
        )
        return model

class TCNHyperModel(HyperModel):
    def __init__(self, input_shape, n_classes, weights=None):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.weights = weights
    
    def build(self, hp):

        # Define the layers
        input_layer = Input(shape=(None, self.input_shape[-1]), name='input_layer')
        tcn_layer = TCN(
            hp.Int('nb_filters', min_value=16, max_value=512, step=16, sampling='linear'),
            hp.Int('kernel_size', min_value=3, max_value=7, step=2, sampling='linear'),
            nb_stacks=1,
            dilations=(1, 2, 4, 8, 16, 32),
            padding='causal', 
            use_skip_connections=True,
            use_batch_norm=True
        )(input_layer)

        # Define an output for each class
        output_layer = []
        for i in range(self.n_classes):
            output_layer.append(Dense(units=1, activation='sigmoid', name=f'output_layer_{i+1}')(tcn_layer))
        
        # Instantiate the model
        model = Model(inputs=input_layer, outputs=output_layer, name='tcn_model')

        # Define losses and metrics
        losses, metrics = {}, {}
        for i in range(self.n_classes):
            if (self.weights == None) or (len(self.weights)==0):
                losses[f'output_layer_{i+1}'] = MyWeightedBinaryCrossentropy(weights=0.01)
            else:
                losses[f'output_layer_{i+1}'] = MyWeightedBinaryCrossentropy(weights=self.weights[f'output_layer_{i+1}'])
            metrics[f'output_layer_{i+1}'] = BinaryAccuracy()

        # Compile the model
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-3)),
            loss=losses,
            metrics=metrics
        )
        return model