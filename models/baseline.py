from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tcn import TCN
from tensorflow.keras.optimizers import Adam
from .losses import MyWeightedBinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

def make_model(input_shape, n_classes, weights=None):

    # Define the layers
    input_layer = Input(shape=(None, input_shape[-1]), name='input_layer')
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
    for i in range(n_classes):
        output_layer.append(Dense(units=1, activation='sigmoid', name=f'output_layer_{i+1}')(tcn_layer))
    
    # Instantiate the model
    model = Model(inputs=input_layer, outputs=output_layer, name='tcn_model')

    # Define losses and metrics
    losses, metrics = {}, {}
    for i in range(n_classes):
        if (weights == None) or (len(weights)==0):
            losses[f'output_layer_{i+1}'] = MyWeightedBinaryCrossentropy(weights=0.01)
        else:
            losses[f'output_layer_{i+1}'] = MyWeightedBinaryCrossentropy(weights=weights[f'output_layer_{i+1}'])
        metrics[f'output_layer_{i+1}'] = BinaryAccuracy()
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=losses,
        metrics=metrics
    )
    return model