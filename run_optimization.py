from numpy.lib.npyio import load
from utils import load_data
from models.hypermodels import BaseTCNHyperModel, TCNHyperModel
from models.baseline import make_model
import numpy as np
import kerastuner as kt
import matplotlib.pyplot as plt

NUM_CLASSES = 2

def run_optimization():

    # Get data
    (x_train, y_train_tmp), (x_val, y_val_tmp) = load_data()

    # Reorganize labels as a dictionary
    # For each class, we expect a numpy array of shape: (n_instances, n_time_steps, n_channels)
    y_train, y_val = {}, {}
    for i in range(NUM_CLASSES):
        y_train[f'output_layer_{i+1}'] = np.expand_dims(y_train_tmp[:,:,i], axis=-1)
        y_val[f'output_layer_{i+1}'] = np.expand_dims(y_val_tmp[:,:,i], axis=-1)
    
    # Get hypermodel
    hypermodel = TCNHyperModel(input_shape=x_train.shape[1:], n_classes=NUM_CLASSES)

    # Define a tuner
    tuner = kt.RandomSearch(
        hypermodel,
        objective="val_loss",
        max_trials=3,
        executions_per_trial=2,
        overwrite=False,
        directory=".\\optimization",
        project_name='optim',
    )

    # Start search
    tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

    # Query the results
    hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
    best_model = tuner.hypermodel.build(hps)
    best_model.summary()
    return best_model

def run_baseline_optimization():

    # Get data
    (x_train, y_train_tmp), (x_val, y_val_tmp) = load_data()

    # Reorganize labels as a dictionary
    # For each class, we expect a numpy array of shape: (n_instances, n_time_steps, n_channels)
    y_train, y_val = {}, {}
    for i in range(NUM_CLASSES):
        y_train[f'output_layer_{i+1}'] = np.expand_dims(y_train_tmp[:,:,i], axis=-1)
        y_val[f'output_layer_{i+1}'] = np.expand_dims(y_val_tmp[:,:,i], axis=-1)
    
    # Get hypermodel
    hypermodel = BaseTCNHyperModel(input_shape=x_train.shape[1:], n_classes=NUM_CLASSES)

    # Set output dir
    output_dir = ".\\models\\test_tune"

    # Define a tuner
    tuner = kt.RandomSearch(
        hypermodel,
        objective="val_loss",
        max_trials=3,
        executions_per_trial=2,
        overwrite=False,
        directory=output_dir,
        project_name='baseline_optim',
    )
    
    # Print search space summary
    tuner.search_space_summary()

    # Start search
    tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

    # Query the results
    hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
    # model = build_model(hps)
    # model.fit(...)
    best_model = tuner.hypermodel.build(hps)
    best_model.summary()

def run_baseline():

    # Get data
    (x_train, y_train_tmp), (x_val, y_val_tmp) = load_data()

    # Reorganize labels as a dictionary
    # For each class, we expect a numpy array of shape: (n_instances, n_time_steps, n_channels)
    y_train, y_val = {}, {}
    for i in range(NUM_CLASSES):
        y_train[f'output_layer_{i+1}'] = np.expand_dims(y_train_tmp[:,:,i], axis=-1)
        y_val[f'output_layer_{i+1}'] = np.expand_dims(y_val_tmp[:,:,i], axis=-1)

    # Make model
    model = make_model(input_shape=x_train.shape[1:], n_classes=NUM_CLASSES)
    
    # Train the model
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=2,
        batch_size=32,
        validation_data=(x_val, y_val)
    )

    # Plot the training loss
    fig, ax = plt.subplots(1, 1)
    ax.plot(history.history['loss'], label='training')
    ax.plot(history.history['val_loss'], label='validation')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.show()

if __name__ == "__main__":
    # run_optimization()
    run_optimization()