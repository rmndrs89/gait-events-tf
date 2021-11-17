import os
import numpy as np

INPUT_DATA_DIR = "Z:\\Keep Control\\Data\\lab dataset\\processed"

WIN_LEN  = 300  # corresponds to 1.5 seconds
STEP_LEN = 50   # corresponds to .25 secdonds

def create_sequences(values, win_len, step_len):
    """Creates sequences of equal length for batch input to the Keras model.

    Parameters
    ----------
    values : (N, D) array_like
        The input data with N time steps across D channels.
    win_len : int
        The window length, or length of the sequence.
    step_len : int
        The step length, or number of samples that the windows slides forward.

    Returns
    -------
    output : (B, win_len, D) array_like
        The output data with B batches of data, each with shape (win_len, D).
    """
    output = []
    for i in range(0, values.shape[0]-win_len+1, step_len):
        output.append(values[i:(i+win_len),:])
    return np.stack(output)

def load_data():
    """Loads data from the Keep Control validation study.

    Returns
    -------
    (x_train, y_train) : (n_instances, n_time_steps, n_channels), (n_instances, n_time_steps, n_classes)
        A numpy array with training data and corresponding labels.
    (x_val, y_val) : (n_instances, n_time_steps, n_channels), (n_instances, n_time_steps, n_classes)
        A numpy array with validation data and corresponding labels.
    """

    train_dir = os.path.join(INPUT_DATA_DIR, 'train')
    val_dir = os.path.join(INPUT_DATA_DIR, 'val')
    test_dir = os.path.join(INPUT_DATA_DIR, 'test')

    train_data, train_labels = [], []

    for train_file in [fname for fname in os.listdir(train_dir) if fname.endswith('.npy')]:
        
        # Load data from current file
        data = np.load(os.path.join(train_dir, train_file))

        # Create sequences of equal length
        sequences = create_sequences(data, WIN_LEN, STEP_LEN)

        # Normalize
        mn = np.mean(sequences[:,:,:-4], axis=1)  # shape: (n_instances, n_channels), use `np.expand_dims` -> (n_instances, 1, n_channels)
        sd = np.std(sequences[:,:,:-4], axis=1)   # shape: (n_instances, n_channels)
        sequences[:,:,:-4] = ( sequences[:,:,:-4] - np.expand_dims(mn, axis=1) ) / np.expand_dims(sd, axis=1)

        # Split data and labels, treat data from left and right as independent (training) instances
        train_data = np.concatenate((train_data, np.concatenate((sequences[:,:,:6], sequences[:,:,6:12]), axis=0)), axis=0) if len(train_data)>0 else np.concatenate((sequences[:,:,:6], sequences[:,:,6:12]), axis=0)
        train_labels = np.concatenate((train_labels, np.concatenate((sequences[:,:,12:14], sequences[:,:,14:]), axis=0)), axis=0) if len(train_labels)>0 else np.concatenate((sequences[:,:,12:14], sequences[:,:,14:]), axis=0)
    
    val_data, val_labels = [], []

    for val_file in [fname for fname in os.listdir(val_dir) if fname.endswith('.npy')]:

         # Load data from current file
        data = np.load(os.path.join(val_dir, val_file))

        # Create sequences of equal length
        sequences = create_sequences(data, WIN_LEN, STEP_LEN)

        # Normalize
        mn = np.mean(sequences[:,:,:-4], axis=1)  # shape: (n_instances, n_channels), use `np.expand_dims` -> (n_instances, 1, n_channels)
        sd = np.std(sequences[:,:,:-4], axis=1)   # shape: (n_instances, n_channels)
        sequences[:,:,:-4] = ( sequences[:,:,:-4] - np.expand_dims(mn, axis=1) ) / np.expand_dims(sd, axis=1)

        # Split data and labels, treat data from left and right as independent (training) instances
        val_data = np.concatenate((val_data, np.concatenate((sequences[:,:,:6], sequences[:,:,6:12]), axis=0)), axis=0) if len(val_data)>0 else np.concatenate((sequences[:,:,:6], sequences[:,:,6:12]), axis=0)
        val_labels = np.concatenate((val_labels, np.concatenate((sequences[:,:,12:14], sequences[:,:,14:]), axis=0)), axis=0) if len(val_labels)>0 else np.concatenate((sequences[:,:,12:14], sequences[:,:,14:]), axis=0)
    
    test_data, test_labels = [], []


    return (train_data, train_labels), (val_data, val_labels)