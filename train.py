#!/venv/bin/python
import h5py
import pandas as pd
import numpy as np
from scipy.fftpack import dct, idct

# Hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 10

# Load the data
def load_file(path: str) -> pd.DataFrame:
    with h5py.File(path, 'r') as hf:
        data_dict = {'activity': [], 'data': []}

        # Iterate over datasets
        for dataset_name in hf.keys():
            dataset = hf[dataset_name]

            # Extract label metadata
            label = dataset.attrs.get('activity', 'No Label')

            # Append dataset name and label to data dictionary
            data_dict['activity'].append(label)
            data_dict['data'].append(dataset[:])  # Extracting data from dataset

    # Create DataFrame from data dictionary
    df = pd.DataFrame(data_dict)

    df['data'] = df['data'].apply(lambda x: np.nan_to_num(x))

    return df

# Function to apply DCT on each axis
def apply_dct(data):
    # Convert signal to a NumPy array (if not already)
    signal_array = np.array(data)
    # Apply DCT along each axis (0 for columns, 1 for rows)
    dct_result = dct(signal_array, axis=0, norm='ortho')  # Apply DCT to columns (X, Y, Z)
    return dct_result

def visualize_data(df):
    # visualize the original and reconstructed data using matplotlib
    import matplotlib.pyplot as plt

    index = df['data'].apply(lambda x: x.shape[0]).idxmax()
    signal = df['data'].iloc[index]

    reconstructed_signal = idct(df['dct_data'].iloc[index], axis=0, norm='ortho')

    t = np.linspace(0, 1, signal.shape[0], endpoint=False)  # 512 samples
    # plot with 3 subplots
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.title('x-axis')
    plt.plot(t, signal[:, 0], label='Original Signal')
    plt.plot(t, reconstructed_signal[:, 0], label='Reconstructed Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title('y-axis')
    plt.plot(t, signal[:, 1], label='Original Signal')
    plt.plot(t, reconstructed_signal[:, 1], label='Reconstructed Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title('z-axis')
    plt.plot(t, signal[:, 2], label='Original Signal')
    plt.plot(t, reconstructed_signal[:, 2], label='Reconstructed Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load the data
    df = load_file('data/acce_data_xyz.h5')

    # Apply DCT to each row of the data
    df['dct_data'] = df['data'].apply(apply_dct)
    
    visualize_data(df)

    



    

