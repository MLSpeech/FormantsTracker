import os
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_formants(pred_file, pred_dir):
    # Read prediction file
    times, f1, f2, f3 = np.loadtxt(os.path.join(pred_dir, pred_file), unpack=True)

    # Plot formants
    plt.figure(figsize=(10, 6))
    plt.plot(times, f1, label='F1', color='r')
    plt.plot(times, f2, label='F2', color='g')
    plt.plot(times, f3, label='F3', color='b')
    plt.legend()
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title(f'Formants for {pred_file}')

    # Save plot as PNG file in the predictions directory
    png_filename = os.path.join(pred_dir, pred_file.replace('.pred', '.png'))
    plt.savefig(png_filename)
    plt.close()

# Directory
pred_dir = 'predictions'

# Plot and save for each prediction file
for pred_file in os.listdir(pred_dir):
    if pred_file.endswith('.pred'):
        plot_and_save_formants(pred_file, pred_dir)
