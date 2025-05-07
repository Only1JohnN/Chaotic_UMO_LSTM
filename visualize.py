import numpy as np
import matplotlib.pyplot as plt
import config

def plot_results():
    """Generate all required plots"""
    data = np.load('data/umo_solution.npy')
    predictions = np.load('data/predictions.npy')
    t = np.linspace(*config.UMO_PARAMS['t_span'], config.UMO_PARAMS['num_points'])
    
    # Time series plots
    plt.figure(figsize=(12, 8))
    for i, var in enumerate(['x', 'y', 'z', 'w']):
        plt.subplot(2, 2, i+1)
        plt.plot(t, data[:, i], 'b', label='Actual')
        plt.plot(t[config.LSTM_CONFIG['window_size']:], predictions[:, i], 'r--', alpha=0.7, label='Predicted')
        plt.title(f'{var} - Actual vs Predicted')
        plt.xlabel('Time')
        plt.legend()
    plt.tight_layout()
    plt.savefig('plots/time_series.png')
    plt.close()
    
    # Error plots
    errors = data[config.LSTM_CONFIG['window_size']:] - predictions
    error_magnitude = np.sqrt(np.sum(errors**2, axis=1))
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for i, var in enumerate(['x', 'y', 'z', 'w']):
        plt.plot(t[config.LSTM_CONFIG['window_size']:], errors[:, i], label=f'e_{var}')
    plt.title('Prediction Errors')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.semilogy(t[config.LSTM_CONFIG['window_size']:], error_magnitude, 'k')
    plt.title('Error Magnitude (Log Scale)')
    plt.xlabel('Time')
    plt.ylabel('||Error||')
    plt.tight_layout()
    plt.savefig('plots/prediction_errors.png')
    plt.close()

if __name__ == "__main__":
    plot_results()