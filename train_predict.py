import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import config
import os
import matplotlib.pyplot as plt  # Added for inline plotting

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)  # Added plots directory creation

def solve_umo_system():
    """Numerically solve the UMO system"""
    print("Solving UMO system numerically...")
    def umo(t, state, a, b, c, d):
        x, y, z, w = state
        return [-y-z, x-w, -a*np.sin(y)-b*z, c*y-d*w]
    
    sol = solve_ivp(
        umo, 
        config.UMO_PARAMS['t_span'], 
        config.UMO_PARAMS['initial_state'],
        args=(config.UMO_PARAMS['a'], config.UMO_PARAMS['b'], 
              config.UMO_PARAMS['c'], config.UMO_PARAMS['d']),
        t_eval=np.linspace(*config.UMO_PARAMS['t_span'], config.UMO_PARAMS['num_points']),
        method='RK45'
    )
    np.save('data/umo_solution.npy', sol.y.T)
    print(f"Numerical solution saved to data/umo_solution.npy (shape: {sol.y.T.shape})")
    return sol.y.T

def create_dataset(data, window_size):
    """Create time-series dataset for LSTM"""
    X, y = [], []
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def train_lstm():
    """Train and save LSTM model"""
    print("\nPreparing LSTM training data...")
    data = np.load('data/umo_solution.npy')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = create_dataset(data_scaled, config.LSTM_CONFIG['window_size'])
    split = int(config.LSTM_CONFIG['train_test_split'] * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
    
    model = Sequential([
        LSTM(config.LSTM_CONFIG['lstm_units'][0], 
             activation='tanh', 
             return_sequences=True, 
             input_shape=(config.LSTM_CONFIG['window_size'], 4)),
        LSTM(config.LSTM_CONFIG['lstm_units'][1], activation='tanh'),
        Dense(4)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    print("Model architecture:")
    model.summary()
    
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=config.LSTM_CONFIG['epochs'],
        batch_size=config.LSTM_CONFIG['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=5)],
        verbose=1  # Ensure training progress is shown
    )
    
    model.save('models/lstm_model.h5')
    print(f"\nModel saved to models/lstm_model.h5")
    
    # Plot training history
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('plots/training_history.png')
    plt.close()
    print("Training history plot saved to plots/training_history.png")
    
    return model, history

def make_predictions(model):
    """Generate and save predictions"""
    print("\nGenerating predictions...")
    data = np.load('data/umo_solution.npy')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data)
    
    # Get the first window of scaled data
    initial_window = data[:config.LSTM_CONFIG['window_size']]
    current_window = scaler.transform(initial_window)
    
    predictions = []
    for i in range(len(data) - config.LSTM_CONFIG['window_size']):
        if i % 1000 == 0:
            print(f"Prediction progress: {i}/{len(data)-config.LSTM_CONFIG['window_size']}")
        
        # Reshape to (1, window_size, 4)
        prediction_input = current_window.reshape(1, config.LSTM_CONFIG['window_size'], 4)
        pred = model.predict(prediction_input, verbose=0)[0]
        predictions.append(pred)
        
        # Update window: remove first element, add new prediction
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1] = pred
    
    predictions = scaler.inverse_transform(np.array(predictions))
    np.save('data/predictions.npy', predictions)
    print(f"Predictions saved to data/predictions.npy")
    return predictions

if __name__ == "__main__":
    # Suppress TensorFlow info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Run pipeline
    solve_umo_system()
    model, history = train_lstm()
    predictions = make_predictions(model)
    
    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"\nTraining completed!")
    print(f"Final training loss: {final_train_loss:.6f}")
    print(f"Final validation loss: {final_val_loss:.6f}")
    
    # Automatically generate visualizations
    import visualize
    visualize.plot_results()
    print("\nVisualizations saved to plots/ directory")