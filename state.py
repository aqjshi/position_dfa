import numpy as np
import pandas as pd
import math
from collections import Counter
from scipy.stats import gaussian_kde

import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
import os
from visualize import save_plane_as_html

def new_id(current_id: int):
    return (current_id + 1)

def prediction(current_unix_time: float, price_mean: float, price_std: float, time_mean: float, time_std: float) -> Tuple[float, float, float, float]:
    price_dev_1 = np.random.normal(0, price_std)
    time_duration_1 = np.random.normal(loc=time_mean, scale=time_std)
    
    price_dev_2 = np.random.normal(0, price_std * 1.5)
    time_duration_2 = np.random.normal(loc=time_mean * 4, scale=time_std)
    
    entry_time = current_unix_time + max(1, abs(time_duration_1))
    entry_price = price_mean + price_dev_1
    
    exit_time = current_unix_time + max(entry_time - current_unix_time + 1, abs(time_duration_2))
    exit_price = price_mean + price_dev_2
    
    return entry_time, entry_price, exit_time, exit_price

def query(current_time, plane: list):
    if not plane:
        return -1, -1, 0

    df = pd.DataFrame(plane, columns=['id', 'lifespan', 'price', 'time'])
    df = df[df['time'] > current_time]
    df = df[df['lifespan'] >= 1]
    
    population = len(df)

    if len(df) < 5:
        return -1, -1, population

    prices = df['price'].values
    times = df['time'].values
    
    H, price_edges, time_edges = np.histogram2d(prices, times, bins=20)
    
    max_count_index = np.unravel_index(np.argmax(H), H.shape)
    
    price_bin_index, time_bin_index = max_count_index
    
    price_min, price_max = price_edges[price_bin_index], price_edges[price_bin_index + 1]
    time_min, time_max = time_edges[time_bin_index], time_edges[time_bin_index + 1]

    peak_subsamples = df[
        (df['price'] >= price_min) & (df['price'] < price_max) &
        (df['time'] >= time_min) & (df['time'] < time_max)
    ]
    
    if peak_subsamples.empty:
        return -1, -1, population
        
    mean_price_peak = peak_subsamples['price'].mean()
    mean_time_peak = peak_subsamples['time'].mean()

    return mean_time_peak, mean_price_peak, population


def initialize_state(asset='AAPL', initial_value=100.0):
    """Creates and returns the initial state as a dictionary."""
    SECONDS_PER_TRADING_YEAR = 252 * 6.5 * 3600
    time_step_seconds = 1
    
    state = {
        'asset': asset,
        'position_id': 0,
        'current_price': initial_value,
        'current_time': 0.0,
        'entry_plane': [],
        'exit_plane': [],
        
        # GBM Parameters
        'drift_mu': 0.05,
        'volatility_sigma': 0.05,
        'time_step_seconds': time_step_seconds,
        'time_step_dt': time_step_seconds / SECONDS_PER_TRADING_YEAR,
        
        # Prediction Parameters
        'p_mean_off': 0.0,
        'MAX_P_STD': 0.5,
        'MAX_T_STD': 10,
        't_mean': 900,
        
        # Sine Wave Parameters for dynamic prediction density/std
        'CONVERGENCE_RATE': 0.0005,
        'WAVE_FREQUENCY': 0.01,
    }
    return state

def batch_cast_predictions(state: dict):
    """Generates new predictions and adds them to the planes."""
    new_entry_plane = state['entry_plane'].copy()
    new_exit_plane = state['exit_plane'].copy()
    
    new_id_start = len(new_entry_plane) + len(new_exit_plane)
    t = state['current_time']

    # Calculate dynamic parameters based on sine wave and decay
    sine_wave = np.sin(state['WAVE_FREQUENCY'] * t)
    decay_factor = np.exp(-state['CONVERGENCE_RATE'] * t)
    scaling_factor = ((1 + sine_wave) / 2) * decay_factor
    final_scale = max(0.01, scaling_factor)
    
    dynamic_p_std = state['MAX_P_STD'] * final_scale
    dynamic_t_std = state['MAX_T_STD'] * final_scale
    
    # Dynamically adjust number of predictions
    MAX_PREDICTIONS, MIN_PREDICTIONS = 50, 5
    density_multiplier = 1.0 - (final_scale * 0.95)
    dynamic_num_predictions = int(MIN_PREDICTIONS + (MAX_PREDICTIONS - MIN_PREDICTIONS) * density_multiplier)
    
    temp_pred_time = state['current_time'] + 1
    
    for i in range(dynamic_num_predictions):
        et, ep, xt, xp = prediction(
            current_unix_time=temp_pred_time,
            price_mean=state['current_price'] + state['p_mean_off'],
            price_std=dynamic_p_std,
            time_mean=state['t_mean'],
            time_std=dynamic_t_std
        )
        
        new_id_val = new_id_start + i
        entry_lifespan_viz = max(0, int(et - state['current_time']))
        exit_lifespan_viz = max(0, int(xt - state['current_time']))
        
        new_entry_plane.append((new_id_val, entry_lifespan_viz, ep, et))
        new_exit_plane.append((new_id_val, exit_lifespan_viz, xp, xt))
        
        temp_pred_time += 1
        
    return {'entry_plane': new_entry_plane, 'exit_plane': new_exit_plane}

def update_planes(state: dict):
    """Filters out past predictions from planes."""
    current_time = state['current_time']
    
    filtered_entry = [p for p in state['entry_plane'] if p[3] > current_time]
    filtered_exit = [p for p in state['exit_plane'] if p[3] > current_time]
    
    return {'entry_plane': filtered_entry, 'exit_plane': filtered_exit}

def step(state: dict):
    """Performs one time step, updating price and time, and returns the new state."""
    new_state = state.copy()
    
    # 1. Update price using GBM
    Z = np.random.standard_normal(1)[0]
    dt = new_state['time_step_dt']
    price_factor = math.exp(
        (new_state['drift_mu'] - 0.5 * new_state['volatility_sigma']**2) * dt +
        new_state['volatility_sigma'] * math.sqrt(dt) * Z
    )
    new_state['current_price'] *= price_factor
    
    # 2. Advance time
    new_state['current_time'] += new_state['time_step_seconds']
    
    # 3. Filter planes based on the *new* current time
    updated_planes = update_planes(new_state)
    new_state.update(updated_planes)
    
    return new_state

def save_snapshot(state: dict, timestep: int):
    """Saves the current state of the prediction planes as HTML files."""
    print(f"\n--- SNAPSHOT t={timestep} (Price: {state['current_price']:.2f}) ---")
    
    entry_time, entry_price, entry_population = query(state['current_time'], state['entry_plane'])
    exit_time, exit_price, exit_population = query(state['current_time'], state['exit_plane'])

    print(f"Entry Query Peak: Time={entry_time:.0f}, Price={entry_price:.4f}, Population={entry_population}")
    print(f"Exit Query Peak: Time={exit_time:.0f}, Price={exit_price:.4f}, Population={exit_population}")

    entry_path = save_plane_as_html(state['entry_plane'], state['current_time'], state['asset'], 'entry')
    print(f"Saved entry plane: {entry_path}")

    exit_path = save_plane_as_html(state['exit_plane'], state['current_time'], state['asset'], 'exit')
    print(f"Saved exit plane: {exit_path}")


def valid(): 
    return True
def fill():
    return True
def update_TSM(): 
    return True
def trigger_TSM(): 
    return True
    
transition_dict = {
    0: {"name": "a", "fn": new_id},
    1: {"name": "b", "fn": prediction},
    2: {"name": "c", "fn": query},
    3: {"name": "d", "fn": valid},
    4: {"name": "e", "fn": fill},
    5: {"name": "f", "fn": update_TSM},
    6: {"name": "g", "fn": trigger_TSM},
}

states_dict = {
    0: {"name": "idle"}, 1: {"name": "price_stream_1"}, 2: {"name": "prediction_plane_1"},
    3: {"name": "top_peak_1"}, 4: {"name": "declare_TSM_1"}, 5: {"name": "active"},
    6: {"name": "price_stream_2"}, 7: {"name": "prediction_plane_2"}, 8: {"name": "top_peak_2"},
    9: {"name": "declare_TSM_2"}, 10: {"name": "exit"},
}

transition_matrix = [
    [1,0,0,0,0,0,0],
    [1,2,1,1,1,1,1],    
    [1,1,3,1,1,1,1],    
    [1,1,1,4,1,1,1],    
    [1,1,1,1,5,1,1],    
    [6,6,6,6,6,6,10],    
    [6,7,6,6,6,6,10],    
    [6,6,8,6,6,6,10],    
    [6,6,6,9,6,6,10],    
    [6,6,6,6,6,5,10],    
    [6,6,6,6,6,5,10],    
    [1,10,10,10,10,10,10],    

]



if __name__ == "__main__":
    os.makedirs("snapshot", exist_ok=True)

    # --- Simulation Parameters ---
    TIME_DURATION = 3600
    PREDICTION_CAST_FREQ = 30
    SNAPSHOT_SAVE_FREQ = 300
    
    # 1. Initialize State using the new function
    state = initialize_state(asset='AAPL', initial_value=150.0)

    print(f"Starting simulation for {state['asset']}. Duration: {TIME_DURATION} seconds.")
    
    for time_step in range(TIME_DURATION + 1):
        # Cast new predictions at a set frequency
        if time_step % PREDICTION_CAST_FREQ == 0:
            updated_planes = batch_cast_predictions(state)
            state.update(updated_planes) # Merge the updated planes back into the main state
        
        # Save a snapshot of the planes for visualization
        if time_step > 0 and time_step % SNAPSHOT_SAVE_FREQ == 0:
            save_snapshot(state, time_step)
        
        # Advance the simulation by one step
        state = step(state)

    print("\nSimulation complete. Check the 'snapshot' folder for HTML files.")