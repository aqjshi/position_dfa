import numpy as np
import pandas as pd
import math
from collections import Counter
from scipy.stats import gaussian_kde

import plotly.graph_objects as go
import plotly.express as px
import os 
from typing import List, Tuple
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import plotly.graph_objects as go
os.makedirs("snapshot", exist_ok=True)


def new_id(current_id: int):
    return (current_id + 1)


def prediction(current_unix_time: float, price_mean: float, price_std: float, time_mean: float, time_std: float) -> Tuple[float, float, float, float]:
    """
    Mocks a deep learning model prediction using GBM-like logic.
    time_mean is treated as the expected DURATION for the event to occur.
    """
    # 1. Generate Leg 1 (Entry) Price and Time DURATION
    price_dev_1 = np.random.normal(0, price_std)
    time_duration_1 = np.random.normal(loc=time_mean, scale=time_std) 
    
    # 2. Generate Leg 2 (Exit) Price and Time DURATION (More volatile, longer duration)
    price_dev_2 = np.random.normal(0, price_std * 1.5) 
    time_duration_2 = np.random.normal(loc=time_mean * 4, scale=time_std) 
    
    # Calculate absolute values and ensure time moves forward
    entry_time = current_unix_time + max(1, abs(time_duration_1)) 
    entry_price = price_mean + price_dev_1
    
    exit_time = current_unix_time + max(entry_time - current_unix_time + 1, abs(time_duration_2))
    exit_price = price_mean + price_dev_2
    
    return entry_time, entry_price, exit_time, exit_price

# The updated query function to include current_time check
def query(current_time, plane: list): # <-- Added current_time parameter
    """
    Aggregates a 'plane' of predictions using 2D binning to find the 
    mean time/price of the single highest density region (the joint peak).

    Args:
        current_time (float): The current simulation time.
        plane (list of tuples): A list of prediction data, where each element is 
                                (prediction_id, remaining_lifespan, price, time).

    Returns:
        tuple: (mean_time_peak, mean_price_peak) of the highest 2D density subsample.
               Returns (-1, -1) if the plane is empty or analysis fails.
    """
    if not plane:
        return -1, -1

    
    df = pd.DataFrame(plane, columns=['id', 'lifespan', 'price', 'time'])
 
    df = df[df['time'] > current_time] 
    df = df[df['lifespan'] >= 1] 
    
    population = len(df)
    # Check if enough valid points remain for stable analysis
    if len(df) < 5: 
        return -1, -1

    # Extract the two dimensions of interest
    prices = df['price'].values
    times = df['time'].values
    
    # --- 1. Compute 2D Density (2D Histogram/Binning) ---
    # Use 20 bins for price and time, creating 20x20 = 400 possible bins (the 2D grid)
    H, price_edges, time_edges = np.histogram2d(
        prices, 
        times, 
        bins=20
    )
    
    # --- 2. Find the Single Highest Density Bin (The Peak) ---
    max_count_index = np.unravel_index(np.argmax(H), H.shape)
    
    price_bin_index = max_count_index[0]
    time_bin_index = max_count_index[1]
    
    # --- 3. Define the Boundaries of the Peak Bin ---
    price_min = price_edges[price_bin_index]
    price_max = price_edges[price_bin_index + 1]
    time_min = time_edges[time_bin_index]
    time_max = time_edges[time_bin_index + 1]

    # --- 4. Filter Data to the Subsample within the Peak Bin ---
    # Filter the original DataFrame to include only the points that fell into this max bin
    peak_subsamples = df[
        (df['price'] >= price_min) & (df['price'] < price_max) &
        (df['time'] >= time_min) & (df['time'] < time_max)
    ]
    
    # --- 5. Calculate Final Output (Mean of the Subsample) ---
    if peak_subsamples.empty:
        return -1, -1
        
    mean_price_peak = peak_subsamples['price'].mean()
    mean_time_peak = peak_subsamples['time'].mean()

    # The function must now return 3 values, as expected by save_snapshot:
    return mean_time_peak, mean_price_peak, population

def not_query():
    # TODO: handle
    return None

def not_active_and_above_threshold(active: bool, entry_price: float, exit_price: float, min_price_diff: float):
    # TODO: return not active and |exit_price - entry_price| > min_price_diff
    pass
def below_threshold(paired_peaks_found: bool, entry_price: float, exit_price: float, min_price_diff: float):
    # TODO: return not paired_peaks_found OR |exit_price - entry_price| < min_price_diff
    pass
def filled(order_status: str):
    # TODO: return order_status == "FILLED"
    pass
def not_filled(order_status: str):
    # TODO: return order_status != "FILLED"
    pass

def active_and_update_tsm(active: bool, market_data: list):
    # TODO: return active and update TSM with market_data
    pass
def trigger_tsm(tsm_signal: bool):
    # TODO: return tsm_signal == "EXIT_TRIGGERED"
    pass
def not_trigger_tsm(tsm_signal: bool):
    # TODO: return tsm_signal != "EXIT_TRIGGERED"
    pass



def save_plane_as_html(plane: List[Tuple], current_time: float, asset: str, plane_type: str) -> str:
    """
    Visualizes the prediction plane as an interactive 3D plot using Plotly.
    Calculates 1D KDE for both Time (X) and Price (Y) and sums the density matrices.
    """

    
    TIME_GRID_RESOLUTION = 20 
    PRICE_GRID_POINTS = 15 

    if len(plane) < 5:
        return f"Skipping {plane_type} plane visualization: Too few points ({len(plane)})."

    df = pd.DataFrame(plane, columns=['id', 'lifespan', 'price', 'time'])
    
    # --- Data Preparation ---
    time_data_seconds = df['time'].values - current_time
    price_data = df['price'].values
    lifespan_data = df['lifespan'].values
    time_data_minutes = time_data_seconds / 60.0
    
    # --- 1. Standardization (Normalization) ---
    time_mean = time_data_minutes.mean()
    time_std = time_data_minutes.std()
    price_mean = price_data.mean()
    price_std = price_data.std()

    time_std_safe = time_std if time_std > 1e-6 else 1.0
    price_std_safe = price_std if price_std > 1e-6 else 1.0

    # Normalize BOTH time and price data
    time_data_norm = (time_data_minutes - time_mean) / time_std_safe
    price_data_norm = (price_data - price_mean) / price_std_safe
    
    # --- 2. Calculate 1D KDEs ---
    
    # scott 
    kde_time = gaussian_kde(time_data_norm, bw_method='scott') 
    kde_price = gaussian_kde(price_data_norm, bw_method='scott') 

    # bw_price = 0.01 
    # bw_time = 0.001 
    
    
    # kde_time = gaussian_kde(time_data_norm, bw_method=bw_time) 
    # kde_price = gaussian_kde(price_data_norm, bw_method=bw_price) 

    x_min_norm, x_max_norm = time_data_norm.min(), time_data_norm.max()
    x_range_norm = x_max_norm - x_min_norm
    x_min_adj_norm = x_min_norm - 0.05 * x_range_norm
    x_max_adj_norm = x_max_norm + 0.05 * x_range_norm
    X_norm_1D = np.linspace(x_min_adj_norm, x_max_adj_norm, TIME_GRID_RESOLUTION)
    
    # Y Grid (Price) Setup
    y_min_norm, y_max_norm = price_data_norm.min(), price_data_norm.max()
    y_range_norm = y_max_norm - y_min_norm
    y_min_adj_norm = y_min_norm - 0.05 * y_range_norm
    y_max_adj_norm = y_max_norm + 0.05 * y_range_norm
    Y_norm_1D = np.linspace(y_min_adj_norm, y_max_adj_norm, PRICE_GRID_POINTS)

    Z_time_1D = kde_time(X_norm_1D)
    Z_price_1D = kde_price(Y_norm_1D)

    Z_matrix_Time = np.tile(Z_time_1D, (PRICE_GRID_POINTS, 1))
    Z_matrix_Price = np.tile(Z_price_1D.reshape(PRICE_GRID_POINTS, 1), (1, TIME_GRID_RESOLUTION))

    # Z_combined: ADD the densities together
    Z_combined = Z_matrix_Time * Z_matrix_Price 
    

    X_matrix = np.tile(X_norm_1D, (PRICE_GRID_POINTS, 1))
    X_final = X_matrix * time_std_safe + time_mean 
    Y_matrix = Y_norm_1D.reshape(PRICE_GRID_POINTS, 1)
    Y_final = Y_matrix * price_std_safe + price_mean 

    # --- 7. Plotting ---
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X_final[0, :], 
        y=Y_final[:, 0], 
        z=Z_combined, # Use the combined Z matrix
        colorscale='Viridis',
        opacity=0.6,
        name='Combined 1D KDE Surface'
    ))

    Z_floor = Z_combined.min() + (Z_combined.max() - Z_combined.min()) * 0.05 
    
    fig.add_trace(go.Scatter3d(
        x=time_data_minutes, 
        y=price_data,
        z=np.full_like(time_data_minutes, Z_floor), 
        mode='markers',
        marker=dict(
            size=5,
            color=lifespan_data, 
            colorscale='Plasma',
            colorbar=dict(title='Lifespan (s)'), 
            opacity=1.0
        ),
        name='Raw Predictions'
    ))
    
    plot_title = f"3D Prediction Plane: {asset} ({plane_type}) - Combined 1D KDE"
    
    # 5. Set Layout and Title
    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title='Time (Relative Minutes)',
            yaxis_title='Price ($)',
            zaxis_title='Combined Density (KDE Value)',
            aspectmode='manual',
            aspectratio=dict(x=5.0, y=1.0, z=0.5) 
        )
    )
    
    output_filename = f"snapshot/time={int(current_time)}_asset={asset}_type={plane_type}.html"
    fig.write_html(output_filename, auto_open=False, include_plotlyjs='cdn')

    return output_filename


class State:
    def __init__(self, asset='AAPL', initial_value = 100, position_id = 0, threshold_pct = .01, prediction_lifespan= 10, query_frequency=30):
        self.asset = asset
        self.position_id = position_id
        self.current_price = initial_value
        self.current_time = 0.0
        self.p_std = 0.5 
        self.t_std = 20  
        self.entry_plane = []
        self.exit_plane = []
        
        # GBM Parameters
        self.drift_mu = 0.05
        self.volatility_sigma = 0.05
        self.time_step_seconds = 1 
        SECONDS_PER_TRADING_YEAR = 252 * 6.5 * 3600 
        self.time_step_dt = self.time_step_seconds / SECONDS_PER_TRADING_YEAR

        self.p_mean_off = 0.0 # Price Mean offset is current price
        
        self.MAX_P_STD = 0.5  # Base Price STD
        self.MAX_T_STD = 10   # Base Time STD (Expected Entry in 15 mins)
        self.t_mean = 900     # Expected Entry in 15 mins (15 * 60)
        
        # Sine Wave parameters
        self.CONVERGENCE_RATE = 0.0005  
        self.WAVE_FREQUENCY = 0.01    

    
    def batch_cast_predictions(self, num_predictions: int, max_lifespan: int):
        new_id_start = len(self.entry_plane) + len(self.exit_plane)

        t = self.current_time
        

        sine_wave = np.sin(self.WAVE_FREQUENCY * t)
        

        decay_factor = np.exp(-self.CONVERGENCE_RATE * t)
        
        scaling_factor = ((1 + sine_wave) / 2) * decay_factor 
      
        min_scale = 0.01 
        final_scale = max(min_scale, scaling_factor)
        
    
        dynamic_p_std = self.MAX_P_STD * final_scale
        dynamic_t_std = self.MAX_T_STD * final_scale
        
    
        MAX_PREDICTIONS = 50 
        MIN_PREDICTIONS = 5   
        

        density_multiplier = 1.0 - (final_scale * 0.95) # Multiply by < 1 to keep it slightly less than 1
                                                        # at max convergence, providing a buffer

        dynamic_num_predictions = int(MIN_PREDICTIONS + (MAX_PREDICTIONS - MIN_PREDICTIONS) * density_multiplier)

        temp_pred_time = self.current_time + 1
        
        for i in range(dynamic_num_predictions):
            # 1. Mock Prediction (Entry/Exit)
            et, ep, xt, xp = prediction(
                current_unix_time=temp_pred_time,
                price_mean=self.current_price + self.p_mean_off,
                price_std=dynamic_p_std, 
                time_mean=self.t_mean,
                time_std=dynamic_t_std  
       
            )
            
            new_id_val = new_id_start + i
            
            # Lifespan for viz is the time delta between event time and *current* time
            entry_lifespan_viz = max(0, int(et - self.current_time))
            exit_lifespan_viz = max(0, int(xt - self.current_time))
            
            # 2. Add to Planes (Entry Leg)
            self.entry_plane.append((new_id_val, entry_lifespan_viz, ep, et))
            
            # 3. Add to Planes (Exit Leg)
            self.exit_plane.append((new_id_val, exit_lifespan_viz, xp, xt))
            
            # 4. Increment the prediction time to simulate the next second's arrival
            temp_pred_time += 1

    def update_planes(self):
        """ Filters out predictions whose predicted event time ('time' field) is in the past. """
        
        # Only keep the prediction if its predicted event time is greater than the current time
        self.entry_plane = [
            (id, lifespan, price, time)
            for id, lifespan, price, time in self.entry_plane
            if time > self.current_time
        ]
        
        self.exit_plane = [
            (id, lifespan, price, time)
            for id, lifespan, price, time in self.exit_plane
            if time > self.current_time
        ]


    def step(self):
        """
        Performs one 1-second price update and updates the simulation time.
        """

        Z = np.random.standard_normal(1)[0]
        dt = self.time_step_dt
        price_factor = math.exp(
            (self.drift_mu - 0.5 * self.volatility_sigma**2) * dt +
            self.volatility_sigma * math.sqrt(dt) * Z
        )
        self.current_price = self.current_price * price_factor
        

        self.current_time += self.time_step_seconds
        
        self.update_planes()
        
    def save_snapshot(self, timestep: int):
        """Saves the current state of the prediction planes as HTML files."""
        
        print(f"\n--- SNAPSHOT t={timestep} (Price: {self.current_price:.2f}) ---")
        
    
        entry_time, entry_price, entry_population = query(self.current_time, self.entry_plane) 
        exit_time, exit_price, exit_population = query(self.current_time, self.exit_plane)  


        print(f"Entry Query Peak: Time={entry_time:.0f}, Price={entry_price:.4f}, entry_population={entry_population}")
        print(f"Exit Query Peak: Time={exit_time:.0f}, Price={exit_price:.4f}, exit_population={exit_population}")

        # Save Entry Plane
        entry_path = save_plane_as_html(
            self.entry_plane, 
            self.current_time, 
            self.asset, 
            'entry'
        )
        print(entry_path)

        # Save Exit Plane
        exit_path = save_plane_as_html(
            self.exit_plane, 
            self.current_time, 
            self.asset, 
            'exit'
        )
        print(exit_path)

