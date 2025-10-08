import numpy as np
import pandas as pd
import math
from collections import Counter
from scipy.stats import gaussian_kde

import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple


def save_plane_as_html(plane: List[Tuple], current_time: float, asset: str, plane_type: str) -> str:
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
    
    # --- CRITICAL CHANGE FOR 2D KDE ---
    # Combine the standardized data into a single 2xN array
    data_2d_norm = np.vstack([time_data_norm, price_data_norm])
    kde_2d = gaussian_kde(data_2d_norm, bw_method='scott') 

    # X Grid (Time) Setup
    x_min_norm, x_max_norm = time_data_norm.min(), time_data_norm.max()
    x_range_norm = x_max_norm - x_min_norm
    x_min_adj_norm = x_min_norm - 0.05 * x_range_norm
    x_max_adj_norm = x_max_norm + 0.05 * x_range_norm
    X_norm_1D = np.linspace(x_min_adj_norm, x_max_adj_norm, TIME_GRID_RESOLUTION)

    y_min_norm, y_max_norm = price_data_norm.min(), price_data_norm.max()
    y_range_norm = y_max_norm - y_min_norm
    y_min_adj_norm = y_min_norm - 0.05 * y_range_norm
    y_max_adj_norm = y_max_norm + 0.05 * y_range_norm
    Y_norm_1D = np.linspace(y_min_adj_norm, y_max_adj_norm, PRICE_GRID_POINTS)


    X_norm_matrix, Y_norm_matrix = np.meshgrid(X_norm_1D, Y_norm_1D)
  
    Z_2D_points = np.vstack([X_norm_matrix.ravel(), Y_norm_matrix.ravel()])
    Z_combined = kde_2d(Z_2D_points).reshape(X_norm_matrix.shape)
    

    X_final = X_norm_matrix * time_std_safe + time_mean 
    Y_final = Y_norm_matrix * price_std_safe + price_mean 


    fig = go.Figure()

    fig.add_trace(go.Surface(

        x=X_final[0, :], # Use a 1D slice for X
        y=Y_final[:, 0], # Use a 1D slice for Y
        z=Z_combined, 
        colorscale='Viridis',
        opacity=0.6,
        name='2D KDE Surface' # Renamed to reflect 2D KDE
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
    
    plot_title = f"3D Prediction Plane: {asset} ({plane_type}) - PROPER 2D KDE"
    
    # 5. Set Layout and Title
    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title='Time (Relative Minutes)',
            yaxis_title='Price ($)',
            zaxis_title='Joint Density (2D KDE Value)', # Renamed Z-axis title
            aspectmode='manual',
            aspectratio=dict(x=5.0, y=1.0, z=0.5) 
        )
    )
    
    output_filename = f"snapshot/time={int(current_time)}_asset={asset}_type={plane_type}.html"
    fig.write_html(output_filename, auto_open=False, include_plotlyjs='cdn')

    return output_filename


