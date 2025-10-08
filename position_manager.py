from state import *

    
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