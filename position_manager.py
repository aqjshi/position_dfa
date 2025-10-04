from state import *
if __name__ == "__main__":
    
    assets = ['AAPL']

    TIME_DURATION = 3600

    PREDICTION_CAST_FREQ = 30
    PREDICTION_BATCH_SIZE = 30
    PREDICTION_LIFESPAN = 600 

    SNAPSHOT_SAVE_FREQ = 300 

    # 2. Initialize State
    state = State(asset=assets[0], initial_value=150.0)

    print(f"Starting simulation for {assets[0]}. Duration: {TIME_DURATION} seconds.")

    
    for time_step in range(TIME_DURATION + 1):
  
        if time_step % PREDICTION_CAST_FREQ == 0:
            state.batch_cast_predictions(PREDICTION_BATCH_SIZE, PREDICTION_LIFESPAN)
 
        if time_step % SNAPSHOT_SAVE_FREQ == 0 and time_step > 0:
            state.save_snapshot(time_step)
        

        state.step()

    print("\nSimulation complete. Check the 'snapshot' folder for HTML files.")