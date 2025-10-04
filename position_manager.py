from state import *


transition_dict = {

    "a": new_id,

    "b": prediction,

    "c": query,

    "d": not_query,

    "e": not_active_and_above_threshold,

    "f": below_threshold,

    "g": not_filled,

    "h": filled,

    "i": active_and_update_tsm,

    "j": trigger_tsm,

    "k": not_trigger_tsm

}


states_dict = {

    0: "idle",

    1: "price_stream",

    2: "prediction_plane",

    3: "top_pair",

    4: "declare",

    5: "active",

    6: "exit"

}


# row = state, col  = function, (row, col) = new state

transition_table = [

    0:  [1,0,0, 0,0,0, 0,0,0, 0,0],

    1:  [1,2,1, 1,1,1, 1,1,1, 1,1],

    2:  [2,2,3, 1,2,2, 2,2,2, 2,2],

    3:  [3,3,3, 3,4,1, 3,3,5, 3,3],

    4:  [4,4,4, 4,4,4, 1,5,4, 4,4],

    5:  [5,5,5, 5,5,5, 5,5,5, 6,1],

    6:  [1,6,6, 6,6,6, 6,6,6, 6,6],  

]



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