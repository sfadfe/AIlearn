import numpy as np
import os
import Double_pendulum as Dp

input_path = "Double_pendulum/RK4_ver/initial_states.txt"
output_dir = "Double_pendulum/ANN_ver/learning_data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

L1, L2 = 1.0, 1.0
dt = 0.05
t_max = 10
steps = int(t_max / dt)

all_trajectories = []

with open(input_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        parts = line.replace(',', ' ').split()
        initial_state = list(map(float, parts))
        
        dp = Dp.Double_pendulum(m1=1.0, m2=1.0, L1=L1, L2=L2, initial_state=initial_state)
        
        trajectory = []
        for i in range(steps):
            trajectory.append(dp.state.copy())
            dp.RK4(dt)
        
        all_trajectories.append(trajectory)

final_data = np.array(all_trajectories)
save_path = os.path.join(output_dir, "RK4.npy")
np.save(save_path, final_data)