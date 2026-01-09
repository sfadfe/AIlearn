import numpy as np
import os
import Double_pendulum as Dp
from tqdm import tqdm

input_path = "Double_pendulum/RK4_ver/initial_states.txt"
output_dir = "Double_pendulum/ANN_ver/learning_data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dt = 0.05
t_max = 10
steps = int(t_max / dt)

trajectories = []

with open(input_path, "r") as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Simulating"):
    line = line.strip()
    if not line:
        continue

    L1 = float(np.random.uniform(0.1, 3.0))
    L2 = float(np.random.uniform(0.1, 3.0))
    m1 = float(np.random.uniform(0.5, 6.5))
    m2 = float(np.random.uniform(0.5, 6.5))

    if np.random.rand() < 0.07:
        m1, m2 = 10.0, 0.1
        if np.random.rand() < 0.5:
            m1, m2 = m2, m1
    
    parts = line.replace(',', ' ').split()
    initial_state = list(map(float, parts))
    
    dp = Dp.Double_pendulum(m1, m2, L1=L1, L2=L2, initial_state=initial_state)
    
    trajectory = []
    for i in range(steps):
        init_data = dp.state.tolist() + [m1, m2, L1, L2]
        trajectory.append(init_data)
        
        dp.RK4(dt)
    
    trajectories.append(trajectory)

print("Saving data...")
final_data = np.array(trajectories, dtype=np.float32)
save_path = os.path.join(output_dir, "RK4.npy")
np.save(save_path, final_data)