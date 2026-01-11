import torch
import numpy as np
from tqdm import tqdm
import time
from PINNs_torch import PINNs, get_device

device = get_device()
print(f"Using device: {device}")

current_time = time.localtime()
timee = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
epochs = 1000000

filepath = "Double_pendulum/ANN_ver/learning_data/RK4.npy"
save_dir = f"Double_pendulum/ANN_ver/models/model_{timee}.pt"
scaler_dir = f"Double_pendulum/ANN_ver/models/scaler_{timee}.npy"

raw_data = np.load(filepath)
X = raw_data[:, :-1, :] 
T = raw_data[:, 1:, :4] 

X_flat = X.reshape(-1, 8)
T_flat = T.reshape(-1, 4)

X_mean = X_flat.mean(axis=0)
X_std = X_flat.std(axis=0) + 1e-8
X_flat_norm = (X_flat - X_mean) / X_std

T_mean = T_flat.mean(axis=0)
T_std = T_flat.std(axis=0) + 1e-8
T_flat_norm = (T_flat - T_mean) / T_std

np.save(scaler_dir, {'X_mean': X_mean, 'X_std': X_std, 'T_mean': T_mean, 'T_std': T_std})

X_train = torch.tensor(X_flat_norm, dtype=torch.float32, device=device)
T_train = torch.tensor(T_flat_norm, dtype=torch.float32, device=device)

T_mean_gpu = torch.tensor(T_mean, device=device, dtype=torch.float32)
T_std_gpu = torch.tensor(T_std, device=device, dtype=torch.float32)

num_data = X_train.shape[0]
batch_size = 8192

model = PINNs(input_size=8, hidden_sizes=[256]*8, output_size=4, 
              X_mean=X_mean, X_std=X_std)
model = model.to(device)

model.register_buffer('T_mean', T_mean_gpu)
model.register_buffer('T_std', T_std_gpu)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=75)

best_loss = float('inf')
best_state_dict = None

lambda_physics = 1e-4 

pbar = tqdm(range(epochs), desc="Training", mininterval=1.0, ascii=True, smoothing=0.1)

try:
    for epoch in pbar:
        indices = torch.randperm(num_data, device=device)
        epoch_loss = 0.0
        iteration = 0
        
        model.train()
        for i in range(0, num_data, batch_size):
            batch_idx = indices[i : i + batch_size]
            x_batch = X_train[batch_idx]
            t_batch = T_train[batch_idx]

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss_data = torch.mean((y_pred - t_batch) ** 2)
            
            y_pred_real = y_pred * model.T_std + model.T_mean
            t_true_real = t_batch * model.T_std + model.T_mean
            
            x_real = x_batch * model.X_std + model.X_mean
            physics_params = x_real[:, 4:]
            
            E_true = model.get_energy(t_true_real, physics_params)
            E_pred = model.get_energy(y_pred_real, physics_params)
            
            loss_physics_raw = torch.mean((E_pred - E_true) ** 2)
            
            loss = loss_data + lambda_physics * loss_physics_raw
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            iteration += 1

        avg_loss = epoch_loss / iteration
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({'Loss': f'{avg_loss:.6f}', 'Best': f'{best_loss:.6f}'})

        if (epoch + 1) % 100 == 0:
            pbar.write(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.8f} | Best: {best_loss:.8f}")

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Saving best model...")

finally:
    if best_state_dict is not None:
        torch.save(best_state_dict, save_dir)
        print(f"Saved: Best Loss {best_loss:.6f}")