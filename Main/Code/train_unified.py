"""
统一的动力学模型训练脚本, 支持确定性 (MBPO) 和概率 (PETS/MPC) 模型.

使用方法:
    python train_unified.py --model_type deterministic
    python train_unified.py --model_type probabilistic
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# ============================================================================
# 1. 模型定义
# ============================================================================

class DeterministicDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)  # 预测状态的变化量 delta_x
        )
        
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None

    def set_normalization_params(self, state_mean, state_std, action_mean, action_std):
        self.state_mean = state_mean if isinstance(state_mean, torch.Tensor) else torch.from_numpy(state_mean)
        self.state_std = state_std if isinstance(state_std, torch.Tensor) else torch.from_numpy(state_std)
        self.action_mean = action_mean if isinstance(action_mean, torch.Tensor) else torch.from_numpy(action_mean)
        self.action_std = action_std if isinstance(action_std, torch.Tensor) else torch.from_numpy(action_std)

    def forward(self, x, u):
        device = next(self.parameters()).device
        x = x.to(device)
        u = u.to(device)
        
        state_mean_d = self.state_mean.to(device)
        state_std_d = self.state_std.to(device)
        action_mean_d = self.action_mean.to(device)
        action_std_d = self.action_std.to(device)

        x_norm = (x - state_mean_d) / state_std_d
        u_norm = (u - action_mean_d) / action_std_d

        xu_norm = torch.cat([x_norm, u_norm], dim=-1)
        delta_x_norm = self.net(xu_norm)

        delta_x = delta_x_norm * state_std_d

        return x + delta_x


class ProbabilisticDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_size)
        
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_head = nn.Linear(hidden_size, state_dim)
        self.log_var_head = nn.Linear(hidden_size, state_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None

    def set_normalization_params(self, state_mean, state_std, action_mean, action_std):
        self.state_mean = state_mean if isinstance(state_mean, torch.Tensor) else torch.from_numpy(state_mean)
        self.state_std = state_std if isinstance(state_std, torch.Tensor) else torch.from_numpy(state_std)
        self.action_mean = action_mean if isinstance(action_mean, torch.Tensor) else torch.from_numpy(action_mean)
        self.action_std = action_std if isinstance(action_std, torch.Tensor) else torch.from_numpy(action_std)

    def forward(self, x, u):
        """返回归一化尺度下的状态变化量的均值和对数方差"""
        device = next(self.parameters()).device
        x = x.to(device)
        u = u.to(device)
        
        state_mean_d = self.state_mean.to(device)
        state_std_d = self.state_std.to(device)
        action_mean_d = self.action_mean.to(device)
        action_std_d = self.action_std.to(device)

        x_norm = (x - state_mean_d) / state_std_d
        u_norm = (u - action_mean_d) / action_std_d

        xu_norm = torch.cat([x_norm, u_norm], dim=-1)
        
        h = self.relu(self.input_layer(xu_norm))
        
        # 带残差连接的隐藏层
        h1 = self.relu(self.hidden1(h))
        h1 = self.dropout(h1)
        h1 = h1 + h
        
        h2 = self.relu(self.hidden2(h1))
        h2 = self.dropout(h2)
        h2 = h2 + h1
        
        h3 = self.relu(self.hidden3(h2))
        h3 = self.dropout(h3)
        h3 = h3 + h2
        
        h4 = self.relu(self.hidden4(h3))
        h4 = self.dropout(h4)
        h4 = h4 + h3
        
        mean_norm = self.mean_head(h4)
        log_var_norm = self.log_var_head(h4)
        
        # 限制log_var的范围, 防止数值不稳定
        log_var_norm = torch.tanh(log_var_norm) * 3

        return mean_norm, log_var_norm

    def predict(self, x, u, num_samples=10):
        """
        Ensemble 预测：多次采样取均值
        """
        mean_norm, log_var_norm = self.forward(x, u)
        
        device = x.device
        state_std_d = self.state_std.to(device)
        
        predictions = []
        for _ in range(num_samples):
            std_norm = torch.exp(0.5 * log_var_norm)
            eps = torch.randn_like(std_norm)
            delta_x_norm_sample = mean_norm + eps * std_norm
            
            delta_x_sample = delta_x_norm_sample * state_std_d
            next_state_sample = x + delta_x_sample
            predictions.append(next_state_sample)
        
        ensemble_prediction = torch.stack(predictions).mean(dim=0)
        
        return ensemble_prediction


# ============================================================================
# 2. 损失函数
# ============================================================================

def gaussian_nll_loss(mean, log_var, target):
    """高斯负对数似然损失"""
    log_var = torch.clamp(log_var, -10, 10)
    variance = torch.exp(log_var)
    sq_diff = (target - mean) ** 2
    
    loss = torch.mean(0.5 * sq_diff / variance + 0.5 * log_var)
    return loss


# ============================================================================
# 3. 训练器类
# ============================================================================

class ModelTrainer:
    def __init__(self, model, model_type, device, lr=0.001):
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        
        if model_type == 'probabilistic':
            # 概率模型: AdamW 优化器, CosineAnnealingWarmRestarts 调度器
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
            self.criterion = gaussian_nll_loss
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        else:
            # 确定性模型: Adam 优化器, ReduceLROnPlateau 调度器
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.criterion = nn.MSELoss()
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', patience=10, factor=0.5, verbose=True
            )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for states, actions, next_states in loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            next_states = next_states.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.model_type == 'probabilistic':
                mean_norm, log_var_norm = self.model(states, actions)
                
                # 目标是归一化后的状态变化量
                target_delta = next_states - states
                state_std_d = self.model.state_std.to(self.device)
                target_delta_norm = target_delta / state_std_d
                
                loss = self.criterion(mean_norm, log_var_norm, target_delta_norm)
            else:
                predicted_next_states = self.model(states, actions)
                
                # 学习状态变化量, 而非下一状态
                target_delta = next_states - states
                predicted_delta = predicted_next_states - states
                
                # 重要修改: 在归一化空间中计算loss, 避免不同维度尺度差异导致的问题
                state_std_d = self.model.state_std.to(self.device)
                target_delta_norm = target_delta / state_std_d
                predicted_delta_norm = predicted_delta / state_std_d
                
                loss = self.criterion(predicted_delta_norm, target_delta_norm)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for states, actions, next_states in loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                next_states = next_states.to(self.device)
                
                if self.model_type == 'probabilistic':
                    mean_norm, log_var_norm = self.model(states, actions)
                    
                    target_delta = next_states - states
                    state_std_d = self.model.state_std.to(self.device)
                    target_delta_norm = target_delta / state_std_d
                    
                    loss = self.criterion(mean_norm, log_var_norm, target_delta_norm)
                else:
                    predicted_next_states = self.model(states, actions)
                    
                    target_delta = next_states - states
                    predicted_delta = predicted_next_states - states
                    
                    # 在归一化空间中计算loss
                    state_std_d = self.model.state_std.to(self.device)
                    target_delta_norm = target_delta / state_std_d
                    predicted_delta_norm = predicted_delta / state_std_d
                    
                    loss = self.criterion(predicted_delta_norm, target_delta_norm)
                
                total_loss += loss.item()
        
        return total_loss / len(loader)




def train_dynamics_model(
    model_type='deterministic',
    data_path='./new_training_data_fixed',
    save_path=None,
    num_epochs=100,
    batch_size=128,
    hidden_size=128,
    learning_rate=0.001
):
    """
    统一的训练函数
    
    Args:
        model_type: 'deterministic' 或 'probabilistic'
        data_path: 数据路径
        save_path: 模型保存路径（如果为None，自动生成）
        num_epochs: 训练轮数
        batch_size: 批次大小
        hidden_size: 隐藏层大小
        learning_rate: 学习率
    """
    
    if save_path is None:
        save_path = f"./saved_models_{model_type}"
    
    os.makedirs(save_path, exist_ok=True)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print(f"[Training Configuration]")
    print("="*80)
    print(f"  Model Type: {model_type.upper()}")
    print(f"  Device: {device}")
    print(f"  Data Path: {data_path}")
    print(f"  Save Path: {save_path}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Learning Rate: {learning_rate}")
    print("="*80)

    # 加载数据
    print("\n[Loading Data...]")
    states = np.load(os.path.join(data_path, "states.npy"))
    actions = np.load(os.path.join(data_path, "actions.npy"))
    next_states = np.load(os.path.join(data_path, "next_states.npy"))
    
    STATE_DIM = states.shape[1]
    ACTION_DIM = actions.shape[1]
    
    print(f"  [OK] States: {states.shape}")
    print(f"  [OK] Actions: {actions.shape}")
    print(f"  [OK] Next States: {next_states.shape}")

    states_tensor = torch.from_numpy(states).float()
    actions_tensor = torch.from_numpy(actions).float()
    next_states_tensor = torch.from_numpy(next_states).float()
    
    dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor)

    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"\n[Data Split]")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("\n[Computing Normalization Parameters...]")
    train_states = torch.stack([s for s, a, ns in train_dataset])
    train_actions = torch.stack([a for s, a, ns in train_dataset])
    
    state_mean = train_states.mean(dim=0).numpy()
    state_std = train_states.std(dim=0).numpy()
    action_mean = train_actions.mean(dim=0).numpy()
    action_std = train_actions.std(dim=0).numpy()

    
    state_std[state_std == 0] = 1.0
    action_std[action_std == 0] = 1.0

    np.save(os.path.join(save_path, "state_mean.npy"), state_mean)
    np.save(os.path.join(save_path, "state_std.npy"), state_std)
    np.save(os.path.join(save_path, "action_mean.npy"), action_mean)
    np.save(os.path.join(save_path, "action_std.npy"), action_std)
    print(f"  [OK] Normalization parameters saved")

    print(f"\n[Initializing {model_type} model...]")
    if model_type == 'probabilistic':
        model = ProbabilisticDynamicsModel(
            state_dim=STATE_DIM, 
            action_dim=ACTION_DIM, 
            hidden_size=hidden_size
        )
    else:
        model = DeterministicDynamicsModel(
            state_dim=STATE_DIM, 
            action_dim=ACTION_DIM, 
            hidden_size=hidden_size
        )
    
    model.set_normalization_params(state_mean, state_std, action_mean, action_std)
    
    trainer = ModelTrainer(model, model_type, device, learning_rate)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\n[Start Training...]")
    print("="*80)
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if model_type == 'probabilistic':
            trainer.scheduler.step()
        else:
            trainer.scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if model_type == 'probabilistic':
               
                model_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'state_mean': model.state_mean,
                    'state_std': model.state_std,
                    'action_mean': model.action_mean,
                    'action_std': model.action_std,
                }
                torch.save(model_checkpoint, os.path.join(save_path, "best_model.pth"))
            else:
               
                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            
            print(f" [OK] [New Best: {best_val_loss:.6f}]")
        else:
            print()

    print("="*80)
    print(f"[Training Complete!]")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    
    
    print(f"\n[Plotting Loss Curve...]")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title(f'{model_type.capitalize()} Model Training Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    curve_path = os.path.join(save_path, "training_curve.png")
    plt.savefig(curve_path, dpi=150)
    print(f"  [OK] Curve saved to: {curve_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("[All Tasks Complete!]")
    print("="*80)
    
    return model, best_val_loss




def main():
    parser = argparse.ArgumentParser(description='训练动力学模型')
    parser.add_argument('--model_type', type=str, default='deterministic',
                        choices=['deterministic', 'probabilistic'],
                        help='模型类型: deterministic 或 probabilistic')
    parser.add_argument('--data_path', type=str, default='./training_data_v3_lowspeed',
                        help='训练数据路径')
    parser.add_argument('--save_path', type=str, default='./saved_models_v3_lowspeed',
                        help='模型保存路径（V3低速优化版）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='隐藏层大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    
    args = parser.parse_args()
    
    train_dynamics_model(
        model_type=args.model_type,
        data_path=args.data_path,
        save_path=args.save_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()

