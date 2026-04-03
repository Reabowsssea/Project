# lstm_controller.py

import torch
import torch.nn as nn

class LSTMController(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=6, num_layers=2):
        """
        input_dim: 输入维度
        hidden_dim: LSTM隐藏单元数
        output_dim: 输出速度控制向量
        num_layers: LSTM层数
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        """
        x: [seq_len, batch, input_dim]
        """
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[-1])  # 取最后时刻
        return out, hidden
