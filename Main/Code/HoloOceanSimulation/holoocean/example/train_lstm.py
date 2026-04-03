# train_lstm.py
import torch
from lstm_controller import LSTMController
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 加载你采集的 CSV
data = np.loadtxt("lstm_training_data.csv", delimiter=",")

# 假设
# 前n列是输入
# 最后6列是监督输出
input_dim = data.shape[1] - 6
X = data[:, :input_dim]
y = data[:, -6:]

# 制作数据集
dataset = TensorDataset(
    torch.tensor(X, dtype=torch.float32).unsqueeze(1),  # seq_len=1
    torch.tensor(y, dtype=torch.float32)
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 模型
model = LSTMController(input_dim=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    for batch_X, batch_y in loader:
        out, _ = model(batch_X)
        loss = loss_fn(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch}, loss={loss.item()}")
torch.save(model.state_dict(), "lstm_controller.pth")
