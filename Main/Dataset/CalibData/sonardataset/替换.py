import pandas as pd

# === 需要你填写 ===
input_csv = r"D:\jianzhi\sonardataset\sonar.csv"   # 把这里改成你的CSV文件名
output_csv = input_csv # 如果想覆盖，直接写成 input_csv 即可

# 读取 CSV（会自动识别文本字段）
df = pd.read_csv(input_csv, dtype=str)

# 将所有元素转成字符串，再进行替换
df = df.applymap(lambda x: x.replace('/root/autodl-tmp', 'D:/jianzhi') if isinstance(x, str) else x)

# 保存
df.to_csv(output_csv, index=False)

print(f"已完成替换，输出文件: {output_csv}")
