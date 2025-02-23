import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("mnist_test.csv")  # Thay thế bằng tên file của bạn

# Lấy một số ảnh mẫu
samples = df.sample(10)  # Chọn ngẫu nhiên 10 ảnh

plt.figure(figsize=(10, 5))
for i, row in enumerate(samples.values):
    image = np.array(row[1:]).reshape(28, 28)  # Bỏ cột nhãn (giả sử cột đầu tiên là nhãn)
    plt.subplot(2, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f"Label: {row[0]}")  # Hiển thị nhãn thực tế
plt.show()
