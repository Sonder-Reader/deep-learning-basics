
---

# PyTorch 入门：环境搭建 + MNIST 加载与可视化

> 本文记录了我从零开始配置 PyTorch 环境、加载 MNIST 数据集、可视化图片以及练习张量操作的完整过程，包括踩坑记录和解决方案。

---

## 1. 为什么要写这篇博客？

MNIST 被称为深度学习的“Hello World”，学习 PyTorch 的第一步自然要从它开始。通过这篇博客，我希望记录下自己搭建环境、处理数据、练习张量操作的完整过程，也希望能帮助到同样刚入门的朋友。

---

## 2. 环境搭建

### 2.1 安装 Anaconda
我使用的是 Windows 系统，从 [Anaconda 官网](https://www.anaconda.com/download) 下载了 64 位安装包。安装时选择了“Just Me”，并**没有勾选**“Add Anaconda to my PATH”，之后通过 Anaconda Prompt 来操作。

### 2.2 创建独立环境
在 Anaconda Prompt 中执行：

```bash
conda create -n pytorch python=3.9 -y
conda activate pytorch
```

创建独立环境的好处是：不同项目可以拥有各自版本的 Python 和库，互不干扰。

### 2.3 安装 PyTorch
由于我的电脑没有 NVIDIA 显卡，我选择了 CPU 版本：

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

安装完成后验证：

```python
import torch
print(torch.__version__)   # 输出 2.3.0
```

### 2.4 安装额外依赖
在后续可视化图片时需要用到 matplotlib，因此提前安装：

```bash
conda install matplotlib -y
```

---

## 3. 加载 MNIST 数据集

### 3.1 数据预处理
```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

- `ToTensor()`：将 PIL 图片或 numpy 数组转换为 PyTorch 张量，并将像素值从 [0,255] 缩放到 [0,1]。
- `Normalize(mean, std)`：用均值 0.1307 和标准差 0.3081 进行标准化，使数据分布接近标准正态分布，有助于模型训练。

### 3.2 下载数据集
```python
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
```

- `root='./data'`：数据保存到当前目录下的 `data` 文件夹。
- `train=True`：加载训练集（60000 张图片）。
- `download=True`：如果本地没有数据，自动下载。下载时先尝试官方源（可能 404），然后自动切换到备用源（Amazon S3），耐心等待即可。
- `DataLoader`：将数据集封装成迭代器，每批 32 张图片，并在每个 epoch 开始时打乱顺序。

### 3.3 查看一个批次
```python
dataiter = iter(trainloader)
images, labels = next(dataiter)
print("images shape:", images.shape)   # torch.Size([32, 1, 28, 28])
print("labels shape:", labels.shape)   # torch.Size([32])
```

---

## 4. 可视化图片

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(8, 5))
for i, ax in enumerate(axes.flat):
    img = images[i].squeeze()          # 去掉通道维度 (1,28,28) -> (28,28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {labels[i].item()}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

![MNIST 前 6 张图片示例](请在这里插入你的截图链接)

运行后可以看到 6 张手写数字图片及其对应标签，说明数据加载正确。

---

## 5. 张量操作练习

张量是 PyTorch 的基础数据结构，熟练掌握张量操作对后续搭建模型至关重要。以下是我手动练习的几个操作：

### 5.1 创建张量
```python
a = torch.tensor([[1, 2], [3, 4]])      # 从列表创建
b = torch.zeros(3, 4)                   # 全零张量
c = torch.randn(2, 3)                   # 标准正态分布随机张量
```

### 5.2 形状变换
```python
x = torch.randn(4, 6)
y = x.view(3, 8)        # 改变形状，元素总数不变（4×6 = 3×8）
z = x.reshape(2, 12)    # 类似 view，但更灵活（可处理非连续内存）
```

### 5.3 索引与切片
```python
print(x[0, :])           # 第一行
print(x[:, 1])           # 第二列
print(x[:2, 2:])         # 前两行，后两列
```

### 5.4 数学运算
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(a + b)                     # 逐元素相加
print(torch.matmul(a.view(1,3), b.view(3,1)))  # 矩阵乘法 (1×3) × (3×1) = 1×1
print(a.sum())                   # 求和
print(b.float().mean())          # 求均值（需要转为浮点）
```

> **注意**：整数张量不能直接求均值，会报错 `RuntimeError`，因为均值结果通常是浮点数。解决方法是用 `.float()` 转换类型。

---

## 6. 踩坑记录

### 6.1 `ModuleNotFoundError: No module named 'matplotlib'`
- **现象**：在 Jupyter Notebook 中运行 `import matplotlib.pyplot as plt` 时提示找不到模块。
- **原因**：新建的 `pytorch` 环境没有安装 matplotlib。
- **解决**：激活环境后执行 `conda install matplotlib -y`，然后重启 Jupyter 内核。

### 6.2 `NameError: name 'axes' is not defined`
- **现象**：可视化代码运行时报错，提示 `axes` 未定义。
- **原因**：我写成了 `fig, axe = plt.subplots(...)`，后面却使用了 `axes`，变量名不一致。
- **解决**：改为 `fig, axes = plt.subplots(...)`，确保变量名统一。

### 6.3 `RuntimeError: mean(): could not infer output dtype...`
- **现象**：对整数张量 `b` 调用 `b.mean()` 报错。
- **原因**：PyTorch 不允许对整数张量直接求均值，因为结果类型不明确。
- **解决**：使用 `b.float().mean()` 先转为浮点类型。

### 6.4 GitHub 推送失败：`remote: Repository not found.`
- **现象**：执行 `git push` 时提示仓库不存在。
- **原因**：我还没有在 GitHub 上创建对应的仓库，或者远程地址填写错误。
- **解决**：先在 GitHub 网页上创建新仓库（不要勾选“初始化 README”），然后使用 `git remote add origin 正确地址` 重新关联，最后推送成功。

---

## 7. 总结与下一步

今天完成了：
- ✅ PyTorch 环境搭建（Anaconda 独立环境）
- ✅ MNIST 数据加载与可视化
- ✅ 张量基础操作练习
- ✅ 代码提交到 GitHub

**下一步**：实现线性分类器（softmax 回归），并继续学习数学建模的基本方法。

---

## 8. 参考资料
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [MNIST 数据集介绍](http://yann.lecun.com/exdb/mnist/)
- [Markdown 语法指南](https://www.markdownguide.org/)

---

**如果你对某个步骤还有疑问，欢迎留言讨论！**


---

