# cat_dog
猫狗图片分类计算机视觉项目
基于PyTorch实现猫狗二分类任务，包含从零搭建CNN、迁移学习、数据增强、多模型对比、模型保存测试、Gradio网页部署全套代码。

## 项目文件说明
- `common.py`：公共工具函数
  - 自定义`CatDogDataset`数据集类，读取图片并自动生成猫狗标签
  - `get_dataloader`：划分训练集/验证集，构建DataLoader
  - `train_one_epoch`：单轮训练函数
  - `evaluate`：模型评估，计算准确率、精确率、召回率、F1分数

- `step1_scratch.py`：从零搭建AlexNet训练模型
  - 手动搭建AlexNet卷积神经网络
  - 不使用预训练权重，完整从头训练
  - 基础图像归一化预处理

- `step2_transfer.py`：迁移学习训练
  - 加载ImageNet预训练AlexNet权重
  - 冻结卷积特征提取层，只训练分类头
  - 修改最后输出层为2分类（猫/狗）

- `step3_aug.py`：数据增强实验
  - 在迁移学习基础上增加图像增强策略
  - 随机水平翻转、随机旋转，扩充训练样本，抑制过拟合

- `step_compare.py`：多模型横向对比实验
  - 支持ResNet18、MobileNetV3、ShuffleNetV2三种轻量网络
  - 统一训练轮次，输出各模型在验证集指标，对比不同模型性能

- `step5_save_test.py`：模型保存与单图预测测试
  - 训练过程保存验证集最优权重`best_model.pth`
  - 实现单张图片推理函数，加载权重进行预测

- `step6_app.py`：Gradio模型部署应用
  - 基于gradio快速搭建简易网页可视化界面
  - 支持上传图片或摄像头拍照，输出猫狗分类概率

> 数据集`datasets`、训练权重`best_model.pth`未上传仓库，需自行准备猫狗图片数据集。

## 环境依赖
```
torch
torchvision
opencv-python
scikit-learn
gradio
```

## 运行说明
1. 准备猫狗图片数据集，放置在`./datasets`文件夹，图片文件名以`cat`或`dog`开头
2. 安装上面列出的python依赖包
3. 按顺序运行脚本：
   - `step1_scratch.py`：从零训练AlexNet
   - `step2_transfer.py`：基础迁移学习
   - `step3_aug.py`：增加数据增强的迁移学习
   - `step_compare.py`：多模型对比
   - `step5_save_test.py`：训练并保存最优模型权重
   - `step6_app.py`：启动Gradio网页分类服务

## 项目原理简述
本项目为图像二分类任务：
- 标签规则：文件名以cat开头 → 标签0（猫）；dog开头 → 标签1（狗）
- 输入图像统一缩放至224×224，使用ImageNet均值方差归一化
- 支持两种训练范式：从零训练CNN、迁移学习微调预训练模型
- 训练完成后可保存模型，最终通过Gradio构建Web交互界面实现图片在线识别

## 注意事项
1. 运行`step5_save_test.py`后才会生成`best_model.pth`，`step6_app.py`依赖该权重文件
2. 网页上传代码时，`.pth`权重文件和数据集文件夹不提交到GitHub
3. 自动优先使用CUDA显卡，无GPU则自动切换CPU运行

