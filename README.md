Paper: [[1506.02640] You Only Look Once: Unified, Real-Time Object Detection (arxiv.org)](https://arxiv.org/abs/1506.02640)

详见 [yolov1原理学习与代码复现 | Hlwdy's blog](https://blog.hlwdy.top/post/yolov1%E5%8E%9F%E7%90%86%E5%AD%A6%E4%B9%A0%E4%B8%8E%E4%BB%A3%E7%A0%81%E5%A4%8D%E7%8E%B0/)

修改预训练的ResNet50权重初始化backbone，来自网上的方法，135个epoch在开头额外加了warm的阶段，另外前期训练冻结了backbone。YOLOv1的loss函数实现已尽量贴近原文。

# 训练

先提前创建res目录，训练得到的权重会保存在./res/yolo.pth。

VOC2007/2012数据集放在dataset目录下，运行`train.py`开始训练。

# 测试

见`run.py`，从数据集中读取一张图片进行目标检测，绘制标注框后输出图片。

在十几个epoch训练之后形成初步的效果：

![检测演示](assets/detect.png)