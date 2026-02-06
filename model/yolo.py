import numpy as np

from model.darknet import DarkNet
from model.resnet import resnet_1024ch

import torch
import torch.nn as nn
import torchvision


class yolo(nn.Module):
    def __init__(self, s, cell_out_ch, backbone_name, pretrain=None):
        """
        return: [s, s, cell_out_ch]
        """

        super(yolo, self).__init__()

        self.s = s
        self.backbone = None
        self.conv = None
        if backbone_name == 'darknet':
            self.backbone = DarkNet()
        elif backbone_name == 'resnet':
            self.backbone = resnet_1024ch(pretrained=pretrain)
        self.backbone_name = backbone_name

        assert self.backbone is not None, 'Wrong backbone name'

        self.fc = nn.Sequential(
            nn.Linear(1024 * s * s, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, s * s * cell_out_ch)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(batch_size, self.s ** 2, -1)
        return x


class yolo_loss:
    """
    YOLOv1 损失函数
    """

    def __init__(self, device, s, b, image_size, num_classes):
        """
        初始化损失函数参数
        :param device: 计算设备（如 'cuda' 或 'cpu'）
        :param s: 网格划分数量 S（如 7）
        :param b: 每个网格预测的边界框数量 B（如 2）
        :param image_size: 输入图像尺寸（如 448）
        :param num_classes: 类别数量 C
        """
        self.device = device
        self.s = s
        self.b = b
        self.image_size = image_size
        self.num_classes = num_classes

    def __call__(self, input, target):
        """
        计算 YOLOv1 总损失
        
        :param input: 模型输出，形状 [batch_size, S*S, B*5 + num_classes]
                      前 B*5 个值：B 个 bbox 的 (cx, cy, w, h, conf)
                      后 num_classes 个值：类别概率（未归一化）
        :param target: 原始标签，类型为 list
                      长度 = batch_size
                      每个元素是 Tensor，形状 [N_i, 5]，格式 [xmin, ymin, xmax, ymax, class_id]
        :return: 五个损失分量的平均值（标量）：
                 (总损失, xy坐标损失, wh尺寸损失, 置信度损失, 分类损失)
        """
        # 获取 batch 大小（target 是 list，所以用 len）
        self.batch_size = len(target)

        # 将原始标签转换为网格格式：每个网格最多一个目标
        # 返回: List[batch_size]，每个元素是长度为 S*S 的 list
        #       每个网格要么是 None（无目标），要么是 Tensor[1, 5]（有目标）
        target_grids = []
        for i in range(self.batch_size):
            grid = self.label_direct2grid_v1(target[i])
            target_grids.append(grid)

        # 初始化各部分损失（用于累加 batch 内所有样本）
        total_loss = torch.zeros(1, device=self.device)
        xy_loss = torch.zeros(1, device=self.device)
        wh_loss = torch.zeros(1, device=self.device)
        conf_loss = torch.zeros(1, device=self.device)
        class_loss = torch.zeros(1, device=self.device)

        # 遍历 batch 中每张图像
        for i in range(self.batch_size):
            # 计算单张图像的损失
            loss_i, xy_i, wh_i, conf_i, cls_i = self.compute_loss_v1(input[i], target_grids[i])
            total_loss += loss_i
            xy_loss += xy_i
            wh_loss += wh_i
            conf_loss += conf_i
            class_loss += cls_i

        # 返回 batch 平均损失
        return (
            total_loss / self.batch_size,
            xy_loss / self.batch_size,
            wh_loss / self.batch_size,
            conf_loss / self.batch_size,
            class_loss / self.batch_size
        )

    def label_direct2grid_v1(self, label):
        """
        将原始标注（xmin, ymin, xmax, ymax, class）转换为所需网格格式。
        YOLOv1 假设每个网格最多一个目标，所以这里只保留第一个落入该网格的目标。
        :param label: Tensor，形状 [N, 5]，每行为 [xmin, ymin, xmax, ymax, class_id]
        :return: List，长度为 S*S。每个元素为：
                 - None：该网格无目标
                 - Tensor [1, 5]：该网格有目标，格式 [cx, cy, w, h, class_id]
        """
        # 初始化 S*S 个网格都为 None
        output = [None for _ in range(self.s ** 2)]
        # 计算每个网格的尺寸
        size = self.image_size // self.s
        # 获取标注框数量
        n_bbox = label.size(0)

        # 如果没有标注框，直接返回全 None
        if n_bbox == 0:
            return output

        # 创建新张量 label_c，用于存储中心坐标格式
        label_c = torch.zeros_like(label)
        # 将 (xmin, ymin, xmax, ymax) 转换为中心点 + 宽高
        # 中心点 x 坐标 = (xmin + xmax) / 2
        label_c[:, 0] = (label[:, 0] + label[:, 2]) / 2
        # 中心点 y 坐标 = (ymin + ymax) / 2
        label_c[:, 1] = (label[:, 1] + label[:, 3]) / 2
        # 宽度 = xmax - xmin（假设 xmax > xmin）
        label_c[:, 2] = label[:, 2] - label[:, 0]
        # 高度 = ymax - ymin（假设 ymax > ymin）
        label_c[:, 3] = label[:, 3] - label[:, 1]
        # 保留类别标签
        label_c[:, 4] = label[:, 4]

        # 确定每个目标中心点所属的网格索引
        # idx_x[i] = 第 i 个目标中心点所在的 x 方向网格索引
        idx_x = [int(label_c[i][0]) // size for i in range(n_bbox)]
        # idx_y[i] = 第 i 个目标中心点所在的 y 方向网格索引
        idx_y = [int(label_c[i][1]) // size for i in range(n_bbox)]

        # 将中心点坐标归一化到 [0, 1]（相对于当前网格）
        # 先对网格 size 取模，得到在网格内的像素偏移，再除以 size 归一化
        label_c[:, 0] = torch.div(torch.fmod(label_c[:, 0], size), size)
        label_c[:, 1] = torch.div(torch.fmod(label_c[:, 1], size), size)
        # 将宽高归一化到整张图像 [0, 1]
        label_c[:, 2] = torch.div(label_c[:, 2], self.image_size)
        label_c[:, 3] = torch.div(label_c[:, 3], self.image_size)

        # 将每个目标分配到对应的网格中（只保留第一个落入该网格的目标）
        for i in range(n_bbox):
            # 计算一维网格索引（行优先展开）
            idx = idx_y[i] * self.s + idx_x[i]
            # 跳过越界情况（理论上不会发生）
            if idx >= self.s ** 2:
                continue
            # 如果该网格还没有目标，就放入这个目标
            if output[idx] is None:
                output[idx] = torch.unsqueeze(label_c[i], dim=0)
            # （官方 Darknet 实际是覆盖为最后一个，但保留第一个更合理）

        return output

    def compute_loss_v1(self, input, target):
        """
        计算单张图像的 YOLOv1 损失
        
        :param input: 单图模型输出，[S*S, B*5 + num_classes]
        :param target: 经过 label_direct2grid_v1 处理后的标签，List[None 或 Tensor[1,5]]
        :return: (总损失, xy_loss, wh_loss, conf_loss, class_loss) —— 都是标量 tensor
        """
        # 定义损失权重（YOLOv1 原文参数）
        lambda_coord = 5.0   # 坐标损失权重
        lambda_noobj = 0.5   # 无物体置信度损失权重

        # 提取预测框部分 [cx, cy, w, h, conf]
        input_bbox = input[:, :self.b * 5].reshape(-1, self.b, 5)
        # 提取分类部分：[S*S, num_classes]
        input_class = input[:, self.b * 5:]

        # 对预测框应用 sigmoid，确保输出在 (0,1) 范围内
        input_bbox = torch.sigmoid(input_bbox)

        # 初始化该图像的各部分损失
        loss = torch.zeros(1, device=self.device)
        xy_loss = torch.zeros(1, device=self.device)
        wh_loss = torch.zeros(1, device=self.device)
        conf_loss = torch.zeros(1, device=self.device)
        class_loss = torch.zeros(1, device=self.device)

        # 遍历每个网格
        for i in range(self.s ** 2):
            # 当前网格无物体
            if target[i] is None:
                # 所有 B 个预测框的置信度都应该趋近于 0
                obj_conf_target = torch.zeros(self.b, device=self.device)
                conf_loss += torch.sum(lambda_noobj * (input_bbox[i, :, 4] - obj_conf_target) ** 2)

            # 当前网格有物体
            else:
                # 计算每个预测框与标注数据的 IoU
                # 获取真实框（[1, 4]）
                true_box = target[i][:, :4]
                # 获取当前网格的 B 个预测框（[B, 4]）
                pred_boxes = input_bbox[i, :, :4]
                ious = self.get_iou_v1(pred_boxes, true_box, i)

                # 找到 IoU 最大的预测框，也就是"负责人"
                best_box_idx = torch.argmax(ious).item()
                best_iou = ious[best_box_idx]

                # 中心点坐标损失
                xy_loss += lambda_coord * (
                    (input_bbox[i, best_box_idx, 0] - target[i][0, 0]) ** 2 +
                    (input_bbox[i, best_box_idx, 1] - target[i][0, 1]) ** 2
                )

                # 尺寸损失，对 w, h 取平方根，添加 1e-8 防止 sqrt(0) 的梯度问题
                wh_loss += lambda_coord * (
                    (torch.sqrt(input_bbox[i, best_box_idx, 2] + 1e-8) - torch.sqrt(target[i][0, 2])) ** 2 +
                    (torch.sqrt(input_bbox[i, best_box_idx, 3] + 1e-8) - torch.sqrt(target[i][0, 3])) ** 2
                )

                # 置信度损失
                # 负责人的置信度目标 = IoU
                conf_loss += (input_bbox[i, best_box_idx, 4] - best_iou) ** 2

                # 注意，虽然网格内有物体，但只有一个预测框对其负责，这里还有对剩下框noobj的置信度损失
                for b in range(self.b):
                    if b != best_box_idx: # 非负责框
                        conf_loss += lambda_noobj * (input_bbox[i, b, 4] - 0) ** 2

                # 类别概率损失
                # 创建 one-hot 目标向量
                true_class = int(target[i][0, 4])
                class_target = torch.zeros(self.num_classes, device=self.device)
                # 只有标注类别概率为 1，其他全 0
                class_target[true_class] = 1.0
                # 计算 MSE 损失
                class_loss += torch.sum((input_class[i] - class_target) ** 2)

        # 总损失 = 所有分量之和
        loss = xy_loss + wh_loss + conf_loss + class_loss
        return loss, xy_loss, wh_loss, conf_loss, class_loss

    def get_iou_v1(self, pred_boxes, true_box, grid_idx):
        """
        计算预测框与真实框的 IoU
        
        :param pred_boxes: [B, 4]，格式 (cx, cy, w, h)，值 ∈ [0,1]（相对网格）
        :param true_box: [1, 4]，格式 (cx, cy, w, h)，值 ∈ [0,1]（相对网格）
        :param grid_idx: 当前网格的一维索引（0 ~ S*S-1）
        :return: [B]，每个预测框与真实框的 IoU 值
        """
        # 获取网格坐标（x 列, y 行）
        grid_x = grid_idx % self.s
        grid_y = grid_idx // self.s

        # 将相对网格的坐标转换为绝对图像坐标（归一化到 [0,1]）
        # 预测框
        pred_abs = pred_boxes.clone()
        pred_abs[:, 0] = (grid_x + pred_abs[:, 0]) / self.s  # 绝对 cx
        pred_abs[:, 1] = (grid_y + pred_abs[:, 1]) / self.s  # 绝对 cy
        # 真实框
        true_abs = true_box.clone()
        true_abs[:, 0] = (grid_x + true_abs[:, 0]) / self.s
        true_abs[:, 1] = (grid_y + true_abs[:, 1]) / self.s

        # 转换为 (xmin, ymin, xmax, ymax) 格式
        pred_xyxy = self.cxcywh_to_xyxy_v1(pred_abs)
        true_xyxy = self.cxcywh_to_xyxy_v1(true_abs)

        # 计算 IoU
        return self.iou_xyxy_v1(pred_xyxy, true_xyxy)

    def cxcywh_to_xyxy_v1(self, boxes):
        """
        将 (cx, cy, w, h) 格式转换为 (xmin, ymin, xmax, ymax)
        :param boxes: [..., 4]，输入格式
        :return: [..., 4]，输出格式
        """
        xmin = boxes[..., 0] - boxes[..., 2] / 2
        ymin = boxes[..., 1] - boxes[..., 3] / 2
        xmax = boxes[..., 0] + boxes[..., 2] / 2
        ymax = boxes[..., 1] + boxes[..., 3] / 2
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

    def iou_xyxy_v1(self, box1, box2):
        """
        计算两组边界框的 IoU
        :param box1: [N, 4]，格式 (xmin, ymin, xmax, ymax)
        :param box2: [M, 4]，格式 (xmin, ymin, xmax, ymax)
        :return: [N]，当 M=1 时（YOLOv1 场景）
        """
        # 扩展 box2 以匹配 box1 的形状（YOLOv1 中 M=1）
        if box2.size(0) == 1:
            box2 = box2.expand_as(box1)

        # 计算交集区域的坐标
        inter_xmin = torch.max(box1[:, 0], box2[:, 0])
        inter_ymin = torch.max(box1[:, 1], box2[:, 1])
        inter_xmax = torch.min(box1[:, 2], box2[:, 2])
        inter_ymax = torch.min(box1[:, 3], box2[:, 3])

        # 计算交集宽高（防止负数）
        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h

        # 计算各自面积
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        # 计算 IoU（加 epsilon 防止除零）
        union = area1 + area2 - inter_area + 1e-6
        iou = inter_area / union
        return iou

def get_iou(bbox1, bbox2):
    """
    :param bbox1: [bbox, bbox, ..] tensor xmin ymin xmax ymax
    :param bbox2:
    :return: area:
    """

    s1 = abs(bbox1[:, 2] - bbox1[:, 0]) * abs(bbox1[:, 3] - bbox1[:, 1])
    s2 = abs(bbox2[:, 2] - bbox2[:, 0]) * abs(bbox2[:, 3] - bbox2[:, 1])

    ious = []
    for i in range(bbox1.shape[0]):
        xmin = np.maximum(bbox1[i, 0], bbox2[:, 0])
        ymin = np.maximum(bbox1[i, 1], bbox2[:, 1])
        xmax = np.minimum(bbox1[i, 2], bbox2[:, 2])
        ymax = np.minimum(bbox1[i, 3], bbox2[:, 3])

        in_w = np.maximum(xmax - xmin, 0)
        in_h = np.maximum(ymax - ymin, 0)

        in_s = in_w * in_h

        iou = in_s / (s1[i] + s2 - in_s)
        ious.append(iou)
    ious = np.array(ious)
    return ious

def nms(bbox, conf_th, iou_th):
    bbox = np.array(bbox.cpu())

    bbox[:, 4] = bbox[:, 4] * bbox[:, 5]

    bbox = bbox[bbox[:, 4] > conf_th]
    order = np.argsort(-bbox[:, 4])

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        iou = get_iou(np.array([bbox[i]]), bbox[order[1:]])[0]
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return bbox[keep]


def output_process(output, image_size, s, b, conf_th, iou_th):
    """
    param: output in batch
    :return output: list[], bbox: xmin, ymin, xmax, ymax, obj_conf, classes_conf, classes
    """
    batch_size = output.size(0)
    size = image_size // s

    output = torch.sigmoid(output)

    # Get Class
    classes_conf, classes = torch.max(output[:, :, b * 5:], dim=2)
    classes = classes.unsqueeze(dim=2).repeat(1, 1, 2).unsqueeze(dim=3)
    classes_conf = classes_conf.unsqueeze(dim=2).repeat(1, 1, 2).unsqueeze(dim=3)
    bbox = output[:, :, :b * 5].reshape(batch_size, -1, b, 5)

    bbox = torch.cat([bbox, classes_conf, classes], dim=3)

    # To Direct
    bbox[:, :, :, [0, 1]] = bbox[:, :, :, [0, 1]] * size
    bbox[:, :, :, [2, 3]] = bbox[:, :, :, [2, 3]] * image_size

    grid_pos = [(j * image_size // s, i * image_size // s) for i in range(s) for j in range(s)]

    def to_direct(bbox):
        for i in range(s ** 2):
            bbox[i, :, 0] = bbox[i, :, 0] + grid_pos[i][0]
            bbox[i, :, 1] = bbox[i, :, 1] + grid_pos[i][1]
        return bbox

    bbox_direct = torch.stack([to_direct(b) for b in bbox])
    bbox_direct = bbox_direct.reshape(batch_size, -1, 7)

    # cxcywh to xyxy
    bbox_direct[:, :, 0] = bbox_direct[:, :, 0] - bbox_direct[:, :, 2] / 2
    bbox_direct[:, :, 1] = bbox_direct[:, :, 1] - bbox_direct[:, :, 3] / 2
    bbox_direct[:, :, 2] = bbox_direct[:, :, 0] + bbox_direct[:, :, 2]
    bbox_direct[:, :, 3] = bbox_direct[:, :, 1] + bbox_direct[:, :, 3]

    bbox_direct[:, :, 0] = torch.maximum(bbox_direct[:, :, 0], torch.zeros(1))
    bbox_direct[:, :, 1] = torch.maximum(bbox_direct[:, :, 1], torch.zeros(1))
    bbox_direct[:, :, 2] = torch.minimum(bbox_direct[:, :, 2], torch.tensor([image_size]))
    bbox_direct[:, :, 3] = torch.minimum(bbox_direct[:, :, 3], torch.tensor([image_size]))

    bbox = [torch.tensor(nms(b, conf_th, iou_th)) for b in bbox_direct]
    bbox = torch.stack(bbox)
    return bbox


if __name__ == "__main__":
    import torch

    # Test yolo
    x = torch.randn([1, 3, 448, 448])

    # B * 5 + n_classes
    net = yolo(7, 2 * 5 + 20, 'resnet', pretrain=None)
    # net = yolo(7, 2 * 5 + 20, 'darknet', pretrain=None)
    print(net)
    out = net(x)
    print(out)
    print(out.size())

    # Test yolo_loss
    # 测试时假设 s=2, class=2
    s = 2
    b = 2
    image_size = 448  # h, w
    input = torch.tensor([[[0.45, 0.24, 0.22, 0.3, 0.35, 0.54, 0.66, 0.7, 0.8, 0.8, 0.17, 0.9],
                           [0.37, 0.25, 0.5, 0.3, 0.36, 0.14, 0.27, 0.26, 0.33, 0.36, 0.13, 0.9],
                           [0.12, 0.8, 0.26, 0.74, 0.8, 0.13, 0.83, 0.6, 0.75, 0.87, 0.75, 0.24],
                           [0.1, 0.27, 0.24, 0.37, 0.34, 0.15, 0.26, 0.27, 0.37, 0.34, 0.16, 0.93]]])
    target = [torch.tensor([[200, 200, 353, 300, 1],
                            [220, 230, 353, 300, 1],
                            [15, 330, 200, 400, 0],
                            [100, 50, 198, 223, 1],
                            [30, 60, 150, 240, 1]], dtype=torch.float)]

    criterion = yolo_loss('cpu', 2, 2, image_size, 2)
    loss = criterion(input, target)
    print(loss)
