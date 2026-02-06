import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import json
import os

from model.yolo import yolo, output_process

class CFG:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    class_path = r'dataset/classes.json'      # 类别文件路径
    model_path = r'res/yolo.pth'              # 模型权重路径
    image_path = r'./dataset/VOC2007/JPEGImages/005775.jpg'
    output_image_path = r'./output_detected.jpg'

    backbone = 'resnet'
    S = 7
    B = 2
    image_size = 448
    conf_th = 0.1
    iou_th = 0.5

    num_classes = 0
    classname = None


def get_transforms(image_size):
    """对图像做变换"""
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size, image_size), antialias=True)
    ])


def draw_boxes(image_pil, boxes, class_names, confidences):
    """在图像上绘制边界框和标签"""
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confidences, class_names):
        label = f"{cls_id} {conf:.2f}"
        color = tuple(torch.randint(0, 255, (3,)).tolist())

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 画标签背景+文字
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill="white", font=font)

    return image_pil


def main():
    device = torch.device(CFG.device)
    print(f"Using device: {device}")

    # 加载类别名
    with open(CFG.class_path, 'r') as f:
        classes_dict = json.load(f)
        CFG.classname = list(classes_dict.keys())
        CFG.num_classes = len(CFG.classname)

    # 初始化模型
    yolo_net = yolo(
        s=CFG.S,
        cell_out_ch=CFG.B * 5 + CFG.num_classes,
        backbone_name=CFG.backbone
    )
    yolo_net.load_state_dict(torch.load(CFG.model_path, map_location=device))
    yolo_net.to(device)
    yolo_net.eval()

    # 加载并预处理图像
    if not os.path.exists(CFG.image_path):
        raise FileNotFoundError(f"Image not found: {CFG.image_path}")

    image_pil = Image.open(CFG.image_path).convert("RGB")
    original_width, original_height = image_pil.size

    transform = get_transforms(CFG.image_size)
    image_tensor = transform(image_pil)  # [C, H, W]
    input_batch = image_tensor.unsqueeze(0).to(device)  # [1, C, H, W]

    # 推理
    with torch.no_grad():
        output = yolo_net(input_batch)
        detections = output_process(
            output.cpu(),
            CFG.image_size,
            CFG.S,
            CFG.B,
            CFG.conf_th,
            CFG.iou_th
        )

    detections = detections.squeeze(0)  # [N, 7]

    if detections.numel() == 0:
        print("No objects detected.")
        image_pil.save(CFG.output_image_path)
        return

    # 还原坐标到原始图像尺寸
    scale_x = original_width / CFG.image_size
    scale_y = original_height / CFG.image_size

    boxes_xyxy = detections[:, :4].clone()
    boxes_xyxy[:, [0, 2]] *= scale_x  # x1, x2
    boxes_xyxy[:, [1, 3]] *= scale_y  # y1, y2

    # 转为整数（绘图需要）
    boxes = boxes_xyxy.tolist()
    boxes = [[int(coord) for coord in box] for box in boxes]

    confidences = detections[:, 4].tolist()
    class_ids = detections[:, 6].long().tolist()
    class_labels = [CFG.classname[cls_id] for cls_id in class_ids]

    # 绘图
    result_img = draw_boxes(image_pil.copy(), boxes, class_labels, confidences)

    # 保存
    result_img.save(CFG.output_image_path)
    print(f"Detection result saved to: {CFG.output_image_path}")

if __name__ == '__main__':
    main()