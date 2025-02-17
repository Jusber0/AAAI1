def preprocess_yolo_output(boxes):
    """将 YOLO 检测到的人的边界框转换为模型的输入格式"""
    features = []
    for box in boxes:
        x, y, w, h = box
        features.append([x, y, w, h, 0, 0, 0])  # 你可以根据需求添加更多特征
    return features
