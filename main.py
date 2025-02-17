import cv2
import torch

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
model.eval()  # 设置模型为评估模式

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 这里可以对摄像头获取的图像进行预处理，例如缩放、归一化等
    # 假设处理后的图像大小为 (batch_size, ip_dim, seq_len) 的形状
    # 你需要根据你的数据结构和模型输入调整这个处理过程
    input_data = preprocess_frame(frame)  # 需要实现这个函数
    
    # 将输入数据转换为 Tensor
    input_tensor = torch.tensor(input_data).double().unsqueeze(0)  # 加上一个维度表示 batch_size
    
    # 进行预测
    with torch.no_grad():  # 在预测时不需要计算梯度
        prediction = lit_model.predict(input_tensor)
    
    # 输出或处理预测结果
    print(f"Predicted: {prediction}")

    # 显示当前帧
    cv2.imshow('Frame', frame)
    
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
