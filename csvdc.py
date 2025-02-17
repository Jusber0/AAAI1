import pickle
import pandas as pd

# 加载 pkl 文件
with open(r"data/JAAD_data/beh_seq_test.pkl", 'rb') as f:
    data = pickle.load(f)

# 假设 data['bbox'] 存储的是目标数据
bbox_data = data['bbox']
intent_data = data['intent']

# 创建一个空的列表来存储每个 frame 的数据
frames = []

# 将数据整理为适合导出的格式
for i, (bbox, intent) in enumerate(zip(bbox_data, intent_data)):
    # 假设每个 bbox 是一个长度为4的列表，intent 是一个列表
    for frame_bbox, frame_intent in zip(bbox, intent):
        # 将 intent 列表转换为单一的数字（0 或 1）
        frame_intent = frame_intent[0]  # 提取 intent 列表中的第一个值

        # 将每个 frame 的数据存入字典中
        frame_data = {
            'Frame': i,
            'BBox_x1': frame_bbox[0],
            'BBox_y1': frame_bbox[1],
            'BBox_x2': frame_bbox[2],
            'BBox_y2': frame_bbox[3],
            'Intent': frame_intent
        }
        frames.append(frame_data)

# 转换为 pandas DataFrame
df = pd.DataFrame(frames)

# 导出为 CSV 文件
df.to_csv("test_data.csv", index=False)

print("数据已导出为 CSV 文件")
