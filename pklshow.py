import pickle

# 加载 pkl 文件
with open(r"data/JAAD_data/beh_seq_val.pkl", 'rb') as f:
    data = pickle.load(f)

# 假设 data['bbox'] 存储的是目标数据
bbox_data = data['bbox']
intent_data = data['intent']

# 打印 bbox 和 intent 对应关系
for i, (bbox, intent) in enumerate(zip(bbox_data, intent_data)):
    print(f"Frame {i}:")
    
    # 打印该帧的 bbox
    print(f"  BBox: {bbox}")
    
    # 只检查该帧的第一个 intent 值
    frame_intent = intent[0]  # 获取第一个 intent 值
    print(f"  Intent: {frame_intent}")  # 打印统一的 intent 值（0 或 1）
    
    print("-" * 178)
