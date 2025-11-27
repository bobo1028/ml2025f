from ultralytics import YOLO
import os

# 進行預測
model = YOLO(r"C:\Users\bobo1\YOLO\runs\detect\train16\weights\best.pt")
results = model.predict(
    source='./datasets/val/images/',
    conf=0.1,
    iou=0.5,
    imgsz=640,
    device=0
)

# 寫出 txt
os.makedirs('./predict_txt/', exist_ok=True)
output_file = open('./predict_txt/images.txt', 'w')

for i in range(len(results)):
    # 取得圖片檔名（不含副檔名）
    filename = os.path.splitext(os.path.basename(results[i].path))[0]

    # 取得預測框座標
    boxes = results[i].boxes
    box_num = len(boxes.cls.tolist())

    # 如果有預測框
    if box_num > 0:
        for j in range(box_num):
            # 取出預測資訊
            label = int(boxes.cls[j].item())       # 類別
            conf  = boxes.conf[j].item()           # 信心度
            x1, y1, x2, y2 = boxes.xyxy[j].tolist() # 邊界框座標

            # 建立一行資料
            line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
            output_file.write(line)

# 關閉輸出檔案
output_file.close()
