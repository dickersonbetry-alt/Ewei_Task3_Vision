import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

def main():
    # 1. 初始化参数配置 [cite: 108-130]
    baseline = 0.06
    # 左相机内参
    fx = 424.21728927
    fy = 424.46025962
    cx = 341.106753
    cy = 250.35323421
    
    # 畸变系数 [cite: 125-130]
    dist_coeffs_left = np.array([-4.19347495e-01, 2.39745507e-01, 5.89055100e-04, 4.05382995e-04, -8.53491128e-02])
    
    # 2. 加载 YOLO 模型
    # 确保 yolo26n.pt 模型文件和视频文件在同级目录下 
    model_path = 'yolo26n.pt'
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，请检查路径！")
        return
    model = YOLO(model_path)

    # 3. 打开视频流 (请将 'test_video.mp4' 替换为压缩包里的真实视频名字)
    video_path = 'video.mp4' 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return

    # 用于保存 Excel 数据的列表
    data_list = []
    frame_count = 0

    print("✅ 成功启动视觉定位系统，正在处理视频流...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1

        # 假设双目视频是左右拼接格式 (Left | Right)
        # 如果是单摄视频，这一步需要根据实际视频源结构进行调整
        h, w, _ = frame.shape
        left_img = frame[:, :w//2]
        right_img = frame[:, w//2:]

        # 对左图进行目标检测，过滤 person 类别 (类别索引通常为 0) 
        results = model(left_img, classes=[0], verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 获取检测框的坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 计算目标中心点像素坐标
                u_left = (x1 + x2) / 2
                v_left = (y1 + y2) / 2

                # ==========================================
                # 简化版双目匹配：
                # 实际工程中需用 SGBM 算法计算稠密视差图。
                # 考核场景下若只检测同一个人，可在右图也跑一次 YOLO 取中心点。
                # 这里提供检测右图并计算视差的逻辑：
                # ==========================================
                right_results = model(right_img, classes=[0], verbose=False)
                if len(right_results) > 0 and len(right_results[0].boxes) > 0:
                    rx1, ry1, rx2, ry2 = right_results[0].boxes[0].xyxy[0].cpu().numpy()
                    u_right = (rx1 + rx2) / 2
                    
                    # 1. 计算视差 d = 左图 x - 右图 x
                    disparity = u_left - u_right
                    
                    if disparity > 0:
                        # 2. 计算深度 Z
                        Z = (fx * baseline) / disparity
                        
                        # 3. 计算三维坐标 X, Y 
                        X = (u_left - cx) * Z / fx
                        Y = (v_left - cy) * Z / fy
                        
                        # 绘制检测框和坐标信息
                        cv2.rectangle(left_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        text = f"X:{X:.2f}m Y:{Y:.2f}m Z:{Z:.2f}m"
                        cv2.putText(left_img, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # 满足条件：每 5 帧记录一次数据 
                        if frame_count % 5 == 0:
                            data_list.append({
                                'Frame': frame_count,
                                'X_m': round(X, 3),
                                'Y_m': round(Y, 3),
                                'Distance_Z_m': round(Z, 3)
                            })

        # 显示画面
        cv2.imshow("MiniROV Binocular Vision", left_img)
        
        # 按 'q' 键可提前退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. 视频处理完毕，释放资源并生成 Excel 
    cap.release()
    cv2.destroyAllWindows()
    
    if data_list:
        df = pd.DataFrame(data_list)
        excel_name = "person_coordinates.xlsx"
        df.to_excel(excel_name, index=False)
        print(f"📊 数据导出成功！文件已保存至: {excel_name}")
    else:
        print("⚠️ 未检测到有效目标，未能生成 Excel。")

if __name__ == "__main__":
    main()