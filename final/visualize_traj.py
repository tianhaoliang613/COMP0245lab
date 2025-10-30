import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- 配置 -----------------
DATA_DIR = Path(__file__).resolve().parent  # CSV 文件所在目录
OUTPUT_DIR = DATA_DIR / "trajectory_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# 轨迹文件和对应的目标点
# 假设 CSV 文件命名为 trajectory_segment_0.csv ... trajectory_segment_3.csv
trajectory_files = [
    "trajectory_segment_0.csv",
    "trajectory_segment_1.csv",
    "trajectory_segment_2.csv",
    "trajectory_segment_3.csv"
]

# 对应目标点 (必须和模拟里一致)
target_points = [
    [0.5, 0.0, 0.1],
    [0.4, 0.2, 0.1],
    [0.4, -0.2, 0.1],
    [0.4, 0.1, 0.1]
]

# ----------------- 可视化 -----------------
for i, file_name in enumerate(trajectory_files):
    csv_path = DATA_DIR / file_name
    if not csv_path.exists():
        print(f"⚠️ File {file_name} not found, skipping.")
        continue

    # 读取 CSV
    traj_df = pd.read_csv(csv_path)
    traj = traj_df[["X","Y","Z"]].to_numpy()

    if traj.shape[0] < 2:
        print(f"⚠️ File {file_name} has too few points, skipping.")
        continue

    target = np.array(target_points[i])
    actual_final = traj[-1]

    # 3D 可视化
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color='red', linewidth=2, label='EE Path')
    ax.scatter(traj[0,0], traj[0,1], traj[0,2], color='green', s=50, marker='^', label='Start')
    ax.scatter(actual_final[0], actual_final[1], actual_final[2], color='orange', s=60, marker='x', label='Actual End')
    ax.scatter(target[0], target[1], target[2], color='blue', s=80, marker='o', label='Target')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"Trajectory Segment {i}")
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()

    # 保存图片
    save_path = OUTPUT_DIR / f"trajectory_segment_{i}_3d.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"💾 Saved 3D trajectory plot: {save_path}")

print("✅ All trajectory plots saved.")
