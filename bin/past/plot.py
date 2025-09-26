import json
import matplotlib.pyplot as plt

# 讀取 JSON 檔
with open("cache/node_growth.json", "r", encoding="utf-8") as f:
    y = json.load(f)

# 建立 X 座標（用索引 1..N）
x = list(range(1, len(y) + 1))

# 畫折線圖並設定軸名
plt.figure()
plt.plot(x, y, marker="o")   # 不指定顏色，符合純 matplotlib 規則
plt.xlabel("edits")           # X 軸名稱
plt.ylabel("reviews")           # Y 軸名稱
plt.title("review counts growth over time/edit")  # 圖表標題
plt.grid(True)
plt.tight_layout()
plt.show()