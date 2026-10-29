# Lecture 9 教学实现：Scaling Laws

- scaling_laws_lab.py 提供固定 loss floor 的 log-linear power-law fit、RMSE/R²/残差、可复现 synthetic data、粗略训练 FLOPs 账本和固定 compute budget 的 N/D tradeoff。
- Notebook 中所有拟合样本均由代码生成并明确标记为 synthetic，不是公开论文数据，也不是实际模型训练观测。
- 拟合时 floor 由调用者固定；同时拟合 floor、amplitude、exponent 会更难辨识，需要更丰富的数据与不确定性分析。
- C ≈ 6ND 是稠密模型的粗略计算账本，不等于 wall-clock 时间；固定 compute 约束也不能单独给出 compute-optimal 配置。

从仓库根目录运行：

~~~bash
python code/lecture_09/scaling_laws_lab.py
python -m unittest discover -s tests/lecture_09 -p "test_*.py" -v
~~~

先检查残差与拟合区间，再讨论外推。不得把 synthetic fit 的指数当作 Kaplan/Chinchilla 论文的复现结果。
