# Lecture 3 教学实现

## 内容

- tiny_transformer.py：最小 decoder-only Transformer，显式暴露 causal attention 权重。
- 不是官方 staff implementation，不是生产级模型，不用于性能对标。

## 运行

从仓库根目录运行：

~~~bash
python code/lecture_03/tiny_transformer.py
python -m unittest discover -s tests/lecture_03 -p 'test_*.py' -v
~~~

需要 PyTorch。若要运行 Notebook 的图示，还需要 matplotlib 和 Jupyter。

## 当前验收边界

当前环境为 PyTorch CPU build。测试验证张量形状、causal mask、未来 token 不泄漏、next-token loss 和反向传播；不代表 GPU 性能已验证。
