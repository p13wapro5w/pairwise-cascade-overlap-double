# pairwise-cascade-overlap-double

## 领域

游戏、图形与多媒体

## 主题

级联重叠双采样

## 说明

分割去重叠、bias 按 texel 与单层采样。

初始状态下 `render/cascade_overlap_double.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
