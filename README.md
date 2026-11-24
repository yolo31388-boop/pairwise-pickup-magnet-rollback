# pairwise-pickup-magnet-rollback

## 领域

游戏、图形与多媒体

## 主题

掉落物磁吸与拾取回滚

## 说明

扫掠吸附、容量前置与稳定槽位。

初始状态下 `gameplay/pickup_magnet_rollback.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
