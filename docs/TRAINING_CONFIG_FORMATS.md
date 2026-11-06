# 训练配置格式

统一训练命令`milplus run --config PATH`接受JSON、YAML配置。
`.json`和`.JSON`使用标准JSON解析器；其他后缀按YAML解析。
两种格式的顶层都必须是映射。省略配置和空YAML保持默认设置；
JSON的`null`、数组及标量会在训练开始前被拒绝。

JSON科学计数法保存为数值，例如：

```json
{
  "epochs": 100,
  "lr": 2e-4,
  "weight_decay": 1e-05,
  "max_instances": null
}
```

这些数值仅示范文件格式，具体训练配置应遵循选用方法的协议。
该示例不表示所有方法采用100轮或相同优化器。

`milplus run-matrix --config PATH`将同一个配置传给各训练子进程，
因此使用相同的格式规则。AKD-PMP的CPU启动策略和DSPA专用子进程
也使用同一读取器。方法输入清单和数据准备schema仍遵循各自协议。

此前把JSON文件交给YAML解析器时，`1e-05`可能变为字符串，
导致优化器初始化失败。原失败日志已保留；本修复保持数值类型，
并通过实际训练、断点恢复和批量调度验证。
