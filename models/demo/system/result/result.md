## 模型对比

| 阶段     | 模型                    | 数据集   | acc  | mae  | 权重目录                 | 测试结果文件               |
|--------|-----------------------|-------|------|------|----------------------|----------------------|
| recall | dssm + dcn (stacked)  | ml-1m | 0.36 | 1.08 | output_model_recall  | recall_infer_result  |
| recall | dssm + dcn (parallel) | ml-1m | 0.30 | 0.91 | output_model_recall1 | recall_infer_result1 |
