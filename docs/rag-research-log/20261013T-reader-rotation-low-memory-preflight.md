# Reader 特征旋转：低内存工程预检

这次完成的是加载与验证方式的修复，尚未得到新 idea 的真实题目效果。
量子特征旋转、经典同预算对照和 Top-5 成员保护损失保持原设计；固定
预训练 Reader 的完整预检还没有完成。

## 服务器观察与处理

原绑定服务器曾短暂恢复。实际容器内存上限为1GiB，没有可见 CUDA；
主机显示的空闲内存不能代表容器可用额度。现有 py310 为 Torch2.7.1、
Transformers4.44.0，和之前 GPU 工作环境不同。没有安装包或切换服务器。

缓存 Reader 的旧 `.bin` 不能直接内存映射。新增转换器只用1MiB缓冲区
搬运原 FP32 字节，限定可识别的连续独立存储，不执行任意 pickle 全局。
服务器转换已完成，全部张量重新打开后逐项哈希一致，原 checkpoint 未改动。
两项转换器测试在服务器通过。官方 checkpoint 中两项 pooler 权重在
当前 DPRReader 架构中不使用，现仅允许这两个明确的多余键；其他缺失、
多余或形状不符的活动参数仍阻断运行。

随后普通低内存加载再次遭遇 SSH 超时，没有看到合成题计算完成。仅凭
这些现象无法判断超时原因，更不能声称已发生 OOM。远程进程状态待核查。

## 已完成的代码与验证

受限 CPU 现在可以按实际 DPR 模块加载映射权重：模块执行前载入，执行
后释放编码器参数；分类头与 span 头保留。原 DPR 运算照常执行，未改
精度或数学定义。该路径不依赖 accelerate；普通驻留加载在已有 accelerate
时使用其低内存选项。一键脚本检查现有 Python、Torch 和 NumPy 桥接兼容性。

现有 lambeq 环境13项测试通过，包括转换字节一致、缓存损坏拒绝、真实
DPR 模块驻留/映射结果对照、量子头梯度，以及完整 runner 的训练/评估
与数值回读。驻留和映射的 FP32 算术对照容差为 rtol=1e-6、atol=1e-7，
不能据此说输出逐 bit 相同。

新留存随机单层、768维 DPR 架构 fixture：
`.cache/quantum_reader_rotation_streaming_architecture_20261013_v1/outputs/20261013T055251674685Z/`。
它包含448个完整数组，8次初始/最终头回放，4次优化器更新回放；无标签
Top-5、span 保持、禁用恢复、冻结参数和完整坐标回读通过。
共享完整 v2 校验和 JSON schema 通过，评估不训练产生一项预期提示。
数值归档 SHA256 为 `651afaf6c8bd7ee3a1268a2821f960a8feb007849abf73547969dc6537c92b2c`。
Python 编译、Bash 语法和3插件计划校验通过。所有下载和真实数据拟合均未执行。

随机权重和显式合成 token 只证明工程流程，不能替代固定预训练模型。
计划仍为 approved，固定 Reader readiness 未认证。最终按模块加载实现
尚未同步到重新失联的服务器。

## 下一步与累积经验

前一轮 bounded residual 无法跨越多数错排分差，启发我们改变特征到分数
的形成过程，并按 Top-5 成员保护已选证据；这一研究选择不因服务器故障
改变。本次加载修复也不算它有效的证据。

服务器可达后先查本轮进程与输出，确认实际运行额度，再同步最终实现，
有时限地执行缓存离线 Reader fixture。完整预检通过后再冻结真实实验的
输入、资源预算以及新/旧头和新/旧损失交叉对照，交给合作者跑真实题目。
目前59题228对监督和全部最终裁决保持冻结，不重开标注队列。

复核命令（项目根目录，已有模型缓存）：

```bash
bash scripts/collab/five_ideas/run_quantum_reader_rotation_preflight.sh
```

该命令仅做工程预检，无需18083 token，不下载模型或数据。输出位于项目
`.cache/quantum_reader_rotation_preflight/<timestamp>/`。摘要凭证在
`.ai-progress/workstreams/rag-selector/refs/reader_rotation_low_memory_20261013_receipt.json`。
