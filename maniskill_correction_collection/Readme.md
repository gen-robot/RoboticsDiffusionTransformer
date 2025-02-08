## Maniskill Correction

#### 环境安装

1. 按照 RDT 方式先安装环境。
2. 安装maniskill

```bash
pip install mani_skill==3.0.0b18
```

#### 修正数据采集

测试环境: StackCube-v1
任务要求：将红色方块堆叠到绿色方块上方。

原指令: "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling."
修正指令: "Mission failure detected! Re-pick up the red cube first. After that continue to tack it on top of a green cube."

修正动作：停下当前在执行的任何事，重新夹起红色方块。

修正过程:
- 正常执行原指令后的 96 步开始进行错误检测，之后每隔 32 步进行一次错误检测。
- 对于每次错误检测，检查方块没有被夹起或者已经被放置到绿色方块上。如果检测到未成功，则执行修正动作。

#### 训练

1. 需要将 data/hdf5_maniskill_dataset.py 中 data_dir 变量改成本地的训练数据目录。
2. 训练 command :

```bash
bash finetune_robotwin_correction.sh {lr} {type}
```

- lr: learning rate, 训练使用 1e-4
- type: 训练模式，共 4 种
    - original: 不含任何修正数据
    - mix: 包含原数据和完整修正 demo 的数据
    - only_correction: 包含原数据和单独修正动作的数据
    - all: 包含原数据，完整修正 demo 和单独修正动作的数据

#### RDT released model 测试

RDT 原论文测试了 ManiSkill 环境中的五个任务 PegInsertionSide，PickCube，StackCube，PlugCharger 和 PushCube。这五个任务的结果可以使用以下命令进行测试。所有命令将会无并行的执行 1000 次测试。

```bash
python maniskill_correction_collection/eval.py -b gpu --pretrained_path {path/to/dir}/rdt-released.pt -e PegInsertionSide-v1
python maniskill_correction_collection/eval.py -b gpu --pretrained_path {path/to/dir}/rdt-released.pt -e PickCube-v1
python maniskill_correction_collection/eval.py -b gpu --pretrained_path {path/to/dir}/rdt-released.pt -e StackCube-v1
python maniskill_correction_collection/eval.py -b gpu --pretrained_path {path/to/dir}/rdt-released.pt -e PlugCharger-v1
python maniskill_correction_collection/eval.py -b gpu --pretrained_path {path/to/dir}/rdt-released.pt -e PushCube-v1
```

结果存储在 `/outs/render/gpu/multi-task/{env}-original/` 中。
以下为 RDT 论文给出的成功率：

||PegInsertionSide|PickCube|StackCube|PlugCharger|PushCube|Mean|
|---|---|---|---|---|---|---|
|RDT|**13.2±0.29%**|**77.2±0.48%**|74.0±0.30%|**1.2±0.07%**|**100±0.00%**|**53.6±0.52%**|

#### 带修正指令的测试

目前带修正指令的测试仅支持 StackCube 任务。

测试命令：
```bash
python maniskill_correction_collection/eval.py -b gpu "--correction" --pretrained_path {Pretrained model path.} --type {Save result name.}

e.g.
python maniskill_correction_collection/eval.py -b gpu --correction --pretrained_path {path/to/dir}/rdt-maniskill-only_correction-125000.pt --type test
```

1. **-b gpu**: Maniskill 模拟使用 gpu 计算。
2. **--correction**: 是否给修正指令。如果加上`--correction`，则会进行错误检测，检测到错误后给修正指令进行修正。若不加，则不会进行错误检测和修正。
3. **--pretrained_path**: 预训练模型路径。其中包括(-xxx表示训练的iter数量)：
    - **rdt-released.pt**: RDT 论文开源出来的原始模型。
    - **rdt-maniskill-original-xxx.pt**: 在开源模型的基础上不增加数据继续训练的模型。
    - **rdt-maniskill-only_correction-xxx.pt**: 在开源模型的基础上增加单独修正动作的数据（1000条，xx_demo表示更少的数据量）后继续训练的模型。
    - **rdt-maniskill-mix{-dagger}-xxx.pt**: 在开源模型的基础上增加完整修正 demo (如果带-dagger，则使用从开始修正到任务结束作为完整 demo。若不带，则为从任务开始到结束包含修正动作作为完整 demo。)继续训练的模型。
    - **rdt-maniskill-all{-dagger}-xxx.pt**在开源模型的基础上增加完整修正 demo 和单独修正动作的数据后继续训练的模型。
4. **--type**: 表示输出结果的名字。结果存储在 `/outs/render/gpu/multi-task/StackCube-v1-{type}/` 中。