#!/bin/bash

# ====================================================================================
#
# 这是一个用于条件化RPG模型训练的两阶段启动脚本。
#
# ====================================================================================

# --- 通用配置 ---
# 您可以根据需要修改这里的 Accelerate 参数
ACCELERATE_ARGS="--main_process_port=0 --num_processes=1 --gpu_ids='0' --num_machines=1 --dynamo_backend=no"

echo "============================================================"
echo "第一步：生成'模型动物园' (如果已生成，可以注释掉此部分)"
echo "============================================================"

# 此步骤会调用我们改造后的通用权重生成脚本，为每个任务训练并保存权重。
# 每个命令都会运行较长时间。建议逐一运行，或者如果您已经生成了权重，可以跳过。

# --- 任务1：为 CIFAR-10 生成 ResNet18 权重 ---
# echo "开始生成 CIFAR-10 权重..."
# accelerate launch ${ACCELERATE_ARGS} ./dataset/cifar10_resnet18/train.py \
#   --dataset CIFAR10 \
#   --model_arch resnet18 \
#   --tag cifar10_resnet18 \
#   --epochs 50 

# --- 任务2：为 CIFAR-100 生成 ResNet18 权重 ---
# echo "开始生成 CIFAR-100 权重..."
# accelerate launch ${ACCELERATE_ARGS} ./dataset/cifar10_resnet18/train.py \
#   --dataset CIFAR100 \
#   --model_arch resnet18 \
#   --tag cifar100_resnet18 \
#   --epochs 50

# --- 任务3：为 SVHN 生成 ResNet18 权重 ---
# echo "开始生成 SVHN 权重..."
# accelerate launch ${ACCELERATE_ARGS} ./dataset/cifar10_resnet18/train.py \
#   --dataset SVHN \
#   --model_arch resnet18 \
#   --tag svhn_resnet18 \
#   --epochs 50


echo "============================================================"
echo "第二步：训练条件化的 RPG 主模型"
echo "============================================================"
echo "确保'模型动物园'已生成，并且主 train.py 中的配置已更新。"

# 此步骤将启动我们最终的主模型训练。
# 它会加载 ModelZooDataset，并使用 DatasetConditionMambaDiffusion 模型进行训练。
accelerate launch ${ACCELERATE_ARGS} ./example/cifar10_resnet18.py

echo "所有流程执行完毕!"