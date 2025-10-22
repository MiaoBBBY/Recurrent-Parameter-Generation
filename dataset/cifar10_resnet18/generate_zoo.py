
# set global seed
import random
import numpy as np
import torch
seed = SEED = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# import
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN # 导入多个数据集类
from tqdm.auto import tqdm
import os
import warnings
import argparse
import json
import timm

warnings.filterwarnings("ignore", category=UserWarning)

import importlib.util

# --- 命令行参数解析 (新增 --custom_model_path 和 --optimizer) ---
# --- 命令行参数解析 (新增 --image_size) ---
parser = argparse.ArgumentParser(description='Generic Model Zoo Trainer')
parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'SVHN'], help='Dataset to use')
parser.add_argument('--model_arch', type=str, default=None, help='Model architecture from timm')
parser.add_argument('--custom_model_path', type=str, default=None, help='Path to custom model definition .py file')
parser.add_argument('--image_size', type=int, default=64, help='Image size for transforms') # <--- 新增
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training')
parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'], help='Optimizer to use')
parser.add_argument('--tag', type=str, required=True, help='A unique tag for this run (e.g., cifar10_cnnsmall)')
args = parser.parse_args()

# load additional config for root paths
config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_file, "r") as f:
    additional_config = json.load(f)

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print("dataset_root: ", additional_config["dataset_root"])
print("model_zoo_root: ", additional_config["model_zoo_root"])

config = {
    "dataset_root": additional_config["dataset_root"],
    "model_zoo_root": additional_config["model_zoo_root"], # 新增：模型动物园的根目录
    "batch_size": args.batch_size,
    "num_workers": 32,
    "learning_rate": args.lr,
    "weight_decay": 0.1,
    "epochs": args.epochs,
    "save_learning_rate": 1e-5,
    "total_save_number": 50,
    "tag": args.tag,
}

# --- 数据集和模型配置 ---
DATASET_CONFIG = {
    'CIFAR10': {'class': CIFAR10, 'num_classes': 10, 'transform_policy': 'cifar10'},
    'CIFAR100': {'class': CIFAR100, 'num_classes': 100, 'transform_policy': 'cifar10'},
    'SVHN': {'class': SVHN, 'num_classes': 10, 'transform_policy': 'svhn'},
}
selected_dataset_config = DATASET_CONFIG[args.dataset]
num_classes = selected_dataset_config['num_classes']
DatasetClass = selected_dataset_config['class']

# --- 数据变换 (使用 args.image_size) ---
print(f"==> Preparing data for {args.dataset} with image size {args.image_size}x{args.image_size}...")
train_transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.RandomCrop(args.image_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy(selected_dataset_config['transform_policy'])),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 根据数据集名称动态实例化
dataset_kwargs = {'root': config["dataset_root"], 'download': True}
if args.dataset == 'SVHN':
    dataset_kwargs['split'] = 'train'
    train_set = DatasetClass(transform=train_transform, **dataset_kwargs)
    dataset_kwargs['split'] = 'test'
    test_set = DatasetClass(transform=test_transform, **dataset_kwargs)
else:
    dataset_kwargs['train'] = True
    train_set = DatasetClass(transform=train_transform, **dataset_kwargs)
    dataset_kwargs['train'] = False
    test_set = DatasetClass(transform=test_transform, **dataset_kwargs)


train_loader = DataLoader(
    dataset=train_set, batch_size=config["batch_size"], num_workers=config["num_workers"],
    shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True
)
test_loader = DataLoader(
    dataset=test_set, batch_size=config["batch_size"], num_workers=config["num_workers"],
    shuffle=False, pin_memory=True, persistent_workers=True
)

# --- 3. 修改：动态模型创建 ---
print(f"==> Building model...")
if args.custom_model_path:
    print(f"Loading custom model from: {args.custom_model_path}")
    spec = importlib.util.spec_from_file_location("custom_model", args.custom_model_path)
    custom_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_model_module)
    model, _ = custom_model_module.Model(num_classes=num_classes)
else:
    print(f"Loading model '{args.model_arch}' from timm for {num_classes} classes...")
    model = timm.create_model(args.model_arch, pretrained=False, num_classes=num_classes)

model = model.to(device)
criterion = nn.CrossEntropyLoss()

# --- 动态优化器创建 ---
if args.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# (训练和测试函数 train(), test() 保持不变)
def train(model=model, optimizer=optimizer, scheduler=scheduler):
    # ... (这部分代码无需修改)
    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),
                                             total=len(train_loader.dataset) // config["batch_size"]):
        # SVHN数据集返回的targets可能格式不同，进行适配
        if isinstance(targets, tuple): targets = targets[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()

# 在您的“通用权重生成脚本”中

@torch.no_grad()
def test(model=model):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    # 我们只测试一个batch来快速定位问题
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # --- [DEBUG] 打印从DataLoader出来的原始形状 ---
        print("\n" + "="*50)
        print(f"--- Debugging Batch {batch_idx} ---")
        print(f"[DEBUG] Shape of 'inputs' from DataLoader: {inputs.shape}")
        print(f"[DEBUG] Shape of 'targets' from DataLoader: {targets.shape}")
        print("="*50 + "\n")

        # SVHN数据集返回的targets可能格式不同，进行适配
        if isinstance(targets, tuple): targets = targets[0]
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 在调用模型之前，再次确认形状
        print(f"[DEBUG] Shape of 'inputs' right before model call: {inputs.shape}")

        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            outputs = model(inputs)
        
        # --- [DEBUG] 打印从模型出来的输出形状 ---
        print(f"[DEBUG] Shape of 'outputs' right after model call: {outputs.shape}\n")
        
        # 程序应该会在这里报错
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicts = outputs.max(1)
        # 这里我们需要处理批次大小不匹配的问题，以便看到准确率
        # 临时修改：如果批次大小不匹配，我们只评估前500个样本
        if outputs.shape[0] != targets.shape[0]:
            print("[DEBUG] Output and target batch sizes mismatch. Evaluating on a subset.")
            # 假设TTA将结果堆叠，我们取每个增强样本的第一个
            step = outputs.shape[0] // targets.shape[0]
            predicts = predicts[::step]

        total += targets.size(0)
        correct += predicts.eq(targets).sum().item()
        
        # 只运行一个batch用于调试
        break 

    loss = test_loss / (batch_idx + 1)
    acc = correct / total
    print(f"Loss: {loss:.4f} | Acc: {acc:.4f}\n")
    model.train()
    return loss, acc


# --- 4. 修改：保存到指定的 "动物园" 目录 ---
def save_train(model=model, optimizer=optimizer):
    model.train()
    # 创建本次运行专属的 checkpoint 保存目录
    save_dir = os.path.join(config["model_zoo_root"], config["tag"])
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # 确定保存间隔
    total_batches = len(train_loader.dataset) // train_loader.batch_size
    save_interval = total_batches // config["total_save_number"]
    if save_interval == 0: save_interval = 1

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if isinstance(targets, tuple): targets = targets[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # Save checkpoint
        if batch_idx % save_interval == 0:
            _, acc, = test(model=model)
            save_path = os.path.join(save_dir, f"{str(batch_idx).zfill(4)}_acc{acc:.4f}_seed{seed:04d}.pth")
            save_state = {key: value.cpu().to(torch.float32) for key, value in model.state_dict().items()}
            torch.save(save_state, save_path)
            print(f"Saved checkpoint: {save_path}")

# main
if __name__ == '__main__':
    if not (args.custom_model_path or args.model_arch):
        raise ValueError("Either --model_arch (for timm) or --custom_model_path must be provided.")
    # ... (主执行逻辑不变) ...
    test(model=model)
    for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
        train(model=model, optimizer=optimizer, scheduler=scheduler)
        test(model=model)
    
    print("\nStarting final save training phase...")
    save_train(model=model, optimizer=optimizer)