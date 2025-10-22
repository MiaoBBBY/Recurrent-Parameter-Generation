# evaluate.py
import torch
import torch.nn as nn
import argparse
import importlib.util
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from tqdm.auto import tqdm

# --- 1. 动态模型加载函数 ---
def get_model_from_arch_string(arch_string, num_classes):
    """根据arch字符串动态加载timm模型或自定义模型"""
    model_source, model_path_or_name = arch_string.split(':', 1)
    
    if model_source == 'timm':
        import timm # 仅在需要时导入
        print(f"Loading reference model '{model_path_or_name}' from timm library...")
        model = timm.create_model(model_path_or_name, num_classes=num_classes)
        return model, None
    
    elif model_source == 'custom':
        print(f"Loading custom reference model from path: {model_path_or_name}")
        spec = importlib.util.spec_from_file_location("custom_model_module", model_path_or_name)
        custom_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_model_module)
        # 假设您的 model.py 文件中有一个名为 Model 的函数
        model, head = custom_model_module.Model(num_classes=num_classes)
        return model, head
    else:
        raise ValueError(f"Unknown model source in TEST_CONFIG: {model_source}")

# --- 2. 动态数据加载函数 (修正版) ---
def get_test_loader(dataset_name, dataset_root, model_arch, batch_size=500, num_workers=8):
    """根据模型架构动态决定图像尺寸"""
    # 【核心修正】根据 model_arch 字符串来判断图像尺寸
    image_size = 32 if 'cnnsmall' in model_arch else 64
    print(f"--- Preparing data with image size: {image_size}x{image_size} ---")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_map = {'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100, 'SVHN': SVHN}
    DatasetClass = dataset_map[dataset_name]
    
    kwargs = {'root': dataset_root, 'download': False, 'transform': transform}
    if dataset_name == 'SVHN':
        kwargs['split'] = 'test'
    else:
        kwargs['train'] = False
        
    test_set = DatasetClass(**kwargs)
    return DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# --- 3. 评估函数 (清洁版) ---
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0, 0, 0
    
    for inputs, targets in tqdm(test_loader, desc=f"Evaluating on {device}", leave=False):
        if isinstance(targets, tuple): targets = targets[0]
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        test_loss += loss.item()
        _, predicts = outputs.max(1)
        total += targets.size(0)
        correct += predicts.eq(targets).sum().item()
        
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    
    print("-" * 50)
    print(f"Evaluation Result for {args.model_arch}:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({correct} / {total})")
    print("-" * 50)

# --- 4. 主执行逻辑 (修正版) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal Model Evaluator')
    parser.add_argument('--model_arch', type=str, required=True, help="Model architecture string (e.g., 'timm:resnet18' or 'custom:path/to/model.py')")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of output classes for the model")
    parser.add_argument('--weights_path', type=str, required=True, help="Path to the generated .pth state_dict file")
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'SVHN'], help="Dataset to evaluate on")
    parser.add_argument('--dataset_root', type=str, default="/home/ubuntu/Desktop/study/RPG-DNNWG/Dataset", help="Root directory of datasets")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建模型
    print(f"Creating model architecture...")
    model, _ = get_model_from_arch_string(args.model_arch, args.num_classes)
    
    # 2. 加载权重
    print(f"Loading weights from {args.weights_path}...")
    state_dict = torch.load(args.weights_path, map_location='cpu',weights_only=True)
    model.load_state_dict(state_dict)
    
    # 3. 准备数据 (将 model_arch 传入)
    print(f"Preparing test data for {args.dataset}...")
    test_loader = get_test_loader(args.dataset, args.dataset_root, args.model_arch)
    
    # 4. 执行评估
    evaluate(model, test_loader, device)