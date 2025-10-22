import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from transformers import CLIPProcessor, CLIPVisionModel
from torch.nn import functional as F
import numpy as np

# 从您原有的 dataset.py 中导入 BaseDataset
from .dataset import BaseDataset

class ModelZooDataset(BaseDataset):
    def __init__(self, model_zoo_root, dataset_root, dim_per_token, 
                 arch_tag=None, # <--- 关键：新增架构标签过滤器
                 num_classes_per_set=10, num_samples_per_class=5, 
                 clip_model_name="openai/clip-vit-base-patch32", **kwargs):
        
        print("="*50)
        print("==> Initializing ModelZooDataset (Filterable Version)...")
        # --- 新增：根据 arch_tag 打印提示信息 ---
        if arch_tag:
            print(f"==> [IMPORTANT] Loading weights ONLY for architecture tag: '{arch_tag}'")
        else:
            print("==> [INFO] No arch_tag provided. Loading ALL weights from the zoo.")
        print("="*50)

        self.model_zoo_root = model_zoo_root
        self.dataset_root = dataset_root
        self.dim_per_token = dim_per_token
        self.arch_tag = arch_tag # 保存过滤器标签
        self.num_classes = num_classes_per_set
        self.num_samples = num_samples_per_class
        
        self.config = self.config.copy()
        self.config.update(kwargs)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, use_safetensors=True)
        self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True).to(self.device).eval()
        self.clip_feature_dim = self.clip_vision_model.config.hidden_size

        self.dataset_map = {
            'cifar10': {'class': CIFAR10, 'split': 'train'},
            'cifar100': {'class': CIFAR100, 'split': 'train'},
            'svhn': {'class': SVHN, 'split': 'train'},
        }

        # 构建 manifest (现在会使用 arch_tag 进行过滤)
        self.manifest = self._build_manifest()
        print(f"==> Found {len(self.manifest)} model weight files matching the filter.")
        
        # 为过滤出的任务计算 structure
        self.structures = {}
        all_task_tags = self.get_all_task_tags()
        if not all_task_tags:
            raise RuntimeError(f"Model Zoo is empty for arch_tag '{arch_tag}' in {self.model_zoo_root}")
        
        for task_tag in all_task_tags:
            print(f"==> Getting structure for task: {task_tag}")
            self.checkpoint_list = [item['weight_path'] for item in self.manifest if item['task_tag'] == task_tag]
            self.structures[task_tag] = self.get_structure()
        
        # 计算序列长度 (因为只处理同一种架构，所以所有样本长度都一样，不再需要计算最大值)
        self.structure = self.structures[all_task_tags[0]] 
        self.sequence_length = self.get_sequence_length()
        print(f"==> Sequence length for this architecture is: {self.sequence_length}")

        self.checkpoint_list = [item['weight_path'] for item in self.manifest]
        self.length = len(self.manifest)
        self.real_length = self.length
        
        print("==> ModelZooDataset initialization complete.")

    def _build_manifest(self):
        manifest = []
        # 如果指定了 arch_tag, 只扫描对应的子目录, 否则扫描全部
        dirs_to_scan = [self.arch_tag] if self.arch_tag else os.listdir(self.model_zoo_root)
        
        for task_dir in dirs_to_scan:
            task_path = os.path.join(self.model_zoo_root, task_dir)
            if not os.path.isdir(task_path): continue
            
            dataset_name = task_dir.split('_')[0].lower()
            if dataset_name not in self.dataset_map: continue
            
            weight_files = glob.glob(os.path.join(task_path, '*.pth'))
            for weight_path in weight_files:
                manifest.append({
                    'weight_path': weight_path,
                    'dataset_name': dataset_name,
                    'task_tag': task_dir
                })
        return manifest
        
    def __getitem__(self, index):
        index = index % self.real_length
        sample_info = self.manifest[index]
        
        # --- 修复您添加的打印语句 ---
        # print(f"[Verification] Loading sample {index}, task_tag is: {sample_info['task_tag']}")
        
        diction = torch.load(sample_info['weight_path'], map_location='cpu')
        
        self.structure = self.structures[sample_info['task_tag']]
        param_tensor_chunked = self.preprocess(diction)

        # 因为现在只加载同一种架构，理论上长度都相等，无需填充
        # 但保留填充逻辑可以增加代码的稳健性
        if param_tensor_chunked.shape[0] < self.sequence_length:
            pad_len = self.sequence_length - param_tensor_chunked.shape[0]
            padding = torch.full((pad_len, self.dim_per_token), fill_value=self.config["fill_value"])
            param_tensor_chunked = torch.cat([param_tensor_chunked, padding], dim=0)

        dataset_features = self._extract_features_for_name(sample_info['dataset_name'])
        
        return param_tensor_chunked, dataset_features

    # ... (get_all_task_tags, get_features_for_task, _extract_features_for_name 等函数保持不变) ...

    def get_all_task_tags(self):
        """获取模型动物园中所有唯一的任务标签。"""
        if not hasattr(self, 'manifest') or not self.manifest: return []
        return sorted(list(set(item['task_tag'] for item in self.manifest)))
        
    def __len__(self):
        return self.length

    @torch.no_grad()
    def get_features_for_task(self, dataset_name):
        """按需为指定的单个任务生成条件特征，供测试时使用。"""
        return self._extract_features_for_name(dataset_name)

    @torch.no_grad()
    def _extract_features_for_name(self, dataset_name):
        """内部辅助函数，加载指定的数据集并提取CLIP特征。"""
        dataset_info = self.dataset_map[dataset_name.lower()]
        DatasetClass = dataset_info['class']
        split = dataset_info.get('split', 'train')
        
        try:
            raw_dataset = DatasetClass(root=self.dataset_root, split=split, download=False)
        except TypeError: # 兼容 CIFAR 等没有 split 参数的数据集
            raw_dataset = DatasetClass(root=self.dataset_root, train=(split == 'train'), download=False)
        
        # 按类别组织数据索引
        class_indices = {}
        targets = np.array(raw_dataset.targets if hasattr(raw_dataset, 'targets') else raw_dataset.labels)
        
        for i, label in enumerate(targets):
            label = int(label) # 确保标签是整数
            if label not in class_indices: 
                class_indices[label] = []
            class_indices[label].append(i)

        unique_classes = list(class_indices.keys())
        
        # 随机选择 num_classes 个类别, 如果总类别数不足则有放回地抽样
        replace_classes = len(unique_classes) < self.num_classes
        selected_classes = np.random.choice(unique_classes, size=self.num_classes, replace=replace_classes)
        
        images_to_process = []
        for cls in selected_classes:
            # 从每个选定类别中随机选择 num_samples 个样本, 如果样本数不足则有放回地抽样
            replace_samples = len(class_indices[cls]) < self.num_samples
            indices = np.random.choice(class_indices[cls], size=self.num_samples, replace=replace_samples)
            for idx in indices:
                image, _ = raw_dataset[idx]
                images_to_process.append(image)
        
        # 使用 CLIP 预处理器处理所有选定的图像
        inputs = self.clip_processor(images=images_to_process, return_tensors="pt").to(self.device)
        
        # 提取特征
        features = self.clip_vision_model(**inputs).pooler_output
        
        # 重塑为 (C, K, D) 的形状并移回CPU
        features = features.view(self.num_classes, self.num_samples, self.clip_feature_dim).cpu()
        
        return features