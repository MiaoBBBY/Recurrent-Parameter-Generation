import sys, os, json
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)
with open("./workspace/config.json", "r") as f:
    additional_config = json.load(f)
USE_WANDB = additional_config["use_wandb"]

from accelerate import Accelerator
accelerator = Accelerator()
print(f"--- 00000000000000000000000000000000000000Manually moved optimizer to device: {accelerator.device} ---")

# set global seed
import random
import numpy as np
import torch
import timm # --- NEW: 引入 timm 用于测试 ---

seed = SEED = 999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# other
import warnings
from _thread import start_new_thread
warnings.filterwarnings("ignore", category=UserWarning)
if USE_WANDB: import wandb
# torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# --- MODIFIED: 导入我们所有的新模块 ---
from dataset.model_zoo_dataset import ModelZooDataset as Dataset 
from model import DatasetConditionMambaDiffusion as Model
from torch.utils.data import DataLoader
from model.diffusion import DDPMSampler, DDIMSampler
EST_CONFIG = {
    'cifar10_resnet18': {
        'test_script': './dataset/cifar10_resnet18/test.py',
        'model_arch': 'resnet18', # 用于在测试时创建参考模型
        'num_classes': 10,
        'dataset_name': 'CIFAR10' # 用于从ModelZooDataset获取条件特征
    },
    'cifar100_resnet18': {
        'test_script': './dataset/cifar100_resnet18bn/test.py', # 确保路径正确
        'model_arch': 'resnet18',
        'num_classes': 100,
        'dataset_name': 'CIFAR100'
    },
    'svhn_resnet18': {
        'test_script': './dataset/cifar10_resnet18/test.py', # SVHN是10分类, 可复用cifar10的测试脚本
        'model_arch': 'resnet18',
        'num_classes': 10,
        'dataset_name': 'SVHN'
    },
}

# --- MODIFIED: 全新的 config 字典 ---
config = {
    "seed": SEED,
    # --- 数据集设置 (使用新数据集) ---
    "dataset": Dataset,
    "dataset_params": {
        "model_zoo_root": additional_config["model_zoo_root"],
        "dataset_root": additional_config["dataset_root"],
        "dim_per_token": 2048,
        "num_classes_per_set": 10,
        "num_samples_per_class": 5,
        "clip_model_name": "openai/clip-vit-base-patch32",
    },
    
# ...
    "cond_stage_config": {
        "target": "model.EmbedData",
        "params": {
            "ckpt_path": None, 
            "enconfig": {"dim_input": 768, "num_inds": 32, "dim_hidden": 128, "num_heads": 2, "ln": False}, # <--- 将 512 修改为 768
            "deconfig": {"num_outputs": 1, "dim_output": 512, "dim_hidden": 128, "num_heads": 2, "ln": False}
        }
    },
# ...


    # --- 训练设置 ---
    "batch_size": 4, # 由于CLIP和SetTransformer会增加内存消耗，建议减小batch size
    "num_workers": 1,
    "total_steps": 80000,
    "learning_rate": 0.00003,
    "weight_decay": 0.0,
    "save_every": 80000 // 20, # 增加保存和测试的频率
    "print_every": 50,
    "checkpoint_save_path": "./checkpoint",

    # --- 模型设置 (MambaDiffusion) ---
    "model_config": {
        "num_permutation": 1, # 条件化模型不再需要permutation
        "d_condition": 1, # 此参数在我们的新模型中不再直接使用
        "d_model": 2048,
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        "diffusion_batch": 512,
        "layer_channels": [1, 32, 64, 128, 64, 32, 1],
        "model_dim": 2048,
        "condition_dim": 2048,
        "kernel_size": 7,
        "T": 1000,
        "beta": (0.0001, 0.02),
        "sample_mode": DDPMSampler,
        
        "forward_once": True,
    },
    "tag": "conditional_rpg_cifar_zoo",
}
print(f"--- 11111111111111111111111111111111111111Manually moved optimizer to device: {accelerator.device} ---")

# --- MODIFIED: 数据加载 ---
print('==> Preparing data from Model Zoo...')
train_set = config["dataset"](**config["dataset_params"])
config["sequence_length"] = train_set.sequence_length
print(f"Sequence length from dataset: {config['sequence_length']}")
print(f"--- 22222222222222222222222222222222222222Manually moved optimizer to device: {accelerator.device} ---")

train_loader = DataLoader(
    dataset=train_set, batch_size=config["batch_size"], num_workers=config["num_workers"],
    persistent_workers=False, drop_last=True, shuffle=True
)
print(f"--- 33333333333333333333333333333333333333Manually moved optimizer to device: {accelerator.device} ---")

# --- MODIFIED: 模型实例化 ---
print('==> Building Dataset-Conditioned model..')
Model.config = config["model_config"]
positional_embedding = train_set.get_position_embedding(positional_embedding_dim=config["model_config"]["d_model"])
model = Model(
    sequence_length=config["sequence_length"],
    positional_embedding=positional_embedding,
    cond_stage_config=config["cond_stage_config"],
    cond_stage_trainable=True # 设为True, 表示将与主模型一起从头训练
)

# --- Optimizer, Scheduler, Accelerator 设置 (保持不变) ---
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0))
scheduler = CosineAnnealingLR(optimizer, T_max=config["total_steps"])
print(f"--- 44444444444444444444444444444444444444Manually moved optimizer to device: {accelerator.device} ---")
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)


model.to(accelerator.device)
print(f"--- Manually moved model to device: {accelerator.device} ---")
# ... (Wandb 设置保持不变) ...

# --- MODIFIED: 训练循环 ---
def train():
    print("==> Start conditional training..")
    model.train()
    current_step = 0
    done = False
    # while not done:
    while not done:
        for batch_idx, (param, dataset_features) in enumerate(train_loader):
            # --- 新增：在调用模型前，显式地将所有数据移动到GPU ---
            device = accelerator.device
            param = param.to(device)
            dataset_features = dataset_features.to(device)
            # --- 修改结束 ---
            # print(f"\n[DEBUG train.py] param device: {param.device}, dataset_features device: {dataset_features.device}")

            optimizer.zero_grad()
            with accelerator.autocast():
                # 现在传入的 param 和 dataset_features 都已确保在GPU上
                loss = model(output_shape=param.shape, x_0=param, condition=dataset_features)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step(current_step)
            
            if accelerator.is_main_process:
                if current_step % config["print_every"] == 0:
                    print(f"Step {current_step}, Loss: {loss.item()}")
                # 定期保存和评估
                if current_step > 0 and current_step % config["save_every"] == 0:
                    print("\nSaving checkpoint and running evaluation...")
                    save_dir = config["checkpoint_save_path"]
                    os.makedirs(save_dir, exist_ok=True)
                    unwrapped_model_state = accelerator.unwrap_model(model).state_dict()
                    save_path = os.path.join(config["checkpoint_save_path"], f"{config['tag']}_step_{current_step}.pth")
                    torch.save(unwrapped_model_state, save_path)
                    print(f"Saved main model to {save_path}")
                    test_all_tasks() # 调用新的测试函数
            current_step += 1
            if current_step >= config["total_steps"]:
                done = True
                break

# --- NEW: 全新的生成和测试函数 ---
def generate_and_test(task_tag):
    print(f"\n==> Generating and Testing for task: {task_tag}")
    model.eval()
    
    task_info = TEST_CONFIG.get(task_tag)
    if not task_info:
        print(f"Warning: Task '{task_tag}' not found in TEST_CONFIG. Skipping test.")
        return

    print("==> Preparing condition features...")
    condition_features = train_set.get_features_for_task(task_info['dataset_name']).unsqueeze(0)
    
    print("==> Generating weights...")
    with torch.no_grad():
        unwrapped_model = accelerator.unwrap_model(model)
        prediction_chunked = unwrapped_model(sample=True, condition=condition_features)
    
    print("==> Post-processing weights to state_dict...")
    # 关键: 复用 train_set 的 postprocess 方法来反归一化和反分块
    train_set.structure = train_set.structures[task_tag] # 设置正确的structure
    generated_state_dict = train_set.postprocess(prediction_chunked)

    save_dir = "./generated_weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_tag}_generated.pth")
    torch.save(generated_state_dict, save_path)
    print(f"==> Saved generated state_dict to {save_path}")

    test_script_path = task_info['test_script']
    if os.path.exists(test_script_path):
        test_command = f"python {test_script_path} {save_path}"
        print(f"==> Executing test command: {test_command}")
        os.system(test_command)
    else:
        print(f"Warning: Test script not found at {test_script_path}. Skipping execution.")
    
    model.train() # 恢复训练模式

def test_all_tasks():
    print("\n\n========================================================")
    print("      STARTING EVALUATION ON ALL CONFIGURED TASKS      ")
    print("========================================================")
    for task_tag in TEST_CONFIG.keys():
        generate_and_test(task_tag)

# --- MODIFIED: 主执行逻辑 ---
if __name__ == '__main__':
    train() # 执行完整的训练
    
    if accelerator.is_main_process:
        print("\nFinal evaluation after training completion...")
        test_all_tasks() # 训练结束后，对所有已定义的任务进行最终的生成和测试

    print("Finished All Processes!")
    exit(0)