import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", message=".*Detected no triton.*", category=UserWarning)

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim_torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# 从 models.py 和 data.py 导入必要的组件
from models import TransformerTorch
from data import grokking_data_torch


# ==============================================================================
# 以下是 TorchTrainer 类和 run_single_experiment 函数的复制
# 它们与 main.py 中的定义完全相同，以确保 MXmain.py 的独立性
# ==============================================================================

class TorchTrainer:
    """
    A parallel trainer that replicates the MLX training flow using PyTorch.
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: optim_torch.Optimizer,
                 classification: bool = False,
                 batch_size: int = 64,
                 device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.classification = classification
        self.batch_size = batch_size

        if classification:
            self.loss_fn = F.cross_entropy
        else:
            self.loss_fn = F.mse_loss

        self.train_error_trace = []
        self.train_acc_trace = []
        self.val_error_trace = []
        self.val_acc_trace = []

    def _make_batches(self, X_torch, T_torch):
        bs = self.batch_size if self.batch_size != -1 else X_torch.shape[0]
        for i in range(0, X_torch.shape[0], bs):
            yield X_torch[i:i + bs], T_torch[i:i + bs]

    def train(self, train_data, val_data, epochs=5, shuffle=True, scheduler=None):
        self.model.train()
        Xtrain_t, Ttrain_t = train_data
        Xtest_t, Ttest_t = val_data

        global_step = 0

        # Basic epoch loop
        epoch_bar = tqdm(range(epochs), desc='Training', unit='epoch', leave=False)
        for _ in epoch_bar:
            self.model.train()
            if shuffle:
                permutation = torch.randperm(Xtrain_t.size(0))
                Xtrain_t = Xtrain_t[permutation]
                Ttrain_t = Ttrain_t[permutation]

            total_loss = 0.0
            total_correct = 0
            for Xb, Tb in self._make_batches(Xtrain_t, Ttrain_t):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(Xb)
                loss = self.loss_fn(outputs, Tb)
                loss.backward()
                self.optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                global_step += 1

                total_loss += loss.item() * Xb.size(0)
                if self.classification:
                    preds = torch.argmax(outputs, dim=1)
                    total_correct += (preds == Tb).sum().item()

            avg_train_loss = total_loss / Xtrain_t.shape[0]
            if self.classification:
                avg_train_acc = total_correct / Xtrain_t.shape[0]
            else:
                avg_train_acc = 0.0

            self.train_error_trace.append(avg_train_loss)
            self.train_acc_trace.append(avg_train_acc)

            # Evaluate
            avg_val_loss, avg_val_acc = self.evaluate((Xtest_t, Ttest_t))
            self.val_error_trace.append(avg_val_loss)
            self.val_acc_trace.append(avg_val_acc)

            postfix = {
                'train_loss': f'{avg_train_loss:.3f}',
                'train_acc': f'{avg_train_acc:.3f}',
                'val_loss': f'{avg_val_loss:.3f}',
                'val_acc': f'{avg_val_acc:.3f}',
            }
            epoch_bar.set_postfix(postfix)

    def evaluate(self, test_data):
        self.model.eval()
        Xtest_t, Ttest_t = test_data
        total_loss, total_correct = 0.0, 0
        with torch.no_grad():
            for Xb, Tb in self._make_batches(Xtest_t, Ttest_t):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)
                outputs = self.model(Xb)
                loss = self.loss_fn(outputs, Tb)
                total_loss += loss.item() * Xb.size(0)
                if self.classification:
                    preds = torch.argmax(outputs, dim=1)
                    total_correct += (preds == Tb).sum().item()
        avg_loss = total_loss / Xtest_t.shape[0]
        if self.classification:
            avg_acc = total_correct / Xtest_t.shape[0]
        else:
            avg_acc = 0.0
        return avg_loss, avg_acc


def run_single_experiment(p, op, train_fraction, depth, dim, heads, dropout, lr, weight_decay, beta1, beta2, warmup,
                          batch_size, epochs, seed, cpu_only):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cpu'
    if not cpu_only and torch.cuda.is_available():
        device = 'cuda'

    Xtrain_torch, Ttrain_torch, Xtest_torch, Ttest_torch = grokking_data_torch(
        p, op=op, train_fraction=train_fraction, device=device)

    kwargs = {
        'depth': depth,
        'dim': dim,
        'heads': heads,
        'n_tokens': p + 2,
        'seq_len': 4,
        'dropout': dropout
    }

    torch_model = TransformerTorch(**kwargs).to(device)

    optimizer_torch = optim_torch.AdamW(
        torch_model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )

    def linear_warmup_lr(current_step):
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        return 1.0

    scheduler = lr_scheduler.LambdaLR(optimizer_torch, lr_lambda=linear_warmup_lr)

    trainer = TorchTrainer(
        torch_model,
        optimizer_torch,
        classification=True,
        batch_size=batch_size,
        device=device
    )
    trainer.train(
        (Xtrain_torch, Ttrain_torch),
        (Xtest_torch, Ttest_torch),
        epochs=epochs,
        shuffle=True,
        scheduler=scheduler
    )

    max_val_acc = max(trainer.val_acc_trace)
    return max_val_acc, trainer


# ==============================================================================
# MXmain.py 的主执行逻辑
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, description="Run multiple grokking experiments.")
    # data args
    parser.add_argument('--p', type=int, default=97, help='prime number')
    parser.add_argument('--op', type=str, default='/',
                        help='operation', choices=['*', '/', '+', '-'])
    # model args
    parser.add_argument('--depth', type=int, default=2, help='depth')
    parser.add_argument('--dim', type=int, default=128, help='dimension')
    parser.add_argument('--heads', type=int, default=4, help='heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # optimizer args
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=1, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2')
    parser.add_argument('--warmup', type=int, default=10, help='warmup steps')
    # training args
    parser.add_argument('-b', '--batch_size', type=int,
                        default=512, help='batch size')
    parser.add_argument('-e', '--epochs', type=int,
                        default=300, help='number of epochs')
    # misc args
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')

    args = parser.parse_args()

    os.makedirs('media', exist_ok=True)

    print("Running multiple experiments for different train fractions (MXmain.py)...")

    # 生成10个从0.1到1.0均匀分布的train_fraction值
    train_fractions_to_test = np.linspace(0.1, 1.0, 10)
    accuracies = []

    # 循环遍历不同的train_fraction值
    for tf in tqdm(train_fractions_to_test, desc="Varying Train Fraction"):
        # 调用复制过来的 run_single_experiment 函数
        current_max_val_acc, _ = run_single_experiment(
            p=args.p, op=args.op, train_fraction=tf,
            depth=args.depth, dim=args.dim, heads=args.heads, dropout=args.dropout,
            lr=args.lr, weight_decay=args.weight_decay, beta1=args.beta1, beta2=args.beta2,
            warmup=args.warmup, batch_size=args.batch_size, epochs=args.epochs,
            seed=args.seed, cpu_only=args.cpu
        )
        accuracies.append(current_max_val_acc)

    # 绘制结果图 (train fraction vs accuracy)
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制散点，带透明度
    ax.plot(train_fractions_to_test, np.array(accuracies) * 100, 'o', alpha=0.6, label='Best Accuracy')

    # 连接散点的虚线
    ax.plot(train_fractions_to_test, np.array(accuracies) * 100, linestyle='--', color='blue', alpha=0.5)

    ax.set_xlabel('Training Data Fraction')
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title(f'Best Validation Accuracy vs. Training Data Fraction (Op: {args.op}, P: {args.p})')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(-5, 105)
    ax.set_xlim(min(train_fractions_to_test) - 0.02, max(train_fractions_to_test) + 0.02)
    ax.legend()
    fig.tight_layout()
    plot_filename = f'media/accuracy_vs_train_fraction_{args.op}_{args.p}_10points.png'
    fig.savefig(plot_filename, dpi=300)
    plt.show()

    print(f"\nPlot '{plot_filename}' saved to 'media/' directory.")