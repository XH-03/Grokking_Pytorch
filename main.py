import argparse
import numpy as np

import matplotlib.pyplot as plt
import os  # Add os import
import warnings

from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", message=".*Detected no triton.*", category=UserWarning)

from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim_torch
import torch.nn.functional as F

import torch.optim.lr_scheduler as lr_scheduler

from models import TransformerTorch
from data import grokking_data_torch

parser = argparse.ArgumentParser(add_help=True)
# data args
parser.add_argument('--p', type=int, default=97, help='prime number')
parser.add_argument('--op', type=str, default='/',
                    help='operation', choices=['*', '/', '+', '-','x^3+xy^2+y'])
parser.add_argument('--train-fraction', type=float,
                    default=0.5, help='train fraction')
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

    def train(self, train_data, val_data, epochs=5, shuffle=True,scheduler=None):
        self.model.train()
        Xtrain_t, Ttrain_t = train_data
        Xtest_t, Ttest_t = val_data

        global_step = 0

        # Basic epoch loop
        epoch_bar = tqdm(range(epochs), desc='Training', unit='epoch')
        for _ in epoch_bar:
            self.model.train()
            if shuffle:
                permutation = torch.randperm(Xtrain_t.size(0))
                Xtrain_t = Xtrain_t[permutation]
                Ttrain_t = Ttrain_t[permutation]

            total_loss = 0.0
            total_correct = 0
            for Xb, Tb in self._make_batches(Xtrain_t, Ttrain_t):
                # Move to device if needed
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(Xb)
                loss = self.loss_fn(outputs, Tb)
                loss.backward()
                self.optimizer.step()

                if scheduler is not None:  # Update scheduler after each step
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


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Xtrain_torch, Ttrain_torch, Xtest_torch, Ttest_torch = grokking_data_torch(
        args.p, op=args.op, train_fraction=args.train_fraction, device='cpu')
    # Already torch tensors

    # Build model(s)
    kwargs = {
        'depth': args.depth,
        'dim': args.dim,
        'heads': args.heads,
        'n_tokens': args.p + 6,
        'seq_len': 4,  # typically X shape is (N, 4) for [a, op, b, '=']
        'dropout': args.dropout
    }

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda'
    torch_model = TransformerTorch(**kwargs).to(device)
    #
    # print("Starting model compilation...")
    # torch_model = torch.compile(torch_model)
    # print("Model compilation finished!")

    optimizer_torch = optim_torch.AdamW(
        torch_model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    # Define the linear warmup function
    def linear_warmup_lr(current_step):
        # Warmup phase: Learning rate linearly increases from 0 to args.lr
        if current_step < args.warmup:
            return float(current_step) / float(max(1, args.warmup))
        # After warmup: Learning rate remains at args.lr
        return 1.0

    # Create the learning rate scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer_torch, lr_lambda=linear_warmup_lr)

    # Initialize the trainer
    trainer = TorchTrainer(
        torch_model,
        optimizer_torch,
        classification=True,
        batch_size=args.batch_size,
        device=device
    )
    trainer.train(
        (Xtrain_torch, Ttrain_torch),
        (Xtest_torch, Ttest_torch),
        epochs=args.epochs,
        shuffle=True,
        scheduler = scheduler
    )

    # Plot results
    os.makedirs('media', exist_ok=True)

    # Smooth the accuracy traces
    train_acc_smooth = gaussian_filter1d(trainer.train_acc_trace, sigma=2)
    val_acc_smooth = gaussian_filter1d(trainer.val_acc_trace, sigma=2)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(np.array(train_acc_smooth) * 100,
            label='train', color='#1b9e77', lw=2)
    ax.plot(np.array(val_acc_smooth) * 100,
            label='val', color='#d95f02', lw=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('media/grokking.png', dpi=300)
    plt.show()


    # Find the maximum accuracy and the corresponding epoch,and print it
    max_train_acc = max(trainer.train_acc_trace)
    max_val_acc = max(trainer.val_acc_trace)
    max_train_epoch = trainer.train_acc_trace.index(max_train_acc) + 1
    max_val_epoch = trainer.val_acc_trace.index(max_val_acc) + 1

    print(f"\n=== Training Results Summary ===")
    print(f"Maximum Training Accuracy: {max_train_acc:.4f} ({max_train_acc * 100:.2f}%) - at Epoch {max_train_epoch}")
    print(f"Maximum Validation Accuracy: {max_val_acc:.4f} ({max_val_acc * 100:.2f}%) - at Epoch {max_val_epoch}")
    print(f"Final Training Accuracy: {trainer.train_acc_trace[-1]:.4f} ({trainer.train_acc_trace[-1] * 100:.2f}%)")
    print(f"Final Validation Accuracy: {trainer.val_acc_trace[-1]:.4f} ({trainer.val_acc_trace[-1] * 100:.2f}%)")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

