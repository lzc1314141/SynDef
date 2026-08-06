import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import os
import json
import time
import re
import random
import sys
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from collections import Counter, deque
import csv

csv.field_size_limit(min(2147483647, sys.maxsize))

FILE_LEVEL_BASE_DIR = "/root/workspace/lzc/SynDef/File-level"

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    try:
        from matplotlib import font_manager as _fm
        _available_fonts = {f.name for f in _fm.fontManager.ttflist}
        _cjk_candidates = [
            'Noto Sans CJK SC', 'Noto Sans CJK CN', 'Noto Sans CJK',
            'Source Han Sans SC', 'Source Han Sans CN', 'Source Han Sans',
            'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
            'AR PL UMing CN', 'AR PL UKai CN',
            'SimHei', 'Microsoft YaHei'
        ]
        _chosen_font = None
        for _name in _cjk_candidates:
            if _name in _available_fonts:
                _chosen_font = _name
                break
        if _chosen_font is not None:
            matplotlib.rcParams['font.sans-serif'] = [_chosen_font, 'DejaVu Sans']
        else:
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        matplotlib.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not installed, training curves will not be plotted")
    MATPLOTLIB_AVAILABLE = False


def create_causal_mask(seq_len, device, dtype=torch.bool):
    try:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
        return mask
    except Exception as e:
        raise RuntimeError(f"Failed to create causal mask: {e}")


def create_padding_mask(input_ids, pad_token_id=0):
    if input_ids is None:
        return None
    try:
        mask = (input_ids != pad_token_id).to(dtype=torch.bool)
        mask = mask.unsqueeze(1).unsqueeze(2)
        return mask
    except Exception as e:
        raise RuntimeError(f"Failed to create padding mask: {e}")


def combine_masks(causal_mask, padding_mask=None, batch_size=None):
    try:
        seq_len = causal_mask.size(-1)
        device = causal_mask.device
        dtype = causal_mask.dtype
        if padding_mask is not None:
            batch_size = padding_mask.size(0)
            causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)
            padding_mask_expanded = padding_mask.expand(-1, -1, seq_len, -1)
            combined_mask = causal_mask_expanded & padding_mask_expanded
        else:
            if batch_size is None:
                batch_size = 1
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)
        return combined_mask.to(dtype=dtype)
    except Exception as e:
        raise RuntimeError(f"Failed to combine masks: {e}")


def validate_mask_dimensions(mask, expected_shape, mask_name="mask"):
    if mask is None:
        return True
    if len(mask.shape) != len(expected_shape):
        raise ValueError(f"{mask_name} dimension error: expected {len(expected_shape)} dimensions, got {len(mask.shape)}")
    for i, (actual, expected) in enumerate(zip(mask.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(f"{mask_name} dimension {i} size error: expected {expected}, got {actual}")
    return True


class TrainingMonitor:
    def __init__(self, save_dir='./checkpoints', experiment_name=None):
        self.save_dir = save_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        print(f"📁 Experiment directory: {self.experiment_dir}")

    def log_epoch(self, epoch, train_loss, val_loss, learning_rate):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(learning_rate)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            return True
        return False

    def save_checkpoint(self, model, optimizer, scheduler, epoch, is_best=False):
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            latest_path = os.path.join(self.experiment_dir, 'latest_checkpoint.pth')
            torch.save(checkpoint, latest_path)
            if is_best:
                best_path = os.path.join(self.experiment_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f"✅ Best model saved: {best_path}")
            if epoch % 10 == 0:
                epoch_path = os.path.join(self.experiment_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, epoch_path)
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        try:
            if checkpoint_path is None:
                checkpoint_path = os.path.join(self.experiment_dir, 'latest_checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint file not found: {checkpoint_path}")
                return 0
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.learning_rates = checkpoint.get('learning_rates', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)
            epoch = checkpoint['epoch']
            print(f"✅ Checkpoint loaded successfully, resuming from epoch {epoch}")
            return epoch + 1
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return 0

    def plot_training_curves(self):
        if not MATPLOTLIB_AVAILABLE or not self.epochs:
            return
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            ax1.plot(self.epochs, self.train_losses, label='Training Loss', color='blue', alpha=0.7)
            ax1.plot(self.epochs, self.val_losses, label='Validation Loss', color='red', alpha=0.7)
            ax1.axvline(x=self.best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {self.best_epoch})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.plot(self.epochs, self.learning_rates, label='Learning Rate', color='orange')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            if len(self.epochs) > 10:
                start_idx = len(self.epochs) // 4
                ax3.plot(self.epochs[start_idx:], self.train_losses[start_idx:], label='Training Loss', color='blue', alpha=0.7)
                ax3.plot(self.epochs[start_idx:], self.val_losses[start_idx:], label='Validation Loss', color='red', alpha=0.7)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss')
                ax3.set_title('Loss Details (Later Epochs)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            ax4.axis('off')
            stats_text = f"""Training Statistics:

Total Epochs: {len(self.epochs)}
Best Validation Loss: {self.best_val_loss:.6f}
Best Model Epoch: {self.best_epoch}
Final Training Loss: {self.train_losses[-1]:.6f}
Final Validation Loss: {self.val_losses[-1]:.6f}
Final Learning Rate: {self.learning_rates[-1]:.2e}

Overfitting Check:
Train-Val Gap: {abs(self.train_losses[-1] - self.val_losses[-1]):.6f}
"""
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace')
            plt.tight_layout()
            plot_path = os.path.join(self.experiment_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Training curves saved: {plot_path}")
        except Exception as e:
            print(f"❌ Failed to plot training curves: {e}")

    def save_training_log(self):
        try:
            log_data = {
                'experiment_name': self.experiment_name,
                'epochs': self.epochs,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch,
                'timestamp': datetime.now().isoformat()
            }
            log_path = os.path.join(self.experiment_dir, 'training_log.json')
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            print(f"📝 Training log saved: {log_path}")
        except Exception as e:
            print(f"❌ Failed to save training log: {e}")


class ProgressTracker:
    def __init__(self, total_epochs, batches_per_epoch, log_interval=100):
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.log_interval = log_interval
        self.current_epoch = 0
        self.current_batch = 0
        self.start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
        self.loss_history = deque(maxlen=100)
        self.speed_history = deque(maxlen=50)
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.peak_memory = 0
        print("📊 Progress tracker initialized")
        print(f"   Total epochs: {total_epochs}")
        print(f"   Batches per epoch: {batches_per_epoch}")
        print(f"   Total training steps: {total_epochs * batches_per_epoch}")

    def start_training(self):
        self.start_time = time.time()
        print(f"\n🚀 Training started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def start_epoch(self, epoch):
        self.current_epoch = epoch
        self.current_batch = 0
        self.epoch_start_time = time.time()
        print(f"\n📚 Epoch {epoch + 1}/{self.total_epochs}")
        print("-" * 60)

    def update_batch(self, batch_idx, loss, lr=None, grad_norm=None):
        self.current_batch = batch_idx
        self.batch_start_time = time.time()
        if not (math.isnan(loss) or math.isinf(loss)):
            self.loss_history.append(loss)
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_epoch = self.current_epoch
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2
            self.peak_memory = max(self.peak_memory, current_memory)
        if batch_idx % self.log_interval == 0:
            self._display_progress(loss, lr, grad_norm)
        if batch_idx % 50 == 0:
            self.save_progress_file()

    def _display_progress(self, loss, lr=None, grad_norm=None):
        current_time = time.time()
        total_batches = self.total_epochs * self.batches_per_epoch
        completed_batches = self.current_epoch * self.batches_per_epoch + self.current_batch
        progress_pct = (completed_batches / total_batches) * 100
        if self.start_time:
            elapsed_time = current_time - self.start_time
            batches_per_sec = completed_batches / elapsed_time if elapsed_time > 0 else 0
            self.speed_history.append(batches_per_sec)
        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else loss
        if self.speed_history and sum(self.speed_history) > 0:
            avg_speed = sum(self.speed_history) / len(self.speed_history)
            remaining_batches = total_batches - completed_batches
            eta_seconds = remaining_batches / avg_speed if avg_speed > 0 else 0
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "Calculating..."
        bar_length = 30
        filled_length = int(bar_length * progress_pct / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r📈 [{bar}] {progress_pct:.1f}%", end="")
        if self.current_batch % self.log_interval == 0:
            print()
            info_lines = [
                f"   Epoch: {self.current_epoch + 1}/{self.total_epochs}",
                f"   Batch: {self.current_batch + 1}/{self.batches_per_epoch}",
                f"   Loss: {loss:.6f} (avg: {avg_loss:.6f})",
                f"   Best Loss: {self.best_loss:.6f} (Epoch {self.best_epoch + 1})"
            ]
            if lr is not None:
                info_lines.append(f"   Learning Rate: {lr:.2e}")
            if grad_norm is not None:
                info_lines.append(f"   Gradient Norm: {grad_norm:.4f}")
            if self.speed_history:
                avg_speed = sum(self.speed_history) / len(self.speed_history)
                info_lines.append(f"   Speed: {avg_speed:.2f} batch/s")
            info_lines.append(f"   ETA: {eta}")
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**2
                info_lines.append(f"   GPU Memory: {current_memory:.0f}MB (peak: {self.peak_memory:.0f}MB)")
            for line in info_lines:
                print(line)
            print("-" * 60)
        sys.stdout.flush()

    def finish_epoch(self, train_loss, val_loss, epoch_time):
        print(f"\n🎯 Epoch {self.current_epoch + 1} complete:")
        print(f"   📊 Training Loss: {train_loss:.6f}")
        print(f"   📊 Validation Loss: {val_loss:.6f}")
        print(f"   ⏱️  Time: {epoch_time:.1f}s")
        if val_loss < self.best_loss:
            print(f"   🏆 New best validation loss!")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.current_epoch
        print("=" * 80)

    def finish_training(self):
        if self.start_time:
            total_time = time.time() - self.start_time
            total_time_str = str(timedelta(seconds=int(total_time)))
            print(f"\n🎉 Training completed! - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print(f"📈 Training Statistics:")
            print(f"   Total time: {total_time_str}")
            print(f"   Average per epoch: {total_time / self.total_epochs:.1f}s")
            print(f"   Best validation loss: {self.best_loss:.6f}")
            print(f"   Best model: Epoch {self.best_epoch + 1}")
            if self.speed_history:
                avg_speed = sum(self.speed_history) / len(self.speed_history)
                total_batches = self.total_epochs * self.batches_per_epoch
                print(f"   Average speed: {avg_speed:.2f} batch/s")
                print(f"   Total batches processed: {total_batches:,}")
            if torch.cuda.is_available():
                print(f"   Peak GPU memory: {self.peak_memory:.0f}MB")
            print("=" * 80)

    def get_progress_info(self):
        total_batches = self.total_epochs * self.batches_per_epoch
        completed_batches = self.current_epoch * self.batches_per_epoch + self.current_batch
        info = {
            'current_epoch': self.current_epoch + 1,
            'total_epochs': self.total_epochs,
            'current_batch': self.current_batch + 1,
            'batches_per_epoch': self.batches_per_epoch,
            'progress_percent': (completed_batches / total_batches) * 100,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch + 1,
            'peak_memory_mb': self.peak_memory
        }
        if self.loss_history:
            info['recent_avg_loss'] = sum(self.loss_history) / len(self.loss_history)
        if self.speed_history:
            info['avg_speed_batch_per_sec'] = sum(self.speed_history) / len(self.speed_history)
        if self.start_time:
            info['elapsed_seconds'] = time.time() - self.start_time
        return info

    def save_progress_file(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(os.getcwd(), 'training_progress.json')
        progress_info = self.get_progress_info()
        progress_info['last_updated'] = datetime.now().isoformat()
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(progress_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save progress file: {e}")


def create_progress_bar(current, total, length=40, prefix='Progress', suffix='Complete'):
    percent = (current / total) * 100
    filled_length = int(length * current / total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    return f'\r{prefix} |{bar}| {percent:.1f}% {suffix}'


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def display_training_status(progress_file='training_progress.json'):
    try:
        if not os.path.exists(progress_file):
            print("❌ Progress file not found")
            return
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        print("📊 Current Training Status:")
        print("-" * 50)
        print(f"Epoch: {progress.get('current_epoch', 0)}/{progress.get('total_epochs', 0)}")
        print(f"Batch: {progress.get('current_batch', 0)}/{progress.get('batches_per_epoch', 0)}")
        print(f"Total Progress: {progress.get('progress_percent', 0):.1f}%")
        print(f"Best Loss: {progress.get('best_loss', 'N/A')}")
        print(f"Best Epoch: {progress.get('best_epoch', 'N/A')}")
        if 'recent_avg_loss' in progress:
            print(f"Recent Average Loss: {progress['recent_avg_loss']:.6f}")
        if 'avg_speed_batch_per_sec' in progress:
            print(f"Average Speed: {progress['avg_speed_batch_per_sec']:.2f} batch/s")
        if 'elapsed_seconds' in progress:
            print(f"Elapsed Time: {format_time(progress['elapsed_seconds'])}")
        if 'peak_memory_mb' in progress:
            print(f"Peak Memory: {progress['peak_memory_mb']:.0f}MB")
        print(f"Last Updated: {progress.get('last_updated', 'Unknown')}")
        print("-" * 50)
    except Exception as e:
        print(f"❌ Failed to read progress file: {e}")


def init_weights(module, init_type='xavier_uniform'):
    if isinstance(module, nn.Linear):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(module.weight, 1.0)


def apply_weight_initialization(model, init_type='xavier_uniform'):
    try:
        for module in model.modules():
            init_weights(module, init_type)
        print(f"✅ Weight initialization complete (type: {init_type})")
    except Exception as e:
        print(f"❌ Weight initialization failed: {e}")
        raise


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attn_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout, pad_token_id=0):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        device = x.device
        causal_mask = create_causal_mask(seq_len, device)
        padding_mask = create_padding_mask(x, self.pad_token_id)
        combined_mask = combine_masks(causal_mask, padding_mask, batch_size)
        if mask is not None:
            validate_mask_dimensions(mask, combined_mask.shape, "extra mask")
            combined_mask = combined_mask & mask
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, combined_mask)
        x = self.ln_f(x)
        output = self.fc_out(x)
        return output


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, pad_token_id=0):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        seq_len = len(input_ids)
        pad_length = self.max_length - 1 - seq_len
        if pad_length > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_length
            target_ids = target_ids + [self.pad_token_id] * pad_length
        input_ids = input_ids[:self.max_length-1]
        target_ids = target_ids[:self.max_length-1]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


class WarmupLRScheduler:
    def __init__(self, optimizer, warmup_steps, d_model):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.d_model ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        step = max(1, self.current_step)
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'd_model': self.d_model,
        }

    def load_state_dict(self, state_dict):
        self.current_step = int(state_dict.get('current_step', 0))
        self.warmup_steps = int(state_dict.get('warmup_steps', self.warmup_steps))
        self.d_model = float(state_dict.get('d_model', self.d_model))
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def train_transformer(model, train_loader, val_loader, epochs=10, learning_rate=1e-4,
                     use_warmup=True, warmup_steps=4000, pad_token_id=0,
                     save_dir='./checkpoints', experiment_name=None, resume_from=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"🚀 Using device: {device}")
    print(f"🔧 GPU count: {torch.cuda.device_count()}")
    monitor = TrainingMonitor(save_dir, experiment_name)
    try:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        if use_warmup:
            scheduler = WarmupLRScheduler(optimizer, warmup_steps, model.d_model)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        start_epoch = 0
        if resume_from:
            start_epoch = monitor.load_checkpoint(model, optimizer, scheduler, resume_from)
        print(f"📚 Starting training (Epochs: {start_epoch} → {epochs})")
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0
            num_batches = len(train_loader)
            for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
                try:
                    input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                    if input_ids.size(0) == 0 or target_ids.size(0) == 0:
                        continue
                    optimizer.zero_grad()
                    outputs = model(input_ids)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️  Warning: invalid loss value at batch {batch_idx}")
                        continue
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if use_warmup:
                        scheduler.step()
                    total_loss += loss.item()
                    if batch_idx % 100 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f'📈 Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{num_batches} | '
                              f'Loss: {loss.item():.4f} | LR: {current_lr:.6f} | '
                              f'Grad Norm: {grad_norm:.4f}')
                except Exception as e:
                    print(f"❌ Error in training batch {batch_idx}: {e}")
                    continue
            try:
                val_loss = evaluate_model(model, val_loader, criterion, device)
            except Exception as e:
                print(f"❌ Error during validation: {e}")
                val_loss = float('inf')
            if not use_warmup and scheduler is not None:
                scheduler.step(val_loss)
            avg_train_loss = total_loss / max(num_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            is_best = monitor.log_epoch(epoch, avg_train_loss, val_loss, current_lr)
            monitor.save_checkpoint(model, optimizer, scheduler, epoch, is_best)
            print(f'🎯 Epoch {epoch+1}/{epochs} complete:')
            print(f'   📊 Training Loss: {avg_train_loss:.6f}')
            print(f'   📊 Validation Loss: {val_loss:.6f}')
            print(f'   📈 Learning Rate: {current_lr:.6f}')
            print(f'   ⏱️  Time: {epoch_time:.1f}s')
            if is_best:
                print(f'   🏆 New best model!')
            print('-' * 80)
        print("🎉 Training complete!")
        monitor.plot_training_curves()
        monitor.save_training_log()
        return monitor
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        monitor.save_checkpoint(model, optimizer, scheduler, epoch, False)
        monitor.plot_training_curves()
        monitor.save_training_log()
        return monitor
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Saving current progress...")
        monitor.save_checkpoint(model, optimizer, scheduler, epoch, False)
        raise


def evaluate_model(model, val_loader, criterion, device):
    if val_loader is None or len(val_loader) == 0:
        print("⚠️  Warning: validation data loader is empty")
        return float('inf')
    model.eval()
    total_loss = 0
    valid_batches = 0
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
                try:
                    input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                    if input_ids.size(0) == 0 or target_ids.size(0) == 0:
                        continue
                    outputs = model(input_ids)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️  Validation batch {batch_idx} produced invalid loss")
                        continue
                    total_loss += loss.item()
                    valid_batches += 1
                except Exception as e:
                    print(f"❌ Error in validation batch {batch_idx}: {e}")
                    continue
        if valid_batches == 0:
            print("⚠️  Warning: no valid validation batches")
            return float('inf')
        return total_loss / valid_batches
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return float('inf')


def safe_model_inference(model, input_tensor, device='cuda', max_retries=3):
    model.eval()
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                if input_tensor.device != device:
                    input_tensor = input_tensor.to(device)
                outputs = model(input_tensor)
                return outputs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt < max_retries - 1:
                print(f"⚠️  GPU out of memory, clearing cache... (attempt {attempt+1}/{max_retries})")
                torch.cuda.empty_cache()
                time.sleep(1)
                continue
            else:
                print(f"❌ Model inference failed: {e}")
                raise
        except Exception as e:
            print(f"❌ Unknown error during inference: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... (attempt {attempt+1}/{max_retries})")
                continue
            raise
    raise RuntimeError("Model inference failed after maximum retries")


def validate_model_config(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length):
    errors = []
    if vocab_size <= 0:
        errors.append(f"Vocabulary size must be > 0, got {vocab_size}")
    if d_model <= 0 or d_model % n_heads != 0:
        errors.append(f"Model dimension must be > 0 and divisible by number of heads, got d_model={d_model}, n_heads={n_heads}")
    if n_heads <= 0:
        errors.append(f"Number of heads must be > 0, got {n_heads}")
    if n_layers <= 0:
        errors.append(f"Number of layers must be > 0, got {n_layers}")
    if d_ff <= 0:
        errors.append(f"Feed-forward dimension must be > 0, got {d_ff}")
    if max_seq_length <= 0:
        errors.append(f"Max sequence length must be > 0, got {max_seq_length}")
    if errors:
        raise ValueError("Model configuration error:\n" + "\n".join(errors))
    return True


def get_model_memory_usage(model, input_shape=(1, 512), device='cuda'):
    try:
        model.eval()
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        dummy_input = torch.randint(0, 1000, input_shape, dtype=torch.long, device=device)
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        if hasattr(torch.cuda, 'max_memory_allocated'):
            activation_memory = torch.cuda.max_memory_allocated() - param_memory - buffer_memory
        else:
            activation_memory = 0
        total_memory = param_memory + buffer_memory + activation_memory
        return {
            'parameters_mb': param_memory / (1024 ** 2),
            'buffers_mb': buffer_memory / (1024 ** 2),
            'activations_mb': activation_memory / (1024 ** 2),
            'total_mb': total_memory / (1024 ** 2)
        }
    except Exception as e:
        print(f"⚠️  Memory usage estimation failed: {e}")
        return None


def check_system_requirements():
    print("🔍 System requirements check:")
    print(f"  PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.version.cuda}")
        print(f"  🔧 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            print(f"    GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    else:
        print("  ⚠️  CUDA not available, training on CPU (slower)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            if memory_gb < 4:
                print(f"  ⚠️  Warning: GPU {i} memory is small ({memory_gb:.1f}GB), consider reducing batch size")
    print("\n💡 Recommended settings:")
    print("  • Small model (<100M params): batch_size=32, d_model=256")
    print("  • Medium model (100M-1B params): batch_size=16, d_model=512")
    print("  • Large model (>1B params): batch_size=4-8, d_model=768+")
    print("  • Using mixed precision can save ~50% VRAM")
    print()


def create_model_summary(model):
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("📋 Model Summary:")
        print(f"  🏗️  Model type: {model.__class__.__name__}")
        print(f"  📊 Total parameters: {total_params:,}")
        print(f"  🎯 Trainable parameters: {trainable_params:,}")
        print(f"  💾 Model size: {total_params * 4 / (1024**2):.1f} MB")
        if hasattr(model, 'd_model'):
            print(f"  🔧 Model dimension: {model.d_model}")
        if hasattr(model, 'transformer_blocks'):
            print(f"  🔧 Number of layers: {len(model.transformer_blocks)}")
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2)
        }
    except Exception as e:
        print(f"❌ Failed to create model summary: {e}")
        return None


def train_with_gradient_accumulation(model, train_loader, val_loader, epochs=10,
                                   accumulation_steps=4, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i, (input_ids, target_ids) in enumerate(train_loader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if i % (100 * accumulation_steps) == 0:
                    print(f'Epoch {epoch+1}, Step {i//accumulation_steps}, Loss: {loss.item()*accumulation_steps:.4f}')


def create_model(vocab_size=50000, d_model=512, n_heads=8, n_layers=6,
                d_ff=2048, max_seq_length=512, dropout=0.1, pad_token_id=0,
                weight_init='xavier_uniform'):
    try:
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        if weight_init is not None:
            apply_weight_initialization(model, weight_init)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model created successfully!")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"💾 Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise


def top_k_sampling(logits, k=50):
    if k <= 0:
        return logits
    values, indices = torch.topk(logits, k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1)
    logits = torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)
    return logits


def top_p_sampling(logits, p=0.9):
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0,
                 sampling_strategy='greedy', top_k=50, top_p=0.9, num_beams=1,
                 device='cuda', pad_token_id=0, eos_token_id=None,
                 repetition_penalty=1.0):
    if sampling_strategy == 'beam_search':
        return beam_search_generate(model, tokenizer, prompt, max_length, num_beams,
                                  device, pad_token_id, eos_token_id)
    model.eval()
    try:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated = input_ids.copy()
        with torch.no_grad():
            for step in range(max_length):
                max_model_length = model.positional_encoding.pe.size(1) - 1
                if input_tensor.size(1) > max_model_length:
                    input_tensor = input_tensor[:, -max_model_length:]
                outputs = model(input_tensor)
                next_token_logits = outputs[0, -1, :]
                if repetition_penalty != 1.0:
                    for token_id in set(generated):
                        if token_id < len(next_token_logits):
                            if next_token_logits[token_id] < 0:
                                next_token_logits[token_id] *= repetition_penalty
                            else:
                                next_token_logits[token_id] /= repetition_penalty
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if sampling_strategy == 'greedy':
                    next_token = torch.argmax(next_token_logits).item()
                elif sampling_strategy == 'random':
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                elif sampling_strategy == 'top_k':
                    filtered_logits = top_k_sampling(next_token_logits.unsqueeze(0), k=top_k)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                elif sampling_strategy == 'top_p':
                    filtered_logits = top_p_sampling(next_token_logits.unsqueeze(0), p=top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
                generated.append(next_token)
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
                input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
                if eos_token_id is not None and next_token == eos_token_id:
                    break
                if next_token == pad_token_id:
                    break
        return tokenizer.decode(generated)
    except Exception as e:
        print(f"Text generation failed: {e}")
        return prompt


def beam_search_generate(model, tokenizer, prompt, max_length=100, num_beams=4,
                        device='cuda', pad_token_id=0, eos_token_id=None):
    model.eval()
    try:
        input_ids = tokenizer.encode(prompt)
        batch_size = 1
        beam_scores = torch.zeros(num_beams, device=device)
        beam_tokens = torch.tensor([input_ids] * num_beams, dtype=torch.long, device=device)
        beam_finished = torch.zeros(num_beams, dtype=torch.bool, device=device)
        with torch.no_grad():
            for step in range(max_length):
                outputs = model(beam_tokens)
                next_token_logits = outputs[:, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                if beam_finished.any():
                    log_probs[beam_finished, :] = float('-inf')
                    if pad_token_id is not None:
                        log_probs[beam_finished, pad_token_id] = 0.0
                vocab_size = log_probs.size(-1)
                candidate_scores = beam_scores.unsqueeze(1) + log_probs
                candidate_scores = candidate_scores.view(-1)
                top_scores, top_indices = torch.topk(candidate_scores, num_beams)
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                new_beam_tokens = []
                new_beam_scores = []
                new_beam_finished = []
                for i in range(num_beams):
                    beam_idx = beam_indices[i]
                    token_idx = token_indices[i]
                    new_tokens = torch.cat([
                        beam_tokens[beam_idx],
                        torch.tensor([token_idx], device=device)
                    ])
                    new_beam_tokens.append(new_tokens)
                    new_beam_scores.append(top_scores[i])
                    finished = beam_finished[beam_idx] or (eos_token_id is not None and token_idx == eos_token_id)
                    new_beam_finished.append(finished)
                beam_tokens = torch.stack(new_beam_tokens)
                beam_scores = torch.stack(new_beam_scores)
                beam_finished = torch.tensor(new_beam_finished, device=device)
                if beam_finished.all():
                    break
        best_beam_idx = torch.argmax(beam_scores)
        best_tokens = beam_tokens[best_beam_idx].tolist()
        return tokenizer.decode(best_tokens)
    except Exception as e:
        print(f"Beam search generation failed: {e}")
        return prompt


def parse_bool_str(value: str) -> bool:
    if value is None:
        return False
    v = str(value).strip().lower()
    return v in {"true", "1", "yes", "y", "t"}


def sanitize_file_level_path(file_path: str) -> str:
    if not file_path:
        return ""
    normalized = file_path.replace("\\", "/").lstrip("./")
    return normalized.replace("/", "_")


def find_file_level_csv_path(base_dir: str, version_key: str) -> Optional[Path]:
    base = Path(base_dir)
    if not base.exists():
        print(f"❌ File-level directory does not exist: {base_dir}")
        return None
    expected = base / f"{version_key}_ground-truth-files_dataset.csv"
    if expected.exists():
        return expected
    version_lower = version_key.lower()
    for candidate in base.glob("*.csv"):
        name = candidate.name.lower()
        if name.startswith(version_lower) and name.endswith("_ground-truth-files_dataset.csv"):
            return candidate
    print(f"❌ CSV for version not found: {version_key} in {base_dir}")
    return None


def load_non_buggy_filenames_from_csv(csv_path: str) -> Set[str]:
    allowed: Set[str] = set()
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return allowed
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"❌ CSV has no valid headers: {csv_path}")
                return allowed
            field_map = {name.lower(): name for name in reader.fieldnames}
            bug_col = field_map.get("bug")
            file_col = field_map.get("file")
            if not bug_col or not file_col:
                print(f"❌ CSV missing required fields (File and Bug): {csv_path}")
                return allowed
            total = 0
            skipped_buggy = 0
            collected = 0
            for row in reader:
                total += 1
                bug_val = row.get(bug_col, "")
                if parse_bool_str(bug_val):
                    skipped_buggy += 1
                    continue
                file_path = row.get(file_col, "")
                if not file_path or not file_path.strip():
                    continue
                if not file_path.endswith(".java"):
                    continue
                sanitized = sanitize_file_level_path(file_path)
                if sanitized:
                    allowed.add(sanitized)
                    collected += 1
        print(f"✅ Non-buggy list: {os.path.basename(csv_path)} | total rows: {total}, buggy: {skipped_buggy}, kept: {collected}")
    except Exception as e:
        print(f"❌ Failed to read CSV: {csv_path} | Error: {e}")
    return allowed


def load_texts_from_file_level_csv(csv_path: str, exclude_buggy: bool = True) -> List[str]:
    texts: List[str] = []
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return texts
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            field_map = {name.lower(): name for name in reader.fieldnames or []}
            bug_col = field_map.get("bug")
            src_col = field_map.get("src")
            if not bug_col or not src_col:
                print(f"❌ CSV missing required fields (Bug and SRC): {csv_path}")
                return texts
            total = 0
            skipped_buggy = 0
            skipped_empty = 0
            for row in reader:
                total += 1
                bug_val = row.get(bug_col, "")
                is_buggy = parse_bool_str(bug_val)
                if exclude_buggy and is_buggy:
                    skipped_buggy += 1
                    continue
                src = row.get(src_col, "")
                if not src or not src.strip():
                    skipped_empty += 1
                    continue
                texts.append(src)
        print(f"✅ Load complete: {os.path.basename(csv_path)} | total rows: {total}, filtered buggy: {skipped_buggy}, empty SRC: {skipped_empty}, kept: {len(texts)}")
    except Exception as e:
        print(f"❌ Failed to read CSV: {csv_path} | Error: {e}")
    return texts


def load_texts_from_file_level_csv_dir(base_dir: str, version_key: str, exclude_buggy: bool = True) -> List[str]:
    target = find_file_level_csv_path(base_dir, version_key)
    if target is None:
        return []
    return load_texts_from_file_level_csv(str(target), exclude_buggy=exclude_buggy)


class JavaCodeTokenizer:
    def __init__(self, vocab_size=50000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.pad_token_id = 0
        self.mask_token_id = 1
        self.unk_token_id = 2
        self.cls_token_id = 3
        self.sep_token_id = 4
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_built = False
        self.java_keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
            'char', 'class', 'const', 'continue', 'default', 'do', 'double',
            'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
            'goto', 'if', 'implements', 'import', 'instanceof', 'int',
            'interface', 'long', 'native', 'new', 'package', 'private',
            'protected', 'public', 'return', 'short', 'static', 'strictfp',
            'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
            'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null'
        }
        self._init_special_tokens()

    def _init_special_tokens(self):
        special_tokens = [
            self.pad_token, self.mask_token, self.unk_token,
            self.cls_token, self.sep_token
        ]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def tokenize_java_code(self, code: str) -> List[str]:
        tokens = []
        code = self._handle_string_literals(code)
        code = self._handle_comments(code)
        pattern = r'''
            (?://.*?$)|
            (?:/\*.*?\*/)|
            (?:"(?:[^"\\]|\\.)*")|
            (?:'(?:[^'\\]|\\.)*')|
            (?:\d+\.?\d*[fFdD]?)|
            (?:0[xX][0-9a-fA-F]+)|
            (?:\w+)|
            (?:[{};(),\[\]<>])|
            (?:[+\-*/=!<>&|^%])|
            (?:\.)|
            (?:\s+)
        '''
        for match in re.finditer(pattern, code, re.MULTILINE | re.VERBOSE):
            token = match.group().strip()
            if token and not token.isspace():
                tokens.append(token)
        return tokens

    def _handle_string_literals(self, code: str) -> str:
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '<STRING>', code)
        code = re.sub(r"'(?:[^'\\]|\\.)*'", '<CHAR>', code)
        return code

    def _handle_comments(self, code: str) -> str:
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    def build_vocab(self, java_files: List[str]):
        print("📚 Building vocabulary...")
        token_counts = Counter()
        processed_files = 0
        for file_path in java_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                tokens = self.tokenize_java_code(code)
                token_counts.update(tokens)
                processed_files += 1
                if processed_files % 1000 == 0:
                    print(f"  Processed {processed_files}/{len(java_files)} files")
            except Exception as e:
                print(f"⚠️  Failed to process file {file_path}: {e}")
                continue
        print(f"✅ Vocabulary building complete, processed {processed_files} files")
        print(f"📊 Found {len(token_counts)} unique tokens")
        most_common = token_counts.most_common(self.vocab_size - len(self.token_to_id))
        current_id = len(self.token_to_id)
        for token, count in most_common:
            if count >= self.min_freq and token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        self.vocab_built = True
        print(f"🎯 Final vocabulary size: {len(self.token_to_id)}")
        java_kw_in_vocab = sum(1 for kw in self.java_keywords if kw in self.token_to_id)
        print(f"📝 Java keyword coverage: {java_kw_in_vocab}/{len(self.java_keywords)}")

    def encode(self, text: str) -> List[int]:
        if not self.vocab_built:
            raise ValueError("Vocabulary not built, call build_vocab() first")
        tokens = self.tokenize_java_code(text)
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.unk_token_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in [self.pad_token, self.cls_token, self.sep_token]:
                    tokens.append(token)
        result = ' '.join(tokens)
        result = re.sub(r'\s+([{}();,\[\]<>])', r'\1', result)
        result = re.sub(r'([{}();,\[\]<>])\s+', r'\1', result)
        return result

    def save_vocab(self, vocab_path: str):
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'vocab_size': len(self.token_to_id),
            'special_tokens': {
                'pad_token_id': self.pad_token_id,
                'mask_token_id': self.mask_token_id,
                'unk_token_id': self.unk_token_id,
                'cls_token_id': self.cls_token_id,
                'sep_token_id': self.sep_token_id
            }
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Vocabulary saved to: {vocab_path}")

    def load_vocab(self, vocab_path: str):
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.token_to_id = vocab_data['token_to_id']
            self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
            special_tokens = vocab_data.get('special_tokens', {})
            self.pad_token_id = special_tokens.get('pad_token_id', 0)
            self.mask_token_id = special_tokens.get('mask_token_id', 1)
            self.unk_token_id = special_tokens.get('unk_token_id', 2)
            self.cls_token_id = special_tokens.get('cls_token_id', 3)
            self.sep_token_id = special_tokens.get('sep_token_id', 4)
            self.vocab_built = True
            print(f"✅ Vocabulary loaded successfully, size: {len(self.token_to_id)}")
        except Exception as e:
            print(f"❌ Failed to load vocabulary: {e}")
            raise


class BasicTokenizer:
    def __init__(self, vocab_size: int = 50000, min_freq: int = 1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.token_to_id: Dict[str, int] = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id
        }
        self.id_to_token: Dict[int, str] = {
            self.pad_token_id: self.pad_token,
            self.unk_token_id: self.unk_token
        }
        self.vocab_built = False

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        pieces = re.findall(r'\w+|[^\s\w]', text, flags=re.UNICODE)
        return [p for p in pieces if p.strip() != '']

    def fit(self, texts: List[str]):
        counts = Counter()
        for t in texts:
            counts.update(self._tokenize(t))
        next_id = len(self.token_to_id)
        for token, cnt in counts.most_common():
            if cnt < self.min_freq:
                continue
            if token in self.token_to_id:
                continue
            if next_id >= self.vocab_size:
                break
            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token
            next_id += 1
        self.vocab_built = True
        print(f"✅ Basic tokenizer vocabulary built, size: {len(self.token_to_id)}")

    def encode(self, text: str) -> List[int]:
        if not self.vocab_built:
            raise ValueError("Vocabulary not built, call fit(texts) first")
        tokens = self._tokenize(text)
        return [self.token_to_id.get(tok, self.unk_token_id) for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
        pieces = []
        for tid in token_ids:
            tok = self.id_to_token.get(int(tid), self.unk_token)
            if tok in (self.pad_token,):
                continue
            pieces.append(tok)
        out = []
        for i, tok in enumerate(pieces):
            if i == 0:
                out.append(tok)
                continue
            if re.match(r'^\w+$', tok) and re.match(r'^\w+$', pieces[i-1]):
                out.append(' ' + tok)
            else:
                out.append(tok)
        return ''.join(out)

    @property
    def vocab_size_current(self) -> int:
        return len(self.token_to_id)

    def save_vocab(self, vocab_path: str):
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'vocab_size': len(self.token_to_id),
            'special_tokens': {
                'pad_token_id': self.pad_token_id,
                'unk_token_id': self.unk_token_id
            }
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Basic tokenizer vocabulary saved to: {vocab_path}")

    def load_vocab(self, vocab_path: str):
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.token_to_id = vocab_data['token_to_id']
            self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
            special_tokens = vocab_data.get('special_tokens', {})
            self.pad_token_id = int(special_tokens.get('pad_token_id', 0))
            self.unk_token_id = int(special_tokens.get('unk_token_id', 1))
            self.vocab_built = True
            print(f"✅ Basic tokenizer vocabulary loaded, size: {len(self.token_to_id)}")
        except Exception as e:
            print(f"❌ Failed to load basic tokenizer vocabulary: {e}")
            raise


class MLMDataset(Dataset):
    def __init__(self, java_files: List[str], tokenizer: JavaCodeTokenizer,
                 max_length: int = 512, mlm_probability: float = 0.15,
                 short_seq_probability: float = 0.1):
        self.java_files = java_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.short_seq_probability = short_seq_probability
        self.samples = self._prepare_samples()
        print(f"📊 MLM dataset built, total samples: {len(self.samples)}")

    def _prepare_samples(self) -> List[List[int]]:
        print("🔄 Preprocessing Java files to build training samples...")
        samples = []
        processed_files = 0
        for file_path in self.java_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                token_ids = self.tokenizer.encode(code)
                if len(token_ids) > self.max_length - 2:
                    step_size = self.max_length // 2
                    for i in range(0, len(token_ids), step_size):
                        chunk = token_ids[i:i + self.max_length - 2]
                        if len(chunk) >= 10:
                            samples.append(chunk)
                else:
                    if len(token_ids) >= 10:
                        samples.append(token_ids)
                processed_files += 1
                if processed_files % 1000 == 0:
                    print(f"  Processed {processed_files}/{len(self.java_files)} files")
            except Exception as e:
                print(f"⚠️  Failed to process file {file_path}: {e}")
                continue
        print(f"✅ Sample preparation complete, total samples: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids = self.samples[idx].copy()
        if random.random() < self.short_seq_probability:
            max_len = random.randint(10, len(token_ids))
            token_ids = token_ids[:max_len]
        token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        input_ids, labels = self._create_masked_lm_predictions(token_ids)
        seq_len = len(input_ids)
        pad_length = self.max_length - seq_len
        if pad_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
            labels.extend([-100] * pad_length)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _create_masked_lm_predictions(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        input_ids = token_ids.copy()
        labels = [-100] * len(token_ids)
        special_tokens = {
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id
        }
        candidates = []
        for i, token_id in enumerate(token_ids):
            if token_id not in special_tokens:
                candidates.append(i)
        num_to_mask = max(1, int(len(candidates) * self.mlm_probability))
        masked_indices = random.sample(candidates, min(num_to_mask, len(candidates)))
        for idx in masked_indices:
            original_token = token_ids[idx]
            labels[idx] = original_token
            rand = random.random()
            if rand < 0.8:
                input_ids[idx] = self.tokenizer.mask_token_id
            elif rand < 0.9:
                vocab_size = len(self.tokenizer.token_to_id)
                random_token = random.randint(5, vocab_size - 1)
                input_ids[idx] = random_token
        return input_ids, labels


def collect_java_files(data_dir: str, file_level_dir: str = FILE_LEVEL_BASE_DIR, exclude_buggy: bool = True) -> List[str]:
    print(f"🔍 Searching for Java files: {data_dir}")
    java_files: List[str] = []
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    non_buggy_cache: Dict[str, Set[str]] = {}
    missing_versions: Set[str] = set()
    skipped_buggy = 0
    skipped_missing = 0
    for java_file in data_path.rglob("*.java"):
        rel_parts = java_file.relative_to(data_path).parts
        if not rel_parts:
            continue
        version_key = rel_parts[0]
        file_name = java_file.name
        if exclude_buggy:
            if version_key in missing_versions:
                skipped_missing += 1
                continue
            if version_key not in non_buggy_cache:
                csv_path = find_file_level_csv_path(file_level_dir, version_key)
                if csv_path is None:
                    missing_versions.add(version_key)
                    skipped_missing += 1
                    continue
                non_buggy_cache[version_key] = load_non_buggy_filenames_from_csv(str(csv_path))
            allowed_files = non_buggy_cache.get(version_key, set())
            if not allowed_files:
                skipped_missing += 1
                continue
            if file_name not in allowed_files:
                skipped_buggy += 1
                continue
        java_files.append(str(java_file))
    if exclude_buggy:
        print(f"📊 Non-buggy filter complete: kept {len(java_files)} files, skipped buggy/unknown {skipped_buggy}, skipped no-CSV {skipped_missing}")
    else:
        print(f"📊 No bug filtering, kept {len(java_files)} files")
    return java_files


def build_code_corpus(data_dir: str, output_dir: str, vocab_size: int = 50000):
    print("🏗️  Building code corpus...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    java_files = collect_java_files(data_dir)
    if not java_files:
        raise ValueError(f"No Java files found in {data_dir}")
    tokenizer = JavaCodeTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(java_files)
    vocab_path = output_path / "vocab.json"
    tokenizer.save_vocab(str(vocab_path))
    files_list_path = output_path / "java_files.json"
    with open(files_list_path, 'w', encoding='utf-8') as f:
        json.dump(java_files, f, ensure_ascii=False, indent=2)
    print(f"💾 Corpus built, saved to: {output_dir}")
    return tokenizer, java_files


class MLMTransformerModel(TransformerModel):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout, pad_token_id=0):
        super().__init__(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout, pad_token_id)

    def forward(self, input_ids, labels=None):
        logits = super().forward(input_ids)
        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(logits.view(-1, self.fc_out.out_features), labels.view(-1))
            outputs = (mlm_loss,) + outputs
        return outputs


def train_mlm_model(model, train_loader, val_loader, tokenizer, epochs=10,
                   learning_rate=1e-4, save_dir='./checkpoints',
                   experiment_name=None, resume_from=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"🚀 Starting MLM training, using device: {device}")
    monitor = TrainingMonitor(save_dir, experiment_name or 'mlm_training')
    progress_tracker = ProgressTracker(epochs, len(train_loader), log_interval=50)
    try:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                               betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos'
        )
        start_epoch = 0
        if resume_from:
            start_epoch = monitor.load_checkpoint(model, optimizer, scheduler, resume_from)
        progress_tracker.start_training()
        for epoch in range(start_epoch, epochs):
            progress_tracker.start_epoch(epoch)
            epoch_start_time = time.time()
            model.train()
            total_loss = 0
            num_batches = len(train_loader)
            valid_batches = 0
            for batch_idx, (input_ids, labels) in enumerate(train_loader):
                try:
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(input_ids, labels=labels)
                    loss = outputs[0]
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️  Warning: batch {batch_idx} produced invalid loss")
                        continue
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()
                    valid_batches += 1
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_tracker.update_batch(batch_idx, loss.item(), current_lr, grad_norm)
                except Exception as e:
                    print(f"❌ Error in training batch {batch_idx}: {e}")
                    continue
            val_loss = evaluate_mlm_model(model, val_loader, device)
            avg_train_loss = total_loss / max(valid_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            progress_tracker.finish_epoch(avg_train_loss, val_loss, epoch_time)
            is_best = monitor.log_epoch(epoch, avg_train_loss, val_loss, current_lr)
            monitor.save_checkpoint(model, optimizer, scheduler, epoch, is_best)
        progress_tracker.finish_training()
        monitor.plot_training_curves()
        monitor.save_training_log()
        final_model_path = os.path.join(monitor.experiment_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': len(tokenizer.token_to_id),
            'model_config': {
                'd_model': model.d_model,
                'n_heads': len(model.transformer_blocks[0].attention.W_q.weight) // model.d_model if model.transformer_blocks else 8,
                'n_layers': len(model.transformer_blocks),
                'd_ff': model.transformer_blocks[0].feed_forward[0].out_features if model.transformer_blocks else 2048,
                'max_seq_length': model.positional_encoding.pe.size(1),
                'dropout': 0.1,
                'pad_token_id': model.pad_token_id
            }
        }, final_model_path)
        tokenizer.save_vocab(os.path.join(monitor.experiment_dir, 'tokenizer_vocab.json'))
        print(f"💾 Final model saved to: {final_model_path}")
        return monitor
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        progress_tracker.finish_training()
        monitor.save_checkpoint(model, optimizer, scheduler, epoch, False)
        return monitor
    except Exception as e:
        print(f"❌ Error during training: {e}")
        progress_tracker.finish_training()
        raise


def evaluate_mlm_model(model, val_loader, device):
    if val_loader is None or len(val_loader) == 0:
        return float('inf')
    model.eval()
    total_loss = 0
    valid_batches = 0
    try:
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model(input_ids, labels=labels)
                loss = outputs[0]
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_batches += 1
        return total_loss / max(valid_batches, 1)
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return float('inf')


def load_mlm_model(model_path: str, tokenizer_path: str):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        model = MLMTransformerModel(
            vocab_size=checkpoint['vocab_size'],
            **model_config
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = JavaCodeTokenizer()
        tokenizer.load_vocab(tokenizer_path)
        print(f"✅ MLM model loaded successfully")
        print(f"📊 Vocabulary size: {len(tokenizer.token_to_id)}")
        print(f"🏗️  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise


def predict_masked_tokens(model, tokenizer, code_with_masks: str, device='cuda', top_k=5):
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    try:
        input_ids = tokenizer.encode(code_with_masks)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs[0]
        mask_positions = []
        for i, token_id in enumerate(input_ids):
            if token_id == tokenizer.mask_token_id:
                mask_positions.append(i)
        predictions = {}
        for pos in mask_positions:
            position_logits = logits[0, pos, :]
            top_tokens = torch.topk(position_logits, top_k)
            predicted_tokens = []
            for score, token_id in zip(top_tokens.values, top_tokens.indices):
                if token_id.item() in tokenizer.id_to_token:
                    token = tokenizer.id_to_token[token_id.item()]
                    predicted_tokens.append((token, float(score)))
            predictions[pos] = predicted_tokens
        return predictions
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return {}


def mlm_main():
    print("🚀 == Java Code Pretraining (MLM) ==")
    print("Masked Language Model pretraining for Java code")
    input_data_dir = "/root/workspace/lzc/SynDef/transform-traindata"
    output_model_dir = "/root/workspace/lzc/SynDef/transform-model"
    print(f"\n📁 Input data directory: {input_data_dir}")
    print(f"💾 Output model directory: {output_model_dir}")
    try:
        print("\n" + "="*60)
        print("📚 Step 1: Build corpus and vocabulary")
        print("="*60)
        tokenizer, java_files = build_code_corpus(
            data_dir=input_data_dir,
            output_dir=output_model_dir,
            vocab_size=50000
        )
        print("\n" + "="*60)
        print("🗂️  Step 2: Create training datasets")
        print("="*60)
        random.shuffle(java_files)
        split_idx = int(0.9 * len(java_files))
        train_files = java_files[:split_idx]
        val_files = java_files[split_idx:]
        print(f"📊 Training files: {len(train_files)}")
        print(f"📊 Validation files: {len(val_files)}")
        train_dataset = MLMDataset(
            java_files=train_files,
            tokenizer=tokenizer,
            max_length=512,
            mlm_probability=0.15
        )
        val_dataset = MLMDataset(
            java_files=val_files,
            tokenizer=tokenizer,
            max_length=512,
            mlm_probability=0.15
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        print("\n" + "="*60)
        print("🏗️  Step 3: Create MLM model")
        print("="*60)
        model = MLMTransformerModel(
            vocab_size=len(tokenizer.token_to_id),
            d_model=768,
            n_heads=12,
            n_layers=12,
            d_ff=3072,
            max_seq_length=512,
            dropout=0.1,
            pad_token_id=tokenizer.pad_token_id
        )
        apply_weight_initialization(model, 'xavier_uniform')
        create_model_summary(model)
        print("\n" + "="*60)
        print("🚀 Step 4: Start MLM pretraining")
        print("="*60)
        monitor = train_mlm_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            epochs=20,
            learning_rate=1e-4,
            save_dir=output_model_dir,
            experiment_name='java_code_mlm'
        )
        print("\n🎉 == Training complete ==")
        print(f"📊 Best validation loss: {monitor.best_val_loss:.6f}")
        print(f"🏆 Best model epoch: {monitor.best_epoch}")
        print(f"💾 Model saved to: {monitor.experiment_dir}")
        print("\n" + "="*60)
        print("🧪 Step 5: Model testing")
        print("="*60)
        test_code_samples = [
            "public class [MASK] { private int value; }",
            "for (int i = 0; i < [MASK]; i++) { }",
            "String result = [MASK].toString();",
            "if (obj == [MASK]) { return false; }"
        ]
        for code in test_code_samples:
            print(f"\nTest code: {code}")
            predictions = predict_masked_tokens(model, tokenizer, code, device='cuda', top_k=3)
            for pos, preds in predictions.items():
                print(f"  Position {pos} predictions: {preds}")
        print("\n✅ == All done ==")
        print("🎯 Java code pretraining model ready!")
        print("💡 Use this model for code completion, code understanding, etc.")
    except Exception as e:
        print(f"\n❌ Program execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def autoregressive_main():
    print("🚀 Autoregressive training entry point")
    print("Aggregate texts from raw code files (no CSV SRC field), build vocabulary and dataset, then train.")
    input_data_dir = "/root/workspace/lzc/SynDef/transform-traindata"
    if not Path(input_data_dir).exists():
        print(f"❌ Raw data directory does not exist: {input_data_dir}")
        return
    java_files = collect_java_files(
        data_dir=input_data_dir,
        file_level_dir=FILE_LEVEL_BASE_DIR,
        exclude_buggy=True
    )
    if not java_files:
        print("❌ No usable source files collected")
        return
    texts: List[str] = []
    for f in java_files:
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as rf:
                content = rf.read()
                if content and content.strip():
                    texts.append(content)
        except Exception as e:
            print(f"⚠️  Read failed, skipping {f}: {e}")
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    if not texts:
        print("❌ No texts available for training")
        return
    print(f"📊 Text samples: {len(texts)} (source: raw files)")
    tokenizer = BasicTokenizer(vocab_size=50000, min_freq=1)
    tokenizer.fit(texts)
    random.shuffle(texts)
    split_idx = int(0.95 * len(texts)) if len(texts) > 100 else max(1, int(0.8 * len(texts)))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:] if split_idx < len(texts) else texts[: max(1, len(texts)//20)]
    print(f"📚 Training samples: {len(train_texts)} | Validation samples: {len(val_texts)}")
    max_seq_len = 512
    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_seq_len, pad_token_id=tokenizer.pad_token_id)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=max_seq_len, pad_token_id=tokenizer.pad_token_id)
    batch_size = 16 if torch.cuda.is_available() else 8
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    vocab_size = tokenizer.vocab_size_current
    d_model = 512
    model = create_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_length=max_seq_len,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        weight_init='xavier_uniform'
    )
    epochs = 5
    learning_rate = 3e-4
    save_dir = str(Path("/root/workspace/lzc/SynDef/transform-model"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    experiment_name = "autoregressive_from_rawfiles"
    monitor = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        use_warmup=True,
        warmup_steps=4000,
        pad_token_id=tokenizer.pad_token_id,
        save_dir=save_dir,
        experiment_name=experiment_name,
        resume_from=None
    )
    print("\n🎉 Autoregressive training complete")
    print(f"🏆 Best validation loss: {monitor.best_val_loss:.6f} @ epoch {monitor.best_epoch}")
    print(f"💾 Output directory: {monitor.experiment_dir}")
    print("To resume MLM, explicitly call mlm_main()")
    try:
        vocab_out = os.path.join(monitor.experiment_dir, 'tokenizer_vocab.json')
        tokenizer.save_vocab(vocab_out)
    except Exception as e:
        print(f"⚠️  Failed to save basic tokenizer vocabulary: {e}")


if __name__ == "__main__":
    autoregressive_main()