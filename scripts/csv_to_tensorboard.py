#!/usr/bin/env python3
"""
Import training metrics from CSV to TensorBoard.

Usage:
    python scripts/csv_to_tensorboard.py --csv checkpoints/chat_model_metrics.csv
    python scripts/csv_to_tensorboard.py --csv checkpoints/chat_model_metrics.csv --log-dir runs/imported
    python scripts/csv_to_tensorboard.py --csv checkpoints/chat_model_metrics.csv --run-name "old_training"
"""
import argparse
import csv
import os
import sys
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Error: tensorboard not installed. Install with: pip install tensorboard")
    sys.exit(1)


# Column name mapping: CSV column -> TensorBoard tag
COLUMN_MAPPING = {
    'epoch': None,  # used as step, not logged
    'train_loss': 'loss/train',
    'val_loss': 'loss/val',
    'train_perplexity': 'perplexity/train',
    'val_perplexity': 'perplexity/val',
    'gap': 'gap',
    'lr': 'learning_rate',
    'grad_norm': 'grad_norm',
    'tokens_per_sec': 'tokens_per_sec',
    'best_loss': 'best_loss',
    'early_stop_patience': 'early_stop_patience',
    'thinking_accuracy': 'thinking/accuracy',
    'thinking_open_acc': 'thinking/open_accuracy',
    'thinking_close_acc': 'thinking/close_accuracy',
    'thinking_coverage': 'thinking/coverage',
    'response_accuracy': 'thinking/response_accuracy',
    'agent_tool_call_acc': 'agent/tool_call_accuracy',
    'agent_observation_acc': 'agent/observation_accuracy',
    'agent_ratio': 'agent/ratio',
    'moe_gate_entropy_norm': 'moe/gate_entropy_norm',
    'mtp_loss': 'mtp/loss',
}

# Per-expert utilization columns (dynamic based on expert count)
EXPERT_PATTERN = 'moe_expert_{id}_util'


def import_csv(csv_path, log_dir, run_name=None):
    """Import CSV metrics to TensorBoard."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Error: CSV is empty")
        sys.exit(1)

    # Detect expert columns
    expert_cols = {}
    for col in rows[0].keys():
        for i in range(20):  # max 20 experts
            pattern = EXPERT_PATTERN.format(id=i)
            if col == pattern:
                expert_cols[col] = f'moe/expert_{i}_utilization'
                break

    # Setup run name
    if not run_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"imported_{timestamp}"

    run_dir = os.path.join(log_dir, run_name)
    print(f"Importing {len(rows)} epochs from {csv_path}")
    print(f"TensorBoard run: {run_dir}")

    writer = SummaryWriter(log_dir=run_dir)

    logged_count = 0
    for row in rows:
        try:
            epoch = int(row['epoch'])
        except (ValueError, KeyError):
            continue

        step = epoch

        for csv_col, tb_tag in COLUMN_MAPPING.items():
            if tb_tag is None:
                continue
            value = row.get(csv_col, '')
            if value == '' or value is None:
                continue
            try:
                writer.add_scalar(tb_tag, float(value), step)
            except (ValueError, TypeError):
                pass

        # Log expert utilization
        for csv_col, tb_tag in expert_cols.items():
            value = row.get(csv_col, '')
            if value == '' or value is None:
                continue
            try:
                writer.add_scalar(tb_tag, float(value), step)
            except (ValueError, TypeError):
                pass

        logged_count += 1

    writer.close()
    print(f"Done! Imported {logged_count} epochs")
    print(f"Run: tensorboard --logdir={log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Import training metrics from CSV to TensorBoard'
    )
    parser.add_argument('--csv', required=True,
                        help='Path to metrics CSV file')
    parser.add_argument('--log-dir', default='runs',
                        help='TensorBoard log directory (default: runs)')
    parser.add_argument('--run-name', default=None,
                        help='Run name (default: auto-generated with timestamp)')

    args = parser.parse_args()
    import_csv(args.csv, args.log_dir, args.run_name)


if __name__ == '__main__':
    main()
