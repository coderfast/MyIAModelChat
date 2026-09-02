#!/usr/bin/env python3
"""
Standalone HTML report generator from training CSV metrics.

Usage:
    python scripts/generate_report.py checkpoints/chat_model_metrics.csv
    python scripts/generate_report.py checkpoints/chat_model_metrics.csv --output report.html
    python scripts/generate_report.py checkpoints/draft_model_metrics.csv --draft
"""
import csv
import json
import os
import sys
import argparse
from datetime import datetime


def read_csv(csv_path):
    """Read CSV metrics file and return parsed data."""
    epochs = []
    train_losses, val_losses = [], []
    train_perplexities, val_perplexities = [], []
    gaps, lrs, grad_norms, tokens_per_secs = [], [], [], []
    thinking_accuracies, thinking_open_accs, thinking_close_accs = [], [], []
    thinking_coverages, response_accuracies = [], []
    agent_tool_call_accs, agent_observation_accs, agent_ratios = [], [], []
    moe_gate_entropy_norms = []
    moe_expert_utils = {}
    mtp_losses = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row.get('epoch', 0)))
                train_losses.append(float(row.get('train_loss', 0)))
                val_loss = row.get('val_loss', '').strip()
                val_losses.append(float(val_loss) if val_loss else None)
                train_perplexities.append(float(row.get('train_perplexity', 0)))
                val_perp = row.get('val_perplexity', '').strip()
                val_perplexities.append(float(val_perp) if val_perp else None)
                gap = row.get('gap', '').strip()
                gaps.append(float(gap) if gap else None)
                lrs.append(float(row.get('lr', 0)))
                grad_norms.append(float(row.get('grad_norm', 0)))
                tokens_per_secs.append(float(row.get('tokens_per_sec', 0)))
                ta = row.get('thinking_accuracy', '').strip()
                thinking_accuracies.append(float(ta) if ta else None)
                to = row.get('thinking_open_acc', '').strip()
                thinking_open_accs.append(float(to) if to else None)
                tc = row.get('thinking_close_acc', '').strip()
                thinking_close_accs.append(float(tc) if tc else None)
                tv = row.get('thinking_coverage', '').strip()
                thinking_coverages.append(float(tv) if tv else None)
                ra = row.get('response_accuracy', '').strip()
                response_accuracies.append(float(ra) if ra else None)
                at = row.get('agent_tool_call_acc', '').strip()
                agent_tool_call_accs.append(float(at) if at else None)
                ao = row.get('agent_observation_acc', '').strip()
                agent_observation_accs.append(float(ao) if ao else None)
                ar = row.get('agent_ratio', '').strip()
                agent_ratios.append(float(ar) if ar else None)
                me = row.get('moe_gate_entropy_norm', '').strip()
                moe_gate_entropy_norms.append(float(me) if me else None)
                for key, value in row.items():
                    if key.startswith('moe_expert_') and key.endswith('_util'):
                        expert_id = key.replace('moe_expert_', '').replace('_util', '')
                        if expert_id not in moe_expert_utils:
                            moe_expert_utils[expert_id] = []
                        ev = value.strip()
                        moe_expert_utils[expert_id].append(float(ev) if ev else None)
                mt = row.get('mtp_loss', '').strip()
                mtp_losses.append(float(mt) if mt else None)
            except (ValueError, TypeError):
                continue

    return {
        'epochs': epochs,
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_perplexities': train_perplexities, 'val_perplexities': val_perplexities,
        'gaps': gaps, 'lrs': lrs, 'grad_norms': grad_norms, 'tokens_per_secs': tokens_per_secs,
        'thinking_accuracies': thinking_accuracies, 'thinking_open_accs': thinking_open_accs,
        'thinking_close_accs': thinking_close_accs, 'thinking_coverages': thinking_coverages,
        'response_accuracies': response_accuracies,
        'agent_tool_call_accs': agent_tool_call_accs, 'agent_observation_accs': agent_observation_accs,
        'agent_ratios': agent_ratios,
        'moe_gate_entropy_norms': moe_gate_entropy_norms, 'moe_expert_utils': moe_expert_utils,
        'mtp_losses': mtp_losses,
    }


def compute_status(train_losses, val_losses):
    """Compute training status from losses."""
    has_val = any(v is not None for v in val_losses)
    if has_val and len(val_losses) >= 2:
        first_val = next(v for v in val_losses if v is not None)
        last_val = next(v for v in reversed(val_losses) if v is not None)
        pct = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
        if last_val < first_val * 0.9:
            return "healthy", f"TRAINING IS GOING WELL - Val loss decreased {abs(pct):.1f}%", "The model is learning and generalizing."
        elif last_val > first_val * 1.1:
            return "overfitting", f"WARNING: OVERFITTING - Val loss increased {pct:.1f}%", "Reduce --val-split or add more data."
        else:
            return "stable", f"TRAINING IS STABLE - Val loss changed {pct:+.1f}%", "Loss is plateauing."
    else:
        if len(train_losses) >= 2:
            first = train_losses[0]
            last = train_losses[-1]
            pct = ((last - first) / first) * 100 if first > 0 else 0
            if last < first * 0.9:
                return "healthy", f"TRAINING IS GOING WELL - Loss decreased {abs(pct):.1f}%", "The model is learning."
            elif last > first * 1.1:
                return "overfitting", f"WARNING: LOSS INCREASING - Loss increased {pct:.1f}%", "Reduce learning rate."
            else:
                return "stable", f"TRAINING IS STABLE - Loss changed {pct:+.1f}%", "Loss plateau."
    return "stable", "TRAINING STATUS", ""


def generate_html(data, checkpoint_name, is_draft=False):
    """Generate HTML report from parsed data."""
    epochs = data['epochs']
    if not epochs:
        return None

    train_losses = data['train_losses']
    val_losses = data['val_losses']
    train_perplexities = data['train_perplexities']
    val_perplexities = data['val_perplexities']
    gaps = data['gaps']
    lrs = data['lrs']
    grad_norms = data['grad_norms']
    tokens_per_secs = data['tokens_per_secs']
    thinking_accuracies = data['thinking_accuracies']
    thinking_open_accs = data['thinking_open_accs']
    thinking_close_accs = data['thinking_close_accs']
    thinking_coverages = data['thinking_coverages']
    response_accuracies = data['response_accuracies']
    agent_tool_call_accs = data['agent_tool_call_accs']
    agent_observation_accs = data['agent_observation_accs']
    agent_ratios = data['agent_ratios']
    moe_gate_entropy_norms = data['moe_gate_entropy_norms']
    moe_expert_utils = data['moe_expert_utils']
    mtp_losses = data['mtp_losses']

    status, status_text, status_detail = compute_status(train_losses, val_losses)
    total_epochs = len(epochs)
    final_train_loss = f"{train_losses[-1]:.4f}" if train_losses else "N/A"
    final_val_loss = next((f"{v:.4f}" for v in reversed(val_losses) if v is not None), "N/A")

    has_thinking = any(v is not None for v in thinking_accuracies)
    has_agent = any(v is not None for v in agent_tool_call_accs)
    has_moe = any(v is not None for v in moe_gate_entropy_norms)
    has_mtp = any(v is not None for v in mtp_losses)

    def safe_list(lst):
        return [v if v is not None else 'null' for v in lst]

    def safe_list_str(lst):
        return json.dumps([v if v is not None else None for v in lst])

    epochs_json = json.dumps(epochs)
    train_losses_json = safe_list_str(train_losses)
    val_losses_json = safe_list_str(val_losses)
    train_ppl_json = safe_list_str(train_perplexities)
    val_ppl_json = safe_list_str(val_perplexities)
    gaps_json = safe_list_str(gaps)
    lrs_json = safe_list_str(lrs)
    grad_norms_json = safe_list_str(grad_norms)
    tps_json = safe_list_str(tokens_per_secs)

    thinking_acc_json = safe_list_str(thinking_accuracies)
    thinking_open_json = safe_list_str(thinking_open_accs)
    thinking_close_json = safe_list_str(thinking_close_accs)
    thinking_cov_json = safe_list_str(thinking_coverages)
    response_acc_json = safe_list_str(response_accuracies)

    agent_tool_json = safe_list_str(agent_tool_call_accs)
    agent_obs_json = safe_list_str(agent_observation_accs)
    agent_ratio_json = safe_list_str(agent_ratios)

    moe_entropy_json = safe_list_str(moe_gate_entropy_norms)
    moe_expert_json = json.dumps({k: [v if v is not None else None for v in vals] for k, vals in moe_expert_utils.items()})

    mtp_loss_json = safe_list_str(mtp_losses)

    # Compute display values
    min_loss = min(train_losses) if train_losses else 0
    min_loss_idx = train_losses.index(min_loss) if train_losses else 0
    min_loss_epoch = epochs[min_loss_idx] if min_loss_idx < len(epochs) else 0
    final_lr = lrs[-1] if lrs else 0
    best_val = next((v for v in reversed(val_losses) if v is not None), None)
    best_val_str = f"{best_val:.4f}" if best_val is not None else "N/A"
    best_val_epoch = epochs[val_losses.index(best_val)] if best_val is not None and best_val in val_losses else "N/A"
    avg_speed = sum(tokens_per_secs) / len(tokens_per_secs) if tokens_per_secs else 0

    report_type = "Draft Model Training Report" if is_draft else "Training Report"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_type} - {checkpoint_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .status {{ padding: 15px 20px; border-radius: 8px; margin: 15px 0; font-weight: bold; border-left: 5px solid; }}
        .status.healthy {{ background: #d4edda; color: #155724; border-color: #28a745; }}
        .status.overfitting {{ background: #f8d7da; color: #721c24; border-color: #dc3545; }}
        .status.stable {{ background: #fff3cd; color: #856404; border-color: #ffc107; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: relative; }}
        .reset-zoom {{ position: absolute; top: 8px; right: 8px; padding: 4px 10px; font-size: 12px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; opacity: 0.8; z-index: 10; }}
        .reset-zoom:hover {{ opacity: 1; background: #2980b9; }}
        .zoom-controls {{ position: absolute; top: 8px; right: 8px; display: flex; gap: 4px; z-index: 10; }}
        .zoom-btn {{ padding: 4px 10px; font-size: 14px; font-weight: bold; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; opacity: 0.8; min-width: 28px; text-align: center; }}
        .zoom-btn:hover {{ opacity: 1; background: #2980b9; }}
        .zoom-btn.reset {{ background: #95a5a6; }}
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        canvas {{ max-height: 300px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }}
        .metric {{ background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .section-divider {{ font-size: 20px; font-weight: bold; color: #333; margin: 30px 0 15px 0; padding: 10px 0; border-bottom: 3px solid #3498db; }}
        .footer {{ text-align: center; margin-top: 30px; padding: 20px; color: #666; font-size: 12px; }}
        @media (max-width: 768px) {{ .charts-grid {{ grid-template-columns: 1fr; }} .summary {{ grid-template-columns: repeat(2, 1fr); }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{report_type}: {checkpoint_name}</h1>
            <div class="status {status}">
                <div>{status_text}</div>
                <div style="font-size: 14px; font-weight: normal; opacity: 0.9;">{status_detail}</div>
            </div>
            <p>Total epochs: {total_epochs} | Final train loss: {final_train_loss} | Final val loss: {final_val_loss}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{final_train_loss}</div>
                <div class="metric-label">Final Train Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_val_loss}</div>
                <div class="metric-label">Final Val Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{min_loss:.4f}</div>
                <div class="metric-label">Best Loss (epoch {min_loss_epoch})</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_lr:.2e}</div>
                <div class="metric-label">Final LR</div>
            </div>
        </div>

        <div class="section-divider">LOSS & PERPLEXITY</div>
        <div class="chart-container">
            <h2>Loss Over Time</h2>
            <canvas id="lossChart"></canvas>
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomIn('lossChart')" title="Zoom In">+</button>
                <button class="zoom-btn" onclick="zoomOut('lossChart')" title="Zoom Out">-</button>
                <button class="zoom-btn reset" onclick="resetZoom('lossChart')" title="Reset Zoom">&#8634;</button>
            </div>
        </div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Perplexity</h2>
                <canvas id="perplexityChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('perplexityChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('perplexityChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('perplexityChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
            <div class="chart-container">
                <h2>Train/Val Gap</h2>
                <canvas id="gapChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('gapChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('gapChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('gapChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h2>Learning Rate</h2>
                <canvas id="lrChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('lrChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('lrChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('lrChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
            <div class="chart-container">
                <h2>Training Speed (tokens/s)</h2>
                <canvas id="speedChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('speedChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('speedChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('speedChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
        </div>

        {f'''
        <div class="section-divider">THINKING TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Thinking Accuracy</h2>
                <canvas id="thinkingAccuracyChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('thinkingAccuracyChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('thinkingAccuracyChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('thinkingAccuracyChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
            <div class="chart-container">
                <h2>Thinking Coverage</h2>
                <canvas id="thinkingCoverageChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('thinkingCoverageChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('thinkingCoverageChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('thinkingCoverageChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
        </div>
        ''' if has_thinking else ''}

        {f'''
        <div class="section-divider">AGENT TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Agent Accuracy</h2>
                <canvas id="agentAccuracyChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('agentAccuracyChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('agentAccuracyChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('agentAccuracyChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
            <div class="chart-container">
                <h2>Agent Token Ratio</h2>
                <canvas id="agentRatioChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('agentRatioChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('agentRatioChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('agentRatioChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
        </div>
        ''' if has_agent else ''}

        {f'''
        <div class="section-divider">MoE TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Gate Entropy (normalized)</h2>
                <canvas id="moeEntropyChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('moeEntropyChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('moeEntropyChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('moeEntropyChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
            <div class="chart-container">
                <h2>Expert Utilization</h2>
                <canvas id="moeUtilChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('moeUtilChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('moeUtilChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('moeUtilChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
        </div>
        ''' if has_moe else ''}

        {f'''
        <div class="section-divider">MTP TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>MTP Loss</h2>
                <canvas id="mtpLossChart"></canvas>
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn('mtpLossChart')" title="Zoom In">+</button>
                    <button class="zoom-btn" onclick="zoomOut('mtpLossChart')" title="Zoom Out">-</button>
                    <button class="zoom-btn reset" onclick="resetZoom('mtpLossChart')" title="Reset Zoom">&#8634;</button>
                </div>
            </div>
        </div>
        ''' if has_mtp else ''}

        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | MyIAModelChat Training Report
        </div>
    </div>

    <script>
        const epochs = {epochs_json};
        const trainLoss = {train_losses_json};
        const valLoss = {val_losses_json};
        const trainPpl = {train_ppl_json};
        const valPpl = {val_ppl_json};
        const gapData = {gaps_json};
        const lrData = {lrs_json};
        const gradNorms = {grad_norms_json};
        const tokensPerSec = {tps_json};
        const thinkingAcc = {thinking_acc_json};
        const thinkingOpen = {thinking_open_json};
        const thinkingClose = {thinking_close_json};
        const thinkingCov = {thinking_cov_json};
        const responseAcc = {response_acc_json};
        const agentTool = {agent_tool_json};
        const agentObs = {agent_obs_json};
        const agentRatio = {agent_ratio_json};
        const moeEntropy = {moe_entropy_json};
        const moeExperts = {moe_expert_json};
        const mtpLoss = {mtp_loss_json};

        const zoomOptions = {{
            zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: 'xy' }},
            pan: {{ enabled: false }}
        }};

        const _panState = {{}};
        function _initPan(id) {{
            const cv = document.getElementById(id);
            if (!cv) return;
            cv.style.cursor = 'grab';
            cv.addEventListener('mousedown', function(e) {{
                if (e.button !== 0) return;
                const ch = Chart.getChart(id);
                if (!ch) return;
                const xs = ch.scales.x, ys = ch.scales.y;
                _panState[id] = {{ active: true, sx: e.clientX, sy: e.clientY, xMin: xs.min, xMax: xs.max, yMin: ys.min, yMax: ys.max }};
                cv.style.cursor = 'grabbing';
                e.preventDefault();
            }});
            cv.addEventListener('mousemove', function(e) {{
                const s = _panState[id];
                if (!s || !s.active) return;
                const ch = Chart.getChart(id);
                if (!ch) return;
                const dx = e.clientX - s.sx, dy = e.clientY - s.sy;
                const cw = cv.width, chh = cv.height;
                const xR = s.xMax - s.xMin, yR = s.yMax - s.yMin;
                ch.zoomScale('x', {{ min: s.xMin - dx / cw * xR, max: s.xMax - dx / cw * xR }}, 'default');
                ch.zoomScale('y', {{ min: s.yMin + dy / chh * yR, max: s.yMax + dy / chh * yR }}, 'default');
                e.preventDefault();
            }});
            window.addEventListener('mouseup', function() {{
                const s = _panState[id];
                if (s && s.active) {{ s.active = false; cv.style.cursor = 'grab'; }}
            }});
        }}

        function resetZoom(id) {{ const c = Chart.getChart(id); if (c) c.resetZoom(); }}
        function zoomIn(id) {{ const c = Chart.getChart(id); if (c) c.zoom(1.2); }}
        function zoomOut(id) {{ const c = Chart.getChart(id); if (c) c.zoom(0.8); }}

        new Chart(document.getElementById('lossChart'), {{
            type: 'line', data: {{ labels: epochs, datasets: [
                {{ label: 'Train Loss', data: trainLoss, borderColor: '#3498db', tension: 0.3, fill: false }},
                {{ label: 'Val Loss', data: valLoss, borderColor: '#e74c3c', tension: 0.3, fill: false }}
            ]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Loss' }} }} }} }}
        }});
        _initPan('lossChart');

        new Chart(document.getElementById('perplexityChart'), {{
            type: 'line', data: {{ labels: epochs, datasets: [
                {{ label: 'Train Perplexity', data: trainPpl, borderColor: '#3498db', tension: 0.3, fill: false }},
                {{ label: 'Val Perplexity', data: valPpl, borderColor: '#e74c3c', tension: 0.3, fill: false }}
            ]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Perplexity' }} }} }} }}
        }});
        _initPan('perplexityChart');

        new Chart(document.getElementById('gapChart'), {{
            type: 'line', data: {{ labels: epochs, datasets: [
                {{ label: 'Train/Val Gap', data: gapData, borderColor: '#f39c12', tension: 0.3, fill: false }}
            ]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Gap' }} }} }} }}
        }});
        _initPan('gapChart');

        new Chart(document.getElementById('lrChart'), {{
            type: 'line', data: {{ labels: epochs, datasets: [
                {{ label: 'Learning Rate', data: lrData, borderColor: '#9b59b6', tension: 0.3, fill: false }}
            ]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Learning Rate' }} }} }} }}
        }});
        _initPan('lrChart');

        new Chart(document.getElementById('speedChart'), {{
            type: 'line', data: {{ labels: epochs, datasets: [
                {{ label: 'Tokens/sec', data: tokensPerSec, borderColor: '#1abc9c', tension: 0.3, fill: false }}
            ]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Tokens/sec' }} }} }} }}
        }});
        _initPan('speedChart');

        {f"new Chart(document.getElementById('thinkingAccuracyChart'), {{ type: 'line', data: {{ labels: epochs, datasets: [{{ label: 'Thinking Acc', data: thinkingAcc, borderColor: '#3498db', tension: 0.3, fill: false }}, {{ label: 'Open Acc', data: thinkingOpen, borderColor: '#2ecc71', tension: 0.3, fill: false }}, {{ label: 'Close Acc', data: thinkingClose, borderColor: '#e74c3c', tension: 0.3, fill: false }}]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Accuracy' }}, min: 0, max: 1 }} }} }} }});" if has_thinking else ''}
        {f"_initPan('thinkingAccuracyChart');" if has_thinking else ''}

        {f"new Chart(document.getElementById('thinkingCoverageChart'), {{ type: 'line', data: {{ labels: epochs, datasets: [{{ label: 'Coverage', data: thinkingCov, borderColor: '#f39c12', tension: 0.3, fill: false }}, {{ label: 'Response Acc', data: responseAcc, borderColor: '#9b59b6', tension: 0.3, fill: false }}]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Value' }}, min: 0, max: 1 }} }} }} }});" if has_thinking else ''}
        {f"_initPan('thinkingCoverageChart');" if has_thinking else ''}

        {f"new Chart(document.getElementById('agentAccuracyChart'), {{ type: 'line', data: {{ labels: epochs, datasets: [{{ label: 'Tool Call Acc', data: agentTool, borderColor: '#3498db', tension: 0.3, fill: false }}, {{ label: 'Observation Acc', data: agentObs, borderColor: '#e74c3c', tension: 0.3, fill: false }}]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Accuracy' }}, min: 0, max: 1 }} }} }} }});" if has_agent else ''}
        {f"_initPan('agentAccuracyChart');" if has_agent else ''}

        {f"new Chart(document.getElementById('agentRatioChart'), {{ type: 'line', data: {{ labels: epochs, datasets: [{{ label: 'Agent Ratio', data: agentRatio, borderColor: '#1abc9c', tension: 0.3, fill: false }}]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Ratio' }} }} }} }} }});" if has_agent else ''}
        {f"_initPan('agentRatioChart');" if has_agent else ''}

        {f"new Chart(document.getElementById('moeEntropyChart'), {{ type: 'line', data: {{ labels: epochs, datasets: [{{ label: 'Gate Entropy (norm)', data: moeEntropy, borderColor: '#3498db', tension: 0.3, fill: false }}]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Entropy' }} }} }} }} }});" if has_moe else ''}
        {f"_initPan('moeEntropyChart');" if has_moe else ''}

        {f"new Chart(document.getElementById('moeUtilChart'), {{ type: 'line', data: {{ labels: epochs, datasets: [{{ label: f'Expert {{k}}', data: v, borderColor: colors[i % colors.length], tension: 0.3, fill: false }} for i, (k, v) in enumerate(moeExperts.items())]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Utilization %' }} }} }} }} }});" if has_moe else ''}
        {f"_initPan('moeUtilChart');" if has_moe else ''}

        {f"new Chart(document.getElementById('mtpLossChart'), {{ type: 'line', data: {{ labels: epochs, datasets: [{{ label: 'MTP Loss', data: mtpLoss, borderColor: '#e74c3c', tension: 0.3, fill: false }}]}}, options: {{ responsive: true, plugins: {{ zoom: zoomOptions }}, scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Loss' }} }} }} }} }});" if has_mtp else ''}
        {f"_initPan('mtpLossChart');" if has_mtp else ''}
    </script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description='Generate HTML training report from CSV metrics')
    parser.add_argument('csv_path', help='Path to CSV metrics file')
    parser.add_argument('--output', '-o', help='Output HTML path (default: same as CSV with .html)')
    parser.add_argument('--draft', action='store_true', help='Mark as draft model report')
    parser.add_argument('--name', default=None, help='Checkpoint name (default: derived from CSV path)')
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        sys.exit(1)

    # Derive checkpoint name
    if args.name:
        checkpoint_name = args.name
    else:
        basename = os.path.splitext(os.path.basename(args.csv_path))[0]
        checkpoint_name = basename.replace('_metrics', '').replace('_draft', '')

    # Output path
    if args.output:
        html_path = args.output
    else:
        html_path = args.csv_path.replace('.csv', '_report.html')

    print(f"Reading CSV: {args.csv_path}")
    data = read_csv(args.csv_path)

    if not data['epochs']:
        print("Error: No epoch data found in CSV")
        sys.exit(1)

    print(f"Generating report: {html_path}")
    html = generate_html(data, checkpoint_name, is_draft=args.draft)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved: {html_path}")
    print(f"  Epochs: {len(data['epochs'])}")
    print(f"  Final train loss: {data['train_losses'][-1]:.4f}" if data['train_losses'] else "")
    print(f"  Open in browser: file:///{os.path.abspath(html_path).replace(os.sep, '/')}")


if __name__ == '__main__':
    main()
