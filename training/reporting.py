"""Reporting mixin: CSV logging and HTML report generation."""
import os
import csv
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportingMixin:
    """Mixin providing CSV metrics logging and HTML report generation.

    Expects the host class to define: self.config.
    """

    def _log_metrics_csv(self, metrics_row, csv_path):
        """Append a row of metrics to CSV file. Creates header if file doesn't exist.
        Correlates epoch numbers when retraining (continues from last session)."""
        file_exists = os.path.exists(csv_path)
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'train_perplexity', 'val_perplexity',
            'gap', 'lr', 'grad_norm', 'tokens_per_sec', 'best_loss', 'early_stop_patience',
            # Thinking metrics
            'thinking_accuracy', 'thinking_open_acc', 'thinking_close_acc',
            'thinking_coverage', 'response_accuracy',
            # Agent metrics
            'agent_tool_call_acc', 'agent_observation_acc', 'agent_ratio',
            # MoE metrics
            'moe_gate_entropy_norm',
            # MTP metrics
            'mtp_loss',
        ]
        # Add per-expert utilization columns dynamically
        for key in sorted(metrics_row.keys()):
            if key.startswith('moe_expert_') and key not in fieldnames:
                fieldnames.append(key)

        # If file exists, migrate missing columns
        if file_exists:
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_fields = reader.fieldnames or []
                    rows = list(reader)

                # Migrate: add missing columns to existing CSV
                missing = [col for col in fieldnames if col not in existing_fields]
                if missing:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in rows:
                            writer.writerow(row)
            except Exception:
                pass

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_row)



    def _generate_training_report(self, csv_path):
        """Generate an HTML report with training graphs from CSV metrics."""
        if not os.path.exists(csv_path):
            return

        # Read CSV data
        epochs = []
        train_losses = []
        val_losses = []
        train_perplexities = []
        val_perplexities = []
        gaps = []
        lrs = []
        grad_norms = []
        tokens_per_secs = []
        # Thinking metrics
        thinking_accuracies = []
        thinking_open_accs = []
        thinking_close_accs = []
        thinking_coverages = []
        response_accuracies = []
        # Agent metrics
        agent_tool_call_accs = []
        agent_observation_accs = []
        agent_ratios = []
        # MoE metrics
        moe_gate_entropy_norms = []
        moe_expert_utils = {}  # expert_id -> list of utilization values
        # MTP metrics
        mtp_losses = []

        try:
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
                        # Thinking metrics
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
                        # Agent metrics
                        at = row.get('agent_tool_call_acc', '').strip()
                        agent_tool_call_accs.append(float(at) if at else None)
                        ao = row.get('agent_observation_acc', '').strip()
                        agent_observation_accs.append(float(ao) if ao else None)
                        ar = row.get('agent_ratio', '').strip()
                        agent_ratios.append(float(ar) if ar else None)
                        # MoE metrics
                        me = row.get('moe_gate_entropy_norm', '').strip()
                        moe_gate_entropy_norms.append(float(me) if me else None)
                        # Per-expert utilization
                        for key, value in row.items():
                            if key.startswith('moe_expert_') and key.endswith('_util'):
                                expert_id = key.replace('moe_expert_', '').replace('_util', '')
                                if expert_id not in moe_expert_utils:
                                    moe_expert_utils[expert_id] = []
                                ev = value.strip()
                                moe_expert_utils[expert_id].append(float(ev) if ev else None)
                        # MTP metrics
                        mt = row.get('mtp_loss', '').strip()
                        mtp_losses.append(float(mt) if mt else None)
                    except (ValueError, TypeError):
                        continue
        except Exception:
            return

        if not epochs:
            return

        # Determine training status
        has_val = any(v is not None for v in val_losses)
        if has_val and len(val_losses) >= 2:
            first_val = next(v for v in val_losses if v is not None)
            last_val = next(v for v in reversed(val_losses) if v is not None)
            val_change_pct = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
            if last_val < first_val * 0.9:
                status = "healthy"
                status_text = f"TRAINING IS GOING WELL - Val loss decreased {abs(val_change_pct):.1f}%"
                status_detail = "The model is learning and generalizing to new data."
            elif last_val > first_val * 1.1:
                status = "overfitting"
                status_text = f"WARNING: OVERFITTING DETECTED - Val loss increased {val_change_pct:.1f}%"
                status_detail = "The model is memorizing training data but failing on new data. Reduce --val-split or add more data."
            else:
                status = "stable"
                status_text = f"TRAINING IS STABLE - Val loss changed {val_change_pct:+.1f}%"
                status_detail = "Loss is plateauing. Try more epochs or adjust learning rate."
        else:
            first_loss = train_losses[0] if train_losses else 0
            last_loss = train_losses[-1] if train_losses else 0
            loss_change_pct = ((last_loss - first_loss) / first_loss) * 100 if first_loss > 0 else 0
            if last_loss < first_loss * 0.9:
                status = "healthy"
                status_text = f"TRAINING IS GOING WELL - Loss decreased {abs(loss_change_pct):.1f}%"
                status_detail = "The model is learning. Add --val-split 0.1 to monitor generalization."
            elif last_loss > first_loss * 1.1:
                status = "overfitting"
                status_text = f"WARNING: LOSS INCREASING - Loss increased {loss_change_pct:.1f}%"
                status_detail = "Training is diverging. Reduce learning rate or check data quality."
            else:
                status = "stable"
                status_text = f"TRAINING IS STABLE - Loss changed {loss_change_pct:+.1f}%"
                status_detail = "Loss plateau. Try more epochs or adjust learning rate."

        # Thinking status
        has_thinking = any(v is not None for v in thinking_accuracies)
        if has_thinking and len(thinking_accuracies) >= 2:
            first_ta = next(v for v in thinking_accuracies if v is not None)
            last_ta = next(v for v in reversed(thinking_accuracies) if v is not None)
            last_tc = next((v for v in reversed(thinking_close_accs) if v is not None), 0)
            if last_ta > 0.8 and last_tc > 0.7:
                thinking_status = "healthy"
                thinking_status_text = f"THINKING IS LEARNING - Accuracy: {last_ta*100:.0f}% (close: {last_tc*100:.0f}%)"
                thinking_status_detail = "Model is learning to generate thinking blocks correctly."
            elif last_ta < 0.5 or last_tc < 0.3:
                thinking_status = "overfitting"
                thinking_status_text = f"THINKING NEEDS ATTENTION - Accuracy: {last_ta*100:.0f}% (close: {last_tc*100:.0f}%)"
                thinking_status_detail = "Model struggles with thinking delimiters. Check thinking data quality."
            else:
                thinking_status = "stable"
                thinking_status_text = f"THINKING IS STABLE - Accuracy: {last_ta*100:.0f}%"
                thinking_status_detail = "Thinking accuracy is moderate. More epochs may help."
        else:
            thinking_status = None
            thinking_status_text = None
            thinking_status_detail = None

        # Agent status
        has_agent = any(v is not None for v in agent_tool_call_accs)
        if has_agent and len(agent_tool_call_accs) >= 2:
            first_at = next(v for v in agent_tool_call_accs if v is not None)
            last_at = next(v for v in reversed(agent_tool_call_accs) if v is not None)
            last_ao = next((v for v in reversed(agent_observation_accs) if v is not None), 0)
            if last_at > 0.7 and last_ao > 0.6:
                agent_status = "healthy"
                agent_status_text = f"AGENT IS LEARNING - Tool Call: {last_at*100:.0f}% | Observation: {last_ao*100:.0f}%"
                agent_status_detail = "Model is learning tool calls and observations correctly."
            elif last_at < 0.4 or last_ao < 0.3:
                agent_status = "overfitting"
                agent_status_text = f"AGENT NEEDS ATTENTION - Tool Call: {last_at*100:.0f}% | Observation: {last_ao*100:.0f}%"
                agent_status_detail = "Model struggles with agentic tokens. Check agent data quality."
            else:
                agent_status = "stable"
                agent_status_text = f"AGENT IS STABLE - Tool Call: {last_at*100:.0f}%"
                agent_status_detail = "Agent accuracy is moderate. More epochs may help."
        else:
            agent_status = None
            agent_status_text = None
            agent_status_detail = None

        # MoE status
        has_moe = any(v is not None for v in moe_gate_entropy_norms)
        if has_moe and len(moe_gate_entropy_norms) >= 2:
            last_entropy = next(v for v in reversed(moe_gate_entropy_norms) if v is not None)
            # Check if experts are balanced (entropy > 0.7 means good balance)
            if last_entropy > 0.7:
                moe_status = "healthy"
                moe_status_text = f"MoE IS BALANCED - Gate entropy: {last_entropy*100:.0f}% of max"
                moe_status_detail = "Experts are being used evenly. Load balancing is working."
            elif last_entropy < 0.4:
                moe_status = "overfitting"
                moe_status_text = f"MoE EXPERT COLLAPSE - Gate entropy: {last_entropy*100:.0f}% of max"
                moe_status_detail = "One or more experts dominate. Increase load_balance_weight."
            else:
                moe_status = "stable"
                moe_status_text = f"MoE IS MODERATE - Gate entropy: {last_entropy*100:.0f}% of max"
                moe_status_detail = "Expert distribution is moderate. Monitor for collapse."
        else:
            moe_status = None
            moe_status_text = None
            moe_status_detail = None

        # MTP status
        has_mtp = any(v is not None for v in mtp_losses)
        if has_mtp and len(mtp_losses) >= 2:
            first_mtp = next(v for v in mtp_losses if v is not None)
            last_mtp = next(v for v in reversed(mtp_losses) if v is not None)
            mtp_change_pct = ((last_mtp - first_mtp) / first_mtp) * 100 if first_mtp > 0 else 0
            if last_mtp < first_mtp * 0.9:
                mtp_status = "healthy"
                mtp_status_text = f"MTP IS LEARNING - Loss decreased {abs(mtp_change_pct):.1f}%"
                mtp_status_detail = "Multi-Token Prediction heads are learning future token patterns."
            elif last_mtp > first_mtp * 1.1:
                mtp_status = "overfitting"
                mtp_status_text = f"MTP LOSS INCREASING - Loss increased {mtp_change_pct:.1f}%"
                mtp_status_detail = "MTP heads may be overfitting. Reduce mtp_loss_weight or add more data."
            else:
                mtp_status = "stable"
                mtp_status_text = f"MTP IS STABLE - Loss changed {mtp_change_pct:+.1f}%"
                mtp_status_detail = "MTP loss is plateauing. More epochs may help."
        else:
            mtp_status = None
            mtp_status_text = None
            mtp_status_detail = None

        # Precompute display values for HTML template (avoid f-string ternary issues)
        final_train_loss_str = f"{train_losses[-1]:.4f}" if train_losses else "N/A"
        final_val_loss_str = f"{next(v for v in reversed(val_losses) if v is not None):.4f}" if has_val else "N/A"
        final_perplexity_str = f"{train_perplexities[-1]:.2f}" if train_perplexities else "N/A"
        final_tokens_str = f"{tokens_per_secs[-1]:.0f}" if tokens_per_secs else "N/A"
        total_epochs_str = str(epochs[-1]) if epochs else "0"

        # Serialize data for JavaScript (Python None → null, proper JS syntax)
        epochs_json = json.dumps(epochs)
        train_losses_json = json.dumps(train_losses)
        val_losses_json = json.dumps(val_losses)
        train_perplexities_json = json.dumps(train_perplexities)
        val_perplexities_json = json.dumps(val_perplexities)
        gaps_json = json.dumps(gaps)
        lrs_json = json.dumps(lrs)
        tokens_per_secs_json = json.dumps(tokens_per_secs)
        grad_norms_json = json.dumps(grad_norms)
        thinking_accuracies_json = json.dumps(thinking_accuracies)
        thinking_open_accs_json = json.dumps(thinking_open_accs)
        thinking_close_accs_json = json.dumps(thinking_close_accs)
        thinking_coverages_json = json.dumps(thinking_coverages)
        response_accuracies_json = json.dumps(response_accuracies)
        agent_tool_call_accs_json = json.dumps(agent_tool_call_accs)
        agent_observation_accs_json = json.dumps(agent_observation_accs)
        agent_ratios_json = json.dumps(agent_ratios)
        moe_gate_entropy_norms_json = json.dumps(moe_gate_entropy_norms)
        moe_expert_utils_json = json.dumps(moe_expert_utils)
        mtp_losses_json = json.dumps(mtp_losses)

        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - {self.checkpoint_name}</title>
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
        .status-main {{ font-size: 18px; margin-bottom: 5px; }}
        .status-detail {{ font-size: 14px; font-weight: normal; opacity: 0.9; }}
        .status-banner {{ display: flex; align-items: center; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.15); }}
        .status-banner.healthy {{ background: linear-gradient(135deg, #28a745, #20c997); color: white; }}
        .status-banner.overfitting {{ background: linear-gradient(135deg, #dc3545, #e83e8c); color: white; }}
        .status-banner.stable {{ background: linear-gradient(135deg, #ffc107, #fd7e14); color: white; }}
        .banner-icon {{ font-size: 36px; margin-right: 20px; opacity: 0.9; }}
        .banner-content {{ flex: 1; }}
        .banner-title {{ font-size: 20px; font-weight: bold; margin-bottom: 5px; }}
        .banner-detail {{ font-size: 14px; opacity: 0.9; }}
        .section-divider {{ font-size: 20px; font-weight: bold; color: #333; margin: 30px 0 15px 0; padding: 10px 0; border-bottom: 3px solid #3498db; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: relative; }}
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
        @media (max-width: 768px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .summary {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div id="google_translate_element" style="position:fixed;top:10px;right:10px;z-index:9999;background:white;padding:5px 10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:13px;"></div>
    <script src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    <script>
    function googleTranslateElementInit() {{
        new google.translate.TranslateElement({{pageLanguage: 'en', includedLanguages: 'es,fr,de,it,pt,ru,ja,ko,zh-CN,ar,hi,th,vi,nl,pl,sv,da,no,fi,tr,uk,cs,ro,hu,el,bg,hr,sk,sl,lt,lv,et,mt,ga,cy,eu,ca,gl,af,sq,bs,is,lb,mk,sr,be,kk,ky,tg,uz,tk,ka,hy,az', autoDisplay: false}}, 'google_translate_element');
    }}
    </script>
    <style>
        .skiptranslate {{ display: inline !important; }}
        .goog-te-gadget {{ font-family: Roboto, sans-serif !important; font-size: 13px !important; }}
        .goog-te-gadget-simple {{ border: 1px solid #ddd !important; border-radius: 6px !important; padding: 2px 8px !important; background: #f8f8f8 !important; }}
        .goog-te-gadget-simple:hover {{ background: #e8e8e8 !important; }}
        .goog-te-combo {{ font-family: Roboto, sans-serif !important; font-size: 13px !important; border: none !important; background: transparent !important; cursor: pointer !important; }}
        body {{ top: 0 !important; }}
    </style>
    <div class="container">
        <div class="header">
            <h1>Training Report: {self.checkpoint_name}</h1>
            <div class="status {status}">
                <div class="status-main">{status_text}</div>
                <div class="status-detail">{status_detail}</div>
            </div>
            <p>Total epochs: {total_epochs_str} | Final train loss: {final_train_loss_str} | Final val loss: {final_val_loss_str}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{final_train_loss_str}</div>
                <div class="metric-label">Final Train Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_val_loss_str}</div>
                <div class="metric-label">Final Val Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_perplexity_str}</div>
                <div class="metric-label">Final Perplexity</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_tokens_str}</div>
                <div class="metric-label">Tokens/sec</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>Loss Over Time</h2>
            <canvas id="lossChart"></canvas>
            <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('lossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('lossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('lossChart')" title="Reset">&#8634;</button></div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h2>Perplexity</h2>
                <canvas id="perplexityChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('perplexityChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('perplexityChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('perplexityChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Train/Val Gap</h2>
                <canvas id="gapChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('gapChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('gapChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('gapChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h2>Learning Rate</h2>
                <canvas id="lrChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('lrChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('lrChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('lrChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Training Speed (tokens/s)</h2>
                <canvas id="speedChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('speedChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('speedChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('speedChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>

        {f'''
        <div class="section-divider">THINKING TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Thinking Accuracy</h2>
                <canvas id="thinkingAccuracyChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('thinkingAccuracyChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('thinkingAccuracyChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('thinkingAccuracyChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Thinking Coverage</h2>
                <canvas id="thinkingCoverageChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('thinkingCoverageChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('thinkingCoverageChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('thinkingCoverageChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
        <div class="status-banner {thinking_status}">
            <div class="banner-icon">{'OK' if thinking_status == 'healthy' else 'WARNING' if thinking_status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{thinking_status_text}</div>
                <div class="banner-detail">{thinking_status_detail}</div>
            </div>
        </div>
        ''' if has_thinking else ''}

        {f'''
        <div class="section-divider">AGENT TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Agent Accuracy</h2>
                <canvas id="agentAccuracyChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('agentAccuracyChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('agentAccuracyChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('agentAccuracyChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Agent Token Ratio</h2>
                <canvas id="agentRatioChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('agentRatioChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('agentRatioChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('agentRatioChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
        <div class="status-banner {agent_status}">
            <div class="banner-icon">{'OK' if agent_status == 'healthy' else 'WARNING' if agent_status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{agent_status_text}</div>
                <div class="banner-detail">{agent_status_detail}</div>
            </div>
        </div>
        ''' if has_agent else ''}

        {f'''
        <div class="section-divider">MoE TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Gate Entropy (normalized)</h2>
                <canvas id="moeEntropyChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('moeEntropyChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('moeEntropyChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('moeEntropyChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Expert Utilization</h2>
                <canvas id="moeUtilChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('moeUtilChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('moeUtilChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('moeUtilChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
        <div class="status-banner {moe_status}">
            <div class="banner-icon">{'OK' if moe_status == 'healthy' else 'WARNING' if moe_status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{moe_status_text}</div>
                <div class="banner-detail">{moe_status_detail}</div>
            </div>
        </div>
        ''' if has_moe else ''}

        {f'''
        <div class="section-divider">MTP TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>MTP Loss</h2>
                <canvas id="mtpLossChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('mtpLossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('mtpLossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('mtpLossChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
        <div class="status-banner {mtp_status}">
            <div class="banner-icon">{'OK' if mtp_status == 'healthy' else 'WARNING' if mtp_status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{mtp_status_text}</div>
                <div class="banner-detail">{mtp_status_detail}</div>
            </div>
        </div>
        ''' if has_mtp else ''}

        <div class="status-banner {status}">
            <div class="banner-icon">{'OK' if status == 'healthy' else 'WARNING' if status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{status_text}</div>
                <div class="banner-detail">{status_detail}</div>
            </div>
        </div>
    </div>

    <script>
        const epochs = {epochs_json};
        const trainLoss = {train_losses_json};
        const valLoss = {val_losses_json};
        const trainPerplexity = {train_perplexities_json};
        const valPerplexity = {val_perplexities_json};
        const gaps = {gaps_json};
        const lrs = {lrs_json};
        const tokensPerSec = {tokens_per_secs_json};
        const gradNorms = {grad_norms_json};
        const thinkingAccuracy = {thinking_accuracies_json};
        const thinkingOpenAcc = {thinking_open_accs_json};
        const thinkingCloseAcc = {thinking_close_accs_json};
        const thinkingCoverage = {thinking_coverages_json};
        const responseAccuracy = {response_accuracies_json};
        const agentToolCallAcc = {agent_tool_call_accs_json};
        const agentObsAcc = {agent_observation_accs_json};
        const agentRatio = {agent_ratios_json};
        const moeEntropyNorm = {moe_gate_entropy_norms_json};
        const moeExpertUtils = {moe_expert_utils_json};
        const mtpLosses = {mtp_losses_json};

        const _orig = {{}};
        const _view = {{}};
        const _hbars = {{}};
        const _vbars = {{}};

        function _sync(chartId) {{
            const c = Chart.getChart(chartId);
            if (!c) return;
            const v = _view[chartId];
            c.options.scales.x.min = v.xMin; c.options.scales.x.max = v.xMax;
            c.options.scales.y.min = v.yMin; c.options.scales.y.max = v.yMax;
            c.update('none');
            _layout(chartId);
        }}

        function _layout(chartId) {{
            const s = _orig[chartId], v = _view[chartId];
            if (!s || !v) return;
            const xPct = (v.xMax - v.xMin) / s.xR * 100;
            const yPct = (v.yMax - v.yMin) / s.yR * 100;
            const xRoom = s.xR - (v.xMax - v.xMin);
            const yRoom = s.yR - (v.yMax - v.yMin);
            const xLeft = xRoom > 0 ? (v.xMin - s.xMin) / xRoom * (100 - xPct) : 0;
            const yTop = yRoom > 0 ? (v.yMin - s.yMin) / yRoom * (100 - yPct) : 0;
            if (_hbars[chartId]) {{ _hbars[chartId].style.width = xPct + '%'; _hbars[chartId].style.left = xLeft + '%'; }}
            if (_vbars[chartId]) {{ _vbars[chartId].style.height = yPct + '%'; _vbars[chartId].style.top = yTop + '%'; }}
        }}

        function _clamp(v, o) {{
            const xW = v.xMax - v.xMin, yW = v.yMax - v.yMin;
            if (xW > o.xR) {{ v.xMin = o.xMin; v.xMax = o.xMax; }}
            else {{ if (v.xMin < o.xMin) {{ v.xMin = o.xMin; v.xMax = v.xMin + xW; }} if (v.xMax > o.xMax) {{ v.xMax = o.xMax; v.xMin = v.xMax - xW; }} }}
            if (yW > o.yR) {{ v.yMin = o.yMin; v.yMax = o.yMax; }}
            else {{ if (v.yMin < o.yMin) {{ v.yMin = o.yMin; v.yMax = v.yMin + yW; }} if (v.yMax > o.yMax) {{ v.yMax = o.yMax; v.yMin = v.yMax - yW; }} }}
        }}

        function _addScrollbars(chartId) {{
            const chart = Chart.getChart(chartId);
            if (!chart) return;
            const xs = chart.scales.x, ys = chart.scales.y;
            const o = {{ xMin: xs.min, xMax: xs.max, xR: xs.max - xs.min, yMin: ys.min, yMax: ys.max, yR: ys.max - ys.min }};
            const v = {{ xMin: xs.min, xMax: xs.max, yMin: ys.min, yMax: ys.max }};
            _orig[chartId] = o; _view[chartId] = v;
            const box = chart.canvas.parentElement;
            if (box.querySelector('[data-sb]')) return;
            box.style.position = 'relative';

            const hTrack = document.createElement('div');
            hTrack.style.cssText = 'position:relative;width:100%;height:6px;background:#ddd;border-radius:3px;margin-top:4px;overflow:visible;';
            const hThumb = document.createElement('div');
            hThumb.style.cssText = 'position:absolute;top:-1px;height:8px;background:#3498db;border-radius:4px;min-width:20px;cursor:grab;transition:none;';
            hTrack.appendChild(hThumb);
            _hbars[chartId] = hThumb;

            const vTrack = document.createElement('div');
            vTrack.style.cssText = 'position:absolute;right:-6px;top:0;width:6px;height:100%;background:#ddd;border-radius:3px;overflow:visible;z-index:5;';
            const vThumb = document.createElement('div');
            vThumb.style.cssText = 'position:absolute;left:-1px;width:8px;background:#3498db;border-radius:4px;min-height:20px;cursor:grab;transition:none;';
            vTrack.appendChild(vThumb);
            _vbars[chartId] = vThumb;

            box.appendChild(hTrack);
            box.appendChild(vTrack);

            let drag = null;
            const onMove = e => {{
                if (!drag) return;
                e.preventDefault();
                const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
                if (drag.axis === 'x') {{
                    const viewW = v.xMax - v.xMin, room = o.xR - viewW;
                    if (room <= 0) return;
                    v.xMin = drag.startMin + dx / drag.trackSize * room;
                    v.xMax = v.xMin + viewW;
                }} else {{
                    const viewH = v.yMax - v.yMin, room = o.yR - viewH;
                    if (room <= 0) return;
                    v.yMin = drag.startMin + dy / drag.trackSize * room;
                    v.yMax = v.yMin + viewH;
                }}
                _clamp(v, o); _sync(chartId);
            }};
            const onUp = () => {{ if (drag) {{ drag = null; document.body.style.cursor = ''; }} }};
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);

            hThumb.addEventListener('mousedown', e => {{ e.preventDefault(); drag = {{ axis: 'x', sx: e.clientX, sy: e.clientY, startMin: v.xMin, trackSize: hTrack.offsetWidth }}; document.body.style.cursor = 'grabbing'; }});
            vThumb.addEventListener('mousedown', e => {{ e.preventDefault(); drag = {{ axis: 'y', sx: e.clientX, sy: e.clientY, startMin: v.yMin, trackSize: vTrack.offsetHeight }}; document.body.style.cursor = 'grabbing'; }});

            hTrack.addEventListener('click', e => {{ if (e.target === hThumb) return; const rect = hTrack.getBoundingClientRect(); const viewW = v.xMax - v.xMin, room = o.xR - viewW; if (room <= 0) return; const pct = (e.clientX - rect.left) / rect.width; v.xMin = o.xMin + pct * room - viewW / 2; v.xMax = v.xMin + viewW; _clamp(v, o); _sync(chartId); }});
            vTrack.addEventListener('click', e => {{ if (e.target === vThumb) return; const rect = vTrack.getBoundingClientRect(); const viewH = v.yMax - v.yMin, room = o.yR - viewH; if (room <= 0) return; const pct = (e.clientY - rect.top) / rect.height; v.yMin = o.yMin + pct * room - viewH / 2; v.yMax = v.yMin + viewH; _clamp(v, o); _sync(chartId); }});

            _layout(chartId);
        }}

        function resetZoom(chartId) {{
            const o = _orig[chartId], v = _view[chartId];
            if (!o || !v) return;
            v.xMin = o.xMin; v.xMax = o.xMax; v.yMin = o.yMin; v.yMax = o.yMax;
            _sync(chartId);
        }}

        function zoomIn(chartId) {{
            const o = _orig[chartId], v = _view[chartId];
            if (!o || !v) return;
            const cx = (v.xMin + v.xMax) / 2, cy = (v.yMin + v.yMax) / 2;
            const hx = (v.xMax - v.xMin) / 2 * 0.7, hy = (v.yMax - v.yMin) / 2 * 0.7;
            v.xMin = cx - hx; v.xMax = cx + hx; v.yMin = cy - hy; v.yMax = cy + hy;
            _clamp(v, o); _sync(chartId);
        }}

        function zoomOut(chartId) {{
            const o = _orig[chartId], v = _view[chartId];
            if (!o || !v) return;
            const cx = (v.xMin + v.xMax) / 2, cy = (v.yMin + v.yMax) / 2;
            const hx = Math.min((v.xMax - v.xMin) / 2 * 1.4, o.xR / 2);
            const hy = Math.min((v.yMax - v.yMin) / 2 * 1.4, o.yR / 2);
            v.xMin = cx - hx; v.xMax = cx + hx; v.yMin = cy - hy; v.yMax = cy + hy;
            _clamp(v, o); _sync(chartId);
        }}

        // Loss Chart
        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Train Loss',
                        data: trainLoss,
                        borderColor: '#3498db',
                        tension: 0.3,
                        fill: false
                    }},
                    {{
                        label: 'Val Loss',
                        data: valLoss,
                        borderColor: '#e74c3c',
                        tension: 0.3,
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Loss' }} }}
                }}
            }}
        }});
        _addScrollbars('lossChart');

        // Perplexity Chart
        new Chart(document.getElementById('perplexityChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Train Perplexity',
                        data: trainPerplexity,
                        borderColor: '#2ecc71',
                        tension: 0.3,
                        fill: false
                    }},
                    {{
                        label: 'Val Perplexity',
                        data: valPerplexity,
                        borderColor: '#e67e22',
                        tension: 0.3,
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Perplexity' }} }}
                }}
            }}
        }});
        _addScrollbars('perplexityChart');

        // Gap Chart
        new Chart(document.getElementById('gapChart'), {{
            type: 'bar',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Val - Train Loss',
                    data: gaps,
                    backgroundColor: gaps.map(v => v === null ? 'transparent' : v > 0 ? 'rgba(231, 76, 60, 0.6)' : 'rgba(46, 204, 113, 0.6)'),
                    borderColor: gaps.map(v => v === null ? 'transparent' : v > 0 ? '#e74c3c' : '#2ecc71'),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Gap' }} }}
                }},
                plugins: {{
                    annotation: {{
                        annotations: {{
                            zeroLine: {{
                                type: 'line',
                                yMin: 0,
                                yMax: 0,
                                borderColor: '#999',
                                borderDash: [5, 5]
                            }}
                        }}
                    }}
                }}
            }}
        }});
        _addScrollbars('gapChart');

        // Learning Rate Chart
        new Chart(document.getElementById('lrChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Learning Rate',
                    data: lrs,
                    borderColor: '#9b59b6',
                    tension: 0.3,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Learning Rate' }} }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                return 'LR: ' + ctx.parsed.y.toFixed(10);
                            }}
                        }}
                    }}
                }}
            }}
        }});
        _addScrollbars('lrChart');

        // Speed Chart
        new Chart(document.getElementById('speedChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Tokens/sec',
                    data: tokensPerSec,
                    borderColor: '#1abc9c',
                    tension: 0.3,
                    fill: true,
                    backgroundColor: 'rgba(26, 188, 156, 0.2)'
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Tokens/sec' }} }}
                }}
            }}
        }});
        _addScrollbars('speedChart');

        // Thinking Accuracy Chart
        if (thinkingAccuracy.some(v => v !== null)) {{
            new Chart(document.getElementById('thinkingAccuracyChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [
                        {{ label: 'Overall', data: thinkingAccuracy, borderColor: '#3498db', tension: 0.3, fill: false }},
                        {{ label: 'Open <|thinking|>', data: thinkingOpenAcc, borderColor: '#2ecc71', tension: 0.3, fill: false }},
                        {{ label: 'Close <|final|>', data: thinkingCloseAcc, borderColor: '#e74c3c', tension: 0.3, fill: false }},
                        {{ label: 'Response', data: responseAccuracy, borderColor: '#9b59b6', tension: 0.3, fill: false }}
                    ]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Accuracy' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
            _addScrollbars('thinkingAccuracyChart');
        }}

        // Thinking Coverage Chart
        if (thinkingCoverage.some(v => v !== null)) {{
            new Chart(document.getElementById('thinkingCoverageChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'Thinking Coverage',
                        data: thinkingCoverage,
                        borderColor: '#f39c12',
                        tension: 0.3,
                        fill: true,
                        backgroundColor: 'rgba(243, 156, 18, 0.2)'
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Coverage' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
            _addScrollbars('thinkingCoverageChart');
        }}

        // Agent Accuracy Chart
        if (agentToolCallAcc.some(v => v !== null)) {{
            new Chart(document.getElementById('agentAccuracyChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [
                        {{ label: 'Tool Call', data: agentToolCallAcc, borderColor: '#e67e22', tension: 0.3, fill: false }},
                        {{ label: 'Observation', data: agentObsAcc, borderColor: '#1abc9c', tension: 0.3, fill: false }}
                    ]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Accuracy' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
            _addScrollbars('agentAccuracyChart');
        }}

        // Agent Ratio Chart
        if (agentRatio.some(v => v !== null)) {{
            new Chart(document.getElementById('agentRatioChart'), {{
                type: 'bar',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'Agent Token Ratio',
                        data: agentRatio,
                        backgroundColor: 'rgba(230, 126, 34, 0.6)',
                        borderColor: '#e67e22',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Ratio' }}, min: 0 }}
                    }}
                }}
            }});
            _addScrollbars('agentRatioChart');
        }}

        // MoE Entropy Chart
        if (moeEntropyNorm.some(v => v !== null)) {{
            new Chart(document.getElementById('moeEntropyChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'Normalized Entropy',
                        data: moeEntropyNorm,
                        borderColor: '#8e44ad',
                        tension: 0.3,
                        fill: true,
                        backgroundColor: 'rgba(142, 68, 173, 0.2)'
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Entropy (0=collapsed, 1=balanced)' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
            _addScrollbars('moeEntropyChart');
        }}

        // MoE Expert Utilization Chart
        if (Object.keys(moeExpertUtils).length > 0) {{
            const expertColors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];
            const datasets = Object.keys(moeExpertUtils).map((key, i) => ({{
                label: 'Expert ' + key,
                data: moeExpertUtils[key],
                backgroundColor: expertColors[i % expertColors.length] + '99',
                borderColor: expertColors[i % expertColors.length],
                borderWidth: 1
            }}));
            new Chart(document.getElementById('moeUtilChart'), {{
                type: 'bar',
                data: {{ labels: epochs, datasets: datasets }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ stacked: true, title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ stacked: true, title: {{ display: true, text: 'Utilization' }}, min: 0 }}
                    }}
                }}
            }});
            _addScrollbars('moeUtilChart');
        }}

        // MTP Loss Chart
        if (mtpLosses.some(v => v !== null)) {{
            new Chart(document.getElementById('mtpLossChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'MTP Loss',
                        data: mtpLosses,
                        borderColor: '#e74c3c',
                        tension: 0.3,
                        fill: true,
                        backgroundColor: 'rgba(231, 76, 60, 0.2)'
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'MTP Loss' }} }}
                    }}
                }}
            }});
            _addScrollbars('mtpLossChart');
        }}
    </script>
</body>
</html>"""

        # Write HTML file
        html_path = csv_path.replace('.csv', '_report.html')
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            if self.rank == 0:
                logger.info(f"  Training report saved to {html_path}")
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"  Could not generate training report: {e}")




    def _log_draft_csv(self, metrics_row, csv_path):
        """Append a row of draft model metrics to CSV file."""
        file_exists = os.path.exists(csv_path)
        fieldnames = [
            'epoch', 'loss', 'lr', 'kd_loss', 'hard_loss', 'tokens_per_sec',
        ]
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_row)



    def _generate_draft_report(self, csv_path):
        """Generate an HTML report for draft model training."""
        if not os.path.exists(csv_path):
            return

        epochs = []
        losses = []
        lrs = []
        kd_losses = []
        hard_losses = []
        tokens_per_secs = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        epochs.append(int(row.get('epoch', 0)))
                        losses.append(float(row.get('loss', 0)))
                        lrs.append(float(row.get('lr', 0)))
                        kl = row.get('kd_loss', '').strip()
                        kd_losses.append(float(kl) if kl else None)
                        hl = row.get('hard_loss', '').strip()
                        hard_losses.append(float(hl) if hl else None)
                        ts = row.get('tokens_per_sec', '').strip()
                        tokens_per_secs.append(float(ts) if ts else 0)
                    except (ValueError, TypeError):
                        continue
        except Exception:
            return

        if not epochs:
            return

        # Determine status
        if len(losses) >= 2:
            change_pct = ((losses[-1] - losses[0]) / losses[0]) * 100 if losses[0] > 0 else 0
            if losses[-1] < losses[0] * 0.9:
                status = "healthy"
                status_text = f"DRAFT TRAINING GOING WELL - Loss decreased {abs(change_pct):.1f}%"
                status_detail = "The draft model is learning from the target model."
            elif losses[-1] > losses[0] * 1.1:
                status = "overfitting"
                status_text = f"WARNING - Loss increased {change_pct:.1f}%"
                status_detail = "Draft model may be overfitting. Try reducing kd_epochs."
            else:
                status = "stable"
                status_text = f"STABLE - Loss changed {change_pct:.1f}%"
                status_detail = "Draft model training is stable."
        else:
            status = "stable"
            status_text = "DRAFT MODEL TRAINING"
            status_detail = "Insufficient data for status determination."

        # Summary values
        final_loss_str = f"{losses[-1]:.4f}" if losses else "N/A"
        final_lr_str = f"{lrs[-1]:.2e}" if lrs else "N/A"
        total_epochs_str = str(epochs[-1]) if epochs else "0"
        min_loss_str = f"{min(losses):.4f}" if losses else "N/A"

        # Serialize data for JavaScript
        epochs_json = json.dumps(epochs)
        losses_json = json.dumps(losses)
        lrs_json = json.dumps(lrs)
        kd_losses_json = json.dumps(kd_losses)
        hard_losses_json = json.dumps(hard_losses)

        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Draft Model Training Report - {self.checkpoint_name}</title>
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
        .status-main {{ font-size: 18px; margin-bottom: 5px; }}
        .status-detail {{ font-size: 14px; font-weight: normal; opacity: 0.9; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: relative; }}
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
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; }}
        .badge-draft {{ background: #17a2b8; color: white; }}
        @media (max-width: 768px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .summary {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div id="google_translate_element" style="position:fixed;top:10px;right:10px;z-index:9999;background:white;padding:5px 10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:13px;"></div>
    <script src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    <script>
    function googleTranslateElementInit() {{
        new google.translate.TranslateElement({{pageLanguage: 'en', includedLanguages: 'es,fr,de,it,pt,ru,ja,ko,zh-CN,ar,hi,th,vi,nl,pl,sv,da,no,fi,tr,uk,cs,ro,hu,el,bg,hr,sk,sl,lt,lv,et,mt,ga,cy,eu,ca,gl,af,sq,bs,is,lb,mk,sr,be,kk,ky,tg,uz,tk,ka,hy,az', autoDisplay: false}}, 'google_translate_element');
    }}
    </script>
    <style>
        .skiptranslate {{ display: inline !important; }}
        .goog-te-gadget {{ font-family: Roboto, sans-serif !important; font-size: 13px !important; }}
        .goog-te-gadget-simple {{ border: 1px solid #ddd !important; border-radius: 6px !important; padding: 2px 8px !important; background: #f8f8f8 !important; }}
        .goog-te-gadget-simple:hover {{ background: #e8e8e8 !important; }}
        .goog-te-combo {{ font-family: Roboto, sans-serif !important; font-size: 13px !important; border: none !important; background: transparent !important; cursor: pointer !important; }}
        body {{ top: 0 !important; }}
    </style>
    <div class="container">
        <div class="header">
            <h1>Draft Model Training Report: {self.checkpoint_name} <span class="badge badge-draft">DRAFT</span></h1>
            <div class="status {status}">
                <div class="status-main">{status_text}</div>
                <div class="status-detail">{status_detail}</div>
            </div>
            <p>Total epochs: {total_epochs_str} | Final loss: {final_loss_str} | Min loss: {min_loss_str}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{final_loss_str}</div>
                <div class="metric-label">Final Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{min_loss_str}</div>
                <div class="metric-label">Min Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_lr_str}</div>
                <div class="metric-label">Final LR</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_epochs_str}</div>
                <div class="metric-label">Total Epochs</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('lossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('lossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('lossChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <canvas id="lrChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('lrChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('lrChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('lrChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <canvas id="kdLossChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('kdLossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('kdLossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('kdLossChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <canvas id="hardLossChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('hardLossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('hardLossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('hardLossChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
    </div>

    <script>
        const epochs = {epochs_json};
        const losses = {losses_json};
        const lrs = {lrs_json};
        const kdLosses = {kd_losses_json};
        const hardLosses = {hard_losses_json};

        const _orig = {{}};
        const _view = {{}};
        const _hbars = {{}};
        const _vbars = {{}};

        function _sync(chartId) {{
            const c = Chart.getChart(chartId);
            if (!c) return;
            const v = _view[chartId];
            c.options.scales.x.min = v.xMin; c.options.scales.x.max = v.xMax;
            c.options.scales.y.min = v.yMin; c.options.scales.y.max = v.yMax;
            c.update('none');
            _layout(chartId);
        }}

        function _layout(chartId) {{
            const s = _orig[chartId], v = _view[chartId];
            if (!s || !v) return;
            const xPct = (v.xMax - v.xMin) / s.xR * 100;
            const yPct = (v.yMax - v.yMin) / s.yR * 100;
            const xRoom = s.xR - (v.xMax - v.xMin);
            const yRoom = s.yR - (v.yMax - v.yMin);
            const xLeft = xRoom > 0 ? (v.xMin - s.xMin) / xRoom * (100 - xPct) : 0;
            const yTop = yRoom > 0 ? (v.yMin - s.yMin) / yRoom * (100 - yPct) : 0;
            if (_hbars[chartId]) {{ _hbars[chartId].style.width = xPct + '%'; _hbars[chartId].style.left = xLeft + '%'; }}
            if (_vbars[chartId]) {{ _vbars[chartId].style.height = yPct + '%'; _vbars[chartId].style.top = yTop + '%'; }}
        }}

        function _clamp(v, o) {{
            const xW = v.xMax - v.xMin, yW = v.yMax - v.yMin;
            if (xW > o.xR) {{ v.xMin = o.xMin; v.xMax = o.xMax; }}
            else {{ if (v.xMin < o.xMin) {{ v.xMin = o.xMin; v.xMax = v.xMin + xW; }} if (v.xMax > o.xMax) {{ v.xMax = o.xMax; v.xMin = v.xMax - xW; }} }}
            if (yW > o.yR) {{ v.yMin = o.yMin; v.yMax = o.yMax; }}
            else {{ if (v.yMin < o.yMin) {{ v.yMin = o.yMin; v.yMax = v.yMin + yW; }} if (v.yMax > o.yMax) {{ v.yMax = o.yMax; v.yMin = v.yMax - yW; }} }}
        }}

        function _addScrollbars(chartId) {{
            const chart = Chart.getChart(chartId);
            if (!chart) return;
            const xs = chart.scales.x, ys = chart.scales.y;
            const o = {{ xMin: xs.min, xMax: xs.max, xR: xs.max - xs.min, yMin: ys.min, yMax: ys.max, yR: ys.max - ys.min }};
            const v = {{ xMin: xs.min, xMax: xs.max, yMin: ys.min, yMax: ys.max }};
            _orig[chartId] = o; _view[chartId] = v;
            const box = chart.canvas.parentElement;
            if (box.querySelector('[data-sb]')) return;
            box.style.position = 'relative';

            const hTrack = document.createElement('div');
            hTrack.style.cssText = 'position:relative;width:100%;height:6px;background:#ddd;border-radius:3px;margin-top:4px;overflow:visible;';
            const hThumb = document.createElement('div');
            hThumb.style.cssText = 'position:absolute;top:-1px;height:8px;background:#3498db;border-radius:4px;min-width:20px;cursor:grab;transition:none;';
            hTrack.appendChild(hThumb);
            _hbars[chartId] = hThumb;

            const vTrack = document.createElement('div');
            vTrack.style.cssText = 'position:absolute;right:-6px;top:0;width:6px;height:100%;background:#ddd;border-radius:3px;overflow:visible;z-index:5;';
            const vThumb = document.createElement('div');
            vThumb.style.cssText = 'position:absolute;left:-1px;width:8px;background:#3498db;border-radius:4px;min-height:20px;cursor:grab;transition:none;';
            vTrack.appendChild(vThumb);
            _vbars[chartId] = vThumb;

            box.appendChild(hTrack);
            box.appendChild(vTrack);

            let drag = null;
            const onMove = e => {{
                if (!drag) return;
                e.preventDefault();
                const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
                if (drag.axis === 'x') {{
                    const viewW = v.xMax - v.xMin, room = o.xR - viewW;
                    if (room <= 0) return;
                    v.xMin = drag.startMin + dx / drag.trackSize * room;
                    v.xMax = v.xMin + viewW;
                }} else {{
                    const viewH = v.yMax - v.yMin, room = o.yR - viewH;
                    if (room <= 0) return;
                    v.yMin = drag.startMin + dy / drag.trackSize * room;
                    v.yMax = v.yMin + viewH;
                }}
                _clamp(v, o); _sync(chartId);
            }};
            const onUp = () => {{ if (drag) {{ drag = null; document.body.style.cursor = ''; }} }};
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);

            hThumb.addEventListener('mousedown', e => {{ e.preventDefault(); drag = {{ axis: 'x', sx: e.clientX, sy: e.clientY, startMin: v.xMin, trackSize: hTrack.offsetWidth }}; document.body.style.cursor = 'grabbing'; }});
            vThumb.addEventListener('mousedown', e => {{ e.preventDefault(); drag = {{ axis: 'y', sx: e.clientX, sy: e.clientY, startMin: v.yMin, trackSize: vTrack.offsetHeight }}; document.body.style.cursor = 'grabbing'; }});

            hTrack.addEventListener('click', e => {{ if (e.target === hThumb) return; const rect = hTrack.getBoundingClientRect(); const viewW = v.xMax - v.xMin, room = o.xR - viewW; if (room <= 0) return; const pct = (e.clientX - rect.left) / rect.width; v.xMin = o.xMin + pct * room - viewW / 2; v.xMax = v.xMin + viewW; _clamp(v, o); _sync(chartId); }});
            vTrack.addEventListener('click', e => {{ if (e.target === vThumb) return; const rect = vTrack.getBoundingClientRect(); const viewH = v.yMax - v.yMin, room = o.yR - viewH; if (room <= 0) return; const pct = (e.clientY - rect.top) / rect.height; v.yMin = o.yMin + pct * room - viewH / 2; v.yMax = v.yMin + viewH; _clamp(v, o); _sync(chartId); }});

            _layout(chartId);
        }}

        function resetZoom(chartId) {{
            const o = _orig[chartId], v = _view[chartId];
            if (!o || !v) return;
            v.xMin = o.xMin; v.xMax = o.xMax; v.yMin = o.yMin; v.yMax = o.yMax;
            _sync(chartId);
        }}

        function zoomIn(chartId) {{
            const o = _orig[chartId], v = _view[chartId];
            if (!o || !v) return;
            const cx = (v.xMin + v.xMax) / 2, cy = (v.yMin + v.yMax) / 2;
            const hx = (v.xMax - v.xMin) / 2 * 0.7, hy = (v.yMax - v.yMin) / 2 * 0.7;
            v.xMin = cx - hx; v.xMax = cx + hx; v.yMin = cy - hy; v.yMax = cy + hy;
            _clamp(v, o); _sync(chartId);
        }}

        function zoomOut(chartId) {{
            const o = _orig[chartId], v = _view[chartId];
            if (!o || !v) return;
            const cx = (v.xMin + v.xMax) / 2, cy = (v.yMin + v.yMax) / 2;
            const hx = Math.min((v.xMax - v.xMin) / 2 * 1.4, o.xR / 2);
            const hy = Math.min((v.yMax - v.yMin) / 2 * 1.4, o.yR / 2);
            v.xMin = cx - hx; v.xMax = cx + hx; v.yMin = cy - hy; v.yMax = cy + hy;
            _clamp(v, o); _sync(chartId);
        }}

        // Loss Chart
        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Total Loss',
                    data: losses,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231,76,60,0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Draft Model - Total Loss' }} }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'Loss' }} }},
                    x: {{ title: {{ display: true, text: 'Epoch' }} }}
                }}
            }}
        }});
        _addScrollbars('lossChart');

        // Learning Rate Chart
        new Chart(document.getElementById('lrChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Learning Rate',
                    data: lrs,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52,152,219,0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Draft Model - Learning Rate' }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                return 'LR: ' + ctx.parsed.y.toFixed(10);
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'LR' }} }},
                    x: {{ title: {{ display: true, text: 'Epoch' }} }}
                }}
            }}
        }});
        _addScrollbars('lrChart');

        // KD Loss Chart (only if KD enabled)
        if (kdLosses.some(v => v !== null)) {{
            new Chart(document.getElementById('kdLossChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'KD Loss',
                        data: kdLosses,
                        borderColor: '#9b59b6',
                        backgroundColor: 'rgba(155,89,182,0.1)',
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Draft Model - Knowledge Distillation Loss' }} }},
                    scales: {{
                        y: {{ title: {{ display: true, text: 'KD Loss' }} }},
                        x: {{ title: {{ display: true, text: 'Epoch' }} }}
                    }}
                }}
            }});
            _addScrollbars('kdLossChart');
        }} else {{
            document.getElementById('kdLossChart').parentElement.style.display = 'none';
        }}

        // Hard Loss Chart (only if KD enabled)
        if (hardLosses.some(v => v !== null)) {{
            new Chart(document.getElementById('hardLossChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'Hard Loss',
                        data: hardLosses,
                        borderColor: '#e67e22',
                        backgroundColor: 'rgba(230,126,34,0.1)',
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Draft Model - Next-Token Loss' }} }},
                    scales: {{
                        y: {{ title: {{ display: true, text: 'Hard Loss' }} }},
                        x: {{ title: {{ display: true, text: 'Epoch' }} }}
                    }}
                }}
            }});
            _addScrollbars('hardLossChart');
        }} else {{
            document.getElementById('hardLossChart').parentElement.style.display = 'none';
        }}
    </script>
</body>
</html>"""

        # Write HTML file
        html_path = csv_path.replace('.csv', '_report.html')
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            if self.rank == 0:
                logger.info(f"  Draft training report saved to {html_path}")
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"  Could not generate draft training report: {e}")




