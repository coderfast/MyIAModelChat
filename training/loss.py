"""Loss computation, metrics, and backward pass mixin."""
import torch


class LossMixin:
    """Mixin providing loss computation, thinking/agent metrics, and backward pass.

    Expects the host class to define: self.tokenizer, self.config,
    self.has_thinking_data, self.thinking_loss_weight, self.use_mixed_precision.
    """

    def _compute_loss(self, model, inputs, targets, criterion):
        """Unified loss computation with thinking-aware, mode-aware, and agentic loss weighting.

        Loss rules by sample type:
        - CONTEXT samples (<|problem|> prefix, no <|thinking|>): tokens before <|final|> get
          weight 0.0, <|final|> and answer tokens get weight 1.0.
        - THINKING samples (contain <|thinking|>): tokens before <|thinking|> get weight 0.0,
          <|thinking|>...<|final|> tokens get thinking_loss_weight, <|final|> and answer
          tokens get weight 1.0.
        - AGENT samples (with <tool_call>): thinking gets thinking_loss_weight, tool_call and
          observation tokens get full weight (1.0), answer tokens get full weight (1.0).
        - MoE models: adds load balancing loss to encourage uniform expert usage.
        """
        # Handle MoE model output (returns logits + gate_scores)
        gate_scores = None
        mtp_logits_list = None
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            if len(outputs) == 2 and isinstance(outputs[1], list):
                # Distinguish MoE+MTP (has mtp_heads) from pure MoE (no mtp_heads)
                if hasattr(model, 'mtp_heads'):
                    outputs, gate_scores_list = outputs
                    gate_scores = model._all_gate_scores if model._all_gate_scores else None
                    # MTP logits stored as side channel by ChatModelMoEMTP.forward()
                    mtp_logits_list = getattr(model, '_mtp_logits', None)
                else:
                    # Pure MoE: second element is gate_scores list
                    outputs, gate_scores_list = outputs
                    gate_scores = model._all_gate_scores if model._all_gate_scores else None
            elif len(outputs) == 2:
                outputs, gate_scores = outputs
        raw_outputs = outputs

        weights = torch.ones_like(targets, dtype=torch.float)

        thinking_id = self.tokenizer.get_thinking_index()
        thinking_end_id = self.tokenizer.get_thinking_end_index()
        problem_id = getattr(self.tokenizer, 'get_problem_index', lambda: -1)()
        final_id = getattr(self.tokenizer, 'get_final_index', lambda: -1)()
        thinking_mode_id = getattr(self.tokenizer, 'get_thinking_mode_index', lambda: -1)()
        tool_call_id = getattr(self.tokenizer, 'get_tool_call_index', lambda: -1)()
        tool_call_end_id = getattr(self.tokenizer, 'get_tool_call_end_index', lambda: -1)()
        tool_result_id = getattr(self.tokenizer, 'get_tool_result_index', lambda: -1)()
        end_id = getattr(self.tokenizer, 'get_end_index', lambda: -1)()
        assistant_id = getattr(self.tokenizer, 'get_assistant_index', lambda: -1)()

        agent_enabled = getattr(self.config, 'agent_enabled', False)
        agent_loss_weight = getattr(self.config, 'agent_loss_weight', 1.0)

        batch_size, seq_len = targets.shape

        for b in range(batch_size):
            has_problem_prefix = False
            has_thinking_prefix = False

            # Detect thinking by presence of <|thinking|> token anywhere in the row
            if thinking_id >= 0 and thinking_id in targets[b]:
                has_thinking_prefix = True
            elif problem_id >= 0:
                first_non_pad = -1
                for s in range(seq_len):
                    tok = targets[b, s].item()
                    if tok != self.tokenizer.get_pad_index() and tok != self.tokenizer.get_unk_index():
                        first_non_pad = s
                        break

                if first_non_pad >= 0:
                    first_token = targets[b, first_non_pad].item()
                    if first_token == problem_id:
                        has_problem_prefix = True

            if has_problem_prefix:
                in_preamble = True
                for s in range(seq_len):
                    token = targets[b, s].item()
                    if token == final_id:
                        in_preamble = False
                        weights[b, s] = 1.0
                    elif in_preamble:
                        weights[b, s] = 0.0

            elif has_thinking_prefix:
                in_preamble = True
                in_thinking = False
                past_answer = False

                for s in range(seq_len):
                    token = targets[b, s].item()

                    if token == thinking_id:
                        in_preamble = False
                        in_thinking = True
                        weights[b, s] = 1.0
                    elif token == thinking_end_id:
                        in_thinking = False
                        weights[b, s] = 1.0
                    elif token == final_id:
                        past_answer = True
                        in_thinking = False
                        weights[b, s] = 1.0
                    elif in_preamble:
                        weights[b, s] = 0.0
                    elif in_thinking:
                        weights[b, s] = self.thinking_loss_weight
                    else:
                        weights[b, s] = 1.0

            elif self.has_thinking_data:
                in_thinking = False
                for s in range(seq_len):
                    token = targets[b, s].item()
                    if token == thinking_end_id:
                        in_thinking = False
                    if token == thinking_id:
                        in_thinking = True
                    if in_thinking and token != thinking_id and token != thinking_end_id:
                        weights[b, s] = self.thinking_loss_weight

            # Agentic masking
            if agent_enabled and tool_call_id >= 0:
                in_tool_call = False
                in_tool_result = False
                for s in range(seq_len):
                    token = targets[b, s].item()
                    if token == tool_call_id:
                        in_tool_call = True
                        in_tool_result = False
                        weights[b, s] = agent_loss_weight
                    elif token == tool_call_end_id:
                        in_tool_call = False
                        in_tool_result = False
                        weights[b, s] = agent_loss_weight
                    elif token == tool_result_id:
                        in_tool_call = False
                        in_tool_result = True
                        weights[b, s] = agent_loss_weight
                    elif end_id >= 0 and token == end_id:
                        in_tool_call = False
                        in_tool_result = False
                        weights[b, s] = agent_loss_weight
                    elif assistant_id >= 0 and token == assistant_id:
                        in_tool_call = False
                        in_tool_result = False
                        weights[b, s] = agent_loss_weight
                    elif in_tool_call:
                        weights[b, s] = agent_loss_weight
                    elif in_tool_result:
                        weights[b, s] = agent_loss_weight

        # Save original 2D targets for MTP loss computation
        targets_2d = targets.clone() if mtp_logits_list and getattr(self.config, 'mtp_enabled', False) else None

        # Flatten outputs and targets
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        targets = targets.contiguous().view(-1)
        weights = weights.contiguous().view(-1)

        # Ignore padded elements
        non_pad_mask = targets.ne(self.tokenizer.get_pad_index())
        outputs = outputs[non_pad_mask]
        targets = targets[non_pad_mask]
        weights = weights[non_pad_mask]

        # Weighted cross-entropy loss
        token_losses = criterion(outputs, targets)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True), raw_outputs, torch.tensor(0.0, device=outputs.device)
        loss = (token_losses * weights).sum() / weight_sum

        # Add load balancing loss for MoE models
        moe_loss = torch.tensor(0.0, device=loss.device)
        if gate_scores is not None and hasattr(model, 'get_load_balancing_loss'):
            moe_loss = model.get_load_balancing_loss(torch.stack(gate_scores))
            loss = loss + self.config.moe_load_balance_weight * moe_loss

        # Add MTP auxiliary loss
        mtp_loss = torch.tensor(0.0, device=loss.device)
        if mtp_logits_list and getattr(self.config, 'mtp_enabled', False) and targets_2d is not None:
            mtp_total = torch.tensor(0.0, device=loss.device)
            mtp_head_count = 0
            for k, head_logits in enumerate(mtp_logits_list):
                shift = k + 2
                if targets_2d.size(1) > shift:
                    mtp_preds = head_logits[:, :-shift].contiguous().view(-1, head_logits.size(-1))
                    mtp_targets = targets_2d[:, shift:].contiguous().view(-1)
                    non_pad = mtp_targets.ne(self.tokenizer.get_pad_index())
                    if non_pad.any():
                        mtp_token_losses = criterion(mtp_preds[non_pad], mtp_targets[non_pad])
                        mtp_loss_k = mtp_token_losses.mean()
                        mtp_total = mtp_total + mtp_loss_k
                        mtp_head_count += 1
            if mtp_head_count > 0:
                mtp_loss = mtp_total / mtp_head_count
                loss = loss + self.config.mtp_loss_weight * mtp_loss

        return loss, raw_outputs, mtp_loss.item()

    def _compute_thinking_metrics(self, model, inputs, targets, device, logits=None):
        """Compute metrics for thinking token generation if thinking data is present."""
        if not self.has_thinking_data:
            return {}

        thinking_id = self.tokenizer.get_thinking_index()
        thinking_end_id = self.tokenizer.get_thinking_end_index()

        if thinking_id < 0 or thinking_end_id < 0:
            return {}

        with torch.no_grad():
            outputs = logits if logits is not None else model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions = outputs.argmax(dim=-1)

            thinking_open_correct = 0
            thinking_close_correct = 0
            total_thinking_positions = 0
            thinking_token_count = 0
            response_token_count = 0
            thinking_correct = 0
            response_correct = 0

            for i in range(targets.size(0)):
                in_thinking = False
                for j in range(targets.size(1)):
                    target_token = targets[i, j].item()
                    pred_token = predictions[i, j].item()

                    if target_token == thinking_id:
                        in_thinking = True
                        total_thinking_positions += 1
                        thinking_token_count += 1
                        if pred_token == thinking_id:
                            thinking_open_correct += 1
                            thinking_correct += 1
                    elif target_token == thinking_end_id:
                        in_thinking = False
                        total_thinking_positions += 1
                        thinking_token_count += 1
                        if pred_token == thinking_end_id:
                            thinking_close_correct += 1
                            thinking_correct += 1
                    elif in_thinking:
                        thinking_token_count += 1
                        if pred_token == target_token:
                            thinking_correct += 1
                    else:
                        response_token_count += 1
                        if pred_token == target_token:
                            response_correct += 1

            metrics = {}
            if total_thinking_positions > 0:
                metrics['thinking_token_accuracy'] = (thinking_open_correct + thinking_close_correct) / total_thinking_positions
                metrics['thinking_open_accuracy'] = thinking_open_correct / max(1, sum(1 for t in targets.flatten() if t.item() == thinking_id))
                metrics['thinking_close_accuracy'] = thinking_close_correct / max(1, sum(1 for t in targets.flatten() if t.item() == thinking_end_id))
                metrics['thinking_positions'] = total_thinking_positions

            if thinking_token_count > 0:
                metrics['thinking_length_avg'] = thinking_token_count / targets.size(0)
                metrics['thinking_coverage'] = thinking_token_count / max(1, thinking_token_count + response_token_count)
                metrics['thinking_token_accuracy_full'] = thinking_correct / thinking_token_count

            if response_token_count > 0:
                metrics['response_token_accuracy'] = response_correct / response_token_count

            return metrics

    def _compute_agent_metrics(self, model, inputs, targets, device, logits=None):
        """Compute metrics for agentic token generation if agent data is present."""
        agent_enabled = getattr(self.config, 'agent_enabled', False)
        if not agent_enabled:
            return {}

        tool_call_id = getattr(self.tokenizer, 'get_tool_call_index', lambda: -1)()
        tool_call_end_id = getattr(self.tokenizer, 'get_tool_call_end_index', lambda: -1)()
        tool_result_id = getattr(self.tokenizer, 'get_tool_result_index', lambda: -1)()

        if tool_call_id < 0:
            return {}

        with torch.no_grad():
            outputs = logits if logits is not None else model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions = outputs.argmax(dim=-1)

            tool_call_count = 0
            tool_call_correct = 0
            tool_name_count = 0
            tool_name_correct = 0
            tool_result_count = 0
            tool_result_correct = 0
            total_agent_tokens = 0

            for i in range(targets.size(0)):
                in_tool_call = False
                in_tool_result = False
                is_tool_name = False

                for j in range(targets.size(1)):
                    target_token = targets[i, j].item()
                    pred_token = predictions[i, j].item()

                    if target_token == tool_call_id:
                        in_tool_call = True
                        is_tool_name = True
                        tool_call_count += 1
                        total_agent_tokens += 1
                        if pred_token == tool_call_id:
                            tool_call_correct += 1
                    elif target_token == tool_call_end_id:
                        in_tool_call = False
                        is_tool_name = False
                        tool_call_count += 1
                        total_agent_tokens += 1
                        if pred_token == tool_call_end_id:
                            tool_call_correct += 1
                    elif target_token == tool_result_id:
                        in_tool_result = True
                        in_tool_call = False
                        tool_result_count += 1
                        total_agent_tokens += 1
                        if pred_token == tool_result_id:
                            tool_result_correct += 1
                    elif in_tool_call:
                        total_agent_tokens += 1
                        if pred_token == target_token:
                            tool_call_correct += 1
                    elif in_tool_result:
                        total_agent_tokens += 1
                        if pred_token == target_token:
                            tool_result_correct += 1

            metrics = {}
            if tool_call_count > 0:
                metrics['agent_tool_call_accuracy'] = tool_call_correct / tool_call_count
                metrics['agent_tool_call_count'] = tool_call_count
            if tool_result_count > 0:
                metrics['agent_observation_accuracy'] = tool_result_correct / tool_result_count
            if total_agent_tokens > 0:
                metrics['agent_total_tokens'] = total_agent_tokens
                metrics['agent_ratio'] = total_agent_tokens / max(1, targets.size(0) * targets.size(1))

            return metrics

    def _backward_pass(self, loss, optimizer, scaler, accumulation_step=1):
        """Unified backward pass handling for mixed and standard precision."""
        loss = loss / accumulation_step

        if self.use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss * accumulation_step
