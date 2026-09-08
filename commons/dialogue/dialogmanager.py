import torch
import torch.nn.functional as F
import logging
import re

logger = logging.getLogger(__name__)

class DialogueManager:
    def __init__(self, model, device, tokenizer,
                 intent_classifier=None, sentiment_analyzer=None, persona=None,
                 top_k=12, top_p=0.8, temperature=0.65, max_len=128,
                 min_length=5, no_repeat_ngram_size=3,
                 pad_token_id=None, eos_token_id=None, unk_token_id=None,
                 default_response="Lo siento, no puedo responder ahora.\nI'm sorry, I can't respond right now.",
                 tool_executor=None, agent_enabled=False, agent_max_iterations=5,
                 moe_enabled=False):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.intent_classifier = intent_classifier
        self.sentiment_analyzer = sentiment_analyzer
        self.persona = persona

        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_len = max_len
        self.min_length = min_length
        self.no_repeat_ngram_size = no_repeat_ngram_size

        self.pad_token_id = pad_token_id if pad_token_id is not None else getattr(tokenizer, "get_pad_index", lambda: None)()
        self.eos_token_id = eos_token_id if eos_token_id is not None else (getattr(tokenizer, "get_eos_index", lambda: None)() if hasattr(tokenizer, "get_eos_index") else None)
        self.unk_token_id = unk_token_id if unk_token_id is not None else getattr(tokenizer, "get_unk_index", lambda: None)()

        self.default_response = default_response

        # GPT-2 standard token IDs
        self.problem_id = getattr(tokenizer, "get_problem_index", lambda: -1)()
        self.final_id = getattr(tokenizer, "get_final_index", lambda: -1)()
        self.user_id = getattr(tokenizer, "get_user_index", lambda: -1)()
        self.assistant_id = getattr(tokenizer, "get_assistant_index", lambda: -1)()
        self.tool_result_id = getattr(tokenizer, "get_tool_result_index", lambda: -1)()
        self.system_id = getattr(tokenizer, "get_system_index", lambda: -1)()
        self.end_id = getattr(tokenizer, "get_end_index", lambda: -1)()
        self.sep_id = getattr(tokenizer, "get_sep_index", lambda: -1)()
        self.thinking_id = getattr(tokenizer, "get_thinking_index", lambda: -1)()
        self.problem_id = getattr(tokenizer, "get_problem_index", lambda: -1)()
        self.final_id = getattr(tokenizer, "get_final_index", lambda: -1)()
        self.thinking_mode_id = getattr(tokenizer, "get_thinking_mode_index", lambda: -1)()

        # Agentic token IDs
        self.tool_call_id = getattr(tokenizer, "get_tool_call_index", lambda: -1)()
        self.tool_call_end_id = getattr(tokenizer, "get_tool_call_end_index", lambda: -1)()

        # Agent support
        self.tool_executor = tool_executor
        self.agent_enabled = agent_enabled
        self.agent_max_iterations = agent_max_iterations
        
        # MoE support
        self.moe_enabled = moe_enabled

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
        logits = logits.clone()
        top_k = min(max(top_k, 0), logits.size(-1))
        if top_k > 0:
            kth_vals, _ = torch.topk(logits, top_k)
            min_kth = kth_vals[..., -1, None]
            logits[logits < min_kth] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits

    def sample_next_token(self, logits, banned_tokens=None, temperature=None):
        if logits is None:
            return None
        temp = temperature if temperature is not None else self.temperature
        logits = logits / max(temp, 1e-8)
        filtered_logits = self.top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
        probs = F.softmax(filtered_logits, dim=-1)

        if banned_tokens is not None and len(banned_tokens) > 0:
            for tok in banned_tokens:
                if 0 <= tok < probs.size(-1):
                    probs[tok] = 0.0
            total = probs.sum()
            if total <= 0 or torch.isnan(total):
                return None
            probs = probs / total

        if torch.isnan(probs).any() or torch.isclose(probs.sum(), torch.tensor(0.0, device=probs.device)).item():
            return int(torch.argmax(logits).item())

        try:
            token = torch.multinomial(probs, num_samples=1)
            return int(token.item())
        except Exception as e:
            logger.debug("Sampling failed, falling back to argmax: %s", e)
            return int(torch.argmax(probs).item())

    def generate_response(self, user_text):
        # Use agent loop if enabled
        if self.agent_enabled and self.tool_executor:
            return self.generate_response_agent(user_text)
        # Comando debug: "debug" o "debug N"
        txt = user_text.strip()
        if txt.lower().startswith("debug"):
            parts = txt.split()
            try:
                n = int(parts[1]) if len(parts) > 1 else 20
            except Exception:
                n = 20
            n = max(1, min(n, 200))

            vocab_items = []
            if hasattr(self.tokenizer, "vocab") and isinstance(self.tokenizer.vocab, dict):
                vocab_items = list(self.tokenizer.vocab.items())
            elif hasattr(self.tokenizer, "vocab_size"):
                vocab_items = []
                for i in range(self.tokenizer.vocab_size):
                    token_str = None
                    try:
                        token_str = self.tokenizer.decode([i])
                    except Exception:
                        try:
                            token_str = self.tokenizer.convert_ids_to_tokens([i])[0]
                        except Exception:
                            token_str = None
                    if token_str is None or (isinstance(token_str, str) and token_str.strip() == ""):
                        token_str = f"<empty:{i}>"
                    vocab_items.append((token_str, i))
            else:
                vocab_items = []

            import random
            if vocab_items:
                sample = random.sample(vocab_items, min(n, len(vocab_items)))
                lines = [f"{tok} -> {idx}" for tok, idx in sample]
                return {'thinking': None, 'response': "DEBUG VOCAB SAMPLE:\n" + "\n".join(lines)}
            else:
                return {'thinking': None, 'response': "DEBUG: tokenizer vocabulary not accessible."}

        # --- intent/sentiment analysis ---
        intent_label = None
        sentiment_label = None
        adjusted_temperature = self.temperature

        try:
            if self.intent_classifier:
                intent_result = self.intent_classifier(user_text)
                if intent_result and len(intent_result) > 0:
                    intent_label = intent_result[0].get('label', None)
            if self.sentiment_analyzer:
                sentiment_result = self.sentiment_analyzer(user_text)
                if sentiment_result and len(sentiment_result) > 0:
                    sentiment_label = sentiment_result[0].get('label', None)
        except Exception:
            pass

        # Adjust temperature based on sentiment
        if sentiment_label:
            try:
                match = re.search(r'(\d+)', sentiment_label)
                if match:
                    stars = int(match.group(1))
                    if stars <= 2:
                        adjusted_temperature = max(0.3, self.temperature - 0.2)
                    elif stars >= 4:
                        adjusted_temperature = min(1.0, self.temperature + 0.1)
            except (ValueError, AttributeError):
                pass

        print(f"Intent: {intent_label} | Sentiment: {sentiment_label} | Temp: {adjusted_temperature:.2f}")

        # Build prompt with chat format (Formato 1): <|user|>text<|end|><|assistant|>
        prompt_text = f"<|user|>{user_text.strip()}<|end|><|assistant|>"
        try:
            input_ids = self.tokenizer.encode(prompt_text)
        except Exception:
            input_ids = [self.unk_token_id] if self.unk_token_id is not None else [0]

        input_ids = [i for i in input_ids if i is not None]
        if len(input_ids) == 0:
            input_ids = [self.unk_token_id] if self.unk_token_id is not None else [0]

        src = torch.LongTensor([input_ids]).to(self.device)

        generated = []
        in_answer_phase = False
        response_tokens = []

        # Pre-allocate token buffer to avoid O(n^2) torch.cat growth
        src_buf = torch.full((1, len(input_ids) + self.max_len), self.pad_token_id or 0, dtype=torch.long, device=self.device)
        src_buf[0, :len(input_ids)] = src
        src_len = len(input_ids)

        try:
            with torch.no_grad():
                for step in range(self.max_len):
                    out = self.model(src_buf[:, :src_len])
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    logits = out[:, -1, :]

                    # Apply dynamic penalization for repeated n-grams
                    banned_tokens = set()
                    if self.no_repeat_ngram_size and len(generated) >= self.no_repeat_ngram_size - 1:
                        penalty = 5.0
                        ngram_size = self.no_repeat_ngram_size
                        prefix = tuple(generated[-(ngram_size - 1):])
                        for i in range(len(generated) - ngram_size + 1):
                            if tuple(generated[i:i + ngram_size - 1]) == prefix:
                                banned_tokens.add(generated[i + ngram_size - 1])
                        for token_id in banned_tokens:
                            if token_id < logits.size(-1):
                                logits[0, token_id] -= penalty

                    next_token = self.sample_next_token(logits.squeeze(0), temperature=adjusted_temperature, banned_tokens=banned_tokens)

                    if next_token is None:
                        break

                    # Stop on agentic end tokens (model finished tool call)
                    if (self.tool_call_end_id >= 0 and next_token == self.tool_call_end_id):
                        break

                    # Stop on <|end|> token (model finished response)
                    if (self.end_id >= 0 and next_token == self.end_id):
                        break

                    # Enforce min_length: don't stop on EOS until minimum length reached
                    if (self.eos_token_id is not None and next_token == self.eos_token_id
                            and len(generated) >= self.min_length):
                        break

                    generated.append(next_token)

                    # Answer phase detection: after <|final|> token
                    if self.final_id >= 0 and next_token == self.final_id:
                        in_answer_phase = True
                        response_tokens = []
                    elif in_answer_phase:
                        response_tokens.append(next_token)

                    # Append token to context for autoregressive generation
                    src_buf[0, src_len] = next_token
                    src_len += 1

                    # If generation keeps outputting only special/tiny tokens, abort
                    if len(generated) >= 3:
                        uniq = set(generated)
                        specials = {tid for tid in (self.pad_token_id, self.unk_token_id) if tid is not None}
                        if uniq.issubset(specials):
                            logger.debug("Generated only special tokens, aborting.")
                            generated = []
                            break

        except Exception as e:
            logger.exception("Error during generation: %s", e)
            return {'thinking': None, 'response': self.default_response}

        logger.debug(f"Raw generated tokens ({len(generated)}): {generated[:20]}...")

        if not generated:
            return {'thinking': None, 'response': self.default_response}

        # Ensure no repeated n-grams in output — retry once if found
        if self.no_repeat_ngram_size and len(generated) >= self.no_repeat_ngram_size:
            def has_repeated_ngram(seq, n):
                if len(seq) < n * 2:
                    return False
                last = tuple(seq[-n:])
                for i in range(len(seq) - n):
                    if tuple(seq[i:i+n]) == last:
                        return True
                return False

            if has_repeated_ngram(generated, self.no_repeat_ngram_size):
                # Strip last n-gram and retry generation for remaining tokens
                generated = generated[:-self.no_repeat_ngram_size]
                response_tokens = response_tokens[:-self.no_repeat_ngram_size] if len(response_tokens) >= self.no_repeat_ngram_size else response_tokens
                # Continue generation from current state
                src_buf_trimmed = src_buf[:, :src_len - self.no_repeat_ngram_size]
                src_len_trimmed = src_len - self.no_repeat_ngram_size
                if src_len_trimmed > 0:
                    try:
                        for step in range(self.max_len - len(generated)):
                            out = self.model(src_buf_trimmed[:, :src_len_trimmed])
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                            logits = out[:, -1, :]
                            next_token = self.sample_next_token(logits.squeeze(0), temperature=adjusted_temperature)
                            if next_token is None:
                                break
                            if (self.eos_token_id is not None and next_token == self.eos_token_id
                                    and len(generated) >= self.min_length):
                                break
                            generated.append(next_token)
                            response_tokens.append(next_token)
                            src_buf_trimmed[0, src_len_trimmed] = next_token
                            src_len_trimmed += 1
                    except Exception:
                        pass  # If retry fails, keep what we have

        # Decode thinking and response separately
        thinking_text = None
        response_text = None

        try:
            # Decode all generated tokens to extract thinking
            full_text = self.tokenizer.decode(generated)

            # Extract thinking from <|thinking|> to <|final|> (or legacy <|answer|>)
            thinking_text = None
            response_text = None
            if '<|thinking|>' in full_text and ('<|final|>' in full_text or '<|answer|>' in full_text):
                end_marker = '<|final|>' if '<|final|>' in full_text else '<|answer|>'
                parts = full_text.split(end_marker)
                thinking_part = parts[0]
                # Remove the <|thinking|> prefix
                if '<|thinking|>' in thinking_part:
                    thinking_text = thinking_part.split('<|thinking|>', 1)[1].strip()
                response_text = parts[1].strip() if len(parts) > 1 else None
            else:
                # Fallback: use response_tokens if no mode tokens found
                if response_tokens:
                    response_text = self.tokenizer.decode(response_tokens)
                else:
                    response_text = full_text

            # Strip trailing <|end|> / <|assistant|> / <|final|> markers from response
            if response_text:
                for marker in ('<|end|>', '<|assistant|>', '<|final|>', '<|answer|>'):
                    if marker in response_text:
                        response_text = response_text.split(marker, 1)[0]
                    response_text = response_text.strip()
        except Exception:
            try:
                response_text = " ".join(self.tokenizer.convert_ids_to_tokens(response_tokens or generated))
            except Exception:
                response_text = " ".join(str(t) for t in (response_tokens or generated))

        logger.debug(f"Decoded response ({len(response_text or '')} chars): '{(response_text or '')[:100]}'")

        if not response_text or response_text.strip() == "":
            return {'thinking': thinking_text, 'response': self.default_response}

        # Ensure minimum response length
        tokens = response_text.split()
        if self.min_length is not None and len(tokens) < self.min_length:
            return {'thinking': thinking_text, 'response': self.default_response}

        return {'thinking': thinking_text, 'response': response_text}

    def generate_response_agent(self, user_text):
        """Agentic response generation with tool call loop.

        Flow:
        1. Generate response with model
        2. If <tool_call> detected → parse, execute, add observation
        3. Loop until no tool_call or max_iterations reached
        4. Return final response
        """
        from commons.tools.tool_executor import format_observation

        context = f"<|user|>{user_text.strip()}<|end|><|assistant|>"
        observations = []

        for iteration in range(self.agent_max_iterations):
            # Generate with model
            raw_output = self._generate_tokens_from_context(context)

            if not raw_output:
                return {'thinking': None, 'response': self.default_response}

            # Check for tool_call
            if not self.tool_executor or not self.tool_executor.has_tool_call(raw_output):
                # No tool call — parse final response
                return self._parse_agent_response(raw_output)

            # Parse tool_call
            tool_name, arguments = self.tool_executor.parse_tool_call(raw_output)

            # Execute tool
            try:
                result = self.tool_executor.execute_tool_call(raw_output)
            except Exception as e:
                result = format_observation(f"Error: {str(e)}")

            observations.append(result)

            # Add observation to context for next iteration
            context = raw_output + result

            logger.info(f"Agent iteration {iteration + 1}: tool={tool_name}")

        # Max iterations reached — force final response
        return {'thinking': f"Agent loop completed after {self.agent_max_iterations} iterations",
                'response': f"Tool calls executed: {len(observations)}"}

    def _generate_tokens_from_context(self, prompt_text):
        """Generate tokens from a prompt context string."""
        try:
            input_ids = self.tokenizer.encode(prompt_text)
        except Exception:
            input_ids = [self.unk_token_id] if self.unk_token_id is not None else [0]

        input_ids = [i for i in input_ids if i is not None]
        if len(input_ids) == 0:
            input_ids = [self.unk_token_id] if self.unk_token_id is not None else [0]

        src = torch.LongTensor([input_ids]).to(self.device)
        generated = []

        src_buf = torch.full((1, len(input_ids) + self.max_len), self.pad_token_id or 0, dtype=torch.long, device=self.device)
        src_buf[0, :len(input_ids)] = src
        src_len = len(input_ids)

        try:
            with torch.no_grad():
                for step in range(self.max_len):
                    out = self.model(src_buf[:, :src_len])
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    logits = out[:, -1, :]

                    next_token = self.sample_next_token(logits.squeeze(0))
                    if next_token is None:
                        break

                    generated.append(next_token)
                    src_buf[0, src_len] = next_token
                    src_len += 1

                    # Stop on final token or EOS
                    if self.final_id >= 0 and next_token == self.final_id:
                        break
        except Exception as e:
            logger.debug(f"Agent generation error: {e}")
            return None

        if not generated:
            return None

        try:
            return self.tokenizer.decode(generated)
        except Exception:
            return " ".join(str(t) for t in generated)

    def _parse_agent_response(self, raw_output):
        """Parse agent response into thinking and response parts."""
        thinking_text = None
        response_text = None

        try:
            # Remove tool_call blocks if any remain
            clean = raw_output
            if '<tool_call>' in clean:
                parts = clean.split('<tool_call>')
                clean = parts[0]
                for part in parts[1:]:
                    if '</tool_call>' in part:
                        clean += part.split('</tool_call>', 1)[1]
                    else:
                        clean += part

            # Remove <|tool_result|> blocks (prefix, run until next turn marker)
            if '<|tool_result|>' in clean:
                parts = clean.split('<|tool_result|>')
                clean = parts[0]
                for part in parts[1:]:
                    marker = None
                    for m in ('<|end|>', '<|assistant|>'):
                        if m in part:
                            marker = m
                            break
                    if marker:
                        clean += part.split(marker, 1)[1]
                    else:
                        clean += part

            # Parse thinking/answer
            thinking_text = None
            response_text = None
            if '<|thinking|>' in clean and ('<|final|>' in clean or '<|answer|>' in clean):
                end_marker = '<|final|>' if '<|final|>' in clean else '<|answer|>'
                parts = clean.split(end_marker)
                thinking_part = parts[0]
                if '<|thinking|>' in thinking_part:
                    thinking_text = thinking_part.split('<|thinking|>', 1)[1].strip()
                response_text = parts[1].strip() if len(parts) > 1 else None
            else:
                response_text = clean.strip()

            # Strip trailing turn markers from response
            if response_text:
                for marker in ('<|end|>', '<|assistant|>', '<|final|>', '<|answer|>'):
                    if marker in response_text:
                        response_text = response_text.split(marker, 1)[0]
                    response_text = response_text.strip()

        except Exception:
            response_text = raw_output

        if not response_text or response_text.strip() == "":
            return {'thinking': thinking_text, 'response': self.default_response}

        return {'thinking': thinking_text, 'response': response_text}

