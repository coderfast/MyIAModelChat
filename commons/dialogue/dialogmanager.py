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
                 default_response="Lo siento, no puedo responder ahora."):
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

        # Mode token IDs for structured generation
        self.context_id = getattr(tokenizer, "get_context_index", lambda: -1)()
        self.answer_id = getattr(tokenizer, "get_answer_index", lambda: -1)()
        self.thinking_mode_id = getattr(tokenizer, "get_thinking_mode_index", lambda: -1)()

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

        if torch.isnan(probs).any() or torch.isclose(probs.sum(), torch.tensor(0.0, device=probs.device)):
            return int(torch.argmax(logits).item())

        try:
            token = torch.multinomial(probs, num_samples=1)
            return int(token.item())
        except Exception as e:
            logger.debug("Sampling failed, falling back to argmax: %s", e)
            return int(torch.argmax(probs).item())

    def generate_response(self, user_text):
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

        # Build prompt with mode tokens: <|thinking|>question<|answer|>
        prompt_text = f"<|thinking|>{user_text.strip()}<|answer|>"
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

                    # Enforce min_length: don't stop on EOS until minimum length reached
                    # Don't stop during thinking phase (before <|answer|>)
                    if (self.eos_token_id is not None and next_token == self.eos_token_id
                            and len(generated) >= self.min_length and in_answer_phase):
                        break

                    generated.append(next_token)

                    # Answer phase detection: after <|answer|> token
                    if self.answer_id >= 0 and next_token == self.answer_id:
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
            
            # Extract thinking from <|thinking|> to <|answer|>
            if '<|thinking|>' in full_text and '<|answer|>' in full_text:
                parts = full_text.split('<|answer|>')
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

