import torch
import torch.nn.functional as F
import logging

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

        self.min_length = min_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.default_response = default_response

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

    def sample_next_token(self, logits, banned_tokens=None):
        if logits is None:
            return None
        logits = logits / max(self.temperature, 1e-8)
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
            n = max(1, min(n, 200))  # límite razonable

            # Intentar obtener vocab list desde tokenizer
            vocab_items = []
            if hasattr(self.tokenizer, "vocab") and isinstance(self.tokenizer.vocab, dict):
                vocab_items = list(self.tokenizer.vocab.items())
            elif hasattr(self.tokenizer, "word2idx") and isinstance(self.tokenizer.word2idx, dict):
                vocab_items = list(self.tokenizer.word2idx.items())
            elif hasattr(self.tokenizer, "trie") and hasattr(self.tokenizer.trie, "get_all_tokens"):
                try:
                    toks = self.tokenizer.trie.get_all_tokens()
                    vocab_items = [(t, self.tokenizer.trie.get_index(t)) for t in toks]
                except Exception:
                    vocab_items = []
            elif hasattr(self.tokenizer, "vocab_size"):
                # Fallback: sample ids and attempt to decode each id to a token string
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

                    # If still empty or None, mark as <empty:id>
                    if token_str is None or (isinstance(token_str, str) and token_str.strip() == ""):
                        token_str = f"<empty:{i}>"

                    vocab_items.append((token_str, i))
            else:
                vocab_items = []

            import random
            if vocab_items:
                sample = random.sample(vocab_items, min(n, len(vocab_items)))
                lines = [f"{tok} -> {idx}" for tok, idx in sample]
                return "DEBUG VOCAB SAMPLE:\n" + "\n".join(lines)
            else:
                return "DEBUG: tokenizer vocabulary not accessible."

        # --- resto de la función (tokenización, generación) se mantiene sin cambios ---
        try:
            if self.intent_classifier:
                _ = self.intent_classifier(user_text)
            if self.sentiment_analyzer:
                _ = self.sentiment_analyzer(user_text)
        except Exception:
            pass

        prompt_text = f"Pregunta: {user_text.strip()}\nRespuesta:"
        try:
            input_ids = self.tokenizer.encode(prompt_text)
        except Exception:
            input_ids = [self.unk_token_id] if self.unk_token_id is not None else [0]

        input_ids = [i for i in input_ids if i is not None]
        if len(input_ids) == 0:
            input_ids = [self.unk_token_id] if self.unk_token_id is not None else [0]

        src = torch.LongTensor([input_ids]).to(self.device)

        generated = []
        try:
            with torch.no_grad():
                for step in range(self.max_len):
                    out = self.model(src)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    logits = out[:, -1, :]

                    # Apply dynamic penalization for repeated n-grams
                    if self.no_repeat_ngram_size and len(generated) >= self.no_repeat_ngram_size - 1:
                        penalty = 5.0  # Adjustable penalty value
                        for token_id in range(logits.size(-1)):
                            cand_seq = generated + [token_id]
                            if self._is_repeated_ngram(cand_seq, self.no_repeat_ngram_size):
                                logits[0, token_id] -= penalty

                    next_token = self.sample_next_token(logits.squeeze(0))

                    if next_token is None:
                        break

                    # Enforce min_length: don't stop on EOS until minimum length reached
                    if self.eos_token_id is not None and next_token == self.eos_token_id and len(generated) >= self.min_length:
                        break

                    generated.append(next_token)

                    # Append token to context for autoregressive generation
                    next_token_tensor = torch.LongTensor([[next_token]]).to(self.device)
                    src = torch.cat([src, next_token_tensor], dim=1)

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
            return self.default_response

        if not generated:
            return self.default_response

        # Ensure no repeated n-grams in output
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
                return self.default_response

        try:
            text = self.tokenizer.decode(generated)
        except Exception:
            try:
                text = " ".join(self.tokenizer.convert_ids_to_tokens(generated))
            except Exception:
                text = " ".join(str(t) for t in generated)

        if not text or text.strip() == "":
            return self.default_response

        # Ensure minimum response length
        tokens = text.split()
        if self.min_length is not None and len(tokens) < self.min_length:
            return self.default_response

        return text

    def _is_repeated_ngram(self, seq, n):
        if n < 1 or len(seq) < n * 2:
            return False
        last = tuple(seq[-n:])
        for i in range(len(seq) - n):
            if tuple(seq[i:i+n]) == last:
                return True
        return False

