"""
Real Thinking Engine based on NLP analysis.
Generates chain-of-thought reasoning by analyzing actual text content,
without depending on external LLM APIs like Ollama.
Supports all EU official languages + Eastern European languages.
"""
import re
import hashlib
import json
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter

from commons.language_utils import LANGUAGE_INDICATORS

logger = logging.getLogger(__name__)

# Module-level compiled regex patterns
_ADJ_PATTERN = re.compile(r'\b\w+\s+\w+(?:\s+\w+)?\b')
_SENT_BOUNDARY_PATTERN = re.compile(r'\s*[.!?]\s+')
_LEADING_TRAILING_PUNCT = re.compile(r'^[\s:;,.—–\-"\“”\'`(){}\[\]]+|[\s:;,.—–\-"\“”\'`(){}\[\]]+$')

# Official spaCy models per language (verified against spaCy 3.8 compatibility table).
# Languages without an official model (bg/sk/sr/be/bs/fo/ga/is/lb/nn/rm/sq...)
# fall back to _GENERIC_MODEL, then to regex analysis.
_LANG_MODEL_MAP = {
    'ca': 'ca_core_news_sm', 'da': 'da_core_news_sm', 'de': 'de_core_news_sm',
    'el': 'el_core_news_sm', 'en': 'en_core_web_sm', 'es': 'es_core_news_sm',
    'fi': 'fi_core_news_sm', 'fr': 'fr_core_news_sm', 'hr': 'hr_core_news_sm',
    'it': 'it_core_news_sm', 'lt': 'lt_core_news_sm', 'mk': 'mk_core_news_sm',
    'nb': 'nb_core_news_sm', 'nl': 'nl_core_news_sm', 'pl': 'pl_core_news_sm',
    'pt': 'pt_core_news_sm', 'ro': 'ro_core_news_sm', 'sl': 'sl_core_news_sm',
    'sv': 'sv_core_news_sm', 'uk': 'uk_core_news_sm', 'ru': 'ru_core_news_sm',
}
_GENERIC_MODEL = 'xx_ent_wiki_sm'
_MAX_SPAN_WORDS = 8
_MAX_CLAIM_WORDS = 20
# Function words that must never appear as standalone concepts
_STOPWORDS = {
    'the', 'a', 'an', 'of', 'to', 'in', 'on', 'and', 'or', 'for', 'is', 'are',
    'as', 'at', 'by', 'it', 'its', 'be', 'not', 'but', 'if', 'we', 'he', 'she',
    'his', 'her', 'has', 'had', 'do', 'does', 'did', 'will', 'can', 'may',
    'that', 'this', 'these', 'those', 'with', 'from', 'their', 'have', 'been',
    'were', 'which', 'into', 'when', 'then', 'than', 'they', 'them', 'there',
    'such', 'also', 'over', 'after', 'between', 'through', 'within', 'while',
    'because', 'would', 'could', 'should', 'under', 'during', 'before', 'both',
    'each', 'other', 'some', 'most', 'more', 'many', 'any', 'all', 'one', 'two',
}

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. Using fallback regex-based analysis. "
                   "Install with: pip install spacy")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTILINGUAL LANGUAGE CONFIGURATION
# Detection indicators imported from commons/language_utils.py
# Thinking-engine-specific fields loaded from JSON config file
# ═══════════════════════════════════════════════════════════════════════════════

def _load_thinking_config() -> Dict[str, Dict]:
    """Load thinking-engine-specific language config from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), 'thinking_engine_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            thinking_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load thinking config from {config_path}: {e}")
        return {}

    # Merge with LANGUAGE_INDICATORS and compile meta_patterns
    merged = dict(LANGUAGE_INDICATORS)
    for lang_code, thinking_fields in thinking_data.items():
        base = dict(LANGUAGE_INDICATORS.get(lang_code, {}))
        base.update(thinking_fields)
        # Compile meta_patterns from strings to regex objects
        raw_patterns = base.pop('meta_patterns', [])
        compiled = []
        for pattern_str in raw_patterns:
            try:
                compiled.append(re.compile(pattern_str, re.IGNORECASE))
            except re.error:
                pass
        base['meta_patterns'] = compiled
        merged[lang_code] = base

    return merged


LANGUAGE_CONFIG = _load_thinking_config()

# Fallback config for unsupported languages (uses English as base)
FALLBACK_CONFIG = LANGUAGE_CONFIG.get('en', {
    'noun_phrases': True, 'verb_phrases': True, 'named_entities': True,
    'temporal_markers': True, 'causal_markers': True, 'contrast_markers': True,
    'meta_patterns': {}
})


class ThinkingEngine:
    """
    Real thinking engine based on NLP analysis.
    Generates chain-of-thought reasoning by analyzing actual text content.
    Supports all EU official languages + Eastern European languages.
    """

    DEPTH_CONFIG = {
        'basic': {'max_concepts': 3, 'max_entities': 2, 'max_steps': 3},
        'adaptive': {'max_concepts': 5, 'max_entities': 3, 'max_steps': 5},
        'detailed': {'max_concepts': 10, 'max_entities': 5, 'max_steps': 7},
    }

    def __init__(self, depth: str = 'adaptive'):
        self.depth = depth
        self.depth_config = self.DEPTH_CONFIG.get(depth, self.DEPTH_CONFIG['adaptive'])
        self._nlp_cache: dict = {}
        self._analysis_cache: dict = {}
        self._cache_maxsize = 500

    def _get_nlp(self, language: str):
        """Lazily load the best spaCy model for `language` (cached per language).

        Fallback cascade: official model for the language -> xx_ent_wiki_sm
        (multilingual) -> None (regex-based analysis).
        Note: uses the raw base language (split regional codes), NOT _REGIONAL_MAP,
        because template fallbacks (ca->es, nb->da) must not limit model choice
        (ca and nb have their own official models).
        """
        if not SPACY_AVAILABLE:
            return None
        base = language.split('-')[0] if language else 'en'
        if not base:
            base = 'en'
        if base in self._nlp_cache:
            return self._nlp_cache[base]
        candidates = []
        official = _LANG_MODEL_MAP.get(base)
        if official:
            candidates.append(official)
        if _GENERIC_MODEL not in candidates:
            candidates.append(_GENERIC_MODEL)
        loaded = None
        for model in candidates:
            try:
                loaded = spacy.load(model)
                logger.info(f"Loaded spaCy model for '{base}': {model}")
                break
            except OSError:
                continue
        if loaded is None:
            wanted = official or _GENERIC_MODEL
            logger.warning(
                f"No spaCy model available for '{base}'. Using fallback regex analysis. "
                f"Install with: python -m spacy download {wanted}"
            )
        self._nlp_cache[base] = loaded
        return loaded

    def _load_nlp_model(self, language: str = 'en'):
        """Backward-compatible wrapper around _get_nlp()."""
        return self._get_nlp(language)

    def _estimate_complexity(self, text: str, analysis: dict) -> str:
        word_count = analysis.get('word_count', 0)
        entity_count = len(analysis.get('entities', []))
        sentence_count = analysis.get('sentence_count', 0)
        has_technical = analysis.get('has_technical_terms', False)
        if word_count < 20 or sentence_count < 2:
            return 'basic'
        if word_count > 100 or entity_count > 5 or (has_technical and word_count > 50):
            return 'detailed'
        return 'adaptive'

    def _get_depth_config(self, text: str, analysis: dict) -> dict:
        if self.depth != 'adaptive':
            return self.depth_config
        complexity = self._estimate_complexity(text, analysis)
        return self.DEPTH_CONFIG.get(complexity, self.depth_config)

    def generate_thinking(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not text or len(text.strip()) < 10:
            return ""
        cache_key = hashlib.md5(f"{text}:{context}".encode()).hexdigest()
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        language = ''
        if context:
            language = context.get('lang') or context.get('language') or ''
        if not language:
            language = self._detect_language(text)
        analysis = self._analyze_text(text, language)
        depth_config = self._get_depth_config(text, analysis)
        key_concepts = self._extract_key_concepts(text, analysis, depth_config)
        content_type = self._detect_content_type(text, analysis)
        thinking = self._build_reasoning(text, analysis, key_concepts, content_type, context, language)
        self._analysis_cache[cache_key] = thinking
        if len(self._analysis_cache) > self._cache_maxsize:
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]
        return thinking

    def _analyze_text(self, text: str, language: Optional[str] = None) -> dict:
        if language is None:
            language = self._detect_language(text)
        nlp = self._get_nlp(language)
        if nlp:
            return self._analyze_text_spacy(text, nlp, language)
        return self._analyze_text_regex(text, language)

    def _analyze_text_spacy(self, text: str, nlp, language: str) -> dict:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ('ORDINAL', 'CARDINAL'):
                continue
            span = self._sanitize_span(ent.text)
            if span:
                entities.append((span, ent.label_))
        noun_phrases = []
        for chunk in doc.noun_chunks:
            span = self._sanitize_span(chunk.text)
            if span:
                noun_phrases.append(span)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        word_count = len(text.split())
        sentence_count = len(sentences)
        return {
            'entities': entities, 'noun_phrases': noun_phrases, 'sentences': sentences,
            'word_count': word_count, 'sentence_count': sentence_count,
            'avg_sentence_length': word_count / max(1, sentence_count),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_questions': '?' in text,
            'has_technical_terms': self._detect_technical_terms(text),
            'language': language,
        }

    def _analyze_text_regex(self, text: str, language: Optional[str] = None) -> dict:
        if language is None:
            language = self._detect_language(text)
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        entities = []
        for i, word in enumerate(words):
            if i > 0 and word[0:1].isupper() and words[i-1][-1:] in '.!?\n':
                continue
            if word[0:1].isupper() and len(word) > 2 and word.isalpha():
                entities.append((word, 'ENTITY'))
        noun_phrases = []
        for match in _ADJ_PATTERN.finditer(text):
            phrase = match.group()
            if len(phrase.split()) >= 2:
                span = self._sanitize_span(phrase)
                if span:
                    noun_phrases.append(span)
        return {
            'entities': entities[:self.depth_config['max_entities'] * 2],
            'noun_phrases': noun_phrases[:self.depth_config['max_concepts'] * 2],
            'sentences': sentences,
            'word_count': len(words), 'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(1, len(sentences)),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_questions': '?' in text,
            'has_technical_terms': self._detect_technical_terms(text),
            'language': language,
        }

    def _sanitize_span(self, raw: str, max_words: int = _MAX_SPAN_WORDS) -> str:
        """Clean a candidate span for use as concept/entity.

        Strips surrounding punctuation (fixes the double-colon 'are: : x' bug),
        rejects sentence-like fragments (contains sentence boundaries or ends
        with a period) and spans longer than max_words words.
        """
        if not raw:
            return ''
        span = _LEADING_TRAILING_PUNCT.sub('', raw.strip())
        span = ' '.join(span.split())
        if not span:
            return ''
        if _SENT_BOUNDARY_PATTERN.search(span):
            return ''
        if span.endswith(('.', '!', '?')):
            return ''
        if len(span.split()) > max_words:
            return ''
        return span

    def _first_claim(self, text: str, max_words: int = _MAX_CLAIM_WORDS) -> str:
        """First sentence of the paragraph, trimmed — grounds the thinking in content."""
        match = re.match(r'\s*([^.!?\n]*[.!?])', text)
        sentence = match.group(1).strip() if match else text.strip()
        sentence = sentence.replace('"', "'").replace('\u201c', "'").replace('\u201d', "'")
        sentence = ' '.join(sentence.split())
        words = sentence.split()
        if len(words) < 5:
            return ''
        if len(words) > max_words:
            sentence = ' '.join(words[:max_words]).rstrip(' ,;:-') + '...'
        return sentence

    def _detect_technical_terms(self, text: str) -> bool:
        patterns = [
            r'\bAPI\b', r'\bSDK\b', r'\bHTTP\b', r'\bJSON\b', r'\bXML\b',
            r'\bSQL\b', r'\bREST\b', r'\bGraphQL\b', r'\bOAuth\b',
            r'\balgorithm\b', r'\bfunction\b', r'\bclass\b', r'\bmethod\b',
            r'\bmodule\b', r'\bdatabase\b', r'\bserver\b', r'\bclient\b', r'\bprotocol\b',
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _extract_key_concepts(self, text: str, analysis: dict, depth_config: dict = None) -> List[Tuple[str, int]]:
        if depth_config is None:
            depth_config = self.depth_config
        candidates = []
        for phrase in analysis.get('noun_phrases', []):
            clean = self._sanitize_span(phrase)
            if clean:
                candidates.append(clean)
        for entity, _ in analysis.get('entities', []):
            clean = self._sanitize_span(entity)
            if clean:
                candidates.append(clean)
        words = text.lower().split()
        word_freq = Counter()
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] += 1
        scored = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen:
                continue
            candidate_words = key.split()
            if all(w in _STOPWORDS for w in candidate_words):
                continue
            seen.add(key)
            score = sum(word_freq.get(w, 0) for w in candidate_words)
            if score > 0:
                scored.append((candidate, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:depth_config['max_concepts']]

    def _detect_content_type(self, text: str, analysis: dict) -> str:
        text_lower = text.lower()
        if analysis.get('has_questions', False):
            return 'qa'
        if analysis.get('has_numbers', False) and analysis.get('has_technical_terms', False):
            return 'technical'
        past_indicators = ['was', 'were', 'had', 'did', 'era', 'fue', 'war', 'hatte', 'byl']
        if any(ind in text_lower for ind in past_indicators):
            return 'narrative'
        imperative_indicators = ['must', 'should', 'need to', 'debe', 'debería', 'muss', 'sollte']
        if any(ind in text_lower for ind in imperative_indicators):
            return 'instructional'
        conversational_indicators = ['hello', 'hi', 'hey', 'hola', 'bonjour', 'hallo', 'ciao']
        if any(ind in text_lower for ind in conversational_indicators):
            return 'conversational'
        return 'factual'

    def _detect_language(self, text: str) -> str:
        """Detect language using indicators from LANGUAGE_CONFIG."""
        text_lower = text.lower()
        best_lang = 'en'
        best_score = 0
        for lang_code, config in LANGUAGE_CONFIG.items():
            score = sum(1 for ind in config.get('indicators', []) if ind in text_lower)
            if score > best_score:
                best_score = score
                best_lang = lang_code
        return best_lang

    # Regional code to base language mapping
    _REGIONAL_MAP = {
        'fr-CH': 'fr', 'fr-FR': 'fr', 'de-CH': 'de', 'it-CH': 'it',
        'rm': 'de',  # Romansh → German family
        'gl': 'es',  # Galician → close to Spanish
        'ca': 'es',  # Catalan → close to Spanish
        'nb': 'da',  # Norwegian Bokmål → close to Danish
        'nn': 'da',  # Norwegian Nynorsk → close to Danish
        'sr-Latn': 'sr', 'sr-Cyrl': 'sr',
        'bs-Latn': 'bs', 'hr-Latn': 'hr',
    }

    def _get_lang_config(self, language: str) -> dict:
        """Get language configuration, fallback to English if unsupported."""
        # Map regional codes to base language
        if language in self._REGIONAL_MAP:
            language = self._REGIONAL_MAP[language]
        elif '-' in language:
            language = language.split('-')[0]
        return LANGUAGE_CONFIG.get(language, FALLBACK_CONFIG)

    def _build_reasoning(self, text: str, analysis: dict, concepts: list,
                        content_type: str, context: Optional[dict],
                        language: str) -> str:
        """Build chain-of-thought reasoning using multilingual config."""
        cfg = self._get_lang_config(language)
        steps = []
        if context and 'title' in context and context['title']:
            steps.append(f"{cfg.get('analyze', 'Analyzing')} {cfg.get('about', 'about')} '{context['title']}'")
        elif concepts:
            steps.append(f"{cfg.get('analyze', 'Analyzing')} {cfg.get('about', 'about')} '{concepts[0][0]}'")
        else:
            steps.append(f"{cfg.get('analyze', 'Analyzing')} {cfg.get('content_of', 'the content')}")
        claim = self._first_claim(text)
        if claim:
            steps.append(f"{cfg.get('claim_prefix', 'The paragraph states')}: \"{claim}\"")
        word_count = analysis.get('word_count', 0)
        sentence_count = analysis.get('sentence_count', 0)
        if sentence_count > 1:
            steps.append(cfg.get('text_contains', '{s} sentences with {w} words').format(s=sentence_count, w=word_count))
        steps.append(cfg.get('type_desc', {}).get(content_type, cfg.get('type_default', 'Relevant information')))
        if concepts:
            steps.append(f"{cfg.get('concepts_prefix', 'Key concepts:')} {', '.join(c[0] for c in concepts[:3])}")
        entities = analysis.get('entities', [])
        seen_entities = set()
        unique_entities = []
        for ent_text, _ in entities:
            key = ent_text.lower()
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(ent_text)
        if unique_entities:
            steps.append(f"{cfg.get('entities_prefix', 'Entities:')} {', '.join(unique_entities[:3])}")
        if context and 'answer' in context and context['answer']:
            answer_words = set(context['answer'].lower().split()[:5])
            thinking_words = set(text.lower().split())
            overlap = answer_words.intersection(thinking_words)
            if overlap:
                steps.append(f"{cfg.get('answer_terms', 'Key terms:')} {', '.join(list(overlap)[:3])}")
        if analysis.get('has_technical_terms', False):
            steps.append(cfg.get('technical_note', 'Technical terminology included'))
        return self._format_steps(steps, cfg)

    def _format_steps(self, steps: list, cfg: dict) -> str:
        """Format reasoning steps using language-specific connectors.

        - Ensures every connector ends with a space (config files historically
          stored 'First,' without trailing space, producing 'First,the ...').
        - Only lowercases the first character of continuation steps, so proper
          nouns keep their casing ('the Treaty of Paris', never 'the treaty').
        """
        if not steps:
            return ""
        connectors = cfg.get('connectors', ['First, ', 'Additionally, ', 'Also, '])
        connectors = [c if c.endswith(' ') else f"{c} " for c in connectors]
        overflow = cfg.get('overflow_connector', 'Also, ')
        if not overflow.endswith(' '):
            overflow = f"{overflow} "
        result = []
        for i, step in enumerate(steps):
            # Avoid doubling terminal punctuation (claims already end with '.'),
            # but ignore trailing quotes so "Analyzing about 'X'" still gets '.'.
            core = step.rstrip().rstrip('\"\'')
            suffix = '' if core[-1:] in ('.', '!', '?') else '.'
            if i == 0:
                result.append(f"{step}{suffix}")
                continue
            continuation = step
            if step[:1].isalpha():
                continuation = step[0].lower() + step[1:]
            connector = connectors[i - 1] if i - 1 < len(connectors) else overflow
            result.append(f"{connector}{continuation}{suffix}")
        return " ".join(result)

    def _format_steps_en(self, steps: list) -> str:
        """Format steps with the English connector set."""
        return self._format_steps(steps, LANGUAGE_CONFIG.get('en', FALLBACK_CONFIG))

    def _format_steps_es(self, steps: list) -> str:
        """Format steps with the Spanish connector set."""
        return self._format_steps(steps, LANGUAGE_CONFIG.get('es', FALLBACK_CONFIG))