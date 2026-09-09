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
        self._nlp = None
        self._analysis_cache: dict = {}
        self._cache_maxsize = 500
        self._load_nlp_model()

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

    def _load_nlp_model(self):
        if not SPACY_AVAILABLE:
            logger.info("Using fallback regex-based NLP analysis")
            return
        models_to_try = [
            'xx_ent_wiki_sm', 'es_core_news_sm', 'en_core_web_sm',
            'fr_core_news_sm', 'de_core_news_sm', 'it_core_news_sm',
            'pt_core_news_sm', 'pl_core_news_sm', 'nl_core_news_sm',
            'sv_core_news_sm', 'fi_core_news_sm', 'el_core_news_sm',
            'ro_core_news_sm', 'hr_core_news_sm', 'bg_core_news_sm',
            'cs_core_news_sm', 'sk_core_news_sm', 'lt_core_news_sm',
            'lv_core_news_sm', 'et_core_news_sm', 'hu_core_news_sm',
            'nb_core_news_sm', 'da_core_news_sm',
        ]
        for model in models_to_try:
            try:
                self._nlp = spacy.load(model)
                logger.info(f"Loaded spaCy model: {model}")
                return
            except OSError:
                continue
        logger.warning("No spaCy model found. Using fallback regex-based analysis.\n"
                       "Install multilingual support: python -m spacy download xx_ent_wiki_sm")
        self._nlp = None

    def generate_thinking(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not text or len(text.strip()) < 10:
            return ""
        cache_key = hashlib.md5(f"{text}:{context}".encode()).hexdigest()
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        analysis = self._analyze_text(text)
        depth_config = self._get_depth_config(text, analysis)
        key_concepts = self._extract_key_concepts(text, analysis, depth_config)
        content_type = self._detect_content_type(text, analysis)
        language = self._detect_language(text)
        thinking = self._build_reasoning(text, analysis, key_concepts, content_type, context, language)
        self._analysis_cache[cache_key] = thinking
        if len(self._analysis_cache) > self._cache_maxsize:
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]
        return thinking

    def _analyze_text(self, text: str) -> dict:
        if self._nlp:
            return self._analyze_text_spacy(text)
        return self._analyze_text_regex(text)

    def _analyze_text_spacy(self, text: str) -> dict:
        doc = self._nlp(text)
        entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 1]
        noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
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
            'language': self._detect_language(text),
        }

    def _analyze_text_regex(self, text: str) -> dict:
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
                noun_phrases.append(phrase)
        return {
            'entities': entities[:self.depth_config['max_entities'] * 2],
            'noun_phrases': noun_phrases[:self.depth_config['max_concepts'] * 2],
            'sentences': sentences,
            'word_count': len(words), 'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(1, len(sentences)),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_questions': '?' in text,
            'has_technical_terms': self._detect_technical_terms(text),
            'language': self._detect_language(text),
        }

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
            candidates.append(phrase.lower())
        for entity, _ in analysis.get('entities', []):
            candidates.append(entity.lower())
        words = text.lower().split()
        word_freq = Counter()
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] += 1
        scored = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            score = sum(word_freq.get(w, 0) for w in candidate.split())
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
        word_count = analysis.get('word_count', 0)
        sentence_count = analysis.get('sentence_count', 0)
        if sentence_count > 1:
            steps.append(cfg.get('text_contains', '{s} sentences with {w} words').format(s=sentence_count, w=word_count))
        steps.append(cfg.get('type_desc', {}).get(content_type, cfg.get('type_default', 'Relevant information')))
        if concepts:
            steps.append(f"{cfg.get('concepts_prefix', 'Key concepts:')} {', '.join(c[0] for c in concepts[:3])}")
        entities = analysis.get('entities', [])
        if entities:
            steps.append(f"{cfg.get('entities_prefix', 'Entities:')} {', '.join(e[0] for e in entities[:3])}")
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
        """Format reasoning steps using language-specific connectors."""
        if not steps:
            return ""
        connectors = cfg.get('connectors', ['First, ', 'Additionally, ', 'Also, '])
        overflow = cfg.get('overflow_connector', 'Also, ')
        result = []
        for i, step in enumerate(steps):
            if i == 0:
                result.append(f"{step}.")
            elif i - 1 < len(connectors):
                result.append(f"{connectors[i-1]}{step.lower()}.")
            else:
                result.append(f"{overflow}{step.lower()}.")
        return " ".join(result)