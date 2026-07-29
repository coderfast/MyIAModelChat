"""
Real Thinking Engine based on NLP analysis.
Generates chain-of-thought reasoning by analyzing actual text content,
without depending on external LLM APIs like Ollama.
"""
import re
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. Using fallback regex-based analysis. "
                   "Install with: pip install spacy")


class ThinkingEngine:
    """
    Motor de thinking real basado en NLP.
    Analiza el contenido real del texto para generar reasoning.
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
        self._analysis_cache: Dict[str, dict] = {}
        self._load_nlp_model()

    def _estimate_complexity(self, text: str, analysis: dict) -> str:
        """
        Estimate text complexity to adjust thinking depth dynamically.
        Returns: 'basic', 'adaptive', or 'detailed'
        """
        word_count = analysis.get('word_count', 0)
        entity_count = len(analysis.get('entities', []))
        sentence_count = analysis.get('sentence_count', 0)
        has_technical = analysis.get('has_technical_terms', False)

        # Simple texts: short, few entities
        if word_count < 20 or sentence_count < 2:
            return 'basic'

        # Complex texts: long, many entities, technical
        if word_count > 100 or entity_count > 5 or (has_technical and word_count > 50):
            return 'detailed'

        # Default
        return 'adaptive'

    def _get_depth_config(self, text: str, analysis: dict) -> dict:
        """Get depth config based on adaptive complexity."""
        if self.depth != 'adaptive':
            return self.depth_config

        complexity = self._estimate_complexity(text, analysis)
        return self.DEPTH_CONFIG.get(complexity, self.depth_config)

    def _load_nlp_model(self):
        """Load spaCy model for NLP analysis."""
        if not SPACY_AVAILABLE:
            logger.info("Using fallback regex-based NLP analysis")
            return

        try:
            self._nlp = spacy.load('es_core_news_sm')
            logger.info("Loaded spaCy model: es_core_news_sm")
        except OSError:
            try:
                self._nlp = spacy.load('en_core_web_sm')
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("No spaCy model found. Download one with:\n"
                               "  python -m spacy download es_core_news_sm\n"
                               "  python -m spacy download en_core_web_sm\n"
                               "Using fallback regex-based analysis.")
                self._nlp = None

    def generate_thinking(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate real thinking by analyzing the actual text content.

        Args:
            text: The text to analyze and generate thinking for
            context: Optional context dict with 'title', 'answer', 'question', etc.

        Returns:
            String with chain-of-thought reasoning
        """
        if not text or len(text.strip()) < 10:
            return ""

        # Check cache
        cache_key = hashlib.md5(f"{text}:{context}".encode()).hexdigest()
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        # Analyze text
        analysis = self._analyze_text(text)

        # Get adaptive depth config
        depth_config = self._get_depth_config(text, analysis)

        # Extract key concepts
        key_concepts = self._extract_key_concepts(text, analysis, depth_config)

        # Detect content type
        content_type = self._detect_content_type(text, analysis)

        # Detect language
        language = self._detect_language(text)

        # Build reasoning
        thinking = self._build_reasoning(text, analysis, key_concepts, content_type, context, language)

        # Cache result
        self._analysis_cache[cache_key] = thinking

        return thinking

    def _analyze_text(self, text: str) -> dict:
        """
        Complete text analysis using spaCy or fallback regex.
        """
        if self._nlp:
            return self._analyze_text_spacy(text)
        else:
            return self._analyze_text_regex(text)

    def _analyze_text_spacy(self, text: str) -> dict:
        """Analysis using spaCy NLP pipeline."""
        doc = self._nlp(text)

        entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 1]
        noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        word_count = len(text.split())
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(1, sentence_count)

        return {
            'entities': entities,
            'noun_phrases': noun_phrases,
            'sentences': sentences,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_questions': '?' in text,
            'has_technical_terms': self._detect_technical_terms(text),
            'language': self._detect_language(text),
        }

    def _analyze_text_regex(self, text: str) -> dict:
        """Fallback analysis using regex when spaCy is not available."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Simple entity detection: capitalized words not at sentence start
        entities = []
        for i, word in enumerate(words):
            if i > 0 and word[0:1].isupper() and words[i-1][-1:] in '.!?\n':
                continue
            if word[0:1].isupper() and len(word) > 2 and word.isalpha():
                entities.append((word, 'ENTITY'))

        # Simple noun phrase detection: adjective + noun patterns
        noun_phrases = []
        adj_pattern = re.compile(r'\b\w+\s+\w+(?:\s+\w+)?\b')
        for match in adj_pattern.finditer(text):
            phrase = match.group()
            if len(phrase.split()) >= 2:
                noun_phrases.append(phrase)

        return {
            'entities': entities[:self.depth_config['max_entities'] * 2],
            'noun_phrases': noun_phrases[:self.depth_config['max_concepts'] * 2],
            'sentences': sentences,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(1, len(sentences)),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_questions': '?' in text,
            'has_technical_terms': self._detect_technical_terms(text),
            'language': self._detect_language(text),
        }

    def _detect_technical_terms(self, text: str) -> bool:
        """Detect if text contains technical terminology."""
        technical_indicators = [
            r'\bAPI\b', r'\bSDK\b', r'\bHTTP\b', r'\bJSON\b', r'\bXML\b',
            r'\bSQL\b', r'\bREST\b', r'\bGraphQL\b', r'\bOAuth\b',
            r'\balgoritmo\b', r'\bfunción\b', r'\bvariable\b', r'\bclase\b',
            r'\bfunction\b', r'\bclass\b', r'\bmethod\b', r'\bmodule\b',
            r'\bdatabase\b', r'\bserver\b', r'\bclient\b', r'\bprotocol\b',
        ]
        for pattern in technical_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_key_concepts(self, text: str, analysis: dict, depth_config: dict = None) -> List[Tuple[str, int]]:
        """
        Extract key concepts using frequency analysis.
        Returns list of (concept, score) sorted by relevance.
        """
        if depth_config is None:
            depth_config = self.depth_config

        candidates = []

        # Add noun phrases as candidates
        for phrase in analysis.get('noun_phrases', []):
            candidates.append(phrase.lower())

        # Add entities as candidates
        for entity, _ in analysis.get('entities', []):
            candidates.append(entity.lower())

        # Count word frequency
        words = text.lower().split()
        word_freq = Counter()
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] += 1

        # Score candidates
        scored = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            candidate_words = candidate.split()
            score = sum(word_freq.get(w, 0) for w in candidate_words)
            if score > 0:
                scored.append((candidate, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:depth_config['max_concepts']]

    def _detect_content_type(self, text: str, analysis: dict) -> str:
        """
        Detect the content type of the text.
        Returns: 'qa', 'technical', 'narrative', 'instructional', 'factual', 'conversational'
        """
        text_lower = text.lower()

        # Questions -> Q&A
        if analysis.get('has_questions', False):
            return 'qa'

        # Numbers + technical terms -> Technical
        if analysis.get('has_numbers', False) and analysis.get('has_technical_terms', False):
            return 'technical'

        # Past tense indicators -> Narrative
        past_indicators = ['was', 'were', 'had', 'did', 'era', 'fue', 'tenía',
                          'hizo', 'came', 'went', 'said', 'told']
        if any(ind in text_lower for ind in past_indicators):
            return 'narrative'

        # Imperative indicators -> Instructional
        imperative_indicators = ['must', 'should', 'need to', 'debe', 'debería',
                                'necesita', 'follow', 'execute', 'run', 'install',
                                'ensure', 'verify', 'check']
        if any(ind in text_lower for ind in imperative_indicators):
            return 'instructional'

        # Conversational indicators
        conversational_indicators = ['hello', 'hi', 'hey', 'hola', 'bye', 'thanks',
                                    'please', 'por favor', 'gracias']
        if any(ind in text_lower for ind in conversational_indicators):
            return 'conversational'

        # Default: factual
        return 'factual'

    def _detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        text_lower = text.lower()

        # Spanish indicators
        spanish_indicators = [' el ', ' la ', ' los ', ' las ', ' de ', ' del ',
                            ' en ', ' un ', ' una ', ' que ', ' es ', ' son ',
                            'porque', 'entonces', 'además', 'sin embargo']
        spanish_count = sum(1 for ind in spanish_indicators if ind in text_lower)

        # English indicators
        english_indicators = [' the ', ' is ', ' are ', ' was ', ' were ',
                            ' have ', ' has ', ' had ', ' that ', ' which ']
        english_count = sum(1 for ind in english_indicators if ind in text_lower)

        if spanish_count > english_count:
            return 'es'
        elif english_count > spanish_count:
            return 'en'
        else:
            return 'es'  # Default to Spanish

    def _build_reasoning(self, text: str, analysis: dict, concepts: list,
                        content_type: str, context: Optional[dict],
                        language: str) -> str:
        """
        Build chain-of-thought reasoning based on actual text analysis.
        """
        if language == 'es':
            return self._build_reasoning_es(text, analysis, concepts, content_type, context)
        else:
            return self._build_reasoning_en(text, analysis, concepts, content_type, context)

    def _build_reasoning_es(self, text: str, analysis: dict, concepts: list,
                           content_type: str, context: Optional[dict]) -> str:
        """Build reasoning in Spanish."""
        steps = []

        # Step 1: Identify the content
        if context and 'title' in context and context['title']:
            steps.append(f"Analizando el contenido '{context['title']}'")
        elif concepts:
            steps.append(f"Analizando contenido sobre '{concepts[0][0]}'")
        else:
            steps.append("Analizando el contenido del texto")

        # Step 2: Describe structure
        word_count = analysis.get('word_count', 0)
        sentence_count = analysis.get('sentence_count', 0)
        if sentence_count > 1:
            steps.append(f"El texto contiene {sentence_count} oraciones con "
                        f"{word_count} palabras en total")

        # Step 3: Identify content type
        type_descriptions = {
            'qa': "Se trata de una pregunta que requiere una respuesta específica",
            'technical': "El contenido es técnico y contiene datos numéricos o术语 especializados",
            'narrative': "El texto presenta una narrativa o descripción de eventos",
            'instructional': "El contenido proporciona instrucciones o directrices a seguir",
            'factual': "Se presenta información factual y objetiva sobre el tema",
            'conversational': "El texto tiene un tono conversacional y coloquial",
        }
        steps.append(type_descriptions.get(content_type, "El contenido presenta información relevante"))

        # Step 4: Mention key concepts
        if concepts:
            top_concepts = [c[0] for c in concepts[:3]]
            steps.append(f"Los conceptos principales identificados son: {', '.join(top_concepts)}")

        # Step 5: Named entities
        entities = analysis.get('entities', [])
        if entities:
            entity_texts = [e[0] for e in entities[:3]]
            steps.append(f"Se mencionan las siguientes entidades: {', '.join(entity_texts)}")

        # Step 6: Connection with context
        if context:
            if 'answer' in context and context['answer']:
                answer_words = set(context['answer'].lower().split()[:5])
                thinking_words = set(text.lower().split())
                overlap = answer_words.intersection(thinking_words)
                if overlap:
                    steps.append(f"Términos clave que conectan con la respuesta: {', '.join(list(overlap)[:3])}")

            if 'question' in context and context['question']:
                question_lower = context['question'].lower()
                if 'qué' in question_lower or 'que' in question_lower:
                    steps.append("La consulta solicita información específica sobre el tema")
                elif 'cómo' in question_lower or 'como' in question_lower:
                    steps.append("La consulta solicita un procedimiento o explicación")
                elif 'por qué' in question_lower or 'porque' in question_lower:
                    steps.append("La consulta busca una razón o explicación causal")

        # Step 7: Technical analysis
        if analysis.get('has_technical_terms', False):
            steps.append("El contenido incluye terminología técnica especializada")

        # Format steps with connectors
        return self._format_steps_es(steps)

    def _build_reasoning_en(self, text: str, analysis: dict, concepts: list,
                           content_type: str, context: Optional[dict]) -> str:
        """Build reasoning in English."""
        steps = []

        # Step 1: Identify the content
        if context and 'title' in context and context['title']:
            steps.append(f"Analyzing the content '{context['title']}'")
        elif concepts:
            steps.append(f"Analyzing content about '{concepts[0][0]}'")
        else:
            steps.append("Analyzing the text content")

        # Step 2: Describe structure
        word_count = analysis.get('word_count', 0)
        sentence_count = analysis.get('sentence_count', 0)
        if sentence_count > 1:
            steps.append(f"The text contains {sentence_count} sentences with "
                        f"{word_count} words in total")

        # Step 3: Identify content type
        type_descriptions = {
            'qa': "This is a question that requires a specific answer",
            'technical': "The content is technical and contains numerical data or specialized terminology",
            'narrative': "The text presents a narrative or description of events",
            'instructional': "The content provides instructions or guidelines to follow",
            'factual': "Factual and objective information is presented about the topic",
            'conversational': "The text has a conversational and colloquial tone",
        }
        steps.append(type_descriptions.get(content_type, "The content presents relevant information"))

        # Step 4: Mention key concepts
        if concepts:
            top_concepts = [c[0] for c in concepts[:3]]
            steps.append(f"Key concepts identified: {', '.join(top_concepts)}")

        # Step 5: Named entities
        entities = analysis.get('entities', [])
        if entities:
            entity_texts = [e[0] for e in entities[:3]]
            steps.append(f"Entities mentioned: {', '.join(entity_texts)}")

        # Step 6: Connection with context
        if context:
            if 'answer' in context and context['answer']:
                answer_words = set(context['answer'].lower().split()[:5])
                thinking_words = set(text.lower().split())
                overlap = answer_words.intersection(thinking_words)
                if overlap:
                    steps.append(f"Key terms connecting with the answer: {', '.join(list(overlap)[:3])}")

            if 'question' in context and context['question']:
                question_lower = context['question'].lower()
                if 'what' in question_lower:
                    steps.append("The query requests specific information about the topic")
                elif 'how' in question_lower:
                    steps.append("The query requests a procedure or explanation")
                elif 'why' in question_lower:
                    steps.append("The query seeks a reason or causal explanation")

        # Step 7: Technical analysis
        if analysis.get('has_technical_terms', False):
            steps.append("The content includes specialized technical terminology")

        # Format steps with connectors
        return self._format_steps_en(steps)

    def _format_steps_es(self, steps: list) -> str:
        """Format reasoning steps in Spanish with logical connectors."""
        if not steps:
            return ""

        connectors = [
            "En primer lugar, ",
            "Además, ",
            "Por otro lado, ",
            "Asimismo, ",
            "Finalmente, ",
            "En conclusión, ",
        ]

        result = []
        for i, step in enumerate(steps):
            if i == 0:
                result.append(f"{step}.")
            elif i < len(connectors):
                result.append(f"{connectors[i-1]}{step.lower()}.")
            else:
                result.append(f"También, {step.lower()}.")

        return " ".join(result)

    def _format_steps_en(self, steps: list) -> str:
        """Format reasoning steps in English with logical connectors."""
        if not steps:
            return ""

        connectors = [
            "First, ",
            "Additionally, ",
            "Furthermore, ",
            "Moreover, ",
            "Finally, ",
            "In conclusion, ",
        ]

        result = []
        for i, step in enumerate(steps):
            if i == 0:
                result.append(f"{step}.")
            elif i < len(connectors):
                result.append(f"{connectors[i-1]}{step.lower()}.")
            else:
                result.append(f"Also, {step.lower()}.")

        return " ".join(result)
