"""
AIML Parser - Resolves AIML 2.0 elements and generates clean training samples.

Resolves: <srai>, <random>, <set>, <get>, <bot>, <star>, <person>,
          <thinking>, <that>, <topic>, <condition>, HTML tags
Expands: wildcards with examples, <random><li> into N samples
Filters: quality pre-contamination
"""
import re
import random
import html
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

# ============================================================
# Wildcard Examples Dictionary
# ============================================================

WILDCARD_EXAMPLES = {
    'greeting': {
        '*': ['HELLO', 'HI', 'GOOD MORNING', 'GOOD AFTERNOON', 'GOOD EVENING', 'HEY', 'HI THERE'],
        '_': ['HELLO THERE', 'HI BOT', 'HEY THERE', 'GOOD MORNING FRIEND'],
        '**': ['SOMETHING TO GREET YOU', 'A FRIENDLY HELLO'],
        '^': ['THERE', 'FRIEND'],
    },
    'farewell': {
        '*': ['GOODBYE', 'BYE', 'SEE YOU LATER', 'GOOD NIGHT', 'TAKE CARE', 'SEE YOU SOON'],
        '_': ['BYE FOR NOW', 'SEE YOU', 'GOODBYE FRIEND', 'TAKE CARE'],
        '**': ['A FAREWELL MESSAGE', 'GOODBYE FOR NOW'],
        '^': ['NOW', 'SOON'],
    },
    'identity': {
        '*': ['A ROBOT', 'AN AI', 'A CHATBOT', 'A COMPUTER PROGRAM', 'A VIRTUAL ASSISTANT', 'AN ASSISTANT'],
        '_': ['AN ARTIFICIAL INTELLIGENCE', 'A MACHINE', 'A SMART BOT', 'YOUR ASSISTANT'],
        '**': ['SOMETHING THAT TALKS', 'A SMART ENTITY'],
        '^': ['ALSO'],
    },
    'question': {
        '*': ['SOMETHING', 'ANYTHING', 'A TOPIC', 'HELP', 'INFO', 'ANSWER', 'QUESTION'],
        '_': ['A QUESTION', 'SOMETHING INTERESTING', 'ANYTHING AT ALL', 'A GOOD TOPIC'],
        '**': ['SOMETHING INTERESTING', 'A GOOD TOPIC', 'ANYTHING YOU WANT', 'SOMETHING NEW'],
        '^': ['ABOUT', 'REGARDING', 'CONCERNING'],
    },
    'opinion': {
        '*': ['CATS', 'DOGS', 'PYTHON', 'MACHINE LEARNING', 'PIZZA', 'MOVIES', 'MUSIC', 'BOOKS', 'FOOD', 'TRAVEL'],
        '_': ['THINGS', 'STUFF', 'EVERYTHING', 'LIFE', 'WORLD', 'NATURE'],
        '**': ['A LOT OF THINGS', 'MANY TOPICS', 'VARIOUS SUBJECTS'],
        '^': ['REALLY', 'VERY MUCH', 'AT ALL'],
    },
    'action': {
        '*': ['LIKE', 'LOVE', 'ENJOY', 'WANT', 'NEED', 'PREFER', 'CHOOSE', 'DO', 'MAKE', 'PLAY'],
        '_': ['DOING THAT', 'GOING THERE', 'MAKING THINGS', 'HELPING PEOPLE'],
        '**': ['DOING SOMETHING', 'MAKING PLANS'],
        '^': ['ALWAYS', 'SOMETIMES', 'OFTEN'],
    },
    'adjective': {
        '*': ['FUNNY', 'SMART', 'NICE', 'WEIRD', 'COOL', 'BRIGHT', 'CLEVER', 'KIND', 'HELPFUL', 'INTERESTING'],
        '_': ['THAT WAY', 'LIKE THAT', 'IN GENERAL', 'MOSTLY'],
        '**': ['SOMEHOW', 'IN CERTAIN WAYS'],
        '^': ['VERY', 'QUITE', 'REALLY'],
    },
    'emotion': {
        '*': ['HAPPY', 'SAD', 'ANGRY', 'EXCITED', 'TIRED', 'WORRIED', 'CALM', 'PROUD', 'GRATEFUL', 'HOPEFUL'],
        '_': ['FEELING THAT WAY', 'THAT EMOTION', 'IN THAT MOOD'],
        '**': ['SOME EMOTION', 'A CERTAIN FEELING'],
        '^': ['A LITTLE', 'VERY', 'EXTREMELY'],
    },
    'time': {
        '*': ['NOW', 'TODAY', 'YESTERDAY', 'TOMORROW', 'MORNING', 'AFTERNOON', 'EVENING', 'NIGHT', 'WEEK', 'MONTH', 'YEAR'],
        '_': ['RIGHT NOW', 'CURRENTLY', 'AT THIS MOMENT', 'LATER'],
        '**': ['SOME TIME', 'A CERTAIN PERIOD'],
        '^': ['EXACTLY', 'AROUND', 'APPROXIMATELY'],
    },
    'place': {
        '*': ['HOME', 'WORK', 'SCHOOL', 'PARK', 'STORE', 'OFFICE', 'BEACH', 'CITY', 'COUNTRY', 'WORLD'],
        '_': ['MY PLACE', 'THAT PLACE', 'SOMEWHERE NICE', 'A GOOD SPOT'],
        '**': ['SOME LOCATION', 'A CERTAIN PLACE'],
        '^': ['NEAR', 'FAR', 'AROUND'],
    },
    'person': {
        '*': ['FRIEND', 'TEACHER', 'DOCTOR', 'ENGINEER', 'ARTIST', 'STUDENT', 'PARENT', 'CHILD', 'NEIGHBOR', 'STRANGER'],
        '_': ['SOMEONE', 'A PERSON', 'MY FRIEND', 'THAT PERSON'],
        '**': ['SOMEONE I KNOW', 'A CERTAIN PERSON'],
        '^': ['ALSO', 'EVEN', 'ESPECIALLY'],
    },
    'object': {
        '*': ['BOOK', 'PHONE', 'COMPUTER', 'TABLE', 'CHAIR', 'DOOR', 'WINDOW', 'CAR', 'BICYCLE', 'KEY'],
        '_': ['SOMETHING', 'THAT THING', 'MY STUFF', 'A NICE OBJECT'],
        '**': ['SOMETHING USEFUL', 'A NICE THING'],
        '^': ['SMALL', 'BIG', 'NEW', 'OLD'],
    },
    'number': {
        '*': ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'TEN', 'HUNDRED', 'THOUSAND', 'MANY', 'FEW'],
        '_': ['A NUMBER', 'SOME AMOUNT', 'THAT MANY', 'SEVERAL'],
        '**': ['SOME AMOUNT', 'A CERTAIN NUMBER'],
        '^': ['ABOUT', 'AROUND', 'EXACTLY'],
    },
    'food': {
        '*': ['PIZZA', 'PASTA', 'RICE', 'BREAD', 'SOUP', 'SALAD', 'FRUIT', 'CHOCOLATE', 'COFFEE', 'TEA'],
        '_': ['SOMETHING TO EAT', 'THAT FOOD', 'MY FAVORITE DISH', 'A GOOD MEAL'],
        '**': ['SOMETHING DELICIOUS', 'A GOOD FOOD'],
        '^': ['REALLY', 'VERY', 'QUITE'],
    },
    'animal': {
        '*': ['CAT', 'DOG', 'BIRD', 'FISH', 'HORSE', 'RABBIT', 'TURTLE', 'SNAKE', 'FROG', 'BEAR'],
        '_': ['THAT ANIMAL', 'SOME PET', 'MY FAVORITE ANIMAL', 'A CUTE CREATURE'],
        '**': ['SOME ANIMAL', 'A LITTLE CREATURE'],
        '^': ['BIG', 'SMALL', 'CUTE', 'WILD'],
    },
    'technology': {
        '*': ['COMPUTER', 'PHONE', 'INTERNET', 'SOFTWARE', 'HARDWARE', 'AI', 'ROBOT', 'PYTHON', 'CODE', 'DATA'],
        '_': ['THAT TECH', 'MODERN TOOLS', 'DIGITAL STUFF', 'MY DEVICE'],
        '**': ['SOMETHING DIGITAL', 'A TECH DEVICE'],
        '^': ['NEW', 'OLD', 'ADVANCED'],
    },
    'generic': {
        '*': ['SOMETHING', 'ANYTHING', 'THAT', 'IT', 'THIS', 'THING', 'STUFF', 'MATTER', 'ASPECT', 'PART'],
        '_': ['SOMETHING NICE', 'A GOOD THING', 'MY FRIEND', 'THAT THING', 'SOMETHING ELSE', 'ANYTHING AT ALL'],
        '**': ['SOMETHING INTERESTING', 'A TOPIC', 'ANYTHING', 'SOMETHING IMPORTANT', 'A GOOD SUBJECT'],
        '^': ['A THING', 'SOMETHING', 'SOME ITEM', 'CERTAIN PART'],
    },
}

CATEGORY_KEYWORDS = {
    'greeting': ['HELLO', 'HI', 'HEY', 'GOOD MORNING', 'GOOD AFTERNOON', 'GOOD EVENING', 'GREETINGS', 'HOWDY', 'WHAT\'S UP'],
    'farewell': ['GOODBYE', 'BYE', 'SEE YOU', 'LATER', 'GOOD NIGHT', 'TAKE CARE', 'SEE YOU LATER', 'CATCH YOU LATER'],
    'identity': ['WHO ARE YOU', 'WHAT ARE YOU', 'YOUR NAME', 'WHAT DO YOU CALL', 'WHAT IS YOUR NAME', 'TELL ME ABOUT YOURSELF', 'INTRODUCE YOURSELF'],
    'question': ['WHAT', 'WHY', 'HOW', 'WHEN', 'WHERE', 'WHO', 'WHICH', 'CAN YOU', 'COULD YOU', 'WOULD YOU', 'DO YOU'],
    'opinion': ['DO YOU LIKE', 'DO YOU THINK', 'FAVORITE', 'BEST', 'WORST', 'PREFER', 'OPINION', 'THOUGHTS', 'FEEL ABOUT'],
    'thanks': ['THANK', 'THANKS', 'APPRECIATE', 'GRATEFUL', 'CHEERS'],
    'yes': ['YES', 'YEAH', 'YEP', 'SURE', 'OK', 'OKAY', 'ALRIGHT', 'CERTAINLY', 'ABSOLUTELY', 'OF COURSE'],
    'no': ['NO', 'NOPE', 'NAH', 'NOT REALLY', 'NOT AT ALL', 'NEGATIVE', 'NO WAY', 'NEVER'],
    'apology': ['SORRY', 'APOLOGIZE', 'MY BAD', 'PARDON', 'EXCUSE ME', 'FORGIVE ME'],
    'emotion': ['HAPPY', 'SAD', 'ANGRY', 'EXCITED', 'TIRED', 'WORRIED', 'FEEL', 'FEELING', 'MOOD', 'EMOTION'],
    'time': ['NOW', 'TODAY', 'YESTERDAY', 'TOMORROW', 'TIME', 'DATE', 'WEEK', 'MONTH', 'YEAR', 'MORNING', 'AFTERNOON', 'EVENING', 'NIGHT'],
    'place': ['WHERE', 'PLACE', 'LOCATION', 'HOME', 'WORK', 'SCHOOL', 'CITY', 'COUNTRY', 'WORLD', 'HERE', 'THERE'],
    'person': ['WHO', 'PERSON', 'FRIEND', 'TEACHER', 'DOCTOR', 'ENGINEER', 'SOMEONE', 'ANYBODY', 'EVERYBODY', 'NOBODY'],
    'object': ['WHAT', 'THING', 'OBJECT', 'ITEM', 'STUFF', 'BELONGING', 'POSSESSION'],
    'number': ['HOW MANY', 'HOW MUCH', 'NUMBER', 'COUNT', 'QUANTITY', 'AMOUNT', 'MANY', 'FEW', 'SOME'],
    'food': ['FOOD', 'EAT', 'DRINK', 'MEAL', 'BREAKFAST', 'LUNCH', 'DINNER', 'SNACK', 'HUNGRY', 'THIRSTY'],
    'animal': ['ANIMAL', 'PET', 'CAT', 'DOG', 'BIRD', 'FISH', 'HORSE', 'CREATURE', 'WILDLIFE'],
    'technology': ['COMPUTER', 'PHONE', 'INTERNET', 'SOFTWARE', 'HARDWARE', 'AI', 'ROBOT', 'CODE', 'PROGRAM', 'TECH'],
    'help': ['HELP', 'ASSIST', 'SUPPORT', 'GUIDE', 'SHOW', 'TEACH', 'EXPLAIN', 'INSTRUCT'],
    'request': ['PLEASE', 'WANT', 'NEED', 'WOULD LIKE', 'CAN I', 'MAY I', 'COULD I'],
    'greeting_response': ['I AM FINE', 'I\'M GOOD', 'NOT BAD', 'DOING WELL', 'GREAT', 'WONDERFUL'],
    'name': ['MY NAME', 'I AM CALLED', 'CALL ME', 'I\'M', 'NAME IS'],
    'age': ['OLD', 'AGE', 'BIRTHDAY', 'BORN', 'YEARS OLD'],
    'weather': ['WEATHER', 'RAIN', 'SUN', 'CLOUDY', 'HOT', 'COLD', 'TEMPERATURE', 'FORECAST', 'STORM', 'SNOW', 'WIND'],
    'music': ['MUSIC', 'SONG', 'BAND', 'SINGER', 'LISTEN', 'PLAY', 'CONCERT', 'ALBUM'],
    'movie': ['MOVIE', 'FILM', 'WATCH', 'CINEMA', 'ACTOR', 'ACTRESS', 'DIRECTOR', 'SCENE'],
    'book': ['BOOK', 'READ', 'AUTHOR', 'STORY', 'NOVEL', 'CHAPTER', 'LIBRARY', 'WRITER'],
    'sport': ['SPORT', 'GAME', 'PLAY', 'TEAM', 'WIN', 'LOSE', 'SCORE', 'MATCH', 'COMPETITION'],
    'travel': ['TRAVEL', 'TRIP', 'VISIT', 'FLIGHT', 'HOTEL', 'VACATION', 'HOLIDAY', 'DESTINATION'],
    'health': ['HEALTH', 'SICK', 'ILL', 'DOCTOR', 'MEDICINE', 'PAIN', 'WELL', 'FIT', 'EXERCISE'],
    'work': ['WORK', 'JOB', 'CAREER', 'OFFICE', 'BOSS', 'COLLEAGUE', 'EMPLOY', 'SALARY', 'MEETING'],
    'education': ['SCHOOL', 'LEARN', 'STUDY', 'TEACHER', 'CLASS', 'EXAM', 'HOMEWORK', 'UNIVERSITY', 'DEGREE'],
    'family': ['FAMILY', 'MOTHER', 'FATHER', 'PARENT', 'CHILD', 'SISTER', 'BROTHER', 'SON', 'DAUGHTER'],
    'love': ['LOVE', 'LIKE', 'CARE', 'ADORE', 'FRIENDSHIP', 'HEART', 'ROMANTIC', 'AFFECTION'],
    'news': ['NEWS', 'HEADLINE', 'REPORT', 'CURRENT', 'EVENT', 'STORY', 'UPDATE', 'INFORMATION'],
    'history': ['HISTORY', 'PAST', 'ANCIENT', 'WAR', 'KING', 'QUEEN', 'EMPIRE', 'CENTURY'],
    'science': ['SCIENCE', 'PHYSICS', 'CHEMISTRY', 'BIOLOGY', 'EXPERIMENT', 'RESEARCH', 'DISCOVER'],
    'math': ['MATH', 'NUMBER', 'CALCULATE', 'EQUATION', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE'],
    'art': ['ART', 'PAINT', 'DRAW', 'COLOR', 'PICTURE', 'SCULPTURE', 'GALLERY', 'MUSEUM'],
    'nature': ['NATURE', 'TREE', 'FLOWER', 'ANIMAL', 'SEA', 'MOUNTAIN', 'FOREST', 'RIVER', 'LAKE'],
    'space': ['SPACE', 'STAR', 'PLANET', 'MOON', 'SUN', 'GALAXY', 'UNIVERSE', 'ASTRONAUT', 'ROCKET'],
    'ocean': ['OCEAN', 'SEA', 'FISH', 'CORAL', 'WAVE', 'BEACH', 'WATER', 'MARINE', 'AQUATIC'],
    'city': ['CITY', 'TOWN', 'STREET', 'BUILDING', 'HOUSE', 'ROAD', 'BRIDGE', 'PARK'],
    'country': ['COUNTRY', 'NATION', 'STATE', 'GOVERNMENT', 'PRESIDENT', 'CAPITAL', 'BORDER'],
    'language': ['LANGUAGE', 'WORDS', 'SPEAK', 'TALK', 'SAY', 'TELL', 'PRONOUNCE', 'GRAMMAR'],
    'color': ['COLOR', 'RED', 'BLUE', 'GREEN', 'YELLOW', 'BLACK', 'WHITE', 'ORANGE', 'PURPLE'],
    'shape': ['SHAPE', 'CIRCLE', 'SQUARE', 'TRIANGLE', 'RECTANGLE', 'ROUND', 'FLAT', 'BIG', 'SMALL'],
}

# Bot property defaults
BOT_DEFAULTS = {
    'name': 'Alice',
    'version': '2.0',
    'species': 'robot',
    'gender': 'female',
    'location': 'internet',
}

# HTML tags to strip
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
HTML_ENTITY_PATTERN = re.compile(r'&\w+;|&#\d+;')

# AIML thinking pattern (NOT training thinking)
AIML_THINKING_PATTERN = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)

# Max SRAI chain depth
MAX_SRAI_DEPTH = 5


class AIMLParser:
    """Parse AIML categories into clean training samples."""

    def __init__(self, bot_properties: Dict[str, str] = None):
        self.bot_properties = bot_properties or BOT_DEFAULTS
        self._category_index: Dict[str, ElementTree.Element] = {}
        self._srai_cache: Dict[str, Optional[str]] = {}
        self._stats = {
            'total_categories': 0,
            'parsed_ok': 0,
            'srai_resolved': 0,
            'srai_failed': 0,
            'random_expanded': 0,
            'wildcards_expanded': 0,
            'discarded': 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def index_categories(self, root: ElementTree.Element) -> Dict[str, ElementTree.Element]:
        """Index all categories by uppercase pattern for <srai> resolution."""
        self._category_index = {}
        for category in root.findall('category'):
            pattern_elem = category.find('pattern')
            if pattern_elem is not None and pattern_elem.text:
                pattern_key = pattern_elem.text.strip().upper()
                self._category_index[pattern_key] = category
        return self._category_index

    def parse_category(
        self,
        category: ElementTree.Element,
        all_categories: Dict[str, ElementTree.Element] = None
    ) -> List[dict]:
        """
        Parse one AIML category into 0-N normalized samples.

        Args:
            category: XML element <category>
            all_categories: Index of all categories for <srai> resolution

        Returns:
            List of sample dicts with 'input_ids' key
        """
        self._stats['total_categories'] += 1

        if all_categories is None:
            all_categories = self._category_index

        # Extract pattern
        pattern_elem = category.find('pattern')
        if pattern_elem is None or not pattern_elem.text:
            self._stats['discarded'] += 1
            return []

        raw_pattern = pattern_elem.text.strip()
        if not raw_pattern:
            self._stats['discarded'] += 1
            return []

        # Extract template
        template_elem = category.find('template')
        if template_elem is None:
            self._stats['discarded'] += 1
            return []

        # Resolve template recursively
        resolved = self._resolve_template(template_elem, all_categories, depth=0)

        # Handle <random> which returns a list
        if isinstance(resolved, list):
            templates = resolved
        elif resolved is None:
            self._stats['discarded'] += 1
            return []
        else:
            templates = [resolved]

        # Process each template
        samples = []
        for template_text in templates:
            if not template_text or len(template_text.strip()) < 2:
                continue

            # Clean pattern
            clean_pattern = self._clean_pattern(raw_pattern)

            # Expand wildcards
            expanded = self._expand_wildcards(clean_pattern, template_text.strip())

            for pattern_exp, template_exp in expanded:
                # Normalize
                sample = self._normalize(pattern_exp, template_exp)

                # Classify quality
                quality = self._classify_quality(pattern_exp, template_exp)
                sample['quality'] = quality

                if quality == 'discardable':
                    self._stats['discarded'] += 1
                    continue

                self._stats['parsed_ok'] += 1
                samples.append(sample)

                # Log sample every 500 parsed
                if self._stats['parsed_ok'] % 500 == 0:
                    logger.info(
                        f"  [Sample #{self._stats['parsed_ok']}] "
                        f"{sample['input_ids'][:80]}..."
                    )

        return samples

    def _resolve_template(
        self,
        element: ElementTree.Element,
        all_categories: Dict[str, ElementTree.Element],
        depth: int
    ) -> Union[str, List[str], None]:
        """
        Recursively resolve AIML template elements.

        Returns: str (single response), List[str] (multiple from <random>), or None (discard)
        """
        if depth > 100:
            return None

        parts = []
        random_options = []

        # Process text before first child
        if element.text:
            parts.append(element.text)

        # Process children
        for child in element:
            tag = child.tag.lower()

            if tag == 'thinking':
                # AIML <thinking> - ELIMINAR completamente (no es training thinking)
                # Preservar tail text
                if child.tail:
                    parts.append(child.tail)
                continue

            elif tag == 'that' or tag == 'topic':
                # Runtime-only elements - ELIMINAR
                if child.tail:
                    parts.append(child.tail)
                continue

            elif tag == 'star':
                # Wildcard capture - replace with placeholder
                idx = child.get('index', '1')
                parts.append(f'[X{idx}]')
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'set':
                # Variable setting - extract value
                value = self._get_text_content(child)
                if value:
                    parts.append(value)
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'get':
                # Variable getting - placeholder
                name = child.get('name', 'unknown')
                parts.append(f'[{name.upper()}]')
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'bot':
                # Bot property - use default
                prop_name = child.get('name', 'name')
                value = self.bot_properties.get(prop_name, prop_name)
                parts.append(value)
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'srai':
                # Symbolic reduction - resolve chain
                target = self._get_text_content(child)
                if target:
                    resolved = self._resolve_srai(target.strip(), all_categories, depth)
                    if resolved is not None:
                        parts.append(resolved)
                        self._stats['srai_resolved'] += 1
                    else:
                        self._stats['srai_failed'] += 1
                        if child.tail:
                            parts.append(child.tail)

            elif tag == 'random':
                # Random responses - collect all <li> options
                for li in child.findall('li'):
                    li_text = self._resolve_template(li, all_categories, depth + 1)
                    if isinstance(li_text, str) and li_text.strip():
                        random_options.append(li_text.strip())
                    elif isinstance(li_text, list):
                        random_options.extend(li_text)

            elif tag == 'li':
                # List item (inside random or condition) - resolve content
                li_text = self._get_text_content(child)
                if li_text and li_text.strip():
                    parts.append(li_text.strip())
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'condition':
                # Conditional - take first valid <li>
                first_li = child.find('li')
                if first_li is not None:
                    li_text = self._get_text_content(first_li)
                    if li_text and li_text.strip():
                        parts.append(li_text.strip())
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'person':
                # Person conversion - basic I/you swap
                text = self._get_text_content(child)
                if text:
                    converted = self._convert_person(text)
                    parts.append(converted)
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'person2':
                # Person2 conversion - basic you/he swap
                text = self._get_text_content(child)
                if text:
                    converted = self._convert_person2(text)
                    parts.append(converted)
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'gender':
                # Gender conversion - basic he/she swap
                text = self._get_text_content(child)
                if text:
                    converted = self._convert_gender(text)
                    parts.append(converted)
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'input':
                # Conversation history - placeholder
                parts.append('[HISTORY]')
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'date':
                # Date - placeholder
                parts.append('[DATE]')
                if child.tail:
                    parts.append(child.tail)

            elif tag == 'eval' or tag == 'system':
                # System commands - placeholder
                parts.append('[SYSTEM]')
                if child.tail:
                    parts.append(child.tail)

            elif tag in ('br', 'br/'):
                # HTML line break
                parts.append(' ')

            elif tag in ('a', 'em', 'b', 'i', 'u', 'p', 'span', 'div', 'table', 'tr', 'td', 'img', 'video', 'audio', 'embed', 'svg'):
                # HTML tags - strip but keep content
                child_text = self._get_text_content(child)
                if child_text:
                    parts.append(child_text)
                if child.tail:
                    parts.append(child.tail)

            else:
                # Unknown element - try to extract text content
                child_text = self._get_text_content(child)
                if child_text:
                    parts.append(child_text)
                if child.tail:
                    parts.append(child.tail)

        # If we have random options, return them as list
        if random_options:
            self._stats['random_expanded'] += 1
            # Prepend/append any surrounding text (text before/after <random>)
            prefix = ''.join(parts).strip()
            if prefix:
                random_options = [f"{prefix} {opt}" for opt in random_options]
            return random_options

        # Join all parts
        result = ''.join(parts).strip()

        # Clean up HTML entities and extra whitespace
        result = html.unescape(result)
        result = re.sub(r'\s+', ' ', result).strip()

        # Remove any remaining AIML thinking tags
        result = AIML_THINKING_PATTERN.sub('', result).strip()

        # Remove HTML tags that slipped through
        result = HTML_TAG_PATTERN.sub('', result).strip()

        return result if result else None

    def _get_text_content(self, element: ElementTree.Element) -> str:
        """Get all text content from an element, including children's text."""
        parts = []
        if element.text:
            parts.append(element.text)
        for child in element:
            child_text = self._get_text_content(child)
            if child_text:
                parts.append(child_text)
            if child.tail:
                parts.append(child.tail)
        return ''.join(parts)

    def _resolve_srai(
        self,
        target_pattern: str,
        all_categories: Dict[str, ElementTree.Element],
        depth: int
    ) -> Optional[str]:
        """
        Resolve <srai> chain. Follows redirects up to MAX_SRAI_DEPTH.
        """
        if depth >= MAX_SRAI_DEPTH:
            return None

        target_key = target_pattern.upper().strip()

        # Check cache
        cache_key = f"{target_key}_{depth}"
        if cache_key in self._srai_cache:
            return self._srai_cache[cache_key]

        # Find target category
        target_category = all_categories.get(target_key)
        if target_category is None:
            self._srai_cache[cache_key] = None
            return None

        # Get target template
        template_elem = target_category.find('template')
        if template_elem is None:
            self._srai_cache[cache_key] = None
            return None

        # Resolve recursively
        result = self._resolve_template(template_elem, all_categories, depth + 1)

        if isinstance(result, list):
            # If random options, take first one for srai resolution
            result = result[0] if result else None

        self._srai_cache[cache_key] = result
        return result

    def _clean_pattern(self, pattern_text: str) -> str:
        """
        Clean pattern by replacing wildcards with placeholders.

        *  → [X]     (matches one or more words)
        _  → [PHRASE] (matches one or more words, higher priority)
        ** → [TEXT]   (matches zero or more words)
        ^  → [OPTIONAL] (matches zero or one word)
        """
        result = pattern_text.strip()

        # Replace ** first (before * to avoid partial match)
        result = result.replace('**', '[TEXT]')

        # Replace ^ (before * to avoid partial match)
        result = result.replace('^', '[OPTIONAL]')

        # Replace remaining * with [X]
        result = result.replace('*', '[X]')

        # Replace _ with [PHRASE]
        result = result.replace('_', '[PHRASE]')

        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _expand_wildcards(self, pattern: str, template: str) -> List[Tuple[str, str]]:
        """
        Generate multiple (pattern, template) pairs by expanding wildcards.

        Returns: List of (expanded_pattern, template) tuples
        """
        # Detect category
        category = self._detect_category(pattern)

        # Get wildcard placeholders in pattern
        has_x = '[X]' in pattern
        has_phrase = '[PHRASE]' in pattern
        has_text = '[TEXT]' in pattern
        has_optional = '[OPTIONAL]' in pattern

        # If no wildcards, return as-is
        if not (has_x or has_phrase or has_text or has_optional):
            return [(pattern, template)]

        # Get examples for this category
        examples = WILDCARD_EXAMPLES.get(category, WILDCARD_EXAMPLES['generic'])

        # Build expansion lists with generic fallbacks
        x_examples = examples.get('*', WILDCARD_EXAMPLES['generic']['*'])
        phrase_examples = examples.get('_', WILDCARD_EXAMPLES['generic']['_'])
        text_examples = examples.get('**', WILDCARD_EXAMPLES['generic']['**'])
        optional_examples = examples.get('^', WILDCARD_EXAMPLES['generic']['^'])

        # Generate combinations
        results = []

        # For patterns with multiple wildcards, limit combinations
        if has_x and has_phrase:
            # 2 wildcards: 2x2 = 4 samples max
            for x_val in x_examples[:2]:
                for phrase_val in phrase_examples[:2]:
                    expanded = pattern.replace('[X]', x_val, 1)
                    expanded = expanded.replace('[PHRASE]', phrase_val, 1)
                    # Replace remaining wildcards with first example
                    expanded = expanded.replace('[X]', x_examples[0])
                    expanded = expanded.replace('[PHRASE]', phrase_examples[0])
                    expanded = expanded.replace('[TEXT]', text_examples[0])
                    expanded = expanded.replace('[OPTIONAL]', optional_examples[0])
                    results.append((expanded, template))
        elif has_x:
            # Single wildcard: 3 samples
            for val in x_examples[:3]:
                expanded = pattern.replace('[X]', val)
                expanded = expanded.replace('[PHRASE]', phrase_examples[0])
                expanded = expanded.replace('[TEXT]', text_examples[0])
                expanded = expanded.replace('[OPTIONAL]', optional_examples[0])
                results.append((expanded, template))
        elif has_phrase:
            for val in phrase_examples[:3]:
                expanded = pattern.replace('[PHRASE]', val)
                expanded = expanded.replace('[X]', x_examples[0])
                expanded = expanded.replace('[TEXT]', text_examples[0])
                expanded = expanded.replace('[OPTIONAL]', optional_examples[0])
                results.append((expanded, template))
        elif has_text:
            for val in text_examples[:2]:
                expanded = pattern.replace('[TEXT]', val)
                expanded = expanded.replace('[X]', x_examples[0])
                expanded = expanded.replace('[PHRASE]', phrase_examples[0])
                expanded = expanded.replace('[OPTIONAL]', optional_examples[0])
                results.append((expanded, template))
        elif has_optional:
            # With optional: 2 samples (with and without)
            expanded_with = pattern.replace('[OPTIONAL]', optional_examples[0])
            expanded_with = expanded_with.replace('[X]', x_examples[0])
            expanded_with = expanded_with.replace('[PHRASE]', phrase_examples[0])
            expanded_with = expanded_with.replace('[TEXT]', text_examples[0])
            results.append((expanded_with, template))
            # Without optional (remove the placeholder)
            expanded_without = pattern.replace('[OPTIONAL]', '')
            expanded_without = re.sub(r'\s+', ' ', expanded_without).strip()
            expanded_without = expanded_without.replace('[X]', x_examples[0])
            expanded_without = expanded_without.replace('[PHRASE]', phrase_examples[0])
            expanded_without = expanded_without.replace('[TEXT]', text_examples[0])
            results.append((expanded_without, template))

        if results:
            self._stats['wildcards_expanded'] += len(results) - 1

        # Apply final safety pass to ensure no wildcards remain
        final_results = []
        for expanded_pattern, expanded_template in results:
            cleaned_pattern = self._ensure_no_wildcards(expanded_pattern)
            final_results.append((cleaned_pattern, expanded_template))

        return final_results if final_results else [(pattern, template)]

    def _ensure_no_wildcards(self, pattern: str) -> str:
        """
        Final safety pass: replace any remaining wildcards with generic fallbacks.
        This ensures no literal wildcards remain in the output.
        """
        # Generic fallbacks for any remaining placeholders
        fallbacks = {
            '[X]': 'SOMETHING',
            '[PHRASE]': 'SOMETHING',
            '[TEXT]': 'SOMETHING',
            '[OPTIONAL]': 'SOMETHING',
        }

        result = pattern
        for wildcard, fallback in fallbacks.items():
            result = result.replace(wildcard, fallback)

        # Handle literal wildcards that weren't converted
        # Replace * with SOMETHING (but not ** which is already handled)
        result = re.sub(r'(?<!\*)\*(?!\*)', 'SOMETHING', result)
        # Replace ** with SOMETHING
        result = result.replace('**', 'SOMETHING')
        # Replace _ with SOMETHING
        result = result.replace('_', 'SOMETHING')
        # Replace ^ with SOMETHING
        result = result.replace('^', 'SOMETHING')

        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _detect_category(self, text: str) -> str:
        """Detect the category of a pattern by keyword matching."""
        text_upper = text.upper()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_upper:
                    return category
        return 'generic'

    def _normalize(self, pattern: str, template: str) -> dict:
        """Normalize sample to standard format."""
        # Ensure no wildcards remain in final output
        clean_pattern = self._ensure_no_wildcards(pattern)
        clean_template = self._ensure_no_wildcards(template)

        # GPT-2 standard chat format (Formato 1): <|user|>...<|end|><|assistant|>...<|end|>
        input_ids = f"<|user|>{clean_pattern}<|end|><|assistant|>{clean_template}<|end|>"

        # Check if wildcards were resolved
        wildcards_resolved = (
            '[X]' not in clean_pattern and
            '[PHRASE]' not in clean_pattern and
            '[TEXT]' not in clean_pattern and
            '[OPTIONAL]' not in clean_pattern and
            '*' not in clean_pattern and
            '_' not in clean_pattern and
            '^' not in clean_pattern
        )

        return {
            'input_ids': input_ids,
            'source': 'AIML',
            'original_pattern': clean_pattern,
            'original_template': clean_template,
            'wildcards_resolved': wildcards_resolved,
            'srai_resolved': False,
            'random_expanded': False,
        }

    def _classify_quality(self, pattern: str, template: str) -> str:
        """
        Classify sample quality: good / fixable / discardable.
        """
        # Discard: pattern is only wildcards
        pattern_clean = re.sub(r'\[.*?\]', '', pattern).strip()
        if len(pattern_clean) < 2:
            return 'discardable'

        # Discard: template too short
        if len(template.strip()) < 3:
            return 'discardable'

        # Discard: template is only a placeholder
        if template.strip() in ('[SYSTEM]', '[DATE]', '[HISTORY]', ''):
            return 'discardable'

        # Fix: HTML tags still present
        if HTML_TAG_PATTERN.search(template):
            return 'fixable'

        # Fix: HTML entities
        if HTML_ENTITY_PATTERN.search(template):
            return 'fixable'

        # Fix: wildcards still present in pattern
        if '[X]' in pattern or '[PHRASE]' in pattern or '[TEXT]' in pattern or '[OPTIONAL]' in pattern:
            return 'fixable'

        # Fix: literal wildcards still present
        if '*' in pattern or '_' in pattern or '^' in pattern:
            return 'fixable'

        # Good
        return 'good'

    def _convert_person(self, text: str) -> str:
        """Basic first/second person conversion."""
        # Simple word-level replacement
        words = text.split()
        result = []
        for word in words:
            lower = word.lower()
            if lower == 'i':
                result.append('You')
            elif lower == 'my':
                result.append('Your')
            elif lower == 'me':
                result.append('you')
            elif lower == 'mine':
                result.append('yours')
            elif lower == 'am':
                result.append('are')
            elif lower == "i'm":
                result.append("you're")
            elif lower == "i've":
                result.append("you've")
            elif lower == "i'll":
                result.append("you'll")
            elif lower == "i'd":
                result.append("you'd")
            elif lower == 'myself':
                result.append('yourself')
            else:
                result.append(word)
        return ' '.join(result)

    def _convert_person2(self, text: str) -> str:
        """Basic second/third person conversion."""
        words = text.split()
        result = []
        for word in words:
            lower = word.lower()
            if lower == 'you':
                result.append('he')
            elif lower == 'your':
                result.append('his')
            elif lower == 'yours':
                result.append('his')
            elif lower == 'yourself':
                result.append('himself')
            else:
                result.append(word)
        return ' '.join(result)

    def _convert_gender(self, text: str) -> str:
        """Basic gender pronoun swap."""
        replacements = {
            'he': 'she', 'she': 'he',
            'him': 'her', 'her': 'him',
            'his': 'her', 'hers': 'his',
            'himself': 'herself', 'herself': 'himself',
        }
        words = text.split()
        result = []
        for word in words:
            lower = word.lower()
            if lower in replacements:
                result.append(replacements[lower])
            else:
                result.append(word)
        return ' '.join(result)


def parse_aiml_file(
    file_path: str,
    bot_properties: Dict[str, str] = None
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Parse a single AIML file into training samples.

    Args:
        file_path: Path to .aiml file
        bot_properties: Optional bot property overrides

    Returns:
        Tuple of (samples_list, stats_dict)
    """
    parser = AIMLParser(bot_properties=bot_properties)

    try:
        tree = ElementTree.parse(file_path)
        root = tree.getroot()
    except ElementTree.ParseError as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return [], {}

    # Index categories for srai resolution
    parser.index_categories(root)
    logger.info(f"  Parsing {file_path} ({len(parser._category_index)} categories)...")

    # Parse all categories
    all_samples = []
    for category in root.findall('category'):
        samples = parser.parse_category(category)
        all_samples.extend(samples)

    return all_samples, parser.stats


def parse_aiml_directory(
    aiml_dir: str,
    bot_properties: Dict[str, str] = None
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Parse all AIML files in a directory.

    Args:
        aiml_dir: Directory containing .aiml files
        bot_properties: Optional bot property overrides

    Returns:
        Tuple of (samples_list, combined_stats_dict)
    """
    import os

    all_samples = []
    combined_stats = {
        'total_categories': 0,
        'parsed_ok': 0,
        'srai_resolved': 0,
        'srai_failed': 0,
        'random_expanded': 0,
        'wildcards_expanded': 0,
        'discarded': 0,
        'files_processed': 0,
    }

    if not os.path.exists(aiml_dir):
        logger.warning(f"AIML directory not found: {aiml_dir}")
        return all_samples, combined_stats

    for filename in os.listdir(aiml_dir):
        if filename.lower().endswith('.aiml'):
            file_path = os.path.join(aiml_dir, filename)
            samples, stats = parse_aiml_file(file_path, bot_properties)
            for s in samples:
                s['source_file'] = Path(filename).stem
            all_samples.extend(samples)

            # Merge stats
            for key in combined_stats:
                if key in stats:
                    combined_stats[key] += stats[key]
            combined_stats['files_processed'] += 1

            logger.info(f"  Parsed {filename}: {len(samples)} samples")

    return all_samples, combined_stats
