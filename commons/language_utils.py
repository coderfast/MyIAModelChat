"""
Multilingual language detection and utilities.

Supports all EU official languages (24) + Eastern European languages.
Reusable across data preparation, thinking engine, and dataset modules.
"""
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION INDICATORS
# EU official languages (24) + Eastern European languages (8)
# ═══════════════════════════════════════════════════════════════════════════════

LANGUAGE_INDICATORS = {
    # ── Romance languages ──────────────────────────────────────────────────────
    'es': {
        'name': 'Spanish',
        'family': 'romance',
        'indicators': [' el ', ' la ', ' los ', ' las ', ' de ', ' del ',
                       ' en ', ' un ', ' una ', ' que ', ' es ', ' son ',
                       'porque', 'entonces', 'además', 'sin embargo'],
    },
    'fr': {
        'name': 'French',
        'family': 'romance',
        'indicators': [' les ', ' des ', ' du ', ' une ', ' est ', ' sont ',
                       'parce que', 'donc', 'cependant', 'en outre',
                       'aussi', 'maintenant', 'après', 'peut',
                       'avant', 'après', 'tout', 'très', 'bien', 'mal',
                       'oui', 'non', 'merci', 'monsieur', 'madame',
                       'famille', 'travail', 'vie', 'personne', 'année',
                       'jour', 'fois', 'partie', 'lieu', 'forme', 'cas',
                       'grand', 'petit', 'nouveau', 'premier', 'toujours',
                       'jamais', 'entre', 'sur', 'chaque', 'autre', 'même',
                       'quelque', 'peut', 'doit', 'aller', 'venir',
                       'savoir', 'vouloir', 'pouvoir', 'devoir', 'mettre',
                       'sembler', 'rester', 'croire', 'avoir', 'arriver',
                       'passer', 'suivre', 'trouver', 'appeler', 'revenir',
                       'prendre', 'connaître', 'vive', 'sentir', 'essayer',
                       'regarder', 'compter', 'commencer', 'attendre',
                       'sortir', 'tomber', 'entrer', 'demander', 'perdre',
                       'servir', 'chercher', 'exister', 'ouvrir', 'fermer',
                       'mourir', 'devenir', 'découvrir', 'défendre', 'offrir',
                       'rappeler', 'accepter', 'partager', 'obtenir',
                       'considérer', 'construire', 'croître', 'entendre',
                       'établir', 'former', 'fonctionner', 'garder',
                       'imaginer', 'inclure', 'indiquer', 'maintenir',
                       'nécessiter', 'payer', 'permettre', 'produire',
                       'recevoir', 'reconnaître', 'résulter', 'suggérer',
                       'supposer', 'terminer', 'traduire', 'utiliser',
                       'expliquer', 'répondre', 'développer', 'présenter',
                       'réaliser', 'important', 'possible', 'nécessaire',
                       'différent', 'suivant', 'précédent', 'principal',
                       'général', 'spécial', 'actuel', 'dernier', 'prochain',
                       'problème', 'solution', 'raison', 'exemple', 'type',
                       'nombre', 'moment', 'manière', 'changement', 'système',
                       'processus', 'résultat', 'information', 'niveau',
                       'qualité', 'quantité', 'valeur', 'groupe', 'équipe',
                       'idée', 'projet', 'programme', 'service', 'produit',
                       'marché', 'pays', 'ville', 'monde', 'gouvernement',
                       'société', 'culture', 'histoire', 'art', 'science',
                       'nature', 'terre', 'eau', 'feu', 'air', 'soleil',
                       'lune', 'étoile', 'mer', 'montagne', 'rivière',
                       'forêt', 'animal', 'plante', 'arbre', 'fleur',
                       'chien', 'chat', 'oiseau', 'poisson', 'maison',
                       'porte', 'fenêtre', 'table', 'chaise', 'lit',
                       'voiture', 'livre', 'papier', 'nourriture', 'pain',
                       'viande', 'lait', 'vin', 'bière', 'café', 'thé',
                       'vêtements', 'chaussures', 'argent', 'travail',
                       'école', 'hôpital', 'magasin', 'parc', 'jardin',
                       'route', 'rue', 'pont', 'bâtiment'],
    },
    'it': {
        'name': 'Italian',
        'family': 'romance',
        'indicators': [' il ', ' la ', ' i ', ' le ', ' di ', ' del ',
                       ' in ', ' un ', ' una ', ' che ', ' è ', ' sono ',
                       'perché', 'quindi', 'tuttavia', 'inoltre'],
    },
    'pt': {
        'name': 'Portuguese',
        'family': 'romance',
        'indicators': [' o ', ' a ', ' os ', ' as ', ' de ', ' do ',
                       ' em ', ' um ', ' uma ', ' que ', ' é ', ' são ',
                       'porque', 'então', 'além disso', 'no entanto'],
    },
    'ro': {
        'name': 'Romanian',
        'family': 'romance',
        'indicators': [' el ', ' ea ', ' ei ', ' ele ', ' de ', ' din ',
                       ' în ', ' un ', ' o ', ' care ', ' este ', ' sunt ',
                       'pentru că', 'deci', 'totuși', 'în plus'],
    },
    'ca': {
        'name': 'Catalan',
        'family': 'romance',
        'indicators': [' el ', ' la ', ' els ', ' les ', ' de ', ' del ',
                       ' en ', ' un ', ' una ', ' que ', ' és ', ' són ',
                       'perquè', 'llavors', 'tanmateix', 'a més'],
    },
    'gl': {
        'name': 'Galician',
        'family': 'romance',
        'indicators': [' o ', ' a ', ' os ', ' as ', ' de ', ' do ',
                       ' en ', ' un ', ' unha ', ' que ', ' é ', ' son ',
                       'porque', 'entón', 'adxemais', 'non obstante'],
    },
    'fr-CH': {
        'name': 'French (Switzerland)',
        'family': 'romance',
        'indicators': [' le ', ' la ', ' les ', ' des ', ' du ', ' de ',
                       ' en ', ' un ', ' une ', ' que ', ' est ', ' sont ',
                       'parce que', 'donc', 'cependant', 'en outre'],
    },
    'rm': {
        'name': 'Romansh',
        'family': 'romance',
        'indicators': [' il ', ' la ', ' ils ', ' las ', ' da ', ' dal ',
                       ' en ', ' in ', ' ul ', ' ina ', ' che ', ' è ', ' sun ',
                       'perchè', 'dunc', 'però', 'enura'],
    },

    # ── Germanic languages ─────────────────────────────────────────────────────
    'en': {
        'name': 'English',
        'family': 'germanic',
        'indicators': [' the ', ' is ', ' are ', ' was ', ' were ', ' have ',
                       ' has ', ' had ', ' that ', ' this ', ' with ',
                       'because', 'therefore', 'however', 'moreover',
                       'also', 'now', 'before', 'after', 'very', 'well',
                       'just', 'only', 'even', 'still', 'already', 'yet',
                       'about', 'been', 'being', 'would', 'could', 'should',
                       'might', 'must', 'shall', 'will ', 'can ',
                       'does', 'did', 'not ', 'but ', 'and ', 'or ',
                       'for ', 'from', 'into', 'more', 'than', 'then',
                       'when', 'what', 'how ', 'why ', 'who ',
                       'where', 'which', 'while', 'their', 'there',
                       'they ', 'them', 'these', 'those', 'some',
                       'each', 'every', 'both', 'other', 'such',
                       'make', 'like', 'long', 'look', 'many', 'much',
                       'take', 'come', 'over', 'think', 'back',
                       'use', 'work', 'first', 'way', 'new', 'want',
                       'give', 'day', 'most', 'find', 'here', 'thing',
                       'tell', 'one', 'two', 'three', 'four', 'five',
                       'people', 'time', 'year', 'know', 'see',
                       'want', 'give', 'use', 'find', 'tell', 'ask',
                       'work', 'seem', 'feel', 'try', 'leave', 'call',
                       'need', 'become', 'keep', 'let', 'begin', 'show',
                       'hear', 'play', 'run', 'move', 'live', 'believe',
                       'hold', 'bring', 'happen', 'write', 'provide',
                       'sit', 'stand', 'lose', 'pay', 'meet', 'include',
                       'continue', 'set', 'learn', 'change', 'lead',
                       'understand', 'watch', 'follow', 'stop', 'create',
                       'speak', 'read', 'allow', 'add', 'spend', 'grow',
                       'open', 'walk', 'win', 'offer', 'remember', 'love',
                       'consider', 'appear', 'buy', 'wait', 'serve',
                       'die', 'send', 'expect', 'build', 'stay', 'fall',
                       'cut', 'reach', 'kill', 'remain', 'suggest',
                       'raise', 'pass', 'sell', 'require', 'report',
                       'decide', 'pull', 'develop', 'agree', 'support',
                       'produce', 'eat', 'apply', 'present', 'available',
                       'likely', 'previous', 'medical', 'insurance',
                       'company', 'water', 'room', 'mother', 'area',
                       'money', 'story', 'fact', 'month', 'lot', 'right',
                       'study', 'book', 'eye', 'job', 'word', 'business',
                       'issue', 'side', 'kind', 'head', 'house', 'service',
                       'friend', 'father', 'power', 'hour', 'game', 'line',
                       'end', 'member', 'law', 'car', 'community', 'name',
                       'president', 'team', 'minute', 'idea', 'body',
                       'information', 'back', 'parent', 'face', 'others',
                       'level', 'office', 'door', 'health', 'person',
                       'art', 'war', 'history', 'party', 'result',
                       'change', 'morning', 'reason', 'research', 'girl',
                       'guy', 'moment', 'air', 'teacher', 'force',
                       'education'],
    },
    'de': {
        'name': 'German',
        'family': 'germanic',
        'indicators': [' der ', ' die ', ' das ', ' den ', ' dem ', ' des ',
                       ' ein ', ' eine ', ' und ', ' ist ', ' sind ',
                       'weil', 'daher', 'jedoch', 'außerdem'],
    },
    'nl': {
        'name': 'Dutch',
        'family': 'germanic',
        'indicators': [' de ', ' het ', ' een ', ' van ', ' in ', ' op ',
                       ' en ', ' is ', ' zijn ', ' dat ',
                       'omdat', 'daarom', 'echter', 'bovendien'],
    },
    'sv': {
        'name': 'Swedish',
        'family': 'germanic',
        'indicators': [' den ', ' det ', ' ett ', ' en ', ' av ', ' i ',
                       ' och ', ' är ', ' som ', ' har ',
                       'eftersom', 'därför', 'men', 'dessutom'],
    },
    'da': {
        'name': 'Danish',
        'family': 'germanic',
        'indicators': [' den ', ' det ', ' en ', ' et ', ' af ', ' i ',
                       ' og ', ' er ', ' som ', ' har ',
                       'fordi', 'derfor', 'men', 'desuden'],
    },
    'nb': {
        'name': 'Norwegian Bokmål',
        'family': 'germanic',
        'indicators': [' den ', ' det ', ' en ', ' et ', ' av ', ' i ',
                       ' og ', ' er ', ' som ', ' har ',
                       'fordi', 'derfor', 'men', 'dessuten'],
    },
    'nn': {
        'name': 'Norwegian Nynorsk',
        'family': 'germanic',
        'indicators': [' den ', ' det ', ' eit ', ' ein ', ' av ', ' i ',
                       ' og ', ' er ', ' som ', ' har ',
                       'fordi', 'derfor', 'men', 'dessutan'],
    },
    'is': {
        'name': 'Icelandic',
        'family': 'germanic',
        'indicators': [' hinn ', ' hið ', ' það ', ' af ', ' í ',
                       ' og ', ' er ', ' sem ', ' hefur ',
                       'vegna', 'því', 'en', 'einnaig'],
    },
    'lb': {
        'name': 'Luxembourgish',
        'family': 'germanic',
        'indicators': [' de ', ' d\'', ' eng ', ' vun ', ' an ',
                       ' an ', ' ass ', ' sinn ', ' déi ',
                       'well', 'dofir', 'mä', 'zousätzlech'],
    },
    'fo': {
        'name': 'Faroese',
        'family': 'germanic',
        'indicators': [' tað ', ' ein ', ' eitt ', ' av ', ' í ',
                       ' og ', ' er ', ' sum ', ' hevur ',
                       'tí', 'tískil', 'men', 'haraf'],
    },

    # ── Slavic languages ───────────────────────────────────────────────────────
    'pl': {
        'name': 'Polish',
        'family': 'slavic',
        'indicators': [' ten ', ' ta ', ' to ', ' te ', ' tym ', ' tej ',
                       ' i ', ' jest ', ' są ', ' nie ',
                       'ponieważ', 'dlatego', 'jednak', 'ponadto'],
    },
    'cs': {
        'name': 'Czech',
        'family': 'slavic',
        'indicators': [' ten ', ' ta ', ' to ', ' ty ', ' tym ', ' té ',
                       ' a ', ' je ', ' jsou ', ' ne ',
                       'protože', 'proto', 'ale', 'kromě toho'],
    },
    'sk': {
        'name': 'Slovak',
        'family': 'slavic',
        'indicators': [' ten ', ' tá ', ' to ', ' tie ', ' tým ', ' tej ',
                       ' a ', ' je ', ' sú ', ' nie ',
                       'pretože', 'preto', 'ale', 'okrem toho'],
    },
    'bg': {
        'name': 'Bulgarian',
        'family': 'slavic',
        'indicators': [' той ', ' тя ', ' то ', ' те ', ' на ', ' в ',
                       ' и ', ' е ', ' са ', ' не ',
                       'защото', 'затова', 'но', 'освен това'],
    },
    'hr': {
        'name': 'Croatian',
        'family': 'slavic',
        'indicators': [' taj ', ' ta ', ' to ', ' ti ', ' tim ', ' te ',
                       ' i ', ' je ', ' su ', ' ne ',
                       'jer', 'stoga', 'međutim', 'štoviše'],
    },
    'sr': {
        'name': 'Serbian',
        'family': 'slavic',
        'indicators': [' тај ', ' та ', ' то ', ' ти ', ' тим ', ' те ',
                       ' и ', ' је ', ' су ', ' не ',
                       'јер', 'стога', 'али', 'осим тога'],
    },
    'sl': {
        'name': 'Slovenian',
        'family': 'slavic',
        'indicators': [' ta ', ' to ', ' ti ', ' te ', ' tega ', ' tej ',
                       ' in ', ' je ', ' so ', ' ne ',
                       'ker', 'zato', 'vendar', 'poleg tega'],
    },
    'bs': {
        'name': 'Bosnian',
        'family': 'slavic',
        'indicators': [' taj ', ' ta ', ' to ', ' ti ', ' tim ', ' te ',
                       ' i ', ' je ', ' su ', ' ne ',
                       'jer', 'stoga', 'međutim', 'štoviše'],
    },
    'mk': {
        'name': 'Macedonian',
        'family': 'slavic',
        'indicators': [' тој ', ' таа ', ' тоа ', ' тие ', ' на ', ' во ',
                       ' и ', ' е ', ' се ', ' не ',
                       'затоа што', 'затоа', 'но', 'покрај тоа'],
    },
    'uk': {
        'name': 'Ukrainian',
        'family': 'slavic',
        'indicators': [' він ', ' вона ', ' воно ', ' вони ', ' на ', ' в ',
                       ' і ', ' є ', ' суть ', ' не ',
                       'тому що', 'тому', 'але', 'крім того'],
    },
    'be': {
        'name': 'Belarusian',
        'family': 'slavic',
        'indicators': [' ён ', ' яна ', ' яно ', ' яны ', ' на ', ' у ',
                       ' і ', ' ёсць ', ' суть ', ' не ',
                       'таму што', 'таму', 'але', 'акрамя таго'],
    },

    # ── Baltic languages ───────────────────────────────────────────────────────
    'lt': {
        'name': 'Lithuanian',
        'family': 'baltic',
        'indicators': [' tas ', ' ta ', ' to ', ' tie ', ' tą ', ' tai ',
                       ' ir ', ' yra ', ' ne ',
                       'nes', 'todėl', 'tačiau', 'be to'],
    },
    'lv': {
        'name': 'Latvian',
        'family': 'baltic',
        'indicators': [' tas ', ' ta ', ' tie ', ' tās ', ' šis ', ' šī ',
                       ' un ', ' ir ', ' ne ',
                       'jo', 'tāpēc', 'tomēr', 'turklāt'],
    },

    # ── Finno-Ugric languages ──────────────────────────────────────────────────
    'fi': {
        'name': 'Finnish',
        'family': 'finno-ugric',
        'indicators': [' se ', ' tämä ', ' tuo ', ' ne ', ' sen ', ' tän ',
                       ' ja ', ' on ', ' ovat ', ' ei ',
                       'koska', 'siksi', 'mutta', 'lisäksi'],
    },
    'et': {
        'name': 'Estonian',
        'family': 'finno-ugric',
        'indicators': [' see ', ' too ', ' need ', ' selle ',
                       ' ja ', ' on ', ' ei ',
                       'sest', 'seetõttu', 'aga', 'lisaks'],
    },
    'hu': {
        'name': 'Hungarian',
        'family': 'finno-ugric',
        'indicators': [' az ', ' egy ', ' és ', ' van ', ' vannak ', ' nem ',
                       'ezt', 'azt', 'ennek', 'annak'],
    },

    # ── Celtic languages ───────────────────────────────────────────────────────
    'ga': {
        'name': 'Irish',
        'family': 'celtic',
        'indicators': [' an ', ' na ', ' agus ', ' é ', ' í ', ' sé ',
                       'sin', 'tá', 'bhí', 'ach'],
    },

    # ── Hellenic ───────────────────────────────────────────────────────────────
    'el': {
        'name': 'Greek',
        'family': 'hellenic',
        'indicators': [' ο ', ' η ', ' το ', ' τα ', ' τον ', ' την ',
                       ' και ', ' είναι ', ' είναι ', ' δεν ',
                       'γιατί', 'επομένως', 'αλλά', 'επίσης'],
    },

    # ── Albanian ───────────────────────────────────────────────────────────────
    'sq': {
        'name': 'Albanian',
        'family': 'indo-european',
        'indicators': [' ai ', ' ajo ', ' ata ', ' ato ', ' të ', ' në ',
                       ' dhe ', ' është ', ' janë ', ' nuk ',
                       'për shkak', 'prandaj', 'por', 'gjithashtu'],
    },
}


def detect_language(text: str) -> str:
    """
    Detect language using weighted indicator scoring.

    Longer indicators and unique words get more weight.
    Short single-letter indicators are excluded to avoid false positives.

    Returns: Language code or 'unknown'
    """
    if not text or not isinstance(text, str):
        return 'unknown'

    text_lower = text.lower()
    best_lang = 'en'
    best_score = 0

    for lang_code, config in LANGUAGE_INDICATORS.items():
        score = 0
        for ind in config['indicators']:
            if ind in text_lower:
                # Weight: length^1.5 gives 3+ chars much more importance
                score += len(ind) ** 1.5
        if score > best_score:
            best_score = score
            best_lang = lang_code

    return best_lang if best_score > 0 else 'unknown'


def get_language_name(lang_code: str) -> str:
    """
    Get human-readable language name from code.

    Args:
        lang_code: ISO 639-1 language code

    Returns:
        Language name or 'Unknown' if not found
    """
    config = LANGUAGE_INDICATORS.get(lang_code)
    return config['name'] if config else 'Unknown'


def get_language_family(lang_code: str) -> str:
    """
    Get language family from code.

    Args:
        lang_code: ISO 639-1 language code

    Returns:
        Language family or 'unknown' if not found
    """
    config = LANGUAGE_INDICATORS.get(lang_code)
    return config['family'] if config else 'unknown'


def get_supported_languages() -> list[str]:
    """
    Get list of all supported language codes.

    Returns:
        List of ISO 639-1 language codes
    """
    return list(LANGUAGE_INDICATORS.keys())


def get_supported_languages_display() -> str:
    """
    Get formatted string of all supported languages for display.

    Returns:
        Formatted string with language codes and names
    """
    lines = []
    for code, config in sorted(LANGUAGE_INDICATORS.items()):
        lines.append(f"  {code:5s} - {config['name']} ({config['family']})")
    return "\n".join(lines)
