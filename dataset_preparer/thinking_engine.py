"""
Real Thinking Engine based on NLP analysis.
Generates chain-of-thought reasoning by analyzing actual text content,
without depending on external LLM APIs like Ollama.
Supports all EU official languages + Eastern European languages.
"""
import re
import hashlib
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
# This config extends with thinking-engine-specific fields (connectors, analyze, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

# Build LANGUAGE_CONFIG by extending LANGUAGE_INDICATORS with thinking-specific fields
LANGUAGE_CONFIG = dict(LANGUAGE_INDICATORS)

# Add thinking-engine-specific fields for languages that need them
LANGUAGE_CONFIG['es'] = {
    **LANGUAGE_INDICATORS['es'],
    'connectors': [
        "En primer lugar, ", "Además, ", "Por otro lado, ",
        "Asimismo, ", "Finalmente, ", "En conclusión, ",
    ],
    'overflow_connector': "También, ",
    'analyze': "Analizando",
    'about': "sobre",
    'content_of': "el contenido del texto",
    'text_contains': "El texto contiene {s} oraciones con {w} palabras en total",
    'type_desc': {
        'qa': "Se trata de una pregunta que requiere una respuesta específica",
        'technical': "El contenido es técnico y contiene datos numéricos o terminología especializada",
        'narrative': "El texto presenta una narrativa o descripción de eventos",
        'instructional': "El contenido proporciona instrucciones o directrices a seguir",
        'factual': "Se presenta información factual y objetiva sobre el tema",
        'conversational': "El texto tiene un tono conversacional y coloquial",
    },
    'type_default': "El contenido presenta información relevante",
    'concepts_prefix': "Los conceptos principales identificados son:",
    'entities_prefix': "Se mencionan las siguientes entidades:",
    'answer_terms': "Términos clave que conectan con la respuesta:",
    'technical_note': "El contenido incluye terminología técnica especializada",
    'question_types': {
        'qué': "La consulta solicita información específica sobre el tema",
        'que': "La consulta solicita información específica sobre el tema",
        'cómo': "La consulta solicita un procedimiento o explicación",
        'como': "La consulta solicita un procedimiento o explicación",
        'por qué': "La consulta busca una razón o explicación causal",
        'porque': "La consulta busca una razón o explicación causal",
        'dónde': "La consulta solicita información sobre una ubicación",
        'donde': "La consulta solicita información sobre una ubicación",
        'cuándo': "La consulta solicita información temporal",
        'cuando': "La consulta solicita información temporal",
        'cuánto': "La consulta solicita información cuantitativa",
        'cuanto': "La consulta solicita información cuantitativa",
    },
    'question_default': "La consulta solicita información específica sobre el tema",
    'meta_patterns': [
        re.compile(r'^(el usuario|the user|el humano|the human)\s*(me\s+)?(saluda|despide|pregunta|pide)', re.IGNORECASE),
        re.compile(r'^(saludo|greeting|despedida|farewell)$', re.IGNORECASE),
        re.compile(r'^(respondo|i respond|contestando|answering)\s*(con|with)', re.IGNORECASE),
        re.compile(r'^(el bot|the bot|asistente|assistant)\s*(responde|answer)', re.IGNORECASE),
        re.compile(r'(el usuario|the user)\s*(escribe|envía|menciona|indica)', re.IGNORECASE),
        re.compile(r'(debo|i should|debería)\s*(responder|contestar|decir|reply|answer)', re.IGNORECASE),
        re.compile(r'(la intención|the intention)\s*(detectada|es|detect)', re.IGNORECASE),
        re.compile(r'(la respuesta|the response)\s*(debe|should|apropiada|correcta)\s*(ser|be|sería)', re.IGNORECASE),
    ],
}

LANGUAGE_CONFIG['fr'] = {
    **LANGUAGE_INDICATORS['fr'],
    'connectors': [
        "Premièrement, ", "De plus, ", "D'autre part, ",
        "Également, ", "Enfin, ", "En conclusion, ",
    ],
    'overflow_connector': "En outre, ",
    'analyze': "Analyse",
    'about': "de",
    'content_of': "le contenu du texte",
    'text_contains': "Le texte contient {s} phrases avec {w} mots au total",
    'type_desc': {
        'qa': "Il s'agit d'une question qui nécessite une réponse spécifique",
        'technical': "Le contenu est technique et contient des données numériques ou une terminologie spécialisée",
        'narrative': "Le texte présente un récit ou une description d'événements",
        'instructional': "Le contenu fournit des instructions ou des directives à suivre",
        'factual': "Des informations factuelles et objectives sont présentées sur le sujet",
        'conversational': "Le texte a un ton conversationnel et familier",
    },
    'type_default': "Le contenu présente des informations pertinentes",
    'concepts_prefix': "Les concepts clés identifiés sont :",
    'entities_prefix': "Les entités mentionnées sont :",
    'answer_terms': "Termes clés liés à la réponse :",
    'technical_note': "Le contenu inclut une terminologie technique spécialisée",
    'question_types': {
        'quoi': "La requête demande des informations spécifiques sur le sujet",
        'comment': "La requête demande une procédure ou une explication",
        'pourquoi': "La requête cherche une raison ou une explication causale",
        'où': "La requête demande des informations sur un emplacement",
        'quand': "La requête demande des informations temporelles",
        'combien': "La requête demande des informations quantitatives",
    },
    'question_default': "La requête demande des informations spécifiques sur le sujet",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['it'] = {
    **LANGUAGE_INDICATORS['it'],
    'connectors': [
        "In primo luogo, ", "Inoltre, ", "D'altra parte, ",
        "Allo stesso modo, ", "Infine, ", "In conclusione, ",
    ],
    'overflow_connector': "Inoltre, ",
    'analyze': "Analisi",
    'about': "di",
    'content_of': "il contenuto del testo",
    'text_contains': "Il testo contiene {s} frasi con {w} parole in totale",
    'type_desc': {
        'qa': "Si tratta di una domanda che richiede una risposta specifica",
        'technical': "Il contenuto è tecnico e contiene dati numerici o terminologia specializzata",
        'narrative': "Il testo presenta una narrazione o descrizione di eventi",
        'instructional': "Il contenuto fornisce istruzioni o linee guida da seguire",
        'factual': "Vengono presentate informazioni fattuali e oggettive sull'argomento",
        'conversational': "Il testo ha un tono conversazionale e colloquiale",
    },
    'type_default': "Il contenuto presenta informazioni rilevanti",
    'concepts_prefix': "I concetti chiave identificati sono:",
    'entities_prefix': "Le entità menzionate sono:",
    'answer_terms': "Termini chiave collegati alla risposta:",
    'technical_note': "Il contenuto include terminologia tecnica specializzata",
    'question_types': {
        'cosa': "La richiesta chiede informazioni specifiche sull'argomento",
        'come': "La richiesta chiede una procedura o una spiegazione",
        'perché': "La richiesta cerca una ragione o una spiegazione causale",
        'dove': "La richiesta chiede informazioni su un luogo",
        'quando': "La richiesta chiede informazioni temporali",
        'quanto': "La richiesta chiede informazioni quantitative",
    },
    'question_default': "La richiesta chiede informazioni specifiche sull'argomento",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['pt'] = {
    **LANGUAGE_INDICATORS['pt'],
    'connectors': [
        "Em primeiro lugar, ", "Além disso, ", "Por outro lado, ",
        "Também, ", "Finalmente, ", "Em conclusão, ",
    ],
    'overflow_connector': "Além disso, ",
    'analyze': "Analisando",
    'about': "sobre",
    'content_of': "o conteúdo do texto",
    'text_contains': "O texto contém {s} frases com {w} palavras no total",
    'type_desc': {
        'qa': "Trata-se de uma pergunta que requer uma resposta específica",
        'technical': "O conteúdo é técnico e contém dados numéricos ou terminologia especializada",
        'narrative': "O texto apresenta uma narrativa ou descrição de eventos",
        'instructional': "O conteúdo fornece instruções ou diretrizes a seguir",
        'factual': "Informações factuais e objetivas são apresentadas sobre o tema",
        'conversational': "O texto tem um tom conversacional e coloquial",
    },
    'type_default': "O conteúdo apresenta informações relevantes",
    'concepts_prefix': "Os conceitos-chave identificados são:",
    'entities_prefix': "As entidades mencionadas são:",
    'answer_terms': "Termos-chave conectados com a resposta:",
    'technical_note': "O conteúdo inclui terminologia técnica especializada",
    'question_types': {
        'o que': "A consulta solicita informações específicas sobre o tema",
        'como': "A consulta solicita um procedimento ou explicação",
        'por que': "A consulta busca uma razão ou explicação causal",
        'onde': "A consulta solicita informações sobre um local",
        'quando': "A consulta solicita informações temporais",
        'quanto': "A consulta solicita informações quantitativas",
    },
    'question_default': "A consulta solicita informações específicas sobre o tema",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['ro'] = {
    **LANGUAGE_INDICATORS['ro'],
    'connectors': [
        "În primul rând, ", "În plus, ", "Pe de altă parte, ",
        "De asemenea, ", "În cele din urmă, ", "În concluzie, ",
    ],
    'overflow_connector': "De asemenea, ",
    'analyze': "Analizând",
    'about': "despre",
    'content_of': "conținutul textului",
    'text_contains': "Textul conține {s} propoziții cu {w} cuvinte în total",
    'type_desc': {
        'qa': "Aceasta este o întrebare care necesită un răspuns specific",
        'technical': "Conținutul este tehnic și conține date numerice sau terminologie de specialitate",
        'narrative': "Textul prezintă o narațiune sau descriere de evenimente",
        'instructional': "Conținutul oferă instrucțiuni sau ghiduri de urmat",
        'factual': "Sunt prezentate informații factuale și obiective despre subiect",
        'conversational': "Textul are un ton conversațional și colocvial",
    },
    'type_default': "Conținutul prezintă informații relevante",
    'concepts_prefix': "Conceptele cheie identificate sunt:",
    'entities_prefix': "Entitățile menționate sunt:",
    'answer_terms': "Termeni cheie asociați cu răspunsul:",
    'technical_note': "Conținutul include terminologie tehnică de specialitate",
    'question_types': {
        'ce': "Interogarea solicită informații specifice despre subiect",
        'cum': "Interogarea solicită o procedură sau o explicație",
        'de ce': "Interogarea caută un motiv sau o explicație cauzală",
        'unde': "Interogarea solicită informații despre o locație",
        'când': "Interogarea solicită informații temporale",
        'cât': "Interogarea solicită informații cantitative",
    },
    'question_default': "Interogarea solicită informații specifice despre subiect",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['ca'] = {
    **LANGUAGE_INDICATORS['ca'],
    'connectors': [
        "En primer lloc, ", "A més, ", "D'altra banda, ",
        "També, ", "Finalment, ", "En conclusió, ",
    ],
    'overflow_connector': "A més, ",
    'analyze': "Analitzant",
    'about': "sobre",
    'content_of': "el contingut del text",
    'text_contains': "El text conté {s} oracions amb {w} paraules en total",
    'type_desc': {
        'qa': "Es tracta d'una pregunta que requereix una resposta específica",
        'technical': "El contingut és tècnic i conté dades numèriques o terminologia especialitzada",
        'narrative': "El text presenta una narració o descripció d'esdeveniments",
        'instructional': "El contingut proporciona instruccions o pautes a seguir",
        'factual': "Es presenten informacions factuals i objectives sobre el tema",
        'conversational': "El text té un to conversacional i col·loquial",
    },
    'type_default': "El contingut presenta informació rellevant",
    'concepts_prefix': "Els conceptes clau identificats són:",
    'entities_prefix': "Les entitats esmentades són:",
    'answer_terms': "Termes clau relacionats amb la resposta:",
    'technical_note': "El contingut inclou terminologia tècnica especialitzada",
    'question_types': {
        'què': "La consulta sol·licita informació específica sobre el tema",
        'com': "La consulta sol·licita un procediment o explicació",
        'per què': "La consulta busca una raó o explicació causal",
        'on': "La consulta sol·licita informació sobre una ubicació",
        'quan': "La consulta sol·licita informació temporal",
        'quant': "La consulta sol·licita informació quantitativa",
    },
    'question_default': "La consulta sol·licita informació específica sobre el tema",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['gl'] = {
    **LANGUAGE_INDICATORS['gl'],
    'connectors': [
        "En primeiro lugar, ", "Adxemais, ", "Por outra banda, ",
        "Tamén, ", "Finalmente, ", "En conclusión, ",
    ],
    'overflow_connector': "Adxemais, ",
    'analyze': "Analizando",
    'about': "sobre",
    'content_of': "o contido do texto",
    'text_contains': "O texto contén {s} oracións con {w} palabras en total",
    'type_desc': {
        'qa': "Trátase dunha pregunta que require unha resposta específica",
        'technical': "O contido é técnico e contén datos numéricos ou terminoloxía especializada",
        'narrative': "O texto presenta unha narración ou descrición de eventos",
        'instructional': "O contido proporciona instrucións ou directrices a seguir",
        'factual': "Preséntanse informacións factuais e obxectivas sobre o tema",
        'conversational': "O texto ten un ton conversacional e coloquial",
    },
    'type_default': "O contido presenta información relevante",
    'concepts_prefix': "Os conceptos clave identificados son:",
    'entities_prefix': "As entidades mencionadas son:",
    'answer_terms': "Termos clave relacionados coa resposta:",
    'technical_note': "O contido inclúe terminoloxía técnica especializada",
    'question_types': {
        'qué': "A consulta solicita información específica sobre o tema",
        'como': "A consulta solicita un procedemento ou explicación",
        'por que': "A consulta busca unha razón ou explicación causal",
        'onde': "A consulta solicita información sobre unha ubicación",
        'cando': "A consulta solicita información temporal",
        'canto': "A consulta solicita información cuantitativa",
    },
    'question_default': "A consulta solicita información específica sobre o tema",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['en'] = {
    **LANGUAGE_INDICATORS['en'],
    'connectors': [
        "First, ", "Additionally, ", "On the other hand, ",
        "Likewise, ", "Finally, ", "In conclusion, ",
    ],
    'overflow_connector': "Also, ",
    'analyze': "Analyzing",
    'about': "about",
    'content_of': "the content of the text",
    'text_contains': "The text contains {s} sentences with {w} words in total",
    'type_desc': {
        'qa': "This is a question that requires a specific answer",
        'technical': "The content is technical and contains numerical data or specialized terminology",
        'narrative': "The text presents a narrative or description of events",
        'instructional': "The content provides instructions or guidelines to follow",
        'factual': "Factual and objective information is presented about the topic",
        'conversational': "The text has a conversational and colloquial tone",
    },
    'type_default': "The content presents relevant information",
    'concepts_prefix': "The key concepts identified are:",
    'entities_prefix': "The mentioned entities are:",
    'answer_terms': "Key terms connected to the answer:",
    'technical_note': "The content includes specialized technical terminology",
    'question_types': {
        'what': "The query requests specific information about the topic",
        'how': "The query requests a procedure or explanation",
        'why': "The query seeks a reason or causal explanation",
        'where': "The query requests information about a location",
        'when': "The query requests temporal information",
        'how much': "The query requests quantitative information",
    },
    'question_default': "The query requests specific information about the topic",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['de'] = {
    **LANGUAGE_INDICATORS['de'],
    'connectors': [
        "Zunächst, ", "Darüber hinaus, ", "Andererseits, ",
        "Ebenso, ", "Schließlich, ", "Zusammenfassend, ",
    ],
    'overflow_connector': "Außerdem, ",
    'analyze': "Analysiere",
    'about': "über",
    'content_of': "den Inhalt des Textes",
    'text_contains': "Der Text enthält {s} Sätze mit {w} Wörtern insgesamt",
    'type_desc': {
        'qa': "Dies ist eine Frage, die eine spezifische Antwort erfordert",
        'technical': "Der Inhalt ist technisch und enthält numerische Daten oder Fachterminologie",
        'narrative': "Der Text präsentiert eine Erzählung oder Beschreibung von Ereignissen",
        'instructional': "Der Inhalt bietet Anweisungen oder Richtlinien zur Befolgung",
        'factual': "Es werden sachliche und objektive Informationen zum Thema präsentiert",
        'conversational': "Der Text hat einen unterhaltenden und umgangssprachlichen Ton",
    },
    'type_default': "Der Inhalt präsentiert relevante Informationen",
    'concepts_prefix': "Die identifizierten Schlüsselkonzepte sind:",
    'entities_prefix': "Die erwähnten Entitäten sind:",
    'answer_terms': "Schlüsselwörter, die mit der Antwort verbunden sind:",
    'technical_note': "Der Inhalt enthält spezialisierte Fachterminologie",
    'question_types': {
        'was': "Die Anfrage fordert spezifische Informationen über das Thema",
        'wie': "Die Anfrage fordert eine Prozedur oder Erklärung",
        'warum': "Die Anfrage sucht einen Grund oder eine Kausalitätserklärung",
        'wo': "Die Anfrage fordert Informationen über einen Ort",
        'wann': "Die Anfrage fordert zeitliche Informationen",
        'wie viel': "Die Anfrage fordert quantitative Informationen",
    },
    'question_default': "Die Anfrage fordert spezifische Informationen über das Thema",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['nl'] = {
    **LANGUAGE_INDICATORS['nl'],
    'connectors': [
        "Ten eerste, ", "Bovendien, ", "Aan de andere kant, ",
        "Evenzo, ", "Tot slot, ", "Tot besluit, ",
    ],
    'overflow_connector': "Ook, ",
    'analyze': "Analyseren",
    'about': "over",
    'content_of': "de inhoud van de tekst",
    'text_contains': "De tekst bevat {s} zinnen met {w} woorden in totaal",
    'type_desc': {
        'qa': "Dit is een vraag die een specifiek antwoord vereist",
        'technical': "De inhoud is technisch en bevat numerieke gegevens of gespecialiseerde terminologie",
        'narrative': "De tekst presenteert een verhaal of beschrijving van gebeurtenissen",
        'instructional': "De inhoud biedt instructies of richtlijnen om te volgen",
        'factual': "Feitelijke en objectieve informatie wordt gepresenteerd over het onderwerp",
        'conversational': "De tekst heeft een informele en alledaagse toon",
    },
    'type_default': "De inhoud presenteert relevante informatie",
    'concepts_prefix': "De geïdentificeerde sleutelconcepten zijn:",
    'entities_prefix': "De genoemde entiteiten zijn:",
    'answer_terms': "Sleutelwoorden die verband houden met het antwoord:",
    'technical_note': "De inhoud bevat gespecialiseerde technische terminologie",
    'question_types': {
        'wat': "Het verzoek vraagt om specifieke informatie over het onderwerp",
        'hoe': "Het verzoek vraagt om een procedure of uitleg",
        'waarom': "Het verzoek zoekt een reden of causale uitleg",
        'waar': "Het verzoek vraagt om informatie over een locatie",
        'wanneer': "Het verzoek vraagt om temporele informatie",
        'hoeveel': "Het verzoek vraagt om kwantitatieve informatie",
    },
    'question_default': "Het verzoek vraagt om specifieke informatie over het onderwerp",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['sv'] = {
    **LANGUAGE_INDICATORS['sv'],
    'connectors': [
        "Först, ", "Dessutom, ", "Å andra sidan, ",
        "Likaså, ", "Slutligen, ", "Sammanfattningsvis, ",
    ],
    'overflow_connector': "Också, ",
    'analyze': "Analyserar",
    'about': "om",
    'content_of': "textens innehåll",
    'text_contains': "Texten innehåller {s} meningar med {w} ord totalt",
    'type_desc': {
        'qa': "Detta är en fråga som kräver ett specifikt svar",
        'technical': "Innehållet är tekniskt och innehåller numeriska data eller specialiserad terminologi",
        'narrative': "Texten presenterar en berättelse eller beskrivning av händelser",
        'instructional': "Innehållet ger instruktioner eller riktlinjer att följa",
        'factual': "Faktabaserad och objektiv information presenteras om ämnet",
        'conversational': "Texten har en avslappnad och daglig ton",
    },
    'type_default': "Innehållet presenterar relevant information",
    'concepts_prefix': "De identifierade nyckelkoncepten är:",
    'entities_prefix': "De nämnda entiteterna är:",
    'answer_terms': "Nyckelord som är kopplade till svaret:",
    'technical_note': "Innehållet innehåller specialiserad teknisk terminologi",
    'question_types': {
        'vad': "Förfrågan begär specifik information om ämnet",
        'hur': "Förfrågan begär en procedur eller förklaring",
        'varför': "Förfrågan söker en anledning eller orsakssamband",
        'var': "Förfrågan begär information om en plats",
        'när': "Förfrågan begär temporal information",
        'hur mycket': "Förfrågan begär kvantitativ information",
    },
    'question_default': "Förfrågan begär specifik information om ämnet",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['da'] = {
    **LANGUAGE_INDICATORS['da'],
    'connectors': [
        "Først, ", "Derudover, ", "På den anden side, ",
        "Ligeledes, ", "Til sidst, ", "Afslutningsvis, ",
    ],
    'overflow_connector': "Også, ",
    'analyze': "Analyserer",
    'about': "om",
    'content_of': "tekstens indhold",
    'text_contains': "Teksten indeholder {s} sætninger med {w} ord i alt",
    'type_desc': {
        'qa': "Dette er et spørgsmål, der kræver et specifikt svar",
        'technical': "Indholdet er teknisk og indeholder numeriske data eller specialiseret terminologi",
        'narrative': "Teksten præsenterer en fortælling eller beskrivelse af begivenheder",
        'instructional': "Indholdet giver instruktioner eller retningslinjer at følge",
        'factual': "Faktuelle og objektive informationer præsenteres om emnet",
        'conversational': "Teksten har en afslappet og dagligdags tone",
    },
    'type_default': "Indholdet præsenterer relevant information",
    'concepts_prefix': "De identificerede nøglekoncepter er:",
    'entities_prefix': "De nævnte enheder er:",
    'answer_terms': "Nøgleord der er forbundet med svaret:",
    'technical_note': "Indholdet indeholder specialiseret teknisk terminologi",
    'question_types': {
        'hvad': "Forespørgslen anmoder om specifik information om emnet",
        'hvordan': "Forespørgslen anmoder om en procedure eller forklaring',",
        'hvorfor': "Forespørgslen søger en årsag eller årsagsforklaring',",
        'hvor': "Forespørgslen anmoder om information om et sted',",
        'hvornår': "Forespørgslen anmoder om tidsmæssig information",
        'hvor meget': "Forespørgslen anmoder om kvantitativ information",
    },
    'question_default': "Forespørgslen anmoder om specifik information om emnet",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['nb'] = {
    **LANGUAGE_INDICATORS['nb'],
    'connectors': [
        "Først, ", "Videre, ", "På den annen side, ",
        "Likeså, ", "Til slutt, ", "Oppsummerende, ",
    ],
    'overflow_connector': "Også, ",
    'analyze': "Analyserer",
    'about': "om",
    'content_of': "tekstens innhold",
    'text_contains': "Teksten inneholder {s} setninger med {w} ord totalt",
    'type_desc': {
        'qa': "Dette er et spørsmål som krever et spesifikt svar",
        'technical': "Innholdet er teknisk og inneholder numeriske data eller spesialisert terminologi",
        'narrative': "Teksten presenterer en fortelling eller beskrivelse av hendelser",
        'instructional': "Innholdet gir instruksjoner eller retningslinjer å følge",
        'factual': "Faktiske og objektive informasjon presenteres om emnet",
        'conversational': "Teksten har en avslappet og dagligdags tone",
    },
    'type_default': "Innholdet presenterer relevant informasjon",
    'concepts_prefix': "De identifiserte nøkkelkonseptene er:",
    'entities_prefix': "De nevnte enhetene er:",
    'answer_terms': "Nøkkelord som er forbundet med svaret:",
    'technical_note': "Innholdet inneholder spesialisert teknisk terminologi",
    'question_types': {
        'hva': "Forespørselen ber om spesifikk informasjon om emnet',",
        'hvordan': "Forespørselen ber om en prosedyre eller forklaring",
        'hvorfor': "Forespørselen søker en årsak eller årsakssammenheng',",
        'hvor': "Forespørselen ber om informasjon om et sted',",
        'når': "Forespørselen ber om tidsmessig informasjon",
        'hvor mye': "Forespørselen ber om kvantitativ informasjon",
    },
    'question_default': "Forespørselen ber om spesifikk informasjon om emnet",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['pl'] = {
    **LANGUAGE_INDICATORS['pl'],
    'connectors': [
        "Po pierwsze, ", "Ponadto, ", "Z drugiej strony, ",
        "Podobnie, ", "Na koniec, ", "Podsumowując, ",
    ],
    'overflow_connector': "Także, ",
    'analyze': "Analizując",
    'about': "o",
    'content_of': "zawartość tekstu",
    'text_contains': "Tekst zawiera {s} zdań z {w} słowami łącznie",
    'type_desc': {
        'qa': "Jest to pytanie wymagające konkretnej odpowiedzi",
        'technical': "Zawartość jest techniczna i zawiera dane liczbowe lub specjalistyczną terminologię",
        'narrative': "Tekst przedstawia narrację lub opis wydarzeń",
        'instructional': "Zawartość zapewnia instrukcje lub wytyczne do przestrzegania',",
        'factual': "Przedstawiono faktyczne i obiektywne informacje na temat",
        'conversational': "Tekst ma ton rozmowny i potoczny",
    },
    'type_default': "Zawartość przedstawia istotne informacje",
    'concepts_prefix': "Zidentyfikowane kluczowe koncepcje to:",
    'entities_prefix': "Wspomniane podmioty to:",
    'answer_terms': "Kluczowe słowa powiązane z odpowiedzią:",
    'technical_note': "Zawartość zawiera specjalistyczną terminologię techniczną",
    'question_types': {
        'co': "Zapytanie prosi o konkretne informacje na temat",
        'jak': "Zapytanie prosi o procedurę lub wyjaśnienie',",
        'dlaczego': "Zapytanie szuka przyczyny lub wyjaśnienia przyczynowego",
        'gdzie': "Zapytanie prosi o informacje o lokalizacji",
        'kiedy': "Zapytanie prosi o informacje czasowe',",
        'ile': "Zapytanie prosi o informacje ilościowe",
    },
    'question_default': "Zapytanie prosi o konkretne informacje na temat",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['cs'] = {
    **LANGUAGE_INDICATORS['cs'],
    'connectors': [
        "Za prvé, ", "Kromě toho, ", "Na druhou stranu, ",
        "Obdobně, ", "Nakonec, ", "Shrnutí, ",
    ],
    'overflow_connector': "Také, ",
    'analyze': "Analyzuji",
    'about': "o",
    'content_of': "obsah textu",
    'text_contains': "Text obsahuje {s} vět s {w} slovy celkem",
    'type_desc': {
        'qa': "Jde o otázku, která vyžaduje konkrétní odpověď",
        'technical': "Obsah je technický a obsahuje numerická data nebo specializovanou terminologii",
        'narrative': "Text prezentuje vyprávění nebo popis událostí",
        'instructional': "Obsah poskytuje instrukce nebo pokyny k dodržování",
        'factual': "Prezentují se faktické a objektivní informace o tématu",
        'conversational': "Text má konverzační a hovorový tón",
    },
    'type_default': "Obsah prezentuje relevantní informace",
    'concepts_prefix': "Identifikované klíčové koncepty jsou:",
    'entities_prefix': "Zmíněné entity jsou:",
    'answer_terms': "Klíčová slova spojená s odpovědí:",
    'technical_note': "Obsah zahrnuje specializovanou technickou terminologii",
    'question_types': {
        'co': "Dotaz žádá konkrétní informace o tématu",
        'jak': "Dotaz žádá postup nebo vysvětlení',",
        'proč': "Dotaz hledá důvod nebo příčinné vysvětlení',",
        'kde': "Dotaz žádá informace o umístění',",
        'kdy': "Dotaz žádá časové informace',",
        'kolik': "Dotaz žádá kvantitativní informace",
    },
    'question_default': "Dotaz žádá konkrétní informace o tématu",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['sk'] = {
    **LANGUAGE_INDICATORS['sk'],
    'connectors': [
        "Po prvé, ", "Okrem toho, ", "Na druhej strane, ",
        "Podobne, ", "Nakoniec, ", "Na záver, ",
    ],
    'overflow_connector': "Tiež, ",
    'analyze': "Analyzujem",
    'about': "o",
    'content_of': "obsah textu",
    'text_contains': "Text obsahuje {s} viet s {w} slovami celkom",
    'type_desc': {
        'qa': "Ide o otázku, ktorá vyžaduje konkrétnu odpoveď",
        'technical': "Obsah je technický a obsahuje numerické dáta alebo špecializovanú terminológiu",
        'narrative': "Text prezentuje rozprávanie alebo popis udalostí",
        'instructional': "Obsah poskytuje inštrukcie alebo pokyny na dodržiavanie",
        'factual': "Prezentujú sa faktické a objektívne informácie o téme",
        'conversational': "Text má konverzačný a hovorový tón",
    },
    'type_default': "Obsah prezentuje relevantné informácie",
    'concepts_prefix': "Identifikované kľúčové koncepty sú:",
    'entities_prefix': "Spomenuté entity sú:",
    'answer_terms': "Kľúčové slová spojené s odpoveďou:",
    'technical_note': "Obsah zahŕňa špecializovanú technickú terminológiu",
    'question_types': {
        'čo': "Dopyt žiada konkrétne informácie o téme",
        'ako': "Dopyt žiada postup alebo vysvetlenie',",
        'prečo': "Dopyt hľadá dôvod alebo príčinné vysvetlenie',",
        'kde': "Dopyt žiada informácie o umiestnení',",
        'kedy': "Dopyt žiada časové informácie',",
        'koľko': "Dopyt žiada kvantitatívne informácie",
    },
    'question_default': "Dopyt žiada konkrétne informácie o téme",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['bg'] = {
    **LANGUAGE_INDICATORS['bg'],
    'connectors': [
        "Първо, ", "Освен това, ", "От друга страна, ",
        "Също така, ", "Накрая, ", "В заключение, ",
    ],
    'overflow_connector': "Също, ",
    'analyze': "Анализирам",
    'about': "относно",
    'content_of': "съдържанието на текста",
    'text_contains': "Текстът съдържа {s} изречения с {w} думи общо",
    'type_desc': {
        'qa': "Това е въпрос, който изисква конкретен отговор",
        'technical': "Съдържанието е техническо и съдържа числови данни или специализирана терминология",
        'narrative': "Текстът представя разказ или описание на събития",
        'instructional': "Съдържанието предоставя инструкции или насоки за следване",
        'factual': "Представени са фактически и обективни информации относно темата",
        'conversational': "Текстът има разговорен и ежедневен тон",
    },
    'type_default': "Съдържанието представя relevantна информация",
    'concepts_prefix': "Идентифицираните ключови концепции са:",
    'entities_prefix': "Споменатите обекти са:",
    'answer_terms': "Ключови думи, свързани с отговора:",
    'technical_note': "Съдържанието включва специализирана техническа терминология",
    'question_types': {
        'какво': "Заявката моли за конкретна информация относно темата",
        'как': "Заявката моли за процедура или обяснение',",
        'защо': "Заявката търси причина или причинно обяснение',",
        'къде': "Заявката моли за информация за местоположение',",
        'кога': "Заявката моли за времева информация',",
        'колко': "Заявката моли за количествена информация",
    },
    'question_default': "Заявката моли за конкретна информация относно темата",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['hr'] = {
    **LANGUAGE_INDICATORS['hr'],
    'connectors': [
        "Prvo, ", "Štoviše, ", "S druge strane, ",
        "Također, ", "Napokon, ", "Zaključno, ",
    ],
    'overflow_connector': "Također, ",
    'analyze': "Analiziram",
    'about': "o",
    'content_of': "sadržaj teksta",
    'text_contains': "Tekst sadrži {s} rečenica s {w} riječi ukupno",
    'type_desc': {
        'qa': "To je pitanje koje zahtijeva specifičan odgovor",
        'technical': "Sadržaj je tehnički i sadrži numeričke podatke ili specijaliziranu terminologiju",
        'narrative': "Tekst prezentira naraciju ili opis događaja",
        'instructional': "Sadržaj pruža upute ili smjernice za praćenje",
        'factual': "Prezentirane su činjenične i objektivne informacije o temi",
        'conversational': "Tekst ima konverzacijski i kolokvijalni ton",
    },
    'type_default': "Sadržaj prezentira relevantne informacije",
    'concepts_prefix': "Identificirani ključni koncepti su:",
    'entities_prefix': "Spomenuti entiteti su:",
    'answer_terms': "Ključne riječi povezane s odgovorom:",
    'technical_note': "Sadržaj uključuje specijaliziranu tehničku terminologiju",
    'question_types': {
        'što': "Upit traži specifične informacije o temi',",
        'kako': "Upit traži postupak ili objašnjenje',",
        'zašto': "Upit traži razlog ili uzročno objašnjenje',",
        'gdje': "Upit traži informacije o lokaciji",
        'kada': "Upit traži vremenske informacije',",
        'koliko': "Upit traži količinske informacije",
    },
    'question_default': "Upit traži specifične informacije o temi",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['sr'] = {
    **LANGUAGE_INDICATORS['sr'],
    'connectors': [
        "Прво, ", "Штавише, ", "С друге стране, ",
        "Такође, ", "Напослетку, ", "Закључно, ",
    ],
    'overflow_connector': "Такође, ",
    'analyze': "Анализiram",
    'about': "о",
    'content_of': "садржај текста",
    'text_contains': "Текст садржи {s} реченица са {w} речи укупно",
    'type_desc': {
        'qa': "То је питање које захтева специфичан одговор",
        'technical': "Садржај је технички и садржи нумеричке податке или специјализовану терминологију",
        'narrative': "Текст презентира нарацију или опис догађаја",
        'instructional': "Садржај пружа упуте или смернице за праћење",
        'factual': "Презентоване су чињеничне и објективне информације о теми",
        'conversational': "Текст има конверзацијски и колоквијални тон",
    },
    'type_default': "Садржај презентира релевантне информације",
    'concepts_prefix': "Идентификовани кључни конCEPTи су:",
    'entities_prefix': "Споменути ентитети су:",
    'answer_terms': "Кључне речи повезане са одговором:",
    'technical_note': "Садржај укључује специјализовану техничку терминологију",
    'question_types': {
        'шта': "Упит тражи специфичне информације о теми",
        'како': "Упит тражи поступак или објашњење",
        'зашто': "Упит тражи разлог или узрочно објашњење',",
        'где': "Упит тражи информације о локацији',",
        'када': "Упит тражи временске информације',",
        'колико': "Упит тражи квантитативне информације",
    },
    'question_default': "Упит тражи специфичне информације о теми",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['sl'] = {
    **LANGUAGE_INDICATORS['sl'],
    'connectors': [
        "Najprej, ", "Poleg tega, ", "Po drugi strani, ",
        "Prav tako, ", "Nazadnje, ", "Zaključno, ",
    ],
    'overflow_connector': "Tudi, ",
    'analyze': "Analiziram",
    'about': "o",
    'content_of': "vsebino besedila",
    'text_contains': "Besedilo vsebuje {s} povedi z {w} besedami skupaj",
    'type_desc': {
        'qa': "To je vprašanje, ki zahteva specifičen odgovor",
        'technical': "Vsebina je tehnična in vsebuje numerične podatke ali specializirano terminologijo",
        'narrative': "Besedilo predstavlja pripoved ali opis dogodkov",
        'instructional': "Vsebina zagotavlja navodila ali smernice za upoštevanje",
        'factual': "Predstavljena so dejstva in objektivne informacije o temi",
        'conversational': "Besedilo ima pogovorni in vsakdanji ton",
    },
    'type_default': "Vsebina predstavlja relevantne informacije",
    'concepts_prefix': "Identificirani ključni koncepti so:",
    'entities_prefix': "Omenjene entitete so:",
    'answer_terms': "Ključne besede, povezane z odgovorom:",
    'technical_note': "Vsebina vključuje specializirano tehnično terminologijo",
    'question_types': {
        'kaj': "Poizvedba zahteva specifične informacije o temi",
        'kako': "Poizvedba zahteva postopek ali pojasnilo',",
        'zakaj': "Poizvedba išče razlog ali vzročno pojasnilo',",
        'kje': "Poizvedba zahteva informacije o lokaciji',",
        'kdaj': "Poizvedba zahteva časovne informacije',",
        'koliko': "Poizvedba zahteva količinske informacije",
    },
    'question_default': "Poizvedba zahteva specifične informacije o temi",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['bs'] = {
    **LANGUAGE_INDICATORS['bs'],
    'connectors': [
        "Prvo, ", "Štoviše, ", "S druge strane, ",
        "Također, ", "Napokon, ", "Zaključno, ",
    ],
    'overflow_connector': "Također, ",
    'analyze': "Analiziram",
    'about': "o",
    'content_of': "sadržaj teksta",
    'text_contains': "Tekst sadrži {s} rečenica sa {w} riječi ukupno",
    'type_desc': {
        'qa': "To je pitanje koje zahtijeva specifičan odgovor",
        'technical': "Sadržaj je tehnički i sadrži numeričke podatke ili specijaliziranu terminologiju",
        'narrative': "Tekst prezentira naraciju ili opis događaja",
        'instructional': "Sadržaj pruža upute ili smjernice za praćenje",
        'factual': "Prezentirane su činjenične i objektivne informacije o temi",
        'conversational': "Tekst ima konverzacijski i kolokvijalni ton",
    },
    'type_default': "Sadržaj prezentira relevantne informacije",
    'concepts_prefix': "Identificirani ključni koncepti su:",
    'entities_prefix': "Spomenuti entiteti su:",
    'answer_terms': "Ključne riječi povezane sa odgovorom:",
    'technical_note': "Sadržaj uključuje specijaliziranu tehničku terminologiju",
    'question_types': {
        'šta': "Upit traži specifične informacije o temi",
        'kako': "Upit traži postupak ili objašnjenje',",
        'zašto': "Upit traži razlog ili uzročno objašnjenje',",
        'gdje': "Upit traži informacije o lokaciji",
        'kada': "Upit traži vremenske informacije",
        'koliko': "Upit traži količinske informacije",
    },
    'question_default': "Upit traži specifične informacije o temi",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['mk'] = {
    **LANGUAGE_INDICATORS['mk'],
    'connectors': [
        "Прво, ", "Покрај тоа, ", "Од друга страна, ",
        "Исто така, ", "На крајот, ", "Заклучно, ",
    ],
    'overflow_connector': "Исто така, ",
    'analyze': "Анализирам",
    'about': "за",
    'content_of': "содржината на текстот",
    'text_contains': "Текстот содржи {s} реченици со {w} зборови вкупно",
    'type_desc': {
        'qa': "Тоа е прашање кое бара конкретен одговор",
        'technical': "Содржината е техничка и содржи нумерички податоци или специјализирана терминологија",
        'narrative': "Текстот претставува нарација или опис на настани",
        'instructional': "Содржината обезбедува упатства или насоки за следење",
        'factual': "Претставени се фактички и објективни информации за темата",
        'conversational': "Текстот има разговорен и секојдневен тон",
    },
    'type_default': "Содржината претставува релевантни информации",
    'concepts_prefix': "Идентификуваните клучни концепти се:",
    'entities_prefix': "Споменатите ентитети се:",
    'answer_terms': "Клучни зборови поврзани со одговорот:",
    'technical_note': "Содржината вклучува специјализирана техничка терминологија",
    'question_types': {
        'што': "Барањето бара специфични информации за темата",
        'како': "Барањето бара постапка или објаснување',",
        'зошто': "Барањето бара причина или причинско објаснување',",
        'каде': "Барањето бара информации за локација",
        'кога': "Барањето бара временски информации",
        'колку': "Барањето бара квантитативни информации",
    },
    'question_default': "Барањето бара специфични информации за темата",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['lt'] = {
    **LANGUAGE_INDICATORS['lt'],
    'connectors': [
        "Pirma, ", "Be to, ", "Kita vertus, ",
        "Taip pat, ", "Galiausiai, ", "Apibendrinant, ",
    ],
    'overflow_connector': "Taip pat, ",
    'analyze': "Analizuoju",
    'about': "apie",
    'content_of': "teksto turinį",
    'text_contains': "Tekste yra {s} sakinių su {w} žodžiais iš viso",
    'type_desc': {
        'qa': "Tai klausimas, reikalaujantis konkretaus atsakymo",
        'technical': "Turinys yra techninis ir turi skaitinius duomenis ar specializuotą terminologiją",
        'narrative': "Tekstas pateikia pasakojimą ar įvykių aprašymą",
        'instructional': "Turinys pateikia instrukcijas ar gaires, kurių reikia laikytis",
        'factual': "Pateikiami faktiniai ir objektyvūs duomenys apie temą",
        'conversational': "Tekstas turi pokalbio ir kasdienį toną",
    },
    'type_default': "Turinys pateikia aktualią informaciją",
    'concepts_prefix': "Identifikuotos pagrindinės koncepcijos:",
    'entities_prefix': "Paminėti subjektai:",
    'answer_terms': "Pagrindiniai žodžiai, susiję su atsakymu:",
    'technical_note': "Turinyje yra specializuota techninė terminologija",
    'question_types': {
        'kas': "Užklausa prašo konkrečios informacijos apie temą",
        'kaip': "Užklausa prašo procedūros ar paaiškinimo',",
        'kodėl': "Užklausa ieško priežasties ar priežastinio paaiškinimo',",
        'kur': "Užklausa prašo informacijos apie vietą",
        'kada': "Užklausa prašo laiko informacijos",
        'kiek': "Užklausa prašo kiekybinės informacijos",
    },
    'question_default': "Užklausa prašo konkrečios informacijos apie temą",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['lv'] = {
    **LANGUAGE_INDICATORS['lv'],
    'connectors': [
        "Vispirms, ", "Turklāt, ", "No otras puses, ",
        "Arī, ", "Visbeidzot, ", "Nobeigumā, ",
    ],
    'overflow_connector': "Arī, ",
    'analyze': "Analizēju",
    'about': "par",
    'content_of': "teksta saturu",
    'text_contains': "Tekstā ir {s} teikumi ar {w} vārdiem kopā",
    'type_desc': {
        'qa': "Tas ir jautājums, kas prasa specifisku atbildi",
        'technical': "Saturs ir tehnisks un satur skaitliskus datus vai specializētu terminoloģiju",
        'narrative': "Teksts prezentē stāstu vai notikumu aprakstu",
        'instructional': "Saturs sniedz instrukcijas vai vadlīnijas, kam jāseko",
        'factual': "Tiek prezentēta faktiska un objektīva informācija par tēmu",
        'conversational': "Tekstam ir sarunvalodas un ikdienišķs tonis",
    },
    'type_default': "Saturs prezentē atbilstošu informāciju",
    'concepts_prefix': "Identificētās galvenās koncepcijas ir:",
    'entities_prefix': "Pieminētās entītes ir:",
    'answer_terms': "Galvenie vārdi, kas saistīti ar atbildi:",
    'technical_note': "Saturs ietver specializētu tehnisko terminoloģiju",
    'question_types': {
        'kas': "Pieprasījums lūdz specifisku informāciju par tēmu",
        'kā': "Pieprasījums lūdz procedūru vai paskaidrojumu',",
        'kāpēc': "Pieprasījums meklē iemeslu vai cēloņu paskaidrojumu',",
        'kur': "Pieprasījums lūdz informāciju par atrašanās vietu',",
        'kad': "Pieprasījums lūdz laika informāciju',",
        'cik': "Pieprasījums lūdz kvantitatīvu informāciju",
    },
    'question_default': "Pieprasījums lūdz specifisku informāciju par tēmu",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['fi'] = {
    **LANGUAGE_INDICATORS['fi'],
    'connectors': [
        "Ensinnäkin, ", "Lisäksi, ", "Toisaalta, ",
        "Samoin, ", "Lopuksi, ", "Yhteenvetona, ",
    ],
    'overflow_connector': "Myös, ",
    'analyze': "Analysoin",
    'about': "aiheesta",
    'content_of': "tekstin sisällön",
    'text_contains': "Teksti sisältää {s} lausetta ja {w} sanaa yhteensä",
    'type_desc': {
        'qa': "Tämä on kysymys, joka vaatii spesifin vastauksen",
        'technical': "Sisältö on teknistä ja sisältää numeerisia tietoja tai erikoistunutta terminologiaa",
        'narrative': "Teksti esittää kertomuksen tai tapahtumien kuvauksen",
        'instructional': "Sisältö tarjoaa ohjeita tai suuntaviivoja noudatettavaksi",
        'factual': "Aiheesta esitetään faktuaalia ja objektiivista tietoa",
        'conversational': "Tekstillä on keskusteleva ja arkinen ääni",
    },
    'type_default': "Sisältö esittää merkityksellistä tietoa",
    'concepts_prefix': "Tunnistetut avainkäsitteet ovat:",
    'entities_prefix': "Mainitut kokonaisuudet ovat:",
    'answer_terms': "Avainsanat, jotka liittyvät vastaukseen:",
    'technical_note': "Sisältö sisältää erikoistunutta teknistä terminologiaa",
    'question_types': {
        'mikä': "Kysely pyytää spesifistä tietoa aiheesta",
        'miten': "Kysely pyytää menettelyä tai selitystä',",
        'miksi': "Kysely etsii syytä tai kausaalista selitystä',",
        'missä': "Kysely pyytää tietoa sijainnista",
        'milloin': "Kysely pyytää aikatietoja',",
        'kuinka paljon': "Kysely pyytää kvantitatiivista tietoa",
    },
    'question_default': "Kysely pyytää spesifistä tietoa aiheesta",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['et'] = {
    **LANGUAGE_INDICATORS['et'],
    'connectors': [
        "Esiteks, ", "Lisaks, ", "Teisalt, ",
        "Samuti, ", "Lõpuks, ", "Kokkuvõttes, ",
    ],
    'overflow_connector': "Samuti, ",
    'analyze': "Analüüsin",
    'about': "teemal",
    'content_of': "teksti sisu",
    'text_contains': "Tekst sisaldab {s} lauset ja {w} sõna kokku",
    'type_desc': {
        'qa': "See on küsimus, mis nõuab spetsiifilist vastust",
        'technical': "Sisu on tehniline ja sisaldab numbrilisi andmeid või spetsialiseeritud terminoloogiat',",
        'narrative': "Tekst esitab jutustuse või sündmuste kirjelduse',",
        'instructional': "Sisu pakub juhiseid või suuniseid järgimiseks',",
        'factual': "Teemal esitatakse faktilist ja objektiivset teavet",
        'conversational': "Tekstil on vestluslik ja igapäevane toon",
    },
    'type_default': "Sisu esitab asjakohast teavet",
    'concepts_prefix': "Tuvastatud võtmekontseptsioonid on:",
    'entities_prefix': "Mainitud üksused on:",
    'answer_terms': "Võtmesõnad, mis on seotud vastusega:",
    'technical_note': "Sisu sisaldab spetsialiseeritud tehnilist terminoloogiat",
    'question_types': {
        'mis': "Päring palub spetsiifilist teavet teema kohta",
        'kuidas': "Päring palub protseduuri või selgitust',",
        'miks': "Päring otsib põhjust või põhjuslikku selgitust',",
        'kus': "Päring palub teavet asukoha kohta',",
        'millal': "Päring palub ajalist teavet',",
        'kui palju': "Päring palub kvantitatiivset teavet",
    },
    'question_default': "Päring palub spetsiifilist teavet teema kohta",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['hu'] = {
    **LANGUAGE_INDICATORS['hu'],
    'connectors': [
        "Először is, ", "Ezenkívül, ", "Másrészt, ",
        "Szintén, ", "Végül, ", "Összefoglalva, ",
    ],
    'overflow_connector': "Szintén, ",
    'analyze': "Elemezem",
    'about': "erről",
    'content_of': "a szöveg tartalmát",
    'text_contains': "A szöveg {s} mondatot tartalmaz, összesen {w} szóval",
    'type_desc': {
        'qa': "Ez egy kérdés, amely konkrét választ igényel",
        'technical': "A tartalom technikai, és számadatokat vagy specializált terminológiát tartalmaz",
        'narrative': "A szöveg elbeszélést vagy események leírását mutatja be",
        'instructional': "A tartalom utasításokat vagy irányelveket nyújt a követéshez",
        'factual': "Tényeken alapuló és objektív információk jelennek meg a témáról",
        'conversational': "A szöveg beszélgetési és hétköznapi hangnemű",
    },
    'type_default': "A tartalom releváns információkat mutat be",
    'concepts_prefix': "Az azonosított kulcsfogalmak:",
    'entities_prefix': "A megemlített entitások:",
    'answer_terms': "Kulcsszavak, amelyek kapcsolódnak a válaszhoz:",
    'technical_note': "A tartalom specializált technikai terminológiát tartalmaz",
    'question_types': {
        'mi': "A kérés konkrét információkat kér a témáról",
        'hogyan': "A kérés eljárást vagy magyarázatot kér',",
        'miért': "A kérés okot vagy ok-okozati magyarázatot keres',",
        'hol': "A kérés információt kér egy helyről',",
        'mikor': "A kérés időbeli információkat kér',",
        'mennyi': "A kérés mennyiségi információkat kér",
    },
    'question_default': "A kérés konkrét információkat kér a témáról",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['ga'] = {
    **LANGUAGE_INDICATORS['ga'],
    'connectors': [
        "Ar dtús, ", "Chomh maith le sin, ", "Ar an taobh eile, ",
        "Chomh maith, ", "Faoi dheireadh, ", "Mar fhocal scoir, ",
    ],
    'overflow_connector': "Chomh maith, ",
    'analyze': "Ag anailísiú",
    'about': "faoi",
    'content_of': "ábhar an téacs",
    'text_contains': "Tá {s} abairte agus {w} focal iomlán sa téacs",
    'type_desc': {
        'qa': "Is ceist é seo a éilíonn freagra ar leith",
        'technical': "Tá an t-ábhar teicniúil agus tá sonraí uimhriúla nó téarmaíocht speisialaithe ann",
        'narrative': "Léiríonn an téacs scéal nó cur síos ar imeachtaí",
        'instructional': "Soláthraíonn an t-ábhar treoracha nó treoirlínte le leanúint",
        'factual': "Tá faisnéis fhiúntach agus oibiachtúil curtha i láthair faoin ábhar",
        'conversational': "Tá tón comhrá agus coiteann ag an téacs",
    },
    'type_default': "Léiríonn an t-ábhar faisnéis thábhachtach",
    'concepts_prefix': "Na coincheapa tábhachtacha a bhraitheadh:",
    'entities_prefix': "Na heintitis a luadh:",
    'answer_terms': "Téarmaí tábhachtacha a bhaineann leis an bhfreagra:",
    'technical_note': "Áirítear sa t-ábhar téarmaíocht theicniúil speisialaithe",
    'question_types': {
        'cad': "Iarrann an fiosrúchán faisnéis ar leith faoin ábhar",
        'conas': "Iarrann an fiosrúchán nós imeartha nó míniú',",
        'cén fáth': "Lorgann an fiosrúchán cúis nó míniú cúiseach',",
        'cá háit': "Iarrann an fiosrúchán faisnéis faoi shuíomh',",
        'cathain': "Iarrann an fiosrúchán faisnéis ama",
        'cé mhéad': "Iarrann an fiosrúchán faisnéis chainníochtúil",
    },
    'question_default': "Iarrann an fiosrúchán faisnéis ar leith faoin ábhar",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['el'] = {
    **LANGUAGE_INDICATORS['el'],
    'connectors': [
        "Πρώτον, ", "Επιπλέον, ", "Από την άλλη πλευρά, ",
        "Επίσης, ", "Τελικά, ", "Συμπερασματικά, ",
    ],
    'overflow_connector': "Επίσης, ",
    'analyze': "Αναλύω",
    'about': "σχετικά με",
    'content_of': "το περιεχόμενο του κειμένου",
    'text_contains': "Το κείμενο περιέχει {s} προτάσεις με {w} λέξεις συνολικά",
    'type_desc': {
        'qa': "Αυτό είναι μια ερώτηση που απαιτεί συγκεκριμένη απάντηση",
        'technical': "Το περιεχόμενο είναι τεχνικό και περιέχει αριθμητικά δεδομένα ή ειδικευμένη τερμινολογία",
        'narrative': "Το κείμενο παρουσιάζει αφήγηση ή περιγραφή γεγονότων",
        'instructional': "Το περιεχόμενο παρέχει οδηγίες ή κατευθυντήριες γραμμές για ακολούθηση",
        'factual': "Παρουσιάζονται γεγοντολογικές και αντικειμενικές πληροφορίες για το θέμα",
        'conversational': "Το κείμενο έχει συζητητικό και καθημερινό τόνο",
    },
    'type_default': "Το περιεχόμενο παρουσιάζει σχετικές πληροφορίες",
    'concepts_prefix': "Τα ταυτοποιημένα βασικά θέματα είναι:",
    'entities_prefix': "Τα αναφερόμενα αντικείμενα είναι:",
    'answer_terms': "Βασικές λέξεις που συνδέονται με την απάντηση:",
    'technical_note': "Το περιεχόμενο περιλαμβάνει ειδικευμένη τεχνική τερμινολογία",
    'question_types': {
        'τι': "Το αίτημα ζητά συγκεκριμένες πληροφορίες για το θέμα",
        'πώς': "Το αίτημα ζητά διαδικασία ή εξήγηση',",
        'γιατί': "Το αίτημα ψάχνει αιτία ή αιτιολογική εξήγηση',",
        'πού': "Το αίτημα ζητά πληροφορίες για τοποθεσία',",
        'πότε': "Το αίτημα ζητά χρονικές πληροφορίες',",
        'πόσο': "Το αίτημα ζητά ποσοτικές πληροφορίες",
    },
    'question_default': "Το αίτημα ζητά συγκεκριμένες πληροφορίες για το θέμα",
    'meta_patterns': [],
}

LANGUAGE_CONFIG['sq'] = {
    **LANGUAGE_INDICATORS['sq'],
    'connectors': [
        "Së pari, ", "Për më tepër, ", "Nga ana tjetër, ",
        "Gjithashtu, ", "Së fundi, ", "Përmbledhtas, ",
    ],
    'overflow_connector': "Gjithashtu, ",
    'analyze': "Duke analizuar",
    'about': "rreth",
    'content_of': "përmbajtjen e tekstit",
    'text_contains': "Teksti përmban {s} fjali me {w} fjalë në total",
    'type_desc': {
        'qa': "Kjo është një pyetje që kërkon një përgjigje specifike",
        'technical': "Përmbajtja është teknike dhe përmban të dhëna numerike ose terminologji të specializuar",
        'narrative': "Teksti paraqet një rrëfim ose përshkrim të ngjarjeve",
        'instructional': "Përmbajtja ofron udhëzime ose udhëzues për t'u ndjekur",
        'factual': "Paraqiten informacione faktike dhe objektive për temën",
        'conversational': "Teksti ka një ton bisedues dhe të përditshëm",
    },
    'type_default': "Përmbajtja paraqet informacione të rëndësishme",
    'concepts_prefix': "Konceptet kryesore të identifikuara janë:",
    'entities_prefix': "Entitetet e përmendura janë:",
    'answer_terms': "Fjalët kryesore të lidhura me përgjigjen:",
    'technical_note': "Përmbajtja përfshin terminologji të specializuar teknike",
    'question_types': {
        'ça': "Kërkesa kërkon informacione specifike për temën",
        'si': "Kërkesa kërkon një procedurë ose shpjegim',",
        'pse': "Kërkesa kërkon një arsye ose shpjegim shkakësor',",
        'ku': "Kërkesa kërkon informacione për një vendndodhje',",
        'kur': "Kërkesa kërkon informacione kohore",
        'sa': "Kërkesa kërkon informacione sasiore",
    },
    'question_default': "Kërkesa kërkon informacione specifike për temën",
    'meta_patterns': [],
}


# Fallback config for unsupported languages (uses English as base)
FALLBACK_CONFIG = LANGUAGE_CONFIG['en']


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

    def _get_lang_config(self, language: str) -> dict:
        """Get language configuration, fallback to English if unsupported."""
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

    # Legacy methods for backward compatibility
    def _build_reasoning_es(self, text, analysis, concepts, content_type, context):
        return self._build_reasoning(text, analysis, concepts, content_type, context, 'es')

    def _build_reasoning_en(self, text, analysis, concepts, content_type, context):
        return self._build_reasoning(text, analysis, concepts, content_type, context, 'en')

    def _format_steps_es(self, steps):
        return self._format_steps(steps, LANGUAGE_CONFIG.get('es', {}))

    def _format_steps_en(self, steps):
        return self._format_steps(steps, LANGUAGE_CONFIG.get('en', {}))