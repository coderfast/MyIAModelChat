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
# EU official languages (24) + Eastern European languages
# ═══════════════════════════════════════════════════════════════════════════════

LANGUAGE_CONFIG = {
    # ── Romance languages ──────────────────────────────────────────────────────
    'es': {
        'name': 'Spanish',
        'family': 'romance',
        'indicators': [' el ', ' la ', ' los ', ' las ', ' de ', ' del ',
                       ' en ', ' un ', ' una ', ' que ', ' es ', ' son ',
                       'porque', 'entonces', 'además', 'sin embargo'],
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
    },
    'fr': {
        'name': 'French',
        'family': 'romance',
        'indicators': [' le ', ' la ', ' les ', ' des ', ' du ', ' de ',
                       ' en ', ' un ', ' une ', ' que ', ' est ', ' sont ',
                       'parce que', 'donc', 'cependant', 'en outre'],
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
    },
    'it': {
        'name': 'Italian',
        'family': 'romance',
        'indicators': [' il ', ' la ', ' i ', ' le ', ' di ', ' del ',
                       ' in ', ' un ', ' una ', ' che ', ' è ', ' sono ',
                       'perché', 'quindi', 'tuttavia', 'inoltre'],
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
    },
    'pt': {
        'name': 'Portuguese',
        'family': 'romance',
        'indicators': [' o ', ' a ', ' os ', ' as ', ' de ', ' do ',
                       ' em ', ' um ', ' uma ', ' que ', ' é ', ' são ',
                       'porque', 'então', 'além disso', 'no entanto'],
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
    },
    'ro': {
        'name': 'Romanian',
        'family': 'romance',
        'indicators': [' el ', ' ea ', ' ei ', ' ele ', ' de ', ' din ',
                       ' în ', ' un ', ' o ', ' care ', ' este ', ' sunt ',
                       'pentru că', 'deci', 'totuși', 'în plus'],
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
    },
    'ca': {
        'name': 'Catalan',
        'family': 'romance',
        'indicators': [' el ', ' la ', ' els ', ' les ', ' de ', ' del ',
                       ' en ', ' un ', ' una ', ' que ', ' és ', ' són ',
                       'perquè', 'llavors', 'tanmateix', 'a més'],
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
    },
    'gl': {
        'name': 'Galician',
        'family': 'romance',
        'indicators': [' o ', ' a ', ' os ', ' as ', ' de ', ' do ',
                       ' en ', ' un ', ' unha ', ' que ', ' é ', ' son ',
                       'porque', 'entón', 'adxemais', 'non obstante'],
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
    },

    # ── Germanic languages ─────────────────────────────────────────────────────
    'en': {
        'name': 'English',
        'family': 'germanic',
        'indicators': [' the ', ' is ', ' are ', ' was ', ' were ',
                       ' have ', ' has ', ' had ', ' that ', ' which '],
        'connectors': [
            "First, ", "Additionally, ", "Furthermore, ",
            "Moreover, ", "Finally, ", "In conclusion, ",
        ],
        'overflow_connector': "Also, ",
        'analyze': "Analyzing",
        'about': "about",
        'content_of': "the text content",
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
        'concepts_prefix': "Key concepts identified:",
        'entities_prefix': "Entities mentioned:",
        'answer_terms': "Key terms connecting with the answer:",
        'technical_note': "The content includes specialized technical terminology",
        'question_types': {
            'what': "The query requests specific information about the topic",
            'how': "The query requests a procedure or explanation",
            'why': "The query seeks a reason or causal explanation",
            'where': "The query requests information about a location",
            'when': "The query requests temporal information",
            'which': "The query requests selection from options",
            'who': "The query requests information about a person",
        },
        'question_default': "The query requests specific information about the topic",
        'meta_patterns': [
            re.compile(r'^(el usuario|the user|el humano|the human)\s*(me\s+)?(saluda|despide|pregunta|pide)', re.IGNORECASE),
            re.compile(r'^(respondo|i respond|contestando|answering)\s*(con|with)', re.IGNORECASE),
            re.compile(r'^(el bot|the bot|asistente|assistant)\s*(responde|answer)', re.IGNORECASE),
            re.compile(r'(debo|i should|debería)\s*(responder|contestar|decir|reply|answer)', re.IGNORECASE),
            re.compile(r'(la intención|the intention)\s*(detectada|es|detect)', re.IGNORECASE),
        ],
    },
    'de': {
        'name': 'German',
        'family': 'germanic',
        'indicators': [' der ', ' die ', ' das ', ' den ', ' dem ',
                       ' ein ', ' eine ', ' ist ', ' sind ', ' haben ',
                       'weil', 'deshalb', 'jedoch', 'außerdem'],
        'connectors': [
            "Erstens, ", "Zudem, ", "Andererseits, ",
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
            'conversational': "Der Text hat einen unterhaltsamen und umgangssprachlichen Ton",
        },
        'type_default': "Der Inhalt präsentiert relevante Informationen",
        'concepts_prefix': "Wichtige erkannte Konzepte:",
        'entities_prefix': "Erwähnte Entitäten:",
        'answer_terms': "Wichtige Begriffe in Verbindung mit der Antwort:",
        'technical_note': "Der Inhalt umfasst spezialisierte Fachterminologie",
        'question_types': {
            'was': "Die Anfrage fordert spezifische Informationen zum Thema an",
            'wie': "Die Anfrage fordert eine Prozedur oder Erklärung an",
            'warum': "Die Anfrage sucht einen Grund oder eine kausale Erklärung",
            'wo': "Die Anfrage fordert Informationen über einen Ort an",
            'wann': "Die Anfrage fordert zeitliche Informationen an",
            'wer': "Die Anfrage fordert Informationen über eine Person an",
        },
        'question_default': "Die Anfrage fordert spezifische Informationen zum Thema an",
        'meta_patterns': [],
    },
    'nl': {
        'name': 'Dutch',
        'family': 'germanic',
        'indicators': [' de ', ' het ', ' een ', ' van ', ' in ',
                       ' is ', ' zijn ', ' hebben ', ' dat ', ' die ',
                       'omdat', 'daarom', 'echter', 'bovendien'],
        'connectors': [
            "Ten eerste, ", "Bovendien, ", "Aan de andere kant, ",
            "Ook, ", "Ten slotte, ", "Samenvattend, ",
        ],
        'overflow_connector': "Bovendien, ",
        'analyze': "Analyseren",
        'about': 'over',
        'content_of': "de inhoud van de tekst",
        'text_contains': "De tekst bevat {s} zinnen met {w} woorden in totaal",
        'type_desc': {
            'qa': "Dit is een vraag die een specifiek antwoord vereist",
            'technical': "De inhoud is technisch en bevat numerieke gegevens of gespecialiseerde terminologie",
            'narrative': "De tekst presenteert een verhaal of beschrijving van gebeurtenissen",
            'instructional': "De inhoud biedt instructies of richtlijnen om te volgen",
            'factual': "Er worden feitelijke en objectieve informatie over het onderwerp gepresenteerd",
            'conversational': "De tekst heeft een gesprekston en een alledaags taalgebruik",
        },
        'type_default': "De inhoud presenteert relevante informatie",
        'concepts_prefix': "Belangrijke geïdentificeerde concepten:",
        'entities_prefix': "Genoemde entiteiten:",
        'answer_terms': "Belangrijke termen in verband met het antwoord:",
        'technical_note': "De inhoud bevat gespecialiseerde technische terminologie",
        'question_types': {
            'wat': "Het verzoek vraagt om specifieke informatie over het onderwerp",
            'hoe': "Het verzoek vraagt om een procedure of uitleg",
            'waarom': "Het verzoek zoekt een reden of causale uitleg",
            'waar': "Het verzoek vraagt om informatie over een locatie",
            'wanneer': "Het verzoek vraagt om temporele informatie",
            'wie': "Het verzoek vraagt om informatie over een persoon",
        },
        'question_default': "Het verzoek vraagt om specifieke informatie over het onderwerp",
        'meta_patterns': [],
    },
    'sv': {
        'name': 'Swedish',
        'family': 'germanic',
        'indicators': [' den ', ' det ', ' en ', ' ett ', ' av ',
                       ' i ', ' är ', ' har ', ' som ', ' och ',
                       'eftersom', 'därför', 'emellertid', 'vidare'],
        'connectors': [
            "För det första, ", "Vidare, ", "Å andra sidan, ",
            "Även, ", "Slutligen, ", "Sammanfattningsvis, ",
        ],
        'overflow_connector': "Vidare, ",
        'analyze': "Analyserar",
        'about': 'om',
        'content_of': "textens innehåll",
        'text_contains': "Texten innehåller {s} meningar med {w} ord totalt",
        'type_desc': {
            'qa': "Detta är en fråga som kräver ett specifikt svar",
            'technical': "Innehållet är tekniskt och innehåller numeriska data eller specialiserad terminologi",
            'narrative': "Texten presenterar en berättelse eller beskrivning av händelser",
            'instructional': "Innehållet ger instruktioner eller riktlinjer att följa",
            'factual': "Faktabaserad och objektiv information presenteras om ämnet",
            'conversational': "Texten har en konversations ton och vardagligt språkbruk",
        },
        'type_default': "Innehållet presenterar relevant information",
        'concepts_prefix': "Viktiga identifierade koncept:",
        'entities_prefix': "Nämnda entiteter:",
        'answer_terms': "Viktiga termer kopplade till svaret:",
        'technical_note': "Innehållet inkluderar specialiserad teknisk terminologi",
        'question_types': {
            'vad': "Förfrågan begär specifik information om ämnet",
            'hur': "Förfrågan begär en procedur eller förklaring",
            'varför': "Förfrågan söker en anledning eller orsaksförklaring",
            'var': "Förfrågan begär information om en plats",
            'när': "Förfrågan begär tidsmässig information",
            'vem': "Förfrågan begär information om en person",
        },
        'question_default': "Förfrågan begär specifik information om ämnet",
        'meta_patterns': [],
    },
    'da': {
        'name': 'Danish',
        'family': 'germanic',
        'indicators': [' den ', ' det ', ' en ', ' et ', ' af ',
                       ' i ', ' er ', ' har ', ' som ', ' og ',
                       'fordi', 'derfor', 'imidlertid', 'desuden'],
        'connectors': [
            "For det første, ", "Desuden, ", "På den anden side, ",
            "Også, ", "Til sidst, ", "Sammenfattende, ",
        ],
        'overflow_connector': "Desuden, ",
        'analyze': "Analiserer",
        'about': 'om',
        'content_of': "tekstens indhold",
        'text_contains': "Teksten indeholder {s} sætninger med {w} ord i alt",
        'type_desc': {
            'qa': "Dette er et spørgsmål, der kræver et specifikt svar",
            'technical': "Indholdet er teknisk og indeholder numeriske data eller specialiseret terminologi",
            'narrative': "Teksten præsenterer en fortælling eller beskrivelse af begivenheder",
            'instructional': "Indholdet giver instruktioner eller retningslinjer at følge",
            'factual': "Faktuelle og objektive informationer præsenteres om emnet",
            'conversational': "Teksten har en samtale-ton og hverdagsligt sprogbrug",
        },
        'type_default': "Indholdet præsenterer relevant information",
        'concepts_prefix': "Vigtige identificerede koncepter:",
        'entities_prefix': "Nævnte entiteter:",
        'answer_terms': "Vigtige termer forbundet med svaret:",
        'technical_note': "Indholdet inkluderer specialiseret teknisk terminologi",
        'question_types': {
            'hvilket': "Forespørgslen anmoder om specifik information om emnet",
            'hvordan': "Forespørgslen anmoder om en procedure eller forklaring",
            'hvorfor': "Forespørgslen søger en årsag eller kausal forklaring",
            'hvor': "Forespørgslen anmoder om information om et sted",
            'hvornår': "Forespørgslen anmoder om tidsmæssig information",
            'hvem': "Forespørgslen anmoder om information om en person",
        },
        'question_default': "Forespørgslen anmoder om specifik information om emnet",
        'meta_patterns': [],
    },
    'nb': {
        'name': 'Norwegian Bokmål',
        'family': 'germanic',
        'indicators': [' den ', ' det ', ' en ', ' et ', ' av ',
                       ' i ', ' er ', ' har ', ' som ', ' og ',
                       'fordi', 'derfor', 'imidlertid', 'videre'],
        'connectors': [
            "For det første, ", "Videre, ", "På den annen side, ",
            "Også, ", "Til slutt, ", "Oppsummert, ",
        ],
        'overflow_connector': "Videre, ",
        'analyze': "Analyserer",
        'about': 'om',
        'content_of': "tekstens innhold",
        'text_contains': "Teksten inneholder {s} setninger med {w} ord totalt",
        'type_desc': {
            'qa': "Dette er et spørsmål som krever et spesifikt svar",
            'technical': "Innholdet er teknisk og inneholder numeriske data eller spesialisert terminologi",
            'narrative': "Teksten presenterer en fortelling eller beskrivelse av hendelser",
            'instructional': "Innholdet gir instruksjoner eller retningslinjer å følge",
            'factual': "Faktiske og objektive informasjoner presenteres om emnet",
            'conversational': "Teksten har en samtale-ton og hverdagslig språkbruk",
        },
        'type_default': "Innholdet presenterer relevant informasjon",
        'concepts_prefix': "Viktige identifiserte konsepter:",
        'entities_prefix': "Nevnte entiteter:",
        'answer_terms': "Viktige termer knyttet til svaret:",
        'technical_note': "Innholdet inkluderer spesialisert teknisk terminologi",
        'question_types': {
            'hva': "Forespørselen ber om spesifikk informasjon om emnet",
            'hvordan': "Forespørselen ber om en prosedyre eller forklaring",
            'hvorfor': "Forespørselen søker en årsak eller årsakssammenheng",
            'hvor': "Forespørselen ber om informasjon om et sted",
            'når': "Forespørselen ber om tidsmessig informasjon",
            'hvem': "Forespørselen ber om informasjon om en person",
        },
        'question_default': "Forespørselen ber om spesifikk informasjon om emnet",
        'meta_patterns': [],
    },
    'is': {
        'name': 'Icelandic',
        'family': 'germanic',
        'indicators': [' þessi ', ' þetta ', ' einn ', ' eitt ', ' af ',
                       ' í ', ' er ', ' hefur ', ' sem ', ' og ',
                       'vegna þess', 'þess vegna', 'hins vegar', 'ennfremur'],
        'connectors': [
            "Í fyrsta lagi, ", "Ennfremur, ", "Hins vegar, ",
            "Einnig, ", "Að lokum, ", "Sammanlega, ",
        ],
        'overflow_connector': "Ennfremur, ",
        'analyze': "Greini",
        'about': 'um',
        'content_of': "efnið í textanum",
        'text_contains': "Textinn inniheldur {s} setningar með {w} orðum samtals",
        'type_desc': {
            'qa': "Þetta er spurning sem krefst svars",
            'technical': "Efnið er tæknilegt og inniheldur töluleg gögn eða sértæka orðaforða",
            'narrative': "Textinn kynnir frásögn eða lýsingu atburða",
            'instructional': "Efnið gefur leiðbeiningar eða reglur til að fylgja",
            'factual': "Staðreyndabundnar og hlutlægar upplýsingar eru kynntar um efnið",
            'conversational': "Textinn hefur samtölutón og daglegt mál",
        },
        'type_default': "Efnið kynnir viðeigandi upplýsingar",
        'concepts_prefix': "Mikilvæg greind hugtök:",
        'entities_prefix': "Taldir einstaklingar:",
        'answer_terms': "Mikilvæg orð tengd svarinu:",
        'technical_note': "Efnið inniheldur sértæka tæknilega orðaforða",
        'question_types': {
            'hvað': "Fyrirspurnin beið um sértækar upplýsingar um efnið",
            'hvernig': "Fyrirspurnin beið um aðferð eða skýringu",
            'af hverju': "Fyrirspurnin leitar að ástæðu eða orsakaskýringu",
            'hvar': "Fyrirspurnin beið um upplýsingar um staðsetningu",
            'hvenær': "Fyrirspurnin beið um tímasettar upplýsingar",
            'hver': "Fyrirspurnin beið um upplýsingar um einstakling",
        },
        'question_default': "Fyrirspurnin beið um sértækar upplýsingar um efnið",
        'meta_patterns': [],
    },
    'lb': {
        'name': 'Luxembourgish',
        'family': 'germanic',
        'indicators': [' de ', ' d\'', ' een ', ' eng ', ' vun ',
                       ' an ', ' ass ', ' hunn ', ' dat ', ' an ',
                       'well', 'dofir', 'awer', 'dernieft'],
        'connectors': [
            "Éischtens, ", "Dernieft, ", "Op der anerer Säit, ",
            "Och, ", "Schlussendlech, ", "Zesummefaassend, ",
        ],
        'overflow_connector': "Dernieft, ",
        'analyze': "Analyséieren",
        'about': 'iwwer',
        'content_of': "den Inhalt vum Text",
        'text_contains': "Den Text enthält {s} Sätz mat {w} Wierder insgesamt",
        'type_desc': {
            'qa': "Dëst eng Fro déi eng spezifesch Äntwert erfuerdert",
            'technical': "Den Inhalt ass technesch a enthält numeresch Donnéeën oder spezialiséiert Terminologie",
            'narrative': "De Text presentéiert eng Erzielung oder Beschreiwung vun Evenementer",
            'instructional': "Den Inhalt gëtt Uweisunge oder Richtlinnen ze verfollegen",
            'factual': "Faktesch an objektiv Informatiounen iwwert d'Thema presentéiert",
            'conversational': "De Text huet en Ënnerhältungston an alldeeglech Sproochgebrauch",
        },
        'type_default': "Den Inhalt presentéiert relevant Informatiounen",
        'concepts_prefix': "Wichteg erkannt Konzepter:",
        'entities_prefix': "Erwähnt Entitéiten:",
        'answer_terms': "Wichteg Begrëffer verbonnen mat der Äntwert:",
        'technical_note': "Den Inhalt ëmfaasst spezialiséiert technesch Terminologie",
        'question_types': {
            'wat': "D'Ufro freet spezifesch Informatiounen iwwert d'Thema",
            'wéi': "D'Ufro freet eng Prozedur oder Erklärung",
            'firwat': "D'Ufro sicht eng Ursaach oder kausal Erklärung",
            'wo': "D'Ufro freet Informatiounen iwwert een Terrain",
            'wéini': "D'Ufro freet zäitlech Informatiounen",
            'wee': "D'Ufro freet Informatiounen iwwert eng Persoun",
        },
        'question_default': "D'Ufro freet spezifesch Informatiounen iwwert d'Thema",
        'meta_patterns': [],
    },

    # ── Slavic languages ───────────────────────────────────────────────────────
    'pl': {
        'name': 'Polish',
        'family': 'slavic',
        'indicators': [' ten ', ' ta ', ' to ', ' tych ', ' z ',
                       ' w ', ' jest ', ' są ', ' ma ', ' i ',
                       'ponieważ', 'dlatego', 'jednak', 'ponadto'],
        'connectors': [
            "Po pierwsze, ", "Ponadto, ", "Z drugiej strony, ",
            "Również, ", "Wreszcie, ", "Podsumowując, ",
        ],
        'overflow_connector': "Ponadto, ",
        'analyze': "Analizując",
        'about': 'o',
        'content_of': "zawartość tekstu",
        'text_contains': "Tekst zawiera {s} zdań z {w} słowami łącznie",
        'type_desc': {
            'qa': "Jest to pytanie wymagające konkretnej odpowiedzi",
            'technical': "Zawartość jest techniczna i zawiera dane liczbowe lub specjalistyczną terminologię",
            'narrative': "Tekst przedstawia narrację lub opis wydarzeń",
            'instructional': "Zawartość zawiera instrukcje lub wytyczne do przestrzegania",
            'factual': "Przedstawiono informacje faktualne i obiektywne na temat",
            'conversational': "Tekst ma ton konwersacyjny i potoczny",
        },
        'type_default': "Zawartość przedstawia istotne informacje",
        'concepts_prefix': "Zidentyfikowane kluczowe pojęcia:",
        'entities_prefix': "Wspomniane jednostki:",
        'answer_terms': "Kluczowe powiązane z odpowiedzią:",
        'technical_note': "Zawartość obejmuje specjalistyczną terminologię techniczną",
        'question_types': {
            'co': "Zapytanie prosi o konkretne informacje na temat",
            'jak': "Zapytanie prosi o procedurę lub wyjaśnienie",
            'dlaczego': "Zapytanie szuka powodu lub wyjaśnienia przyczynowego",
            'gdzie': "Zapytanie prosi o informacje o lokalizacji",
            'kiedy': "Zapytanie prosi o informacje czasowe",
            'ile': "Zapytanie prosi o informacje ilościowe",
        },
        'question_default': "Zapytanie prosi o konkretne informacje na temat",
        'meta_patterns': [],
    },
    'cs': {
        'name': 'Czech',
        'family': 'slavic',
        'indicators': [' tento ', ' tato ', ' toto ', ' těch ', ' z ',
                       ' v ', ' je ', ' jsou ', ' má ', ' a ',
                       'protože', 'proto', 'avšak', 'kromě toho'],
        'connectors': [
            "Za prvé, ", "Kromě toho, ", "Na druhou stranu, ",
            "Také, ", "Nakonec, ", "Shrnutím, ",
        ],
        'overflow_connector': "Kromě toho, ",
        'analyze': "Analyzuji",
        'about': 'o',
        'content_of': "obsah textu",
        'text_contains': "Text obsahuje {s} vět s {w} slovy celkem",
        'type_desc': {
            'qa': "Jde o otázku, která vyžaduje konkrétní odpověď",
            'technical': "Obsah je technický a obsahuje numerická data nebo specializovanou terminologii",
            'narrative': "Text předkládá vyprávění nebo popis událostí",
            'instructional': "Obsah poskytuje instrukce nebo pokyny k dodržování",
            'factual': "Prezentují se faktické a objektivní informace o tématu",
            'conversational': "Text má konverzační a hovorový tón",
        },
        'type_default': "Obsah předkládá relevantní informace",
        'concepts_prefix': "Identifikované klíčové koncepty:",
        'entities_prefix': "Zmíněné entity:",
        'answer_terms': "Klíčové pojmy spojené s odpovědí:",
        'technical_note': "Obsah zahrnuje specializovanou technickou terminologii",
        'question_types': {
            'co': "Dotaz žádá o konkrétní informace o tématu",
            'jak': "Dotaz žádá o postup nebo vysvětlení",
            'proč': "Dotaz hledá důvod nebo kauzální vysvětlení",
            'kde': "Dotaz žádá o informace o umístění",
            'kdy': "Dotaz žádá o časové informace",
            'kolik': "Dotaz žádá o kvantitativní informace",
        },
        'question_default': "Dotaz žádá o konkrétní informace o tématu",
        'meta_patterns': [],
    },
    'sk': {
        'name': 'Slovak',
        'family': 'slavic',
        'indicators': [' tento ', ' táto ', ' toto ', ' tých ', ' z ',
                       ' v ', ' je ', ' sú ', ' má ', ' a ',
                       'pretože', 'preto', 'avšak', 'okrem toho'],
        'connectors': [
            "Po prvé, ", "Okrem toho, ", "Na druhú stranu, ",
            "Tiež, ", "Nakoniec, ", "Zhrnutím, ",
        ],
        'overflow_connector': "Okrem toho, ",
        'analyze': "Analyzujem",
        'about': 'o',
        'content_of': "obsah textu",
        'text_contains': "Text obsahuje {s} viet s {w} slovami celkovo",
        'type_desc': {
            'qa': "Ide o otázku, ktorá vyžaduje konkrétnu odpoveď",
            'technical': "Obsah je technický a obsahuje numerické dáta alebo špecializovanú terminológiu",
            'narrative': "Text predkladá rozprávanie alebo popis udalostí",
            'instructional': "Obsah poskytuje inštrukcie alebo pokyny na dodržiavanie",
            'factual': "Prezentujú sa faktické a objektívne informácie o téme",
            'conversational': "Text má konverzačný a hovorový tón",
        },
        'type_default': "Obsah predkladá relevantné informácie",
        'concepts_prefix': "Identifikované kľúčové koncepty:",
        'entities_prefix': "Spomínané entity:",
        'answer_terms': "Kľúčové pojmy spojené s odpoveďou:",
        'technical_note': "Obsah zahŕňa špecializovanú technickú terminológiu",
        'question_types': {
            'čo': "Otázka žiada o konkrétne informácie o téme",
            'ako': "Otázka žiada o postup alebo vysvetlenie",
            'prečo': "Otázka hľadá dôvod alebo kauzálne vysvetlenie",
            'kde': "Otázka žiada o informácie o umiestnení",
            'kedy': "Otázka žiada o časové informácie",
            'koľko': "Otázka žiada o kvantitatívne informácie",
        },
        'question_default': "Otázka žiada o konkrétne informácie o téme",
        'meta_patterns': [],
    },
    'bg': {
        'name': 'Bulgarian',
        'family': 'slavic',
        'indicators': [' този ', ' тази ', ' това ', ' тези ', ' от ',
                       ' в ', ' е ', ' са ', ' има ', ' и ',
                       'защото', 'затова', 'обаче', 'освен това'],
        'connectors': [
            "Първо, ", "Освен това, ", "От друга страна, ",
            "Също, ", "Накрая, ", "В обобщение, ",
        ],
        'overflow_connector': "Освен това, ",
        'analyze': "Анализирайки",
        'about': 'за',
        'content_of': "съдържанието на текста",
        'text_contains': "Текстът съдържа {s} изречения с {w} думи общо",
        'type_desc': {
            'qa': "Това е въпрос, който изисква конкретен отговор",
            'technical': "Съдържанието е техническо и съдържа числови данни или специализирана терминология",
            'narrative': "Текстът представя разказ или описание на събития",
            'instructional': "Съдържанието предоставя инструкции или насоки за следване",
            'factual': "Представени са фактологични и обективна информация за темата",
            'conversational': "Текстът има разговорен и неформален тон",
        },
        'type_default': "Съдържанието представя подходяща информация",
        'concepts_prefix': "Идентифицирани ключови концепции:",
        'entities_prefix': "Споменати субекти:",
        'answer_terms': "Ключови думи свързани с отговора:",
        'technical_note': "Съдържанието включва специализирана техническа терминология",
        'question_types': {
            'какво': "Заявката иска конкретна информация за темата",
            'как': "Заявката иска процедура или обяснение",
            'защо': "Заявката търси причина или причинно обяснение",
            'къде': "Заявката иска информация за местоположение",
            'кога': "Заявката иска времева информация",
            'колко': "Заявката иска количествена информация",
        },
        'question_default': "Заявката иска конкретна информация за темата",
        'meta_patterns': [],
    },
    'hr': {
        'name': 'Croatian',
        'family': 'slavic',
        'indicators': [' ovaj ', ' ova ', ' ovo ', ' ovih ', ' od ',
                       ' u ', ' je ', ' su ', ' ima ', ' i ',
                       'jer', 'stoga', 'međutim', 'osim toga'],
        'connectors': [
            "Prvo, ", "Osim toga, ", "S druge strane, ",
            "Također, ", "Na kraju, ", "Ukratko, ",
        ],
        'overflow_connector': "Osim toga, ",
        'analyze': "Analizirajući",
        'about': 'o',
        'content_of': "sadržaj teksta",
        'text_contains': "Tekst sadrži {s} rečenica s {w} riječi ukupno",
        'type_desc': {
            'qa': "Ovo je pitanje koje zahtijeva specifičan odgovor",
            'technical': "Sadržaj je tehnički i sadrži numeričke podatke ili specijaliziranu terminologiju",
            'narrative': "Tekst predstavlja pripovijest ili opis događaja",
            'instructional': "Sadržaj pruža upute ili smjernice za praćenje",
            'factual': "Predstavljene su činjenične i objektivne informacije o temi",
            'conversational': "Tekst ima konverzacionalni i kolokvijalni ton",
        },
        'type_default': "Sadržaj predstavlja relevantne informacije",
        'concepts_prefix': "Identificirani ključni koncepti:",
        'entities_prefix': "Spomenuti entiteti:",
        'answer_terms': "Ključne riječi povezane s odgovorom:",
        'technical_note': "Sadržaj uključuje specijaliziranu tehničku terminologiju",
        'question_types': {
            'što': "Upit traži specifične informacije o temi",
            'kako': "Upit traži postupak ili objašnjenje",
            'zašto': "Upit traži razlog ili uzročno objašnjenje",
            'gdje': "Upit traži informacije o lokaciji",
            'kada': "Upit traži vremenske informacije",
            'koliko': "Upit traži količinske informacije",
        },
        'question_default': "Upit traži specifične informacije o temi",
        'meta_patterns': [],
    },
    'sr': {
        'name': 'Serbian',
        'family': 'slavic',
        'indicators': [' овај ', ' ова ', ' ово ', ' ових ', ' од ',
                       ' у ', ' је ', ' су ', ' има ', ' и ',
                       'јер', 'стога', 'међутим', 'осим тога'],
        'connectors': [
            "Прво, ", "Осим тога, ", "С друге стране, ",
            "Такође, ", "На крају, ", "Укратко, ",
        ],
        'overflow_connector': "Осим тога, ",
        'analyze': "Анализирајући",
        'about': 'о',
        'content_of': "садржај текста",
        'text_contains': "Текст садржи {s} реченица са {w} речи укупно",
        'type_desc': {
            'qa': "Ово је питање које захтева специфичан одговор",
            'technical': "Садржај је технички и садржи нумеричке податке или специјализовану терминологију",
            'narrative': "Текст представља причу или опис догађаја",
            'instructional': "Садржај пружа упутства или смернице за праћење",
            'factual': "Представљене су чињеничне и објективне информације о теми",
            'conversational': "Текст има конверзацијски и колоквијални тон",
        },
        'type_default': "Садржај представља релевантне информације",
        'concepts_prefix': "Идентификовани кључни концепти:",
        'entities_prefix': "Поменути ентитети:",
        'answer_terms': "Кључне речи повезане са одговором:",
        'technical_note': "Садржај укључује специјализовану техничку терминологију",
        'question_types': {
            'шта': "Упит тражи специфичне информације о теми",
            'како': "Упит тражи поступак или објашњење",
            'зашто': "Упит тражи разлог или узрочно објашњење",
            'где': "Упит тражи информације о локацији",
            'када': "Упит тражи временске информације",
            'колико': "Упит тражи квантитативне информације",
        },
        'question_default': "Упит тражи специфичне информације о теми",
        'meta_patterns': [],
    },
    'sl': {
        'name': 'Slovenian',
        'family': 'slavic',
        'indicators': [' ta ', ' ta ', ' to ', ' teh ', ' od ',
                       ' v ', ' je ', ' so ', ' ima ', ' in ',
                       'ker', 'zato', 'vendar', 'poleg tega'],
        'connectors': [
            "Najprej, ", "Poleg tega, ", "Po drugi strani, ",
            "Tudi, ", "Nazadnje, ", "V povzetku, ",
        ],
        'overflow_connector': "Poleg tega, ",
        'analyze': "Analiziram",
        'about': 'o',
        'content_of': "vsebino besedila",
        'text_contains': "Besedilo vsebuje {s} povedi z {w} besedami skupaj",
        'type_desc': {
            'qa': "To je vprašanje, ki zahteva specifičen odgovor",
            'technical': "Vsebina je tehnična in vsebuje numerične podatke ali specializirano terminologijo",
            'narrative': "Besedilo predstavlja pripoved ali opis dogodkov",
            'instructional': "Vsebina zagotavlja navodila ali smernice za sledenje",
            'factual': "Predstavljena so dejstva in objektivne informacije o temi",
            'conversational': "Besedilo ima pogovorni in neformalni ton",
        },
        'type_default': "Vsebina predstavlja relevantne informacije",
        'concepts_prefix': "Prepoznani ključni koncepti:",
        'entities_prefix': "Omenjeni entiteti:",
        'answer_terms': "Ključne besede povezane z odgovorom:",
        'technical_note': "Vsebina vključuje specializirano tehnično terminologijo",
        'question_types': {
            'kaj': "Povpraševanje zahteva specifične informacije o temi",
            'kako': "Povpraševanje zahteva postopek ali pojasnilo",
            'zakaj': "Povpraševanje išče razlog ali vzročno pojasnilo",
            'kje': "Povpraševanje zahteva informacije o lokaciji",
            'kdaj': "Povpraševanje zahteva časovne informacije",
            'koliko': "Povpraševanje zahteva količinske informacije",
        },
        'question_default': "Povpraševanje zahteva specifične informacije o temi",
        'meta_patterns': [],
    },
    'bs': {
        'name': 'Bosnian',
        'family': 'slavic',
        'indicators': [' ovaj ', ' ova ', ' ovo ', ' ovih ', ' od ',
                       ' u ', ' je ', ' su ', ' ima ', ' i ',
                       'jer', 'stoga', 'međutim', 'osim toga'],
        'connectors': [
            "Prvo, ", "Osim toga, ", "S druge strane, ",
            "Također, ", "Na kraju, ", "Ukratko, ",
        ],
        'overflow_connector': "Osim toga, ",
        'analyze': "Analizirajući",
        'about': 'o',
        'content_of': "sadržaj teksta",
        'text_contains': "Tekst sadrži {s} rečenica sa {w} riječi ukupno",
        'type_desc': {
            'qa': "Ovo je pitanje koje zahtijeva specifičan odgovor",
            'technical': "Sadržaj je tehnički i sadrži numeričke podatke ili specijaliziranu terminologiju",
            'narrative': "Tekst predstavlja priču ili opis događaja",
            'instructional': "Sadržaj pruža upute ili smjernice za praćenje",
            'factual': "Predstavljene su činjenične i objektivne informacije o temi",
            'conversational': "Tekst ima konverzacionalni i kolokvijalni ton",
        },
        'type_default': "Sadržaj predstavlja relevantne informacije",
        'concepts_prefix': "Identificirani ključni koncepti:",
        'entities_prefix': "Spomenuti entiteti:",
        'answer_terms': "Ključne riječi povezane sa odgovorom:",
        'technical_note': "Sadržaj uključuje specijaliziranu tehničku terminologiju",
        'question_types': {
            'što': "Upit traži specifične informacije o temi",
            'kako': "Upit traži postupak ili objašnjenje",
            'zašto': "Upit traži razlog ili uzročno objašnjenje",
            'gdje': "Upit traži informacije o lokaciji",
            'kada': "Upit traži vremenske informacije",
            'koliko': "Upit traži količinske informacije",
        },
        'question_default': "Upit traži specifične informacije o temi",
        'meta_patterns': [],
    },
    'mk': {
        'name': 'Macedonian',
        'family': 'slavic',
        'indicators': [' овој ', ' оваа ', ' ова ', ' овие ', ' од ',
                       ' во ', ' е ', ' се ', ' има ', ' и ',
                       'затоа', 'сепак', 'покрај тоа', 'исто така'],
        'connectors': [
            "Прво, ", "Покрај тоа, ", "Од друга страна, ",
            "Исто така, ", "На крајот, ", "Во заклучок, ",
        ],
        'overflow_connector': "Покрај тоа, ",
        'analyze': "Анализирајќи",
        'about': 'за',
        'content_of': "содржината на текстот",
        'text_contains': "Текстот содржи {s} реченици со {w} зборови вкупно",
        'type_desc': {
            'qa': "Ова е прашање кое бара специфичен одговор",
            'technical': "Содржината е техничка и содржи нумерички податоци или специјализирана терминологија",
            'narrative': "Текстот претставува расказ или опис на настани",
            'instructional': "Содржината обезбедува упатства или насоки за следење",
            'factual': "Претставени се фактички и објективни информации за темата",
            'conversational': "Текстот има разговорен и колоквијален тон",
        },
        'type_default': "Содржината претставува релевантни информации",
        'concepts_prefix': "Идентифицирани клучни концепти:",
        'entities_prefix': "Споменати ентитети:",
        'answer_terms': "Клучни зборови поврзани со одговорот:",
        'technical_note': "Содржината вклучува специјализирана техничка терминологија",
        'question_types': {
            'што': "Прашањето бара специфични информации за темата",
            'како': "Прашањето бара постапка или објаснување",
            'зошто': "Прашањето бара причина или причинско објаснување",
            'каде': "Прашањето бара информации за локација",
            'кога': "Прашањето бара временски информации",
            'колку': "Прашањето бара квантитативни информации",
        },
        'question_default': "Прашањето бара специфични информации за темата",
        'meta_patterns': [],
    },
    'cnr': {
        'name': 'Montenegrin',
        'family': 'slavic',
        'indicators': [' овај ', ' ова ', ' ово ', ' ових ', ' од ',
                       ' у ', ' је ', ' су ', ' има ', ' и ',
                       'јер', 'стога', 'међутим', 'осим тога'],
        'connectors': [
            "Прво, ", "Осим тога, ", "С друге стране, ",
            "Такође, ", "На крају, ", "Укратко, ",
        ],
        'overflow_connector': "Осим тога, ",
        'analyze': "Анализирајући",
        'about': 'о',
        'content_of': "садржај текста",
        'text_contains': "Текст садржи {s} реченица са {w} речи укупно",
        'type_desc': {
            'qa': "Ово је питање које захтева специфичан одговор",
            'technical': "Садржај је технички и садржи нумеричке податке или специјализовану терминологију",
            'narrative': "Текст представља причу или опис догађаја",
            'instructional': "Садржај пружа упутства или смернице за праћење",
            'factual': "Представљене су чињеничне и објективне информације о теми",
            'conversational': "Текст има конверзацијски и колоквијални тон",
        },
        'type_default': "Садржај представља релевантне информације",
        'concepts_prefix': "Идентификовани кључни концепти:",
        'entities_prefix': "Поменути ентитети:",
        'answer_terms': "Кључне речи повезане са одговором:",
        'technical_note': "Садржај укључује специјализовану техничку терминологију",
        'question_types': {
            'што': "Упит тражи специфичне информације о теми",
            'како': "Упит тражи поступак или објашњење",
            'зашто': "Упит тражи разлог или узрочно објашњење",
            'где': "Упит тражи информације о локацији",
            'када': "Упит тражи временске информације",
            'колико': "Упит тражи квантитативне информације",
        },
        'question_default': "Упит тражи специфичне информације о теми",
        'meta_patterns': [],
    },

    # ── Baltic languages ───────────────────────────────────────────────────────
    'lt': {
        'name': 'Lithuanian',
        'family': 'baltic',
        'indicators': [' šis ', ' ši ', ' šie ', ' šios ', ' iš ',
                       ' yra ', ' buvo ', ' turi ', ' ir ', ' kad ',
                       'nes', 'todėl', 'tačiau', 'be to'],
        'connectors': [
            "Pirma, ", "Be to, ", "Kita vertus, ",
            "Taip pat, ", "Galiausiai, ", "Apibendrinant, ",
        ],
        'overflow_connector': "Be to, ",
        'analyze': "Analizuojant",
        'about': 'apie',
        'content_of': "teksto turinį",
        'text_contains': "Tekste yra {s} sakinių su {w} žodžiais iš viso",
        'type_desc': {
            'qa': "Tai klausimas, reikalaujantis konkretaus atsakymo",
            'technical': "Turinys yra techninis ir jose yra skaitmeniniai duomenys arba specializuota terminologija",
            'narrative': "Tekstas pristato pasakojimą arba įvykių aprašymą",
            'instructional': "Turinys pateikia instrukcijas ar gaires laikytis",
            'factual': "Pateikiami faktiniai ir objektyvūs duomenys apie temą",
            'conversational': "Tekstas turi pokalbio ir kasdieninį toną",
        },
        'type_default': "Turinys pristato svarbią informaciją",
        'concepts_prefix': "Identifikuoti pagrindiniai konceptai:",
        'entities_prefix': "Paminėtos entitetės:",
        'answer_terms': "Pagrindiniai susiję su atsakymu:",
        'technical_note': "Turinys apima specializuotą techninę terminologiją",
        'question_types': {
            'kas': "Užklausa prašo konkrečios informacijos apie temą",
            'kaip': "Užklausa prašo procedūros ar paaiškinimo",
            'kodėl': "Užklausa ieško priežasties ar priežastinio paaiškinimo",
            'kur': "Užklausa prašo informacijos apie vietą",
            'kada': "Užklausa prašo laiko informacijos",
            'kiek': "Užklausa prašo kiekybinės informacijos",
        },
        'question_default': "Užklausa prašo konkrečios informacijos apie temą",
        'meta_patterns': [],
    },
    'lv': {
        'name': 'Latvian',
        'family': 'baltic',
        'indicators': [' šis ', ' šī ', ' šie ', ' šīs ', ' no ',
                       ' ir ', ' bija ', ' ir ', ' un ', ' ka ',
                       'jo', 'tāpēc', 'tomēr', 'turklāt'],
        'connectors': [
            "Pirmkārt, ", "Turklāt, ", "No otras puses, ",
            "Arī, ", "Visbeidzot, ", "Kopsavilkumā, ",
        ],
        'overflow_connector': "Turklāt, ",
        'analyze': "Analizējot",
        'about': 'par',
        'content_of': "teksta saturu",
        'text_contains': "Tekstā ir {s} teikumi ar {vārdiem} vārdiem kopā",
        'type_desc': {
            'qa': "Šis ir jautājums, kam nepieciešama konkrēta atbilde",
            'technical': "Saturs ir tehnisks un satur ciparus datus vai specializētu terminoloģiju",
            'narrative': "Teksts sniedz stāstu vai notikumu aprakstu",
            'instructional': "Saturs sniedz instrukcijas vai vadlīnijas, ko ievērot",
            'factual': "Tiek sniegta faktiska un objektīva informācija par tēmu",
            'conversational': "Tekstam ir sarunvalodas un ikdienišķs tonis",
        },
        'type_default': "Saturs sniedz atbilstošu informāciju",
        'concepts_prefix': "Identificētie galvenie koncepti:",
        'entities_prefix': "Minētās entītes:",
        'answer_terms': "Galvenie vārdi, kas saistīti ar atbildi:",
        'technical_note': "Saturs ietver specializētu tehnikas terminoloģiju",
        'question_types': {
            'kas': "Pieprasījums lūdz konkrētu informāciju par tēmu",
            'kā': "Pieprasījums lūdz procedūru vai paskaidrojumu",
            'kāpēc': "Pieprasījums meklē iemeslu vai cēloņu paskaidrojumu",
            'kur': "Pieprasījums lūdz informāciju par atrašanās vietu",
            'kad': "Pieprasījums lūdz laika informāciju",
            'cik': "Pieprasījums lūdz kvantitatīvu informāciju",
        },
        'question_default': "Pieprasījums lūdz konkrētu informāciju par tēmu",
        'meta_patterns': [],
    },

    # ── Finno-Ugric languages ──────────────────────────────────────────────────
    'fi': {
        'name': 'Finnish',
        'family': 'finno-ugric',
        'indicators': [' tämä ', ' tällä ', ' näitä ', ' on ', ' ovat ',
                       ' oli ', ' ja ', ' että ', ' tai ',
                       'koska', 'siksi', 'kuitenkin', 'myös'],
        'connectors': [
            "Ensinnäkin, ", "Myös, ", "Toisaalta, ",
            "Lisäksi, ", "Lopuksi, ", "Yhteenvetona, ",
        ],
        'overflow_connector': "Lisäksi, ",
        'analyze': "Analysoiden",
        'about': 'aiheesta',
        'content_of': "tekstin sisällön",
        'text_contains': "Teksti sisältää {s} lauseetta ja {w} sanaa yhteensä",
        'type_desc': {
            'qa': "Tämä on kysymys, joka vaatii erityisen vastauksen",
            'technical': "Sisältö on teknistä ja sisältää numeerisia tietoja tai erityistä terminologiaa",
            'narrative': "Teksti esittää kertomuksen tai tapahtumien kuvauksen",
            'instructional': "Sisältö tarjoaa ohjeita tai suuntaviivoja noudatettavaksi",
            'factual': "Aiheesta esitetään tosiasiallisia ja objektiivisia tietoja",
            'conversational': "Tekstillä on keskusteleva ja arkikielinen sävy",
        },
        'type_default': "Sisältö esittää relevantteja tietoja",
        'concepts_prefix': "Tunnistetut avainkäsitteet:",
        'entities_prefix': "Mainitut entiteetit:",
        'answer_terms': "Avainsanat, jotka liittyvät vastaukseen:",
        'technical_note': "Sisältö sisältää erityistä teknistä terminologiaa",
        'question_types': {
            'mikä': "Kysely pyytää aiheesta erityisiä tietoja",
            'miten': "Kysely pyytää menettelyä tai selitystä",
            'miksi': "Kysely etsii syyt tai kausaalista selitystä",
            'missä': "Kysely pyytää tietoja sijainnista",
            'milloin': "Kysely pyytää aikatietoja",
            'kuinka': "Kysely pyytää kvantitatiivisia tietoja",
        },
        'question_default': "Kysely pyytää aiheesta erityisiä tietoja",
        'meta_patterns': [],
    },
    'et': {
        'name': 'Estonian',
        'family': 'finno-ugric',
        'indicators': [' see ', ' selle ', ' neid ', ' on ', ' olid ',
                       ' oli ', ' ja ', ' et ',
                       'sellepärast', 'kuid', 'lisaks', 'seetõttu'],
        'connectors': [
            "Esiteks, ", "Lisaks, ", "Teisalt, ",
            "Samuti, ", "Lõpuks, kokkuvõtlikult, ",
        ],
        'overflow_connector': "Lisaks, ",
        'analyze': "Analüüsides",
        'about': 'teemal',
        'content_of': "teksti sisu",
        'text_contains': "Tekst sisaldab {s} lauset ja {w} sõna kokku",
        'type_desc': {
            'qa': "See on küsimus, mis nõuab konkreetset vastust",
            'technical': "Sisu on tehniline ja sisaldab numbrilisi andmeid või spetsialiseeritud terminoloogiat",
            'narrative': "Tekst esitab jutustuse või sündmuste kirjelduse",
            'instructional': "Sisu pakub juhiseid või suuniseid järgimiseks",
            'factual': "Teemast esitatakse faktidel põhinevaid ja objektiivseid andmeid",
            'conversational': "Tekstil on vestluslik ja igapäevane toon",
        },
        'type_default': "Sisu esitab asjakohast teavet",
        'concepts_prefix': "Tuvastatud põhimõisted:",
        'entities_prefix': "Mainitud üksused:",
        'answer_terms': "Põhimõtted, mis on seotud vastusega:",
        'technical_note': "Sisu hõlmab spetsialiseeritud tehnilist terminoloogiat",
        'question_types': {
            'mis': "Päring palub teemal konkreetset teavet",
            'kuidas': "Päring palub protseduuri või selgitust",
            'miks': "Päring otsib põhjust või põhjuslikku selgitust",
            'kus': "Päring palub teavet asukoha kohta",
            'millal': "Päring palub ajalist teavet",
            'mitu': "Päring palub kvantitatiivset teavet",
        },
        'question_default': "Päring palub teemal konkreetset teavet",
        'meta_patterns': [],
    },
    'hu': {
        'name': 'Hungarian',
        'family': 'finno-ugric',
        'indicators': [' ez ', ' ez a ', ' ezt ', ' van ', ' vannak ',
                       ' volt ', ' és ', ' hogy ',
                       'mert', 'ezért', 'azonban', 'emellett'],
        'connectors': [
            "Először is, ", "Emellett, ", "Másrészt, ",
            "Szintén, ", "Végül, ", "Összefoglalva, ",
        ],
        'overflow_connector': "Emellett, ",
        'analyze': "Elemezve",
        'about': 'arról',
        'content_of': "a szöveg tartalmát",
        'text_contains': "A szöveg {s} mondatot tartalmaz, összesen {w} szóval",
        'type_desc': {
            'qa': "Ez egy olyan kérdés, amely konkrét választ igényel",
            'technical': "A tartalom technikai, és számadatokat vagy szakszókincset tartalmaz",
            'narrative': "A szöveg elbeszélést vagy események leírását mutatja be",
            'instructional': "A tartalom utasításokat vagy iránymutatásokat nyújt a követéshez",
            'factual': "A témáról tényeken alapuló és objektív információk jelennek meg",
            'conversational': "A szöveg beszélgetési és köznyelvi hangnemű",
        },
        'type_default': "A tartalom releváns információkat mutat be",
        'concepts_prefix': "Azonosított kulcsfogalmak:",
        'entities_prefix': "Említett entitások:",
        'answer_terms': "Kulcsszavak, amelyek a válasszal kapcsolatosak:",
        'technical_note': "A tartalom szakszerű technikai szókincset tartalmaz",
        'question_types': {
            'mi': "A kérés konkrét információkat kér a témáról",
            'hogyan': "A kérés eljárást vagy magyarázatot kér",
            'miért': "A kérés okot vagy ok-okozati magyarázatot keres",
            'hol': "A kérés információkat kér egy helyről",
            'mikor': "A kérés időbeli információkat kér",
            'mennyi': "A kérés mennyiségi információkat kér",
        },
        'question_default': "A kérés konkrét információkat kér a témáról",
        'meta_patterns': [],
    },

    # ── Celtic languages ───────────────────────────────────────────────────────
    'ga': {
        'name': 'Irish',
        'family': 'celtic',
        'indicators': [' an ', ' na ', ' is ', ' agus ', ' le ',
                       ' ar ', ' tá ', ' bhí ', ' go ',
                       'mar gheall ar', 'dá bhrí sin', 'ach', 'chomh maith le'],
        'connectors': [
            "Ar an gcéad dul síos, ", "Chomh maith le sin, ",
            "Ar an taobh eile, ",
            "Freisin, ", "Sa deireadh, ", "Mar achoimre, ",
        ],
        'overflow_connector': "Chomh maith le sin, ",
        'analyze': "Ag anailísiú",
        'about': 'faoi',
        'content_of': "ábhar an téacs",
        'text_contains': "Tá {s} abairt agus {focal} focal san iomlán sa téacs",
        'type_desc': {
            'qa': "Is ceist é seo a éilíonn freagra ar leith",
            'technical': "Is ábhar teicniúil é agus tá sonraí uimhriúla nó téarmaíocht speisialaithe ann",
            'narrative': "Léiríonn an téacs scéal nó cur síos ar imeachtaí",
            'instructional': "Soláthraíonn an t-ábhar treoracha nó treoirlínte le leanúint",
            'factual': "Cuirtear faisnéis fhíriciúil agus thimpeallach ar fáil faoin ábhar",
            'conversational': "Tá ton comhrá agus coitianta ag an téacs",
        },
        'type_default': "Léiríonn an t-ábhar faisnéis thábhachtach",
        'concepts_prefix': "Coincheapa tábhachtacha aitheanta:",
        'entities_prefix': "Ainmneacha a luaitear:",
        'answer_terms': "Téarmaí tábhachtacha a bhaineann leis an freagra:",
        'technical_note': "Áirítear leis an ábhar téarmaíocht theicniúil speisialaithe",
        'question_types': {
            'cad': "Iarrann an t-iarratas faisnéis shonrach faoin ábhar",
            'conas': "Iarrann an t-iarratas nós imeartha nó míniú",
            'cén fáth': "Lorgaíonn an t-iarratas cúis nó míniú cúiseach",
            'cén áit': "Iarrann an t-iarratas faisnéis faoin áit",
            'cathain': "Iarrann an t-iarratas faisnéis ama",
            'cé mhéad': "Iarrann an t-iarratas faisnéis chainníochtúil",
        },
        'question_default': "Iarrann an t-iarratas faisnéis shonrach faoin ábhar",
        'meta_patterns': [],
    },

    # ── Hellenic ────────────────────────────────────────────────────────────────
    'el': {
        'name': 'Greek',
        'family': 'hellenic',
        'indicators': [' ο ', ' η ', ' το ', ' τα ', ' των ',
                       ' είναι ', ' έχουν ', ' και ', ' που ',
                       'επειδή', 'επομένως', 'ωστόσο', 'επίσης'],
        'connectors': [
            "Πρώτον, ", "Επίσης, ", "Από την άλλη πλευρά, ",
            "Επίσης, ", "Τέλος, ", "Συνολικά, ",
        ],
        'overflow_connector': "Επίσης, ",
        'analyze': "Αναλύοντας",
        'about': 'σχετικά με',
        'content_of': "το περιεχόμενο του κειμένου",
        'text_contains': "Το κείμενο περιέχει {s} προτάσεις με {w} λέξεις συνολικά",
        'type_desc': {
            'qa': "Αυτό είναι μια ερώτηση που απαιτεί συγκεκριμένη απάντηση",
            'technical': "Το περιεχόμενο είναι τεχνικό και περιέχει αριθμητικά δεδομένα ή εξειδικευμένη ορολογία",
            'narrative': "Το κείμενο παρουσιάζει μια αφήγηση ή περιγραφή γεγονότων",
            'instructional': "Το περιεχόμενο παρέχει οδηγίες ή οδηγίες για ακολουθία",
            'factual': "Παρουσιάζονται πραγματικές και αντικειμενικές πληροφορίες για το θέμα",
            'conversational': "Το κείμενο έχει συνομιλιακό και καθημερινό τόνο",
        },
        'type_default': "Το περιεχόμενο παρουσιάζει σχετικές πληροφορίες",
        'concepts_prefix': "Τα αναγνωρισμένα βασικά θέματα:",
        'entities_prefix': "Αναφερόμενες οντότητες:",
        'answer_terms': "Βασικοί όρος που σχετίζονται με την απάντηση:",
        'technical_note': "Το περιεχόμενο περιλαμβάνει εξειδικευμένη τεχνική ορολογία",
        'question_types': {
            'τι': "Το αίτημα ζητάει συγκεκριμένες πληροφορίες για το θέμα",
            'πώς': "Το αίτημα ζητάει μια διαδικασία ή εξήγηση",
            'γιατί': "Το αίτημα ψάχνει για λόγο ή αιτιολογική εξήγηση",
            'πού': "Το αίτημα ζητάει πληροφορίες για τοποθεσία",
            'πότε': "Το αίτημα ζητάει χρονικές πληροφορίες",
            'πόσο': "Το αίτημα ζητάει ποσοτικές πληροφορίες",
        },
        'question_default': "Το αίτημα ζητάει συγκεκριμένες πληροφορίες για το θέμα",
        'meta_patterns': [],
    },

    # ── Albanian ────────────────────────────────────────────────────────────────
    'sq': {
        'name': 'Albanian',
        'family': 'albanian',
        'indicators': [' këtë ', ' këta ', ' këto ', ' të ',
                       ' është ', ' janë ', ' ka ', ' dhe ',
                       'sepse', 'prandaj', 'megjithatë', 'gjithashtu'],
        'connectors': [
            "Së pari, ", "Gjithashtu, ", "Nga ana tjetër, ",
            "Po ashtu, ", "Së fundi, ", "Përmbledhur, ",
        ],
        'overflow_connector': "Gjithashtu, ",
        'analyze': "Duke analizuar",
        'about': 'rreth',
        'content_of': "përmbajtjen e tekstit",
        'text_contains': "Teksti përmban {s} fjali me {fjalë} fjalë gjithsej",
        'type_desc': {
            'qa': "Kjo është një pyetje që kërkon një përgjigje specifike",
            'technical': "Përmbajtja është teknike dhe përmban të dhëna numerike ose terminologji të specializuar",
            'narrative': "Teksti paraqet një rrëfim ose përshkrim të ngjarjeve",
            'instructional': "Përmbajtja ofron udhëzime ose udhëzues për ndjekje",
            'factual': "Paraqiten informacione faktike dhe objektive për temën",
            'conversational': "Teksti ka një ton bisedues dhe të përditshëm",
        },
        'type_default': "Përmbajtja paraqet informacione relevante",
        'concepts_prefix': "Konceptet kryesore të identifikuara:",
        'entities_prefix': "Entitetet e përmendura:",
        'answer_terms': "Termet kryesore të lidhura me përgjigjen:",
        'technical_note': "Përmbajtja përfshin terminologji teknike të specializuar",
        'question_types': {
            'çfarë': "Kërkesa kërkon informacione specifike për temën",
            'si': "Kërkesa kërkon një procedurë ose shpjegim",
            'pse': "Kërkesa kërkon një arsye ose shpjegim shkak-pasojë",
            'ku': "Kërkesa kërkon informacione për një vendndodhje",
            'kur': "Kërkesa kërkon informacione kohore",
            'sa': "Kërkesa kërkon informacione sasiore",
        },
        'question_default': "Kërkesa kërkon informacione specifike për temën",
        'meta_patterns': [],
    },
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

        # Try multilingual spaCy models in priority order
        models_to_try = [
            'xx_ent_wiki_sm',      # Multilingual (best for EU languages)
            'es_core_news_sm',     # Spanish
            'en_core_web_sm',      # English
            'fr_core_news_sm',     # French
            'de_core_news_sm',     # German
            'it_core_news_sm',     # Italian
            'pt_core_news_sm',     # Portuguese
            'pl_core_news_sm',     # Polish
            'nl_core_news_sm',     # Dutch
            'sv_core_news_sm',     # Swedish
            'fi_core_news_sm',     # Finnish
            'el_core_news_sm',     # Greek
            'ro_core_news_sm',     # Romanian
            'hr_core_news_sm',     # Croatian
            'bg_core_news_sm',     # Bulgarian
            'cs_core_news_sm',     # Czech
            'sk_core_news_sm',     # Slovak
            'lt_core_news_sm',     # Lithuanian
            'lv_core_news_sm',     # Latvian
            'et_core_news_sm',     # Estonian
            'hu_core_news_sm',     # Hungarian
            'nb_core_news_sm',     # Norwegian
            'da_core_news_sm',     # Danish
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
        else:
            return self._analyze_text_regex(text)

    def _analyze_text_spacy(self, text: str) -> dict:
        doc = self._nlp(text)
        entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 1]
        noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        word_count = len(text.split())
        sentence_count = len(sentences)
        return {
            'entities': entities,
            'noun_phrases': noun_phrases,
            'sentences': sentences,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': word_count / max(1, sentence_count),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_questions': '?' in text,
            'has_technical_terms': self._detect_technical_terms(text),
            'language': self._detect_language(text),
        }

    def _analyze_text_regex(self, text: str) -> dict:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
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
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(1, len(sentences)),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_questions': '?' in text,
            'has_technical_terms': self._detect_technical_terms(text),
            'language': self._detect_language(text),
        }

    def _detect_technical_terms(self, text: str) -> bool:
        technical_indicators = [
            r'\bAPI\b', r'\bSDK\b', r'\bHTTP\b', r'\bJSON\b', r'\bXML\b',
            r'\bSQL\b', r'\bREST\b', r'\bGraphQL\b', r'\bOAuth\b',
            r'\balgoritmo\b', r'\balgoritmus\b', r'\balgorithm\b', r'\balgorithme\b', r'\balgoritmo\b',
            r'\bfunción\b', r'\bfunkce\b', r'\bfonction\b', r'\bfunctie\b', r'\bfunktsioon\b',
            r'\bfunction\b', r'\bclass\b', r'\bmethod\b', r'\bmodule\b',
            r'\bdatabase\b', r'\bserver\b', r'\bclient\b', r'\bprotocol\b',
            r'\bAPI\b', r'\bHTTP\b', r'\bJSON\b', r'\bXML\b',
        ]
        for pattern in technical_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

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
            candidate_words = candidate.split()
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
        # Multilingual past tense indicators
        past_indicators = [
            'was', 'were', 'had', 'did',  # English
            'era', 'fue', 'tenía', 'hizo',  # Spanish
            'était', 'avait', 'fit',  # French
            'war', 'hatte', 'machte',  # German
            'era', 'aveva', 'fece',  # Italian
            'era', 'tinha', 'fez',  # Portuguese
            'era', 'avea', 'făcut',  # Romanian
            'byl', 'měl', 'udělal',  # Czech
            'bol', 'mal', 'urobil',  # Slovak
            'byl', 'miał', 'zrobił',  # Polish
            'olli', ' oli', 'teki',  # Finnish
            'olid', 'omal', 'tegi',  # Estonian
            'was', 'hadde', 'gjorde',  # Norwegian/Danish
            'var', 'hade', 'gjorde',  # Swedish
            ' היה', 'היה', 'עשה',  # Hebrew (not EU but common)
            'was', 'havde', 'gjorde',  # Danish
        ]
        if any(ind in text_lower for ind in past_indicators):
            return 'narrative'
        # Multilingual imperative indicators
        imperative_indicators = [
            'must', 'should', 'need to',  # English
            'debe', 'debería', 'necesita',  # Spanish
            'doit', 'devrait', 'besoin',  # French
            'muss', 'sollte', 'benötigt',  # German
            'deve', 'dovrebbe', 'necessita',  # Italian
            'deve', 'deveria', 'precisa',  # Portuguese
            'trebuie', 'ar trebui', 'necesită',  # Romanian
            'musí', 'měl by', 'potřebuje',  # Czech
            'musí', 'mal by', 'potrebuje',  # Slovak
            'musi', 'powinien', 'potrzebuje',  # Polish
            'täytyä', 'pitäisi', 'tarvitsee',  # Finnish
            'peab', 'peaks', 'vajab',  # Estonian
            'må', 'bør', 'trenger',  # Norwegian/Danish
            'måste', 'bör', 'behöver',  # Swedish
        ]
        if any(ind in text_lower for ind in imperative_indicators):
            return 'instructional'
        # Multilingual conversational indicators
        conversational_indicators = [
            'hello', 'hi', 'hey',  # English
            'hola', 'buenos',  # Spanish
            'bonjour', 'salut',  # French
            'hallo', 'guten',  # German
            'ciao', 'salve',  # Italian
            'olá', 'bom',  # Portuguese
            'salut', 'bună',  # Romanian
            'ahoj', 'dobrý',  # Czech/Slovak
            'cześć', 'dzień',  # Polish
            'hei', 'moi',  # Finnish
            'tere', 'tere',  # Estonian
            'hei', 'hallo',  # Norwegian/Danish
            'hej', 'hallå',  # Swedish
            'γεια', 'χαίρομαι',  # Greek
            'përshëndetje',  # Albanian
        ]
        if any(ind in text_lower for ind in conversational_indicators):
            return 'conversational'
        return 'factual'

    def _detect_language(self, text: str) -> str:
        """Detect language using indicators from LANGUAGE_CONFIG."""
        text_lower = text.lower()
        best_lang = 'en'
        best_score = 0

        for lang_code, config in LANGUAGE_CONFIG.items():
            score = sum(1 for ind in config['indicators'] if ind in text_lower)
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

        # Step 1: Identify the content
        if context and 'title' in context and context['title']:
            steps.append(f"{cfg['analyze']} {cfg['about']} '{context['title']}'")
        elif concepts:
            steps.append(f"{cfg['analyze']} {cfg['about']} '{concepts[0][0]}'")
        else:
            steps.append(f"{cfg['analyze']} {cfg['content_of']}")

        # Step 2: Describe structure
        word_count = analysis.get('word_count', 0)
        sentence_count = analysis.get('sentence_count', 0)
        if sentence_count > 1:
            steps.append(cfg['text_contains'].format(s=sentence_count, w=word_count))

        # Step 3: Identify content type
        steps.append(cfg['type_desc'].get(content_type, cfg['type_default']))

        # Step 4: Mention key concepts
        if concepts:
            top_concepts = [c[0] for c in concepts[:3]]
            steps.append(f"{cfg['concepts_prefix']} {', '.join(top_concepts)}")

        # Step 5: Named entities
        entities = analysis.get('entities', [])
        if entities:
            entity_texts = [e[0] for e in entities[:3]]
            steps.append(f"{cfg['entities_prefix']} {', '.join(entity_texts)}")

        # Step 6: Connection with context
        if context:
            if 'answer' in context and context['answer']:
                answer_words = set(context['answer'].lower().split()[:5])
                thinking_words = set(text.lower().split())
                overlap = answer_words.intersection(thinking_words)
                if overlap:
                    steps.append(f"{cfg['answer_terms']} {', '.join(list(overlap)[:3])}")

            if 'question' in context and context['question']:
                question_lower = context['question'].lower()
                for q_marker, q_desc in cfg.get('question_types', {}).items():
                    if q_marker in question_lower:
                        steps.append(q_desc)
                        break
                else:
                    steps.append(cfg.get('question_default', ''))

        # Step 7: Technical analysis
        if analysis.get('has_technical_terms', False):
            steps.append(cfg['technical_note'])

        return self._format_steps(steps, cfg)

    def _format_steps(self, steps: list, cfg: dict) -> str:
        """Format reasoning steps using language-specific connectors."""
        if not steps:
            return ""
        connectors = cfg['connectors']
        overflow = cfg['overflow_connector']
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
        return self._format_steps(steps, LANGUAGE_CONFIG['es'])

    def _format_steps_en(self, steps):
        return self._format_steps(steps, LANGUAGE_CONFIG['en'])
