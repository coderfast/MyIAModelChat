"""
Thinking quality validation module - V2 with advanced detection and multilingual support.
Validates that generated thinking is real reasoning, not meta-commentary or re-declaration.
Supports all EU official languages + Eastern European languages.
"""
import re
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class QualityResult:
    """Result of quality validation for a single sample."""
    valid: bool
    score: float
    issues: List[str] = field(default_factory=list)


@dataclass
class BatchQualityReport:
    """Report of quality validation for a batch of samples."""
    total: int = 0
    valid: int = 0
    too_short: int = 0
    meta_commentary: int = 0
    re_declaration: int = 0
    placeholder: int = 0
    no_derivation: int = 0
    low_quality: int = 0

    @property
    def valid_rate(self) -> float:
        return self.valid / max(1, self.total)

    def summary(self) -> str:
        return (
            f"Quality: {self.valid}/{self.total} valid ({self.valid_rate:.1%}) | "
            f"Too short: {self.too_short} | Meta: {self.meta_commentary} | "
            f"Re-declaration: {self.re_declaration} | Placeholder: {self.placeholder} | "
            f"No derivation: {self.no_derivation} | Low quality: {self.low_quality}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MULTILINGUAL PATTERNS — EU official languages + Eastern European
# ═══════════════════════════════════════════════════════════════════════════════

# Meta-commentary patterns (bypass detection)
META_PATTERNS = [
    # Spanish
    re.compile(r'^(el usuario|el humano)\s*(me\s+)?(saluda|despide|pregunta|pide)', re.IGNORECASE),
    re.compile(r'^(saludo|despedida)$', re.IGNORECASE),
    re.compile(r'^(respondo|contestando)\s*(con|with)', re.IGNORECASE),
    re.compile(r'^(el bot|asistente)\s*(responde|debe)', re.IGNORECASE),
    re.compile(r'(el usuario)\s*(escribe|envía|menciona|indica)', re.IGNORECASE),
    re.compile(r'(debo|debería)\s*(responder|contestar|decir)', re.IGNORECASE),
    re.compile(r'(la intención)\s*(detectada|es)', re.IGNORECASE),
    re.compile(r'(la respuesta)\s*(debe|apropiada|correcta)\s*(ser|es)', re.IGNORECASE),
    # English
    re.compile(r'^(the user)\s*(greets|asks|says|writes|indicates)', re.IGNORECASE),
    re.compile(r'^(greeting|farewell)$', re.IGNORECASE),
    re.compile(r'^(i respond|answering)\s*(with|by)', re.IGNORECASE),
    re.compile(r'^(the bot|assistant)\s*(should|must|responds)', re.IGNORECASE),
    re.compile(r'(i should)\s*(respond|answer|reply|say)', re.IGNORECASE),
    re.compile(r'(the intention)\s*(detected|is)', re.IGNORECASE),
    # French
    re.compile(r'^(l\'utilisateur)\s*(salué|demande|dit|écrit)', re.IGNORECASE),
    re.compile(r'^(salutation|au revoir)$', re.IGNORECASE),
    re.compile(r'^(je réponds|répondant)\s*(avec|par)', re.IGNORECASE),
    re.compile(r'^(le bot|l\'assistant)\s*(doit|répond)', re.IGNORECASE),
    # German
    re.compile(r'^(der Benutzer)\s*(begrüßt|fragt|sagt|schreibt)', re.IGNORECASE),
    re.compile(r'^(Begrüßung|Verabschiedung)$', re.IGNORECASE),
    re.compile(r'^(ich antworte|antwortend)\s*(mit)', re.IGNORECASE),
    re.compile(r'^(der Bot|Assistent)\s*(soll|muss|antwortet)', re.IGNORECASE),
    # Italian
    re.compile(r'^(l\'utente)\s*(saluta|chiede|dice|scrive)', re.IGNORECASE),
    re.compile(r'^(saluto|addio)$', re.IGNORECASE),
    re.compile(r'^(rispondo|rispondendo)\s*(con)', re.IGNORECASE),
    re.compile(r'^(il bot|l\'assistente)\s*(deve|risponde)', re.IGNORECASE),
    # Portuguese
    re.compile(r'^(o utilizador|o usuário)\s*(sauda|pergunta|diz|escreve)', re.IGNORECASE),
    re.compile(r'^(saudação|despedida)$', re.IGNORECASE),
    re.compile(r'^(respondo|respondendo)\s*(com)', re.IGNORECASE),
    re.compile(r'^(o bot|o assistente)\s*(deve|responde)', re.IGNORECASE),
    # Romanian
    re.compile(r'^(utilizatorul)\s*(salută|întreabă|scrie)', re.IGNORECASE),
    re.compile(r'^(salut|la revedere)$', re.IGNORECASE),
    re.compile(r'^(răspund|răspunzând)\s*(cu)', re.IGNORECASE),
    re.compile(r'^(botul|asistentul)\s*(trebuie|răspunde)', re.IGNORECASE),
    # Polish
    re.compile(r'^(użytkownik)\s*(wita|pyta|pisze)', re.IGNORECASE),
    re.compile(r'^(powitanie|pożegnanie)$', re.IGNORECASE),
    re.compile(r'^(odpowiadam|odpowiadając)\s*(z|na)', re.IGNORECASE),
    re.compile(r'^(bot|asystent)\s*(powinien|musi|odpowiada)', re.IGNORECASE),
    # Czech
    re.compile(r'^(uživatel)\s*(zdraví|ptá se|píše)', re.IGNORECASE),
    re.compile(r'^(přivítání|rozloučení)$', re.IGNORECASE),
    re.compile(r'^(odpovídám|odpovídaje)\s*(s)', re.IGNORECASE),
    re.compile(r'^(bot|asistent)\s*(má|musí|odpovídá)', re.IGNORECASE),
    # Dutch
    re.compile(r'^(de gebruiker)\s*(groet|vraagt|schrijft)', re.IGNORECASE),
    re.compile(r'^(begroeting|afscheid)$', re.IGNORECASE),
    re.compile(r'^(ik antwoord|antwoordend)\s*(met)', re.IGNORECASE),
    re.compile(r'^(de bot|de assistent)\s*(moet|antwoordt)', re.IGNORECASE),
    # Swedish
    re.compile(r'^(användaren)\s*(hälsar|frågar|skriver)', re.IGNORECASE),
    re.compile(r'^(hälsning|farväl)$', re.IGNORECASE),
    re.compile(r'^(jag svarar|svarande)\s*(med)', re.IGNORECASE),
    re.compile(r'^(botten|assistenten)\s*(ska|måste|svarar)', re.IGNORECASE),
    # Greek
    re.compile(r'^(ο χρήστης)\s*(χαιρετά/α\?|ρωτά|γράφει)', re.IGNORECASE),
    re.compile(r'^(χαιρετισμός|αποχαιρετισμός)$', re.IGNORECASE),
    # Bulgarian
    re.compile(r'^(потребителят)\s*(поздравява|пита|пише)', re.IGNORECASE),
    re.compile(r'^(поздрав|сбогом)$', re.IGNORECASE),
    # Croatian/Serbian/Bosnian/Montenegrin
    re.compile(r'^(korisnik)\s*(pozdravlja|pita|piše)', re.IGNORECASE),
    re.compile(r'^(pozdrav|doviđenja)$', re.IGNORECASE),
    # Slovenian
    re.compile(r'^(uporabnik)\s*(pozdravlja|vpraša|piše)', re.IGNORECASE),
    re.compile(r'^(pozdrav|nasvidenje)$', re.IGNORECASE),
    # Lithuanian
    re.compile(r'^(vartotojas)\s*(sveikina|klausia|rašo)', re.IGNORECASE),
    re.compile(r'^(sveikinimas|atsisveikinimas)$', re.IGNORECASE),
    # Latvian
    re.compile(r'^(lietotājs)\s*(sveicina|jautā|raksta)', re.IGNORECASE),
    re.compile(r'^(sveiciens|atvadīšanās)$', re.IGNORECASE),
    # Estonian
    re.compile(r'^(kasutaja)\s*(tervitab|küsib|kirjutab)', re.IGNORECASE),
    re.compile(r'^(tervitus| hüvasti)$', re.IGNORECASE),
    # Finnish
    re.compile(r'^(käyttäjä)\s*(tervehtii|kysyy|kirjoittaa)', re.IGNORECASE),
    re.compile(r'^(tervehdys|hyvästit)$', re.IGNORECASE),
    # Hungarian
    re.compile(r'^(a felhasználó)\s*(köszönt|kérdez|ír)', re.IGNORECASE),
    re.compile(r'^(köszöntés|búcsú)$', re.IGNORECASE),
    # Irish
    re.compile(r'^(an t-úsáideoir)\s*(beannaithe|ial|scríobhann)', re.IGNORECASE),
    re.compile(r'^(beannacht|slán)$', re.IGNORECASE),
    # Albanian
    re.compile(r'^(përdoruesi)\s*(përshëndet|pyet|shkruan)', re.IGNORECASE),
    re.compile(r'^(përshëndetje|mirupafshim)$', re.IGNORECASE),
    # Catalan
    re.compile(r'^(l\'usuari)\s*(saluda|pregunta|escriu)', re.IGNORECASE),
    re.compile(r'^(salut|adeu)$', re.IGNORECASE),
    # Galician
    re.compile(r'^(o usuario)\s*(saúda|pregunta|escribe)', re.IGNORECASE),
    re.compile(r'^(saúdo|adeus)$', re.IGNORECASE),
    # Icelandic
    re.compile(r'^(notandinn)\s*(heilsar|spyr|skrifar)', re.IGNORECASE),
    re.compile(r'^(kveðja|bless)$', re.IGNORECASE),
    # Macedonian
    re.compile(r'^(корисникот)\s*(поздравува|праша пишува)', re.IGNORECASE),
    re.compile(r'^(поздрав|здраво)$', re.IGNORECASE),
]

# Re-declaration patterns (bypass detection) — multilingual
RE_DECLARATION_PATTERNS = [
    # Spanish
    re.compile(r'la pregunta es:.*la respuesta', re.IGNORECASE),
    re.compile(r'la pregunta:.*respuesta correcta', re.IGNORECASE),
    re.compile(r'pregunta del usuario:.*respuesta del bot', re.IGNORECASE),
    re.compile(r'la respuesta correcta es', re.IGNORECASE),
    re.compile(r'se presenta información sobre:.*la respuesta contiene', re.IGNORECASE),
    re.compile(r'esto indica que la información.*puede responderse', re.IGNORECASE),
    # English
    re.compile(r'the question is:.*the answer', re.IGNORECASE),
    re.compile(r'question:.*correct answer', re.IGNORECASE),
    re.compile(r'the correct answer is', re.IGNORECASE),
    re.compile(r'information about:.*the answer contains', re.IGNORECASE),
    # French
    re.compile(r'la question est:.*la réponse', re.IGNORECASE),
    re.compile(r'la bonne réponse est', re.IGNORECASE),
    # German
    re.compile(r'die frage lautet:.*die antwort', re.IGNORECASE),
    re.compile(r'die richtige antwort ist', re.IGNORECASE),
    # Italian
    re.compile(r'la domanda è:.*la risposta', re.IGNORECASE),
    re.compile(r'la risposta corretta è', re.IGNORECASE),
    # Portuguese
    re.compile(r'a pergunta é:.*a resposta', re.IGNORECASE),
    re.compile(r'a resposta correta é', re.IGNORECASE),
    # Romanian
    re.compile(r'întrebarea este:.*răspunsul', re.IGNORECASE),
    re.compile(r'răspunsul corect este', re.IGNORECASE),
    # Polish
    re.compile(r'pytanie brzmi:.*odpowiedź', re.IGNORECASE),
    re.compile(r'prawidłowa odpowiedź to', re.IGNORECASE),
    # Czech
    re.compile(r'otázka zní:.*odpověď', re.IGNORECASE),
    re.compile(r'správná odpověď je', re.IGNORECASE),
    # Dutch
    re.compile(r'de vraag is:.*het antwoord', re.IGNORECASE),
    re.compile(r'het juiste antwoord is', re.IGNORECASE),
    # Swedish
    re.compile(r'frågan är:.*svaret', re.IGNORECASE),
    re.compile(r'detta rätt svar är', re.IGNORECASE),
    # Greek
    re.compile(r'η ερώτηση είναι:.*η απάντηση', re.IGNORECASE),
    re.compile(r'η σωστή απάντηση είναι', re.IGNORECASE),
    # Bulgarian
    re.compile(r'въпросът е:.*отговорът', re.IGNORECASE),
    re.compile(r'правилният отговор е', re.IGNORECASE),
    # Croatian/Serbian/Bosnian
    re.compile(r'pitanje je:.*odgovor', re.IGNORECASE),
    re.compile(r'točan odgovor je', re.IGNORECASE),
    # Finnish
    re.compile(r'kysymys on:.*vastaus', re.IGNORECASE),
    re.compile(r'oikea vastaus on', re.IGNORECASE),
    # Hungarian
    re.compile(r'a kérdés:.*a válasz', re.IGNORECASE),
    re.compile(r'a helyes válasz', re.IGNORECASE),
]

# Placeholder patterns (generic filler) — multilingual
PLACEHOLDER_PATTERNS = [
    # Word count placeholders (Spanish)
    re.compile(r'contiene \d+ palabras', re.IGNORECASE),
    re.compile(r'con \d+ palabras de contenido', re.IGNORECASE),
    re.compile(r'texto con \d+ palabras', re.IGNORECASE),
    # Word count placeholders (English)
    re.compile(r'contains \d+ words', re.IGNORECASE),
    re.compile(r'with \d+ words of content', re.IGNORECASE),
    re.compile(r'text with \d+ words', re.IGNORECASE),
    # Word count placeholders (French)
    re.compile(r'contient \d+ mots', re.IGNORECASE),
    # Word count placeholders (German)
    re.compile(r'enthält \d+ Wörter', re.IGNORECASE),
    # Word count placeholders (Italian)
    re.compile(r'contiene \d+ parole', re.IGNORECASE),
    # Word count placeholders (Portuguese)
    re.compile(r'contém \d+ palavras', re.IGNORECASE),
    # Word count placeholders (Romanian)
    re.compile(r'conține \d+ cuvinte', re.IGNORECASE),
    # Word count placeholders (Polish)
    re.compile(r'zawiera \d+ słów', re.IGNORECASE),
    # Word count placeholders (Czech)
    re.compile(r'obsahuje \d+ slov', re.IGNORECASE),
    # Word count placeholders (Dutch)
    re.compile(r'bevat \d+ woorden', re.IGNORECASE),
    # Word count placeholders (Swedish)
    re.compile(r'innehåller \d+ ord', re.IGNORECASE),
    # Word count placeholders (Greek)
    re.compile(r'περιέχει \d+ λέξεις', re.IGNORECASE),
    # Generic relevance filler (multilingual)
    re.compile(r'información relevante.*puede ser utilizada', re.IGNORECASE),
    re.compile(r'contiene información que debe ser procesada', re.IGNORECASE),
    re.compile(r'relevant information.*can be used', re.IGNORECASE),
    re.compile(r'contains information that must be processed', re.IGNORECASE),
    re.compile(r'informations pertinentes.*peuvent être utilisées', re.IGNORECASE),
    re.compile(r'relevante Informationen.*verwendet werden können', re.IGNORECASE),
    re.compile(r'informazioni rilevanti.*possono essere utilizzate', re.IGNORECASE),
    re.compile(r'informações relevantes.*podem ser utilizadas', re.IGNORECASE),
    re.compile(r'informații relevante.*pot fi utilizate', re.IGNORECASE),
    re.compile(r'istotne informacje.*mogą być wykorzystane', re.IGNORECASE),
    re.compile(r'důležité informace.*mohou být použity', re.IGNORECASE),
    re.compile(r'relevante informatie.*kan worden gebruikt', re.IGNORECASE),
    re.compile(r'relevant information.*kan användas', re.IGNORECASE),
    re.compile(r'sχετικές πληροφορίες.*μπορούν να χρησιμοποιηθούν', re.IGNORECASE),
    # Generic structural labels (multilingual)
    re.compile(r'^.*presenta información (técnica|académica)\.?$', re.IGNORECASE),
    re.compile(r'^.*contiene (datos|información) sobre el tema\.?$', re.IGNORECASE),
    re.compile(r'^.*presents (technical|academic) information\.?$', re.IGNORECASE),
    re.compile(r'^.*contains (data|information) about the topic\.?$', re.IGNORECASE),
    re.compile(r'^.*präsentiert (technische|akademische) Informationen\.?$', re.IGNORECASE),
    re.compile(r'^.*enthält (Daten|Informationen) zum Thema\.?$', re.IGNORECASE),
    re.compile(r'^.*presenta (informazioni tecniche|accademiche)\.?$', re.IGNORECASE),
    re.compile(r'^.*contiene (dati|informazioni) sull\'argomento\.?$', re.IGNORECASE),
    re.compile(r'^.*apresenta informações (técnicas|acadêmicas)\.?$', re.IGNORECASE),
    re.compile(r'^.*contém (dados|informações) sobre o tema\.?$', re.IGNORECASE),
    re.compile(r'^.*prezintă informații (tehnice|academice)\.?$', re.IGNORECASE),
    re.compile(r'^.*conține (date|informații) despre subiect\.?$', re.IGNORECASE),
    re.compile(r'^.*zawiera informacje (techniczne|naukowe)\.?$', re.IGNORECASE),
    re.compile(r'^.*obsahuje (data|informace) o tématu\.?$', re.IGNORECASE),
    re.compile(r'^.*bevat (technische|academische) informatie\.?$', re.IGNORECASE),
    re.compile(r'^.*innehåller (data|information) om ämnet\.?$', re.IGNORECASE),
]

# Step indicators (reasoning markers) — multilingual
STEP_INDICATORS = [
    # Spanish
    'porque', 'por lo tanto', 'primero', 'paso', 'análisis',
    'entonces', 'sin embargo', 'además', 'en cambio', 'consiste',
    'analizando', 'identificando', 'observando', 'detectando',
    'conceptos', 'entidades', 'términos', 'secciones',
    # English
    'because', 'therefore', 'first', 'step', 'analysis',
    'however', 'additionally', 'furthermore', 'consists',
    'analyzing', 'identifying', 'observing', 'detecting',
    'concepts', 'entities', 'terms', 'sections',
    # French
    'parce que', 'donc', 'premièrement', 'étape', 'analyse',
    'cependant', 'de plus', 'en outre', 'consiste',
    'analysant', 'identifiant', 'observant', 'détectant',
    'concepts', 'entités', 'termes', 'sections',
    # German
    'weil', 'deshalb', 'zuerst', 'schritt', 'analyse',
    'jedoch', 'außerdem', 'ferner', 'besteht',
    'analysierend', 'identifizierend', 'beobachtend', 'erkennend',
    'konzepte', 'entitäten', 'begriffe', 'abschnitte',
    # Italian
    'perché', 'quindi', 'prima', 'passo', 'analisi',
    'tuttavia', 'inoltre', 'consiste',
    'analizzando', 'identificando', 'osservando', 'rilevando',
    'concetti', 'entità', 'termini', 'sezioni',
    # Portuguese
    'porque', 'portanto', 'primeiro', 'passo', 'análise',
    'no entanto', 'além disso', 'consiste',
    'analisando', 'identificando', 'observando', 'detectando',
    'conceitos', 'entidades', 'termos', 'seções',
    # Romanian
    'deoarece', 'prin urmare', 'primul', 'pas', 'analiză',
    'totuși', 'în plus', 'constă',
    'analizând', 'identificând', 'observând', 'detectând',
    'concepte', 'entități', 'termeni', 'secțiuni',
    # Polish
    'ponieważ', 'dlatego', 'pierwszy', 'krok', 'analiza',
    'jednak', 'ponadto', 'polega na',
    'analizując', 'identyfikując', 'obserwując', 'wykrywając',
    'pojęcia', 'encje', 'terminy', 'sekcje',
    # Czech
    'protože', 'proto', 'první', 'krok', 'analýza',
    'avšak', 'kromě toho', 'spočívá v',
    'analyzováním', 'identifikováním', 'pozorováním', 'zjišťováním',
    'koncepty', 'entity', 'termíny', 'sekce',
    # Dutch
    'omdat', 'daarom', 'eerst', 'stap', 'analyse',
    'echter', 'bovendien', 'bestaat uit',
    'analyserend', 'identificerend', 'observerend', 'detecterend',
    'concepten', 'entiteiten', 'termen', 'secties',
    # Swedish
    'eftersom', 'därför', 'först', 'steg', 'analys',
    'emellertid', 'vidare', 'består av',
    'analyserande', 'identifierande', 'observerande', 'upptäckande',
    'koncept', 'entiteter', 'termer', 'avsnitt',
    # Greek
    'επειδή', 'επομένως', 'πρώτα', 'βήμα', 'ανάλυση',
    'ωστόσο', 'επίσης', 'αποτελείται από',
    'αναλύοντας', 'αναγνωρίζοντας', 'παρατηρώντας', 'ανιχνεύοντας',
    'έννοιες', 'οντότητες', 'όροι', 'ενότητες',
    # Bulgarian
    'защото', 'затова', 'първо', 'стъпка', 'анализ',
    'обаче', 'освен това', 'състои се от',
    'анализирайки', 'идентифицирайки', 'наблюдавайки', 'откривайки',
    'концепции', 'субекти', 'термини', 'раздели',
    # Croatian/Serbian/Bosnian/Montenegrin
    'jer', 'stoga', 'prvo', 'korak', 'analiza',
    'međutim', 'osim toga', 'sastoji se od',
    'analizirajući', 'identificirajući', 'promatrajući', 'otkrivajući',
    'koncepti', 'entiteti', 'termini', 'odjeljci',
    # Finnish
    'koska', 'siksi', 'ensin', 'vaihe', 'analyysi',
    'kuitenkin', 'myös', 'koostuu',
    'analysoiden', 'tunnistaen', 'havainnoiden', 'havaiten',
    'käsitteet', 'entiteetit', 'termit', 'osiot',
    # Hungarian
    'mert', 'ezért', 'először', 'lépés', 'elemzés',
    'azonban', 'emellett', 'áll',
    'elemezve', 'azonosítva', 'megfigyelve', 'észlelve',
    'fogalmak', 'entitások', 'kifejezések', 'szakaszok',
    # Lithuanian
    'nes', 'todėl', 'pirma', 'žingsnis', 'analizė',
    'tačiau', 'be to', 'susideda iš',
    'analizuojant', 'identifikuojant', 'stebint', 'aptinkant',
    'konceptai', 'entitetės', 'terminai', 'skyriai',
    # Latvian
    'jo', 'tāpēc', 'vispirms', 'solis', 'analīze',
    'tomēr', 'turklāt', 'sastāv no',
    'analizējot', 'identificējot', 'novērojot', 'konstatējot',
    'koncepcijas', 'entītes', 'termini', 'nodaļas',
    # Estonian
    'sest', 'seetõttu', 'kõigepealt', 'samm', 'analüüs',
    'kuid', 'lisaks', 'koosneb',
    'analüüsides', 'tuvastades', 'vaadeldes', 'avastades',
    'kontseptsioonid', 'üksused', 'terminid', 'jaotised',
    # Albanian
    'sepse', 'prandaj', 'së pari', 'hapi', 'analiza',
    'megjithatë', 'gjithashtu', 'përbëhet nga',
    'duke analizuar', 'duke identifikuar', 'duke vëzhguar', 'duke zbuluar',
    'koncepte', 'entitete', 'terma', 'pjesë',
    # Catalan
    'perquè', 'per tant', 'primer', 'pas', 'anàlisi',
    'tanmateix', 'a més', 'consisteix en',
    'analitzant', 'identificant', 'observant', 'detectant',
    'conceptes', 'entitats', 'termes', 'seccions',
    # Galician
    'porque', 'polo tanto', 'primeiro', 'paso', 'análise',
    'non obstante', 'adxemais', 'consiste en',
    'analizando', 'identificando', 'observando', 'detectando',
    'conceptos', 'entidades', 'termos', 'seccions',
    # Icelandic
    'vegna þess', 'þess vegna', 'fyrst', 'skref', 'greining',
    'hins vegar', 'ennfremur', 'samanstendur af',
    'greinandi', 'greinandi', 'athugandi', 'uppgötvaandi',
    'hugtök', 'einindin', 'skilmálar', 'kaflar',
    # Macedonian
    'затоа', 'прво', 'чекор', 'анализа',
    'сепак', 'покрај тоа', 'се состои од',
    'анализирајќи', 'идентифицирајќи', 'набљудувајќи', 'откривајќи',
    'концепти', 'субјекти', 'термини', 'оддели',
]

# Logical connectors for validation — multilingual
LOGICAL_CONNECTORS = [
    # Spanish
    'en primer lugar', 'además', 'por otro lado', 'finalmente',
    'por lo tanto', 'sin embargo', 'asimismo',
    # English
    'first', 'additionally', 'furthermore', 'finally',
    'therefore', 'however', 'moreover',
    # French
    'premièrement', 'de plus', 'd\'autre part', 'enfin',
    'par conséquent', 'cependant', 'également',
    # German
    'erstens', 'zudem', 'andererseits', 'schließlich',
    'deshalb', 'jedoch', 'ebenso',
    # Italian
    'in primo luogo', 'inoltre', 'd\'altra parte', 'infine',
    'quindi', 'tuttavia', 'allo stesso modo',
    # Portuguese
    'em primeiro lugar', 'além disso', 'por outro lado', 'finalmente',
    'portanto', 'no entanto', 'também',
    # Romanian
    'în primul rând', 'în plus', 'pe de altă parte', 'în cele din urmă',
    'prin urmare', 'totuși', 'de asemenea',
    # Polish
    'po pierwsze', 'ponadto', 'z drugiej strony', 'wreszcie',
    'dlatego', 'jednak', 'również',
    # Czech
    'za prvé', 'kromě toho', 'na druhou stranu', 'nakonec',
    'proto', 'avšak', 'také',
    # Dutch
    'ten eerste', 'bovendien', 'aan de andere kant', 'ten slotte',
    'daarom', 'echter', 'ook',
    # Swedish
    'för det första', 'vidare', 'å andra sidan', 'slutligen',
    'därför', 'emellertid', 'även',
    # Greek
    'πρώτον', 'επίσης', 'από την άλλη πλευρά', 'τέλος',
    'επομένως', 'ωστόσο', 'επίσης',
    # Bulgarian
    'първо', 'освен това', 'от друга страна', 'накрая',
    'затова', 'обаче', 'също така',
    # Croatian/Serbian/Bosnian
    'prvo', 'osim toga', 's druge strane', 'na kraju',
    'stoga', 'međutim', 'također',
    # Finnish
    'ensinnäkin', 'lisäksi', 'toisaalta', 'lopuksi',
    'siksi', 'kuitenkin', 'myös',
    # Hungarian
    'először is', 'emellett', 'másrészt', 'végül',
    'ezért', 'azonban', 'szintén',
    # Lithuanian
    'pirma', 'be to', 'kita vertus', 'galiausiai',
    'todėl', 'tačiau', 'taip pat',
    # Latvian
    'pirmkārt', 'turklāt', 'no otras puses', 'visbeidzot',
    'tāpēc', 'tomēr', 'arī',
    # Estonian
    'esiteks', 'lisaks', 'teisalt', 'lõpuks',
    'seetõttu', 'kuid', 'samuti',
    # Albanian
    'së pari', 'gjithashtu', 'nga ana tjetër', 'së fundi',
    'prandaj', 'megjithatë', 'po ashtu',
    # Catalan
    'en primer lloc', 'a més', 'd\'altra banda', 'finalment',
    'per tant', 'tanmateix', 'també',
    # Galician
    'en primeiro lugar', 'adxemais', 'por outra banda', 'finalmente',
    'polo tanto', 'non obstante', 'tamén',
]


def validate_thinking(thinking: str, answer: str = '', question: str = '') -> QualityResult:
    """
    Validate that thinking is real reasoning, not meta-commentary or re-declaration.
    V2 with advanced detection of bypass patterns.
    """
    issues = []
    score = 0.0

    # Check minimum length
    if not thinking or len(thinking.strip()) < 15:
        return QualityResult(valid=False, score=0.0, issues=['too_short'])

    # Length scoring
    if len(thinking) < 30:
        issues.append('short')
        score += 0.2
    else:
        score += 0.4

    # Check for meta-commentary (original patterns)
    is_meta = False
    for pattern in META_PATTERNS:
        if pattern.match(thinking.strip()):
            is_meta = True
            issues.append('meta_commentary')
            break

    # NEW: Check for re-declaration (bypass detection)
    is_re_declaration = False
    if not is_meta:
        for pattern in RE_DECLARATION_PATTERNS:
            if pattern.search(thinking):
                is_re_declaration = True
                issues.append('re_declaration')
                break

    # NEW: Check for placeholder content
    is_placeholder = False
    if not is_meta and not is_re_declaration:
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(thinking):
                is_placeholder = True
                issues.append('placeholder')
                break

    # Check for reasoning indicators
    has_steps = any(ind in thinking.lower() for ind in STEP_INDICATORS)

    # Check for vocabulary diversity (not just re-declaration)
    has_vocabulary_diversity = False
    if answer:
        answer_words = set(answer.lower().split())
        thinking_words = set(thinking.lower().split())
        # If thinking has many unique words not in answer, it's diverse
        unique_thinking = thinking_words - answer_words
        has_vocabulary_diversity = len(unique_thinking) > len(thinking_words) * 0.3

    # Check for answer derivation
    has_answer_derivation = False
    if answer:
        answer_words = set(answer.lower().split()[:5])
        thinking_words = set(thinking.lower().split())
        has_answer_derivation = bool(answer_words and answer_words.intersection(thinking_words))

    # Check for logical connectors (multilingual)
    has_logical_connectors = any(connector in thinking.lower() for connector in LOGICAL_CONNECTORS)

    # Scoring based on checks
    if is_meta and not has_steps and not has_answer_derivation:
        score -= 0.3  # Penalty for pure meta-commentary
    elif is_re_declaration:
        score -= 0.2  # Penalty for re-declaration
    elif is_placeholder:
        score -= 0.1  # Penalty for placeholder
    elif not is_meta and not is_re_declaration and not is_placeholder:
        score += 0.3  # Bonus for non-meta content

    if has_answer_derivation:
        score += 0.2
    elif answer:
        issues.append('no_derivation')

    if has_vocabulary_diversity:
        score += 0.2

    if has_logical_connectors:
        score += 0.1

    if has_steps:
        score += 0.2

    # Final validation
    valid = (
        score >= 0.7 and
        not (is_meta and not has_steps and not has_answer_derivation) and
        not is_re_declaration and
        not is_placeholder
    )

    return QualityResult(valid=valid, score=max(0.0, score), issues=issues)


def validate_thinking_batch(samples: List[Dict[str, Any]]) -> BatchQualityReport:
    """Validate a batch of samples with thinking."""
    report = BatchQualityReport()

    for sample in samples:
        thinking = sample.get('thinking', '')
        answer = sample.get('output', sample.get('answer', ''))
        question = sample.get('input', sample.get('question', ''))

        report.total += 1
        result = validate_thinking(thinking, answer, question)

        if result.valid:
            report.valid += 1
        else:
            if 'too_short' in result.issues:
                report.too_short += 1
            if 'meta_commentary' in result.issues:
                report.meta_commentary += 1
            if 're_declaration' in result.issues:
                report.re_declaration += 1
            if 'placeholder' in result.issues:
                report.placeholder += 1
            if 'no_derivation' in result.issues:
                report.no_derivation += 1
            if result.score < 0.5:
                report.low_quality += 1

    return report


def filter_low_quality(samples: List[Dict[str, Any]], min_score: float = 0.5) -> List[Dict[str, Any]]:
    """Filter out low-quality thinking samples."""
    filtered = []
    for sample in samples:
        thinking = sample.get('thinking', '')
        answer = sample.get('output', sample.get('answer', ''))
        result = validate_thinking(thinking, answer)
        if result.score >= min_score:
            filtered.append(sample)
    return filtered
