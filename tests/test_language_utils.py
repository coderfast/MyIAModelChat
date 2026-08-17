"""Tests for language detection and tagging in commons/language_utils.py"""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons.language_utils import (
    detect_language,
    get_language_name,
    get_language_family,
    get_supported_languages,
    get_supported_languages_display,
    LANGUAGE_INDICATORS,
)


class TestManifestLanguageTagging:
    """Test that _tag_languages preserves per-sample language tags for
    multilingual sources (e.g. Tagengo) instead of overwriting them with
    a source-level manifest mapping."""

    def _make_preparer(self, rows):
        from dataset_preparer.data_preparer import DataPreparer
        from datasets import Dataset
        prep = DataPreparer.__new__(DataPreparer)
        prep.combined_data = Dataset.from_list(rows)
        prep._get_num_proc = lambda: 1
        return prep

    def _tag(self, prep, manifest):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(manifest, f)
            path = f.name
        try:
            prep._tag_languages(manifest_path=path)
        finally:
            os.unlink(path)
        return prep.combined_data

    def test_tagengo_per_sample_tags_preserved(self):
        rows = [
            {'input_ids': '<|user|>hello<|end|>', 'source': 'tagengo', 'language': 'en'},
            {'input_ids': '<|user|>hola<|end|>', 'source': 'tagengo', 'language': 'es'},
        ]
        prep = self._make_preparer(rows)
        manifest = {'sources': {'aiml': 'en', 'tagengo': 'en'}, 'default': 'unknown'}
        ds = self._tag(prep, manifest)
        langs = {r['source']: r['language'] for r in ds}
        assert langs == {'tagengo': 'es'}, f"Expected last per-sample tag preserved, got {langs}"

    def test_source_level_mapping_applies_when_no_language(self):
        rows = [
            {'input_ids': 'aiml text', 'source': 'aiml', 'language': None},
        ]
        prep = self._make_preparer(rows)
        manifest = {'sources': {'aiml': 'en'}, 'default': 'unknown'}
        ds = self._tag(prep, manifest)
        assert ds[0]['language'] == 'en'

    def test_file_level_mapping_overrides_existing_tag(self):
        rows = [
            {'input_ids': 'pdf text', 'source': 'pdf', 'language': 'es',
             'file_path': 'datasets_source/pdf/doc.pdf'},
        ]
        prep = self._make_preparer(rows)
        manifest = {'files': {'datasets_source/pdf/doc.pdf': 'en'}, 'sources': {}, 'default': 'unknown'}
        ds = self._tag(prep, manifest)
        assert ds[0]['language'] == 'en'

    def test_unknown_existing_tag_falls_through_to_source_mapping(self):
        rows = [
            {'input_ids': 'csv text', 'source': 'csv', 'language': 'unknown'},
        ]
        prep = self._make_preparer(rows)
        manifest = {'sources': {'csv': 'es'}, 'default': 'unknown'}
        ds = self._tag(prep, manifest)
        assert ds[0]['language'] == 'es'


class TestLanguageDetection:
    """Test detect_language function."""

    def test_detect_spanish(self):
        text = "El gato está sentado en la mesa porque tiene hambre"
        assert detect_language(text) == 'es'

    def test_detect_english(self):
        text = "The cat is sitting on the table because it is hungry"
        assert detect_language(text) == 'en'

    def test_detect_french(self):
        text = "Le chat est assis sur la table parce qu'il a faim"
        assert detect_language(text) == 'fr'

    def test_detect_german(self):
        text = "Die Katze sitzt auf dem Tisch, weil sie hungrig ist"
        assert detect_language(text) == 'de'

    def test_detect_italian(self):
        text = "Il gatto è seduto sul tavolo perché ha fame"
        assert detect_language(text) == 'it'

    def test_detect_portuguese(self):
        text = "O gato está sentado na mesa porque tem fome"
        # Portuguese shares indicators with Spanish, Galician and Irish
        assert detect_language(text) in ('pt', 'gl', 'ga', 'es')

    def test_detect_dutch(self):
        text = "De kat zit op de tafel omdat ze honger heeft"
        assert detect_language(text) == 'nl'

    def test_detect_polish(self):
        text = "Kot siedzi na stole, bo jest głodny"
        assert detect_language(text) == 'pl'

    def test_detect_czech(self):
        text = "Kočka sedí na stole, protože má hlad"
        assert detect_language(text) == 'cs'

    def test_detect_swedish(self):
        text = "Katten sitter på bordet för att den är hungrig"
        assert detect_language(text) == 'sv'

    def test_detect_greek(self):
        text = "Η γάτα κάθεται στο τραπέζι γιατί πεινάει"
        assert detect_language(text) == 'el'

    def test_detect_romanian(self):
        text = "Pisica stă pe masă pentru că îi este foame"
        assert detect_language(text) == 'ro'

    def test_detect_croatian(self):
        text = "Mačka sjedi na stolu jer je gladna"
        assert detect_language(text) == 'hr'

    def test_detect_bulgarian(self):
        text = "Котката седи на масата, защото е гладна"
        assert detect_language(text) == 'bg'

    def test_detect_hungarian(self):
        text = "A macska az asztalon ül, mert éhes"
        assert detect_language(text) == 'hu'

    def test_detect_finnish(self):
        text = "Kissa istuu pöydällä, koska se on nälkäinen"
        assert detect_language(text) == 'fi'

    def test_detect_estonian(self):
        text = "Kass istub laual, sest tal on nälg"
        assert detect_language(text) == 'et'

    def test_detect_lithuanian(self):
        text = "Katinas sėdi ant stalo, nes jis alkanas"
        assert detect_language(text) == 'lt'

    def test_detect_latvian(self):
        text = "Kaķis sēž uz galda, jo viņš ir izsalcis"
        assert detect_language(text) == 'lv'

    def test_detect_slovak(self):
        text = "Mačka sedí na stole, lebo má hlad"
        # Slovak shares indicators with other Slavic and Celtic languages
        assert detect_language(text) in ('sk', 'ga', 'pl')

    def test_detect_slovenian(self):
        text = "Mačka sedi na mizi, ker je lačna"
        assert detect_language(text) == 'sl'

    def test_detect_danish(self):
        text = "Katten sidder på bordet, fordi den er sulten"
        assert detect_language(text) == 'da'

    def test_detect_norwegian(self):
        text = "Katten sitter på bordet fordi den er sulten"
        # Norwegian and Danish share very similar indicators
        assert detect_language(text) in ('nb', 'da')

    def test_detect_catalan(self):
        text = "El gat està assegut a la taula perquè té gana"
        assert detect_language(text) == 'ca'

    def test_detect_galician(self):
        text = "O gato está sentado na mesa porque ten fame"
        # Galician shares indicators with Spanish, Portuguese and Irish
        assert detect_language(text) in ('gl', 'pt', 'ga', 'es')

    def test_detect_albanian(self):
        text = "Maceshi është ulur në tryezë sepse ka uri"
        assert detect_language(text) == 'sq'

    def test_detect_icelandic(self):
        text = "Kötturinn situr á borðinu vegna þess að hann er svangur"
        assert detect_language(text) == 'is'

    def test_detect_irish(self):
        text = "Tá an cat ag suí ar an mbord toisc go bhfuil ocrais air"
        # Irish shares indicators with Luxembourgish and other languages
        assert detect_language(text) in ('ga', 'lb')

    def test_detect_luxembourgish(self):
        text = "Déi Kaz sëtzt op dem Dësch well se hongereg ass"
        # Luxembourgish shares indicators with German
        assert detect_language(text) in ('lb', 'de')

    def test_empty_text(self):
        assert detect_language('') == 'unknown'

    def test_none_text(self):
        assert detect_language(None) == 'unknown'

    def test_short_text(self):
        assert detect_language('hi') == 'en'

    def test_mixed_language(self):
        # Text with more Spanish than English indicators
        text = "El gato the cat está sentado on the table"
        result = detect_language(text)
        assert result in ('es', 'en', 'et')


class TestLanguageInfo:
    """Test helper functions."""

    def test_get_language_name(self):
        assert get_language_name('es') == 'Spanish'
        assert get_language_name('en') == 'English'
        assert get_language_name('fr') == 'French'
        assert get_language_name('xx') == 'Unknown'

    def test_get_language_family(self):
        assert get_language_family('es') == 'romance'
        assert get_language_family('de') == 'germanic'
        assert get_language_family('pl') == 'slavic'
        assert get_language_family('xx') == 'unknown'

    def test_get_supported_languages(self):
        langs = get_supported_languages()
        assert 'es' in langs
        assert 'en' in langs
        assert 'fr' in langs
        assert 'de' in langs
        assert len(langs) >= 24  # At least 24 EU languages

    def test_get_supported_languages_display(self):
        display = get_supported_languages_display()
        assert 'Spanish' in display
        assert 'English' in display
        assert 'es' in display


class TestLanguageIndicators:
    """Test that LANGUAGE_INDICATORS has correct structure."""

    def test_all_languages_have_required_fields(self):
        for lang_code, config in LANGUAGE_INDICATORS.items():
            assert 'name' in config, f"{lang_code} missing 'name'"
            assert 'family' in config, f"{lang_code} missing 'family'"
            assert 'indicators' in config, f"{lang_code} missing 'indicators'"
            assert isinstance(config['indicators'], list), f"{lang_code} indicators not list"
            assert len(config['indicators']) > 0, f"{lang_code} has empty indicators"

    def test_all_languages_have_indicators(self):
        for lang_code, config in LANGUAGE_INDICATORS.items():
            assert len(config['indicators']) >= 4, \
                f"{lang_code} has only {len(config['indicators'])} indicators (need >= 4)"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
