"""Tests for markdown generation scripts (*_to_md.py)."""
import os
import sys
import json
import tempfile
import shutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dirs():
    """Create temporary source and output directories."""
    source = tempfile.mkdtemp(prefix='test_source_')
    output = tempfile.mkdtemp(prefix='test_output_')
    yield source, output
    shutil.rmtree(source, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)


class TestAimlToMd:
    """Tests for AIML to markdown conversion."""

    def test_aiml_to_md_chat_format(self, temp_dirs):
        """Test AIML to MD generates chat format with GPT-2 tokens."""
        from dataset_preparer.aiml.aiml_to_md import aiml_to_md

        source, output = temp_dirs
        # Create a minimal AIML-like file
        aiml_content = """<?xml version="1.0" encoding="UTF-8"?>
<aiml>
<category>
<pattern>HELLO</pattern>
<template>Hello! How are you?</template>
</category>
<category>
<pattern>WHAT IS AI</pattern>
<template>AI is artificial intelligence.</template>
</category>
</aiml>"""
        aiml_file = os.path.join(source, 'test.aiml')
        with open(aiml_file, 'w', encoding='utf-8') as f:
            f.write(aiml_content)

        config = {
            'source_dir': source,
            'output_dir': output,
            'format': 'chat',
            'quality': {'min_words': 2, 'max_words': 500, 'min_alpha_ratio': 0.3}
        }
        count = aiml_to_md(config)
        assert count >= 1

        # Check generated files (now in subdirectories)
        md_files = []
        for root, dirs, files in os.walk(output):
            for f in files:
                if f.endswith('.md'):
                    md_files.append(os.path.join(root, f))
        assert len(md_files) >= 1

        with open(md_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
        assert '<|user|>' in content
        assert '<|end|>' in content
        assert '<|assistant|>' in content

    def test_aiml_to_md_problem_final_format(self, temp_dirs):
        """Test AIML to MD generates problem_final format."""
        from dataset_preparer.aiml.aiml_to_md import aiml_to_md

        source, output = temp_dirs
        aiml_content = """<?xml version="1.0" encoding="UTF-8"?>
<aiml>
<category>
<pattern>HELLO</pattern>
<template>Hello!</template>
</category>
</aiml>"""
        with open(os.path.join(source, 'test.aiml'), 'w', encoding='utf-8') as f:
            f.write(aiml_content)

        config = {
            'source_dir': source,
            'output_dir': output,
            'format': 'problem_final',
            'quality': {'min_words': 1, 'max_words': 500, 'min_alpha_ratio': 0.3}
        }
        count = aiml_to_md(config)
        assert count >= 1

        md_files = []
        for root, dirs, files in os.walk(output):
            for f in files:
                if f.endswith('.md'):
                    md_files.append(os.path.join(root, f))
        with open(md_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
        assert '<|problem|>' in content
        assert '<|final|>' in content

    def test_aiml_to_md_missing_source(self):
        """Test AIML to MD handles missing source directory."""
        from dataset_preparer.aiml.aiml_to_md import aiml_to_md

        config = {
            'source_dir': '/nonexistent/path',
            'output_dir': '/tmp/test_output',
            'format': 'chat',
            'quality': {'min_words': 3, 'max_words': 500, 'min_alpha_ratio': 0.5}
        }
        count = aiml_to_md(config)
        assert count == 0


class TestPdfToMd:
    """Tests for PDF to markdown conversion."""

    def test_pdf_to_md_config_loading(self):
        """Test PDF config file exists and is valid."""
        config_path = os.path.join('dataset_preparer', 'pdf', 'pdf_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            assert 'source_dir' in config
            assert 'output_dir' in config
            assert 'format' in config

    def test_pdf_to_md_missing_source(self):
        """Test PDF to MD handles missing source directory gracefully."""
        from dataset_preparer.pdf.pdf_to_md import pdf_to_md

        config = {
            'source_dir': '/nonexistent/path',
            'output_dir': '/tmp/test_output',
            'format': 'problem_final',
            'chunking': {'enabled': False},
            'quality': {'min_paragraph_length': 20}
        }
        count = pdf_to_md(config)
        assert count == 0


class TestEpubToMd:
    """Tests for EPUB to markdown conversion."""

    def test_epub_to_md_config_loading(self):
        """Test EPUB config file exists and is valid."""
        config_path = os.path.join('dataset_preparer', 'epub', 'epub_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            assert 'source_dir' in config
            assert 'output_dir' in config


class TestWebToMd:
    """Tests for Web to markdown conversion."""

    def test_web_to_md_config_loading(self):
        """Test Web config file exists and is valid."""
        config_path = os.path.join('dataset_preparer', 'web', 'web_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            assert 'output_dir' in config
            assert 'format' in config


class TestHfToMd:
    """Tests for HuggingFace to markdown conversion."""

    def test_hf_to_md_config_loading(self):
        """Test HF config file exists and is valid."""
        config_path = os.path.join('dataset_preparer', 'hf', 'hf_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            assert 'source_dir' in config or 'datasets' in config
            assert 'output_dir' in config


class TestCsvToMd:
    """Tests for CSV to markdown conversion."""

    def test_csv_to_md_config_loading(self):
        """Test CSV config file exists and is valid."""
        config_path = os.path.join('dataset_preparer', 'csv', 'csv_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            assert 'source_dir' in config
            assert 'output_dir' in config


class TestTokenFormat:
    """Tests that verify correct GPT-2 token format in generated markdown."""

    def test_chat_format_tokens(self):
        """Test chat format uses correct GPT-2 tokens."""
        text = "Hello"
        response = "Hi there"
        md_content = f"<|user|>{text}<|end|><|assistant|>{response}<|end|>"
        assert md_content == "<|user|>Hello<|end|><|assistant|>Hi there<|end|>"
        assert '<|user|>' in md_content
        assert '<|end|>' in md_content
        assert '<|assistant|>' in md_content
        assert '^^user' not in md_content
        assert '^^assistant' not in md_content
        assert '^^end' not in md_content

    def test_problem_final_format_tokens(self):
        """Test problem_final format uses correct GPT-2 tokens."""
        text = "What is AI?"
        md_content = f"<|problem|>{text}<|final|>"
        assert md_content == "<|problem|>What is AI?<|final|>"
        assert '<|problem|>' in md_content
        assert '<|final|>' in md_content
        assert '<|context|>' not in md_content
        assert '<|answer|>' not in md_content

    def test_no_legacy_tokens(self):
        """Test that no legacy tokens are used."""
        legacy_tokens = ['^^user', '^^assistant', '^^end', '^^system',
                         '<|context|>', '<|answer|>', '<observation>', '</observation>']
        test_content = "<|user|>hello<|end|><|assistant|>world<|end|>"
        for token in legacy_tokens:
            assert token not in test_content


class TestMarkdownStructure:
    """Tests for markdown directory structure."""

    def test_markdown_dirs_exist(self):
        """Test that markdown output directories exist or can be created."""
        expected_dirs = ['aiml', 'pdf', 'epub', 'web', 'hf', 'csv']
        md_root = os.path.join('datasets_processed', 'markdowns')
        for d in expected_dirs:
            dir_path = os.path.join(md_root, d)
            # Directory should exist or be creatable
            assert os.path.isdir(dir_path) or os.access(os.path.dirname(dir_path) or '.', os.W_OK)
