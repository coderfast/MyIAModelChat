# AIML Usage Guide

## AIML Files in This Project

### AIML Directory Structure
The `datasets_source/aiml/` directory contains AIML files for dialogue patterns and the `dataset_preparer/aiml/parser.py` module processes these files.

### Major AIML Files

**Core Dialogue**
- `bot.aiml`: Bot profile and identity (434KB)
- `atomic.aiml`: Atomic dialogue patterns (407KB)
- `ai.aiml`: AI-related patterns (42KB)
- `alice.aiml`: ALICE bot responses (33KB)
- `astrology.aiml`: Astrology patterns (2.4KB)

### AIML Pattern Format

Basic AIML pattern:
```xml
<category>
    <pattern>HELLO</pattern>
    <template>Hi there!</template>
</category>
```

With wildcards (resolved by parser):
```xml
<category>
    <pattern>MY NAME IS *</pattern>
    <template>Nice to meet you, <star/>!</template>
</category>
```

With `<srai>` reduction:
```xml
<category>
    <pattern>HI</pattern>
    <srai>HELLO</srai>
</category>
```

With `<random>` responses:
```xml
<category>
    <pattern>HELLO</pattern>
    <template>
        <random>
            <li>Hi there!</li>
            <li>Hello!</li>
            <li>Hey!</li>
        </random>
    </template>
</category>
```

## Parser Features

The AIML Parser (`dataset_preparer/aiml/parser.py`) resolves:
- `<srai>` chains (max depth 5)
- `<random><li>` → generates N samples (one per `<li>`)
- Wildcards `*`, `_`, `**`, `^` → expanded with contextual examples
- `<thinking>` → eliminated (AIML's `<thinking>` element is stripped to avoid collision with training format)
- `<set>`, `<get>`, `<bot>` → resolved with defaults
- HTML tags → stripped

## Integrating AIML with the Model

### Current Integration (main.py)
- AIML files are parsed via `dataset_preparer/aiml/parser.py`
- Patterns resolved and expanded into training samples
- Combined with other data sources
- Used to train the neural model

### Usage in DialogueManager
- Intent and sentiment detected first
- Context maintained across turns
- Persona modeling applied
- Neural model generates response
- AIML patterns used for fallback/validation

## Best Practices

1. **Keep patterns simple and clear**: More specific patterns should be listed first
2. **Use wildcards efficiently**: `*` matches any text, `_` matches single word
3. **Organize by category**: Group related patterns in separate files
4. **Test patterns**: Verify reduction files work correctly
5. **Update reduction files**: Keep pattern reduction rules up-to-date
6. **Use context (that/topic)**: Make conversations more coherent

---

*Updated: 2026-07-30 - Reflects AIML 2.0 parser implementation*
