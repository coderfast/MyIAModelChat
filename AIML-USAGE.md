# AIML Usage Guide

## AIML Files in This Project

### AIML Directory Structure
The `aiml/` directory contains ~60 AIML files for dialogue patterns and the `aimlloder.py` module loads these files.

### Major AIML Files

**Core Dialogue**
- `salutations.aiml`: Greeting patterns
- `default.aiml`: Default responses
- `that.aiml`: Context-aware responses based on what bot said
- `continuation.aiml`: Conversation continuation patterns

**Knowledge Categories**
- `alice.aiml`: ALICE bot responses
- `knowledge.aiml`: General knowledge responses
- `science.aiml`: Science-related patterns
- `history.aiml`: Historical information
- `geography.aiml`: Geographic information
- `movies.aiml`: Movie-related patterns
- `music.aiml`: Music-related patterns
- `sports.aiml`: Sports-related patterns
- `food.aiml`: Food-related patterns

**NLP Processing**
- `reduction0.safe.aiml` - `reduction4.safe.aiml`: Pattern reduction and normalization
- `reductions-update.aiml`: Updated reductions

**Personality**
- `bot_profile.aiml`: Bot personality definition
- `client_profile.aiml`: Client profile handling
- `personality.aiml`: Personality traits
- `emotion.aiml`: Emotional responses

### AIML Pattern Format

Basic AIML pattern:
```xml
<category>
    <pattern>HELLO</pattern>
    <template>Hi there!</template>
</category>
```

With wildcards:
```xml
<category>
    <pattern>MY NAME IS *</pattern>
    <template>Nice to meet you, <star/>!</template>
</category>
```

With context:
```xml
<category>
    <pattern>HELLO</pattern>
    <that>*</that>
    <template>Hello back!</template>
</category>
```

## Integrating AIML with the Model

### Current Integration (main_train.py)
- AIML files are loaded via `aimlloder.py`
- Patterns converted to training data
- Combined with Hugging Face datasets
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

## Development AIML Files

The `aiml_dev/` directory contains AIML files with associated `.datasets` files:
- Used for development and testing
- Allows dataset tracking with patterns
- Enables iteration on patterns before moving to production
