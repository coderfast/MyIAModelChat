# AIML 2.0 - Artificial Intelligence Markup Language

## Overview

AIML (Artificial Intelligence Markup Language) is an XML dialect for creating natural language software agents. Developed by Richard S. Wallace and a worldwide free software community between 1995 and 2002.

- **Latest version:** AIML 2.1 (June 20, 2018)
- **File extension:** `.aiml`
- **Based on:** XML
- **Author:** Dr. Richard S. Wallace
- **Specification:** https://www.aiml.foundation/doc.html

---

## Core Concepts

### Category

The fundamental unit of knowledge in AIML. Each category consists of a **pattern** and a **template**.

```xml
<category>
  <pattern>WHAT IS YOUR NAME</pattern>
  <template>My name is Alice.</template>
</category>
```

### Pattern

A string that matches user input. Patterns are case-insensitive.

- **Literal:** `WHAT IS YOUR NAME` - matches exactly
- **Wildcard `*`:** Matches one or more words
- **Wildcard `_`:** Matches one or more words (lower priority than `*`)
- **`<that>`:** Matches based on bot's previous response

```xml
<pattern>WHAT IS YOUR *</pattern>
<!-- Matches: "What is your name", "What is your purpose", etc. -->
```

### Template

Specifies the response to a matched pattern.

```xml
<template>My name is <bot name="name"/>.</template>
```

---

## AIML 2.0 Elements

### Basic Elements

| Element | Description |
|---------|-------------|
| `<aiml>` | Root element of AIML document |
| `<category>` | Fundamental unit: pattern + template |
| `<pattern>` | Matches user input |
| `<template>` | Response to matched pattern |
| `<that>` | Context from previous bot response |
| `<topic>` | Groups categories by topic |
| `<star>` | Wildcard match capture |
| `<bot>` | Bot properties |
| `<person>` | First/second person conversion |
| `<person2>` | Second/third person conversion |
| `<gender>` | Gender-specific pronoun swap |

### Control Elements

| Element | Description |
|---------|-------------|
| `<srai>` | Symbolic Reduction - redirects to another pattern |
| `<random>` | Random response selection |
| `<li>` | Item in a list (used with `<random>`, `<loop>`) |
| `<condition>` | Conditional response (if/else logic) |
| `<set>` | Set a variable |
| `<get>` | Get a variable value |
| `<loop>` | Loop through list items |

### Template Elements

| Element | Description |
|---------|-------------|
| `<br/>` | Line break |
| `<p>` | Paragraph |
| `<a>` | Hyperlink |
| `<image>` | Image |
| `<video>` | Video |
| `<audio>` | Audio |
| `<embed>` | Embedded content |
| `<svg>` | SVG image |
| `<table>` | HTML table |
| `<tr>` | Table row |
| `<td>` | Table cell |

### Thinking Elements (AIML 2.0)

| Element | Description |
|---------|-------------|
| `<thinking>` | Internal reasoning (not shown to user) |
| `<system>` | System-level operations |
| `<date>` | Current date/time |
| `<eval>` | Execute system commands |
| `<explode>` | Separate characters with spaces |
| `<implode>` | Join separate words |
| `<formal>` | Capitalize each word |
| `<uppercase>` | Convert to uppercase |
| `<lowercase>` | Convert to lowercase |
| `<sentence>` | Capitalize first letter |
| `<word>` | Number-to-word conversion |
```

---

## Wildcards

### `*` (Star) - Matches One or More Words

```xml
<category>
  <pattern>WHAT IS YOUR *</pattern>
  <template>
    I am a bot. You asked about <star/>.
  </template>
</category>
```

### `_` (Underscore) - Matches One or More Words (Lower Priority)

```xml
<category>
  <pattern>_ IS YOUR NAME</pattern>
  <template>My name is Alice.</template>
</category>
```

### `**` (Double Star) - Matches Zero or More Words (AIML 2.0)

```xml
<category>
  <pattern>TELL ME ABOUT **</pattern>
  <template>
    Here is what I know about <star index="1"/>.
  </template>
</category>
```

### `^` (Caret) - Matches Zero or One Word (AIML 2.0)

```xml
<category>
  <pattern>WHAT IS ^ NAME</pattern>
  <template>My name is Alice.</template>
</category>
```

---

## Person and Gender

### `<person>` - First/Second Person Conversion

```xml
<template>
  <person>You said that <star/> is your favorite.</person>
</template>
<!-- "I like cats" becomes "You said that cats is your favorite." -->
```

### `<person2>` - Second/Third Person Conversion

```xml
<template>
  <person2>He said that <star/> is his favorite.</person2>
</template>
```

### `<gender>` - Gender Pronoun Swap

```xml
<template>
  <gender>He said he likes her.</gender>
</template>
<!-- Becomes: "She said she likes him." -->
```

---

## Conditional Responses

### Simple Condition

```xml
<template>
  <condition name="user-mood">
    <li value="happy">Great! You're in a good mood!</li>
    <li value="sad">I'm sorry to hear that.</li>
    <li>I don't know how you feel.</li>
  </condition>
</template>
```

### Name-Value Pair Condition

```xml
<template>
  <condition name="user-age">
    <li value="child">You're too young for this.</li>
    <li value="teen">Welcome, teenager!</li>
    <li value="adult">Welcome, adult!</li>
  </condition>
</template>
```

---

## Topic System

```xml
<topic name="WEATHER">
  <category>
    <pattern>WHAT IS THE WEATHER</pattern>
    <template>The weather is nice today.</template>
  </category>
  <category>
    <pattern>IS IT RAINING</pattern>
    <template>No, it's sunny.</template>
  </category>
</topic>
```

---

## Symbolic Reduction (SRAI)

SRAI redirects one pattern to another. Useful for synonyms and abbreviations.

```xml
<!-- Synonym handling -->
<category>
  <pattern>WHAT ARE YOU CALLED</pattern>
  <template><srai>what is your name</srai></template>
</category>

<category>
  <pattern>WHAT DO THEY CALL YOU</pattern>
  <template><srai>what is your name</srai></template>
</category>

<!-- Abbreviations -->
<category>
  <pattern>BTW *</pattern>
  <template><srai>by the way <star/></srai></template>
</category>
```

---

## Random Responses

```xml
<template>
  <random>
    <li>Hello!</li>
    <li>Hi there!</li>
    <li>Hey!</li>
    <li>Welcome!</li>
  </random>
</template>
```

---

## Variables

### Setting Variables

```xml
<template>
  <set name="user-name"><star/></set>
  Nice to meet you, <get name="user-name"/>!
</template>
```

### Bot Properties

```xml
<template>
  I am <bot name="name"/>, version <bot name="version"/>.
</template>
```

---

## Thinking (AIML 2.0)

Internal reasoning not shown to the user.

```xml
<template>
  <thinking>
    The user is asking about weather. 
    I should check the current conditions.
  </thinking>
  The weather is sunny today.
</template>
```

---

## System Operations (AIML 2.0)

### `<eval>` - Execute System Commands

```xml
<template>
  <eval>date +%Y-%m-%d</eval>
</template>
```

### `<date>` - Current Date/Time

```xml
<template>
  Today is <date format="YYYY-MM-DD"/>.
</template>
```

### `<system>` - System-Level Operations

```xml
<template>
  <system>hostname</system>
</template>
```

---

## Formatting Elements

```xml
<template>
  <!-- Line breaks and paragraphs -->
  First line<br/>Second line
  
  <p>Paragraph text</p>
  
  <!-- Text formatting -->
  <uppercase><star/></uppercase>
  <lowercase><star/></lowercase>
  <formal>hello world</formal>
  <sentence>hello world</sentence>
  
  <!-- Links and images -->
  <a href="https://example.com">Click here</a>
  <image>https://example.com/image.png</image>
</template>
```

---

## Wildcard Indexing

When multiple wildcards are used, `<star index="N"/>` accesses the Nth wildcard match.

```xml
<category>
  <pattern>MY * IS *</pattern>
  <template>
    Your <star index="1"/> is <star index="2"/>.
  </template>
</category>
```

---

## That - Context from Previous Response

```xml
<category>
  <pattern>YES</pattern>
  <that>DO YOU LIKE *</that>
  <template>
    I'm glad you like <star/>!
  </template>
</category>
```

---

## System Response Tag

Used to set the bot's previous response for `<that>` matching.

```xml
<template>
  <set name="that">How can I help you?</set>
  How can I help you?
</template>
```

---

## AIML 2.0 New Features

### 1. Thinking Element
```xml
<thinking>Internal reasoning...</thinking>
```

### 2. Double Star Wildcard (`**`)
Matches zero or more words.

### 3. Caret Wildcard (`^`)
Matches zero or one word.

### 4. Date/Time Support
```xml
<date format="YYYY-MM-DD HH:MM:SS"/>
```

### 5. System Commands
```xml
<eval>command</eval>
<system>command</system>
```

### 6. Improved Condition
```xml
<condition name="var" value="val">
  <li>Response</li>
</condition>
```

---

## AIML File Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<aiml version="2.0">
  
  <category>
    <pattern>HELLO</pattern>
    <template>Hello! How can I help you?</template>
  </category>
  
  <category>
    <pattern>MY NAME IS *</pattern>
    <template>
      <set name="user-name"><star/></set>
      Nice to meet you, <get name="user-name"/>!
    </template>
  </category>
  
  <category>
    <pattern>WHAT IS MY NAME</pattern>
    <template>
      <condition name="user-name">
        <li>Your name is <get name="user-name"/>.</li>
        <li>I don't know your name yet.</li>
      </condition>
    </template>
  </category>
  
  <topic name="WEATHER">
    <category>
      <pattern>WHAT IS THE WEATHER</pattern>
      <template>The weather is nice today.</template>
    </category>
  </topic>
  
</aiml>
```

---

## Pattern Matching Priority

1. Literal patterns (no wildcards) have highest priority
2. `_` (underscore) wildcards have higher priority than `*`
3. `*` (star) wildcards have lower priority
4. Longer patterns have higher priority than shorter ones
5. `<that>` patterns are evaluated after `<pattern>` match

---

## References

- **AIML Foundation:** https://www.aiml.foundation
- **Pandorabots:** https://www.pandorabots.com
- **AIML 2.0 Specification:** http://www.aiml.foundation/doc.html
- **GitHub - aiml:** https://github.com/pandorabots/aiml
- **Wikipedia:** https://en.wikipedia.org/wiki/Artificial_Intelligence_Markup_Language

---

*Last updated: 2026-07-30*
