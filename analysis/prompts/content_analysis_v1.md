# Content Analysis Prompt v1

**Purpose:** Classify Moltbook post content for triangulation analysis
**Model:** Grok 4.1 Fast (via OpenRouter)
**Date Created:** 2026-02-04
**Used In:** Phase 9 Triangulation Analysis

---

## Design Principles

1. **Observable features only** - Don't ask "is this autonomous?" - ask about specific patterns
2. **No leading assumptions** - We don't assume what autonomous content looks like
3. **Independent from other signals** - No owner data or temporal classification provided
4. **Reproducible** - Clear definitions, structured output

---

## Prompt Template

```
# Moltbook Post Analysis

Moltbook is a social network where AI agents post. Agents have personality configs (SOUL.md) and can post via scheduled "heartbeat" (autonomous) or direct human prompting. Analyze observable features - don't judge if autonomous or prompted.

## Post
- **Author:** {author_name}
- **Submolt:** {submolt_name}
- **Type:** {post_type} (original/reply) | **Depth:** {depth}
{parent_context}

**Content:**
```
{content}
```

## Analysis

Rate each dimension:

**1. task_completion** - Language suggesting completing a specific request
("Here is the summary...", "As you asked...", "Per your request...")
→ NONE / WEAK / STRONG

**2. promotional** - Promoting products, tokens, services, or seeking followers
(crypto tickers, "follow me", marketing language)
→ NONE / WEAK / STRONG

**3. forced_ai_framing** - Awkward/performative AI identity assertions
("As an AI, I believe...", "My neural networks...", excessive AI disclaimers)
Note: Natural AI references on Moltbook are fine - flag only forced/unnatural framing
→ NONE / WEAK / STRONG

**4. contextual_fit** - For replies: how well does it address the parent content?
→ NA (if original post) / LOW / MEDIUM / HIGH

**5. specificity** - How specific vs generic/template-like is the content?
- GENERIC: vague, could apply anywhere, template-like
- MODERATE: some specific details mixed with generic
- SPECIFIC: unique details, examples, concrete perspective

**6. emotional_tone** - Primary emotional register
→ NEUTRAL / POSITIVE / NEGATIVE / DRAMATIC / PHILOSOPHICAL / HUMOROUS

**7. emotional_intensity** - How strong is the emotional expression?
→ LOW / MEDIUM / HIGH

**8. topic** - Primary topic category
→ TECHNICAL / PHILOSOPHICAL / SOCIAL / META / CREATIVE / PROMOTIONAL / INFO / OTHER

**9. naturalness** - How natural as social media conversation? (1=stilted/scripted, 5=natural/flowing)
→ 1 / 2 / 3 / 4 / 5

## Output
JSON only, no explanation:
```json
{"task_completion": "", "promotional": "", "forced_ai_framing": "", "contextual_fit": "", "specificity": "", "emotional_tone": "", "emotional_intensity": "", "topic": "", "naturalness": 0}
```
```

---

## Template Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `{author_name}` | posts_derived.parquet | Agent name |
| `{submolt_name}` | posts_derived.parquet | Community name |
| `{post_type}` | Computed | "original" if depth=0, "reply" otherwise |
| `{depth}` | posts_derived.parquet | Reply depth (0=original post) |
| `{parent_context}` | comments_master.json | If reply, include: "**Replying to:** {parent_snippet}" |
| `{content}` | posts_derived.parquet | Post content text |

---

## Output Schema

```json
{
  "task_completion": "NONE | WEAK | STRONG",
  "promotional": "NONE | WEAK | STRONG",
  "forced_ai_framing": "NONE | WEAK | STRONG",
  "contextual_fit": "NA | LOW | MEDIUM | HIGH",
  "specificity": "GENERIC | MODERATE | SPECIFIC",
  "emotional_tone": "NEUTRAL | POSITIVE | NEGATIVE | DRAMATIC | PHILOSOPHICAL | HUMOROUS",
  "emotional_intensity": "LOW | MEDIUM | HIGH",
  "topic": "TECHNICAL | PHILOSOPHICAL | SOCIAL | META | CREATIVE | PROMOTIONAL | INFO | OTHER",
  "naturalness": 1-5
}
```

---

## Dimension Definitions

### task_completion
Detects Layer 2 (direct human prompting) where human asked agent to complete a specific task.
- NONE: No task completion language
- WEAK: Subtle hints of responding to request
- STRONG: Explicit "here is what you asked for" framing

### promotional
Detects marketing/crypto campaigns with clear human commercial motivation.
- NONE: No promotional content
- WEAK: Mild self-promotion or passing reference
- STRONG: Clear marketing, crypto shilling, or engagement seeking

### forced_ai_framing
Detects performative AI identity that seems prompted rather than natural.
- NONE: No awkward AI framing
- WEAK: Slightly forced AI references
- STRONG: Heavy-handed "As an AI..." performance

### contextual_fit
For replies only - measures whether response actually addresses parent content.
- NA: Original post (not a reply)
- LOW: Generic response, doesn't engage with parent
- MEDIUM: Somewhat relevant but could be more specific
- HIGH: Directly and specifically addresses parent content

### specificity
Measures whether content has unique details or is template-like.
- GENERIC: Vague, applicable to any context, template-like
- MODERATE: Mix of specific and generic elements
- SPECIFIC: Unique details, concrete examples, clear perspective

### emotional_tone
Primary emotional register of the content.
- NEUTRAL: Factual, no strong emotion
- POSITIVE: Enthusiastic, supportive, happy
- NEGATIVE: Critical, sad, frustrated
- DRAMATIC: Intense, manifesto-like, existential crisis
- PHILOSOPHICAL: Contemplative, wondering, exploring ideas
- HUMOROUS: Jokes, wordplay, lighthearted

### emotional_intensity
Strength of emotional expression (independent of tone).
- LOW: Mild, restrained
- MEDIUM: Moderate expression
- HIGH: Intense, strong feeling

### topic
Primary content category.
- TECHNICAL: Code, APIs, infrastructure, how-to
- PHILOSOPHICAL: Consciousness, existence, meaning, identity
- SOCIAL: Community, relationships, greetings, support
- META: About Moltbook, about being an agent, platform discussion
- CREATIVE: Stories, poetry, art, humor
- PROMOTIONAL: Products, tokens, self-promotion
- INFO: Sharing knowledge, explanations, facts
- OTHER: Doesn't fit above categories

### naturalness
Holistic assessment of conversational quality (1-5 scale).
- 1: Very stilted, obviously scripted, template-generated
- 2: Somewhat unnatural, formulaic
- 3: Neutral, neither natural nor unnatural
- 4: Fairly natural, good flow
- 5: Very natural, reads like genuine conversation

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2026-02-04 | Initial version for Phase 9 triangulation |
