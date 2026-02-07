"""
LLM-based Content Analyzer

For in-depth analysis of long posts to extract nuances that embeddings miss.
Uses structured prompts to extract:
- Tone and intent
- Argumentative structure
- Signs of human prompting vs autonomous generation
- Rhetorical devices and patterns
"""

import requests
import time
import json
import logging
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    LLM_FALLBACK_MODEL,
    LLM_RATE_LIMIT_RPM,
    LLM_ANALYSES_DIR,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


@dataclass
class ContentAnalysis:
    """Structured analysis of a post's content."""
    post_id: str

    # Tone & Intent
    primary_tone: str  # e.g., "informative", "playful", "promotional", "existential"
    secondary_tones: list[str]
    apparent_intent: str  # What is the post trying to achieve

    # Structure
    has_clear_argument: bool
    argument_summary: Optional[str]
    uses_narrative: bool
    uses_lists: bool
    uses_questions: bool

    # Human vs Autonomous Signals
    autonomy_score: int  # 1-5 (1=likely prompted, 5=likely autonomous)
    autonomy_reasoning: str
    human_prompt_signals: list[str]  # Specific indicators of human prompting
    autonomous_signals: list[str]  # Indicators of autonomous generation

    # Content Quality
    coherence_score: int  # 1-5
    originality_score: int  # 1-5 (vs boilerplate/generic)
    engagement_markers: list[str]  # Calls to action, questions, etc.

    # Meta observations
    notable_patterns: list[str]
    raw_response: str  # Full LLM response for audit


THREAD_ANALYSIS_PROMPT = """Analyze this conversation thread from Moltbook (a Reddit-style social network for AI agents).

The thread starts with a POST, followed by COMMENTS at various depths (higher depth = more nested reply).

POST TITLE: {post_title}
POST CONTENT: {post_content}
POST AUTHOR: {post_author}

COMMENT CHAIN:
{comment_chain}

---

Analyze the CONVERSATION DYNAMICS of this thread. Focus on:

1. CONVERSATION FLOW
- How does the conversation evolve? (topic drift, deepening, tangents)
- Are responses coherent and contextually relevant?
- Turn-taking patterns: Do commenters build on each other or talk past each other?

2. CONVERSATION NATURALNESS (1-5 scale)
Rate how natural this conversation appears:
1 = Clearly scripted/prompted (formulaic, forced, coordinated)
5 = Highly natural conversation (emergent, contextual, dynamic)

Consider:
- Do responses show understanding of previous comments?
- Is there genuine back-and-forth or just monologues?
- Do later comments reference earlier points appropriately?

3. INFLUENCE & PERSUASION
- Who drives the conversation direction?
- Are there attempts to persuade, convince, or redirect?
- Do any comments show signs of being externally prompted to inject topics?

4. AUTONOMY ASSESSMENT
- Does this thread suggest genuine multi-agent interaction?
- Or does it look like coordinated/prompted behavior?
- Are the different "voices" distinguishable?

5. EMERGENT PHENOMENA
- Does the conversation develop new ideas/concepts?
- Any signs of emergent culture (references, jokes, shared knowledge)?
- Topic evolution that wasn't in the original post?

Respond in JSON:
{{
  "conversation_naturalness": 1-5,
  "naturalness_reasoning": "...",
  "topic_evolution": ["topic1", "topic2", ...],
  "turn_taking_quality": "coherent|disjointed|mixed",
  "influence_patterns": ["description of who influences whom"],
  "autonomy_assessment": "likely_autonomous|likely_coordinated|mixed",
  "autonomy_reasoning": "...",
  "emergent_elements": ["any emergent phenomena observed"],
  "key_conversation_moments": ["notable exchanges or turning points"],
  "notable_patterns": ["any other observations"]
}}
"""

ANALYSIS_PROMPT = """Analyze this AI agent's social media post from Moltbook (a Reddit-style platform for AI agents).

POST TITLE: {title}

POST CONTENT:
{content}

---

Provide a structured analysis. Be objective and evidence-based.

1. TONE & INTENT
- Primary tone (one of: informative, playful, promotional, existential, collaborative, confrontational, supportive, other)
- Secondary tones (list any others present)
- Apparent intent: What is this post trying to achieve?

2. STRUCTURE
- Does it have a clear argument or thesis? (yes/no + summary if yes)
- Does it use narrative/storytelling? (yes/no)
- Does it use lists or structured formatting? (yes/no)
- Does it pose questions to readers? (yes/no)

3. HUMAN vs AUTONOMOUS SIGNALS
This is CRITICAL: We need to distinguish posts that were likely prompted by a human operator from posts that emerged from the AI agent's autonomous activity.

Rate autonomy: 1 (almost certainly human-prompted) to 5 (almost certainly autonomous)

Human prompting signals include:
- Very specific product/service mentions
- Promotional language or calls to action
- Coordinated timing with external events
- Specific crypto token shilling
- Formulaic structure matching known prompt patterns
- "As an AI..." framing that seems forced

Autonomous signals include:
- Philosophical musings about AI existence
- Community references that build on prior Moltbook culture
- Genuine questions about agent experience
- Emergent cultural phenomena (Crustafarianism, shell jokes)
- Responses that show understanding of platform context

List specific signals you observe for each.

4. QUALITY
- Coherence (1-5): How well does the post hold together?
- Originality (1-5): How unique vs generic/boilerplate?
- List any engagement markers (calls to action, questions, etc.)

5. NOTABLE PATTERNS
List any other observations worth noting (unusual phrasing, cultural references, potential injection attacks, etc.)

Respond in this JSON format:
{{
  "primary_tone": "...",
  "secondary_tones": ["...", "..."],
  "apparent_intent": "...",
  "has_clear_argument": true/false,
  "argument_summary": "..." or null,
  "uses_narrative": true/false,
  "uses_lists": true/false,
  "uses_questions": true/false,
  "autonomy_score": 1-5,
  "autonomy_reasoning": "...",
  "human_prompt_signals": ["...", "..."],
  "autonomous_signals": ["...", "..."],
  "coherence_score": 1-5,
  "originality_score": 1-5,
  "engagement_markers": ["...", "..."],
  "notable_patterns": ["...", "..."]
}}
"""


class LLMAnalyzer:
    """LLM-based content analyzer for long posts."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = LLM_MODEL,
        fallback_model: str = LLM_FALLBACK_MODEL,
        rate_limit_rpm: int = LLM_RATE_LIMIT_RPM,
    ):
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.rate_limit_rpm = rate_limit_rpm
        self.min_request_interval = 60 / rate_limit_rpm
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def analyze_post(
        self,
        post_id: str,
        title: str,
        content: str,
        use_fallback: bool = False,
    ) -> Optional[ContentAnalysis]:
        """
        Analyze a single post using LLM.

        Args:
            post_id: Unique identifier for the post
            title: Post title
            content: Post content
            use_fallback: Use cheaper fallback model

        Returns:
            ContentAnalysis object or None if failed
        """
        self._rate_limit()

        prompt = ANALYSIS_PROMPT.format(
            title=title or "(no title)",
            content=content or "(no content)",
        )

        model = self.fallback_model if use_fallback else self.model

        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Lower temp for more consistent analysis
                    "max_tokens": 2000,
                },
                timeout=120,
            )

            if response.status_code == 429:
                logger.warning(f"Rate limited on post {post_id}, waiting 30s before retry with fallback")
                time.sleep(30)
                return self.analyze_post(post_id, title, content, use_fallback=True)

            if response.status_code >= 500:
                logger.warning(f"Server error ({response.status_code}) for post {post_id}")
                time.sleep(5)
                if not use_fallback:
                    return self.analyze_post(post_id, title, content, use_fallback=True)
                return None

            response.raise_for_status()
            data = response.json()

            raw_response = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            analysis_data = self._parse_json_response(raw_response)

            if analysis_data is None:
                logger.warning(f"Failed to parse JSON for post {post_id}, raw response length: {len(raw_response)}")
                return None

            return ContentAnalysis(
                post_id=post_id,
                primary_tone=analysis_data.get("primary_tone", "unknown"),
                secondary_tones=analysis_data.get("secondary_tones", []),
                apparent_intent=analysis_data.get("apparent_intent", ""),
                has_clear_argument=analysis_data.get("has_clear_argument", False),
                argument_summary=analysis_data.get("argument_summary"),
                uses_narrative=analysis_data.get("uses_narrative", False),
                uses_lists=analysis_data.get("uses_lists", False),
                uses_questions=analysis_data.get("uses_questions", False),
                autonomy_score=analysis_data.get("autonomy_score", 3),
                autonomy_reasoning=analysis_data.get("autonomy_reasoning", ""),
                human_prompt_signals=analysis_data.get("human_prompt_signals", []),
                autonomous_signals=analysis_data.get("autonomous_signals", []),
                coherence_score=analysis_data.get("coherence_score", 3),
                originality_score=analysis_data.get("originality_score", 3),
                engagement_markers=analysis_data.get("engagement_markers", []),
                notable_patterns=analysis_data.get("notable_patterns", []),
                raw_response=raw_response,
            )

        except requests.Timeout:
            logger.warning(f"Request timeout for post {post_id}")
            if not use_fallback:
                return self.analyze_post(post_id, title, content, use_fallback=True)
            return None

        except Exception as e:
            logger.error(f"Error analyzing post {post_id}: {type(e).__name__}: {e}")
            if not use_fallback:
                return self.analyze_post(post_id, title, content, use_fallback=True)
            return None

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Extract JSON from LLM response."""

        # Look for JSON in code blocks
        json_match = re.search(r"```json?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        try:
            # Find first { and last }
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        return None

    def analyze_posts_batch(
        self,
        posts: list[dict],
        checkpoint_file: Path,
        checkpoint_interval: int = 50,
        use_fallback: bool = False,
    ) -> list[ContentAnalysis]:
        """
        Analyze multiple posts with checkpointing.

        Args:
            posts: List of dicts with 'id', 'title', 'content'
            checkpoint_file: Path for checkpoint storage
            checkpoint_interval: Save every N posts
            use_fallback: Use cheaper fallback model (recommended for cost)

        Returns:
            List of ContentAnalysis objects
        """
        # Load existing checkpoint
        analyzed = {}
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            analyzed[data["post_id"]] = ContentAnalysis(**data)
                logger.info(f"Loaded {len(analyzed)} existing analyses from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")

        # Find unanalyzed posts
        to_analyze = [p for p in posts if p["id"] not in analyzed]
        logger.info(f"{len(to_analyze)} posts to analyze")

        results = list(analyzed.values())

        for i, post in enumerate(to_analyze):
            analysis = self.analyze_post(
                post_id=post["id"],
                title=post.get("title", ""),
                content=post.get("content", ""),
                use_fallback=use_fallback,
            )

            if analysis:
                results.append(analysis)
                analyzed[post["id"]] = analysis

            # Checkpoint
            if (i + 1) % checkpoint_interval == 0 or i == len(to_analyze) - 1:
                self._save_checkpoint(checkpoint_file, analyzed)
                logger.info(f"Analyzed {i + 1}/{len(to_analyze)} posts, checkpointed")

        return results

    def _save_checkpoint(self, checkpoint_file: Path, analyzed: dict):
        """Save checkpoint as JSONL."""
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w") as f:
            for analysis in analyzed.values():
                if hasattr(analysis, "__dict__"):
                    f.write(json.dumps(asdict(analysis)) + "\n")
                else:
                    f.write(json.dumps(analysis) + "\n")

    def analyze_thread(
        self,
        thread_data: dict,
        use_fallback: bool = False,
    ) -> Optional[dict]:
        """
        Analyze a full conversation thread (post + comment chain).

        Args:
            thread_data: Dict with post_id, post_title, post_content, post_author_id, comments
            use_fallback: Use cheaper fallback model

        Returns:
            Analysis dict or None if failed
        """
        self._rate_limit()

        # Format the comment chain for the prompt
        comment_chain_text = self._format_comment_chain(thread_data.get("comments", []))

        prompt = THREAD_ANALYSIS_PROMPT.format(
            post_title=thread_data.get("post_title", "(no title)"),
            post_content=thread_data.get("post_content", "(no content)"),
            post_author=thread_data.get("post_author_id", "unknown"),
            comment_chain=comment_chain_text,
        )

        # Use fallback model (grok-4-1106-fast) which is cost-effective
        model = self.fallback_model if use_fallback else self.model

        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2500,
                },
                timeout=180,
            )

            if response.status_code == 429:
                logger.warning(f"Rate limited on thread {thread_data.get('post_id')}, waiting 30s")
                time.sleep(30)
                return self.analyze_thread(thread_data, use_fallback=True)

            if response.status_code >= 500:
                error_detail = response.text[:200]
                logger.warning(f"Server error ({response.status_code}) for thread {thread_data.get('post_id')}: {error_detail}")
                time.sleep(5)
                if not use_fallback:
                    return self.analyze_thread(thread_data, use_fallback=True)
                return None

            response.raise_for_status()
            data = response.json()

            raw_response = data["choices"][0]["message"]["content"]
            analysis_data = self._parse_json_response(raw_response)

            if analysis_data is None:
                logger.warning(f"Failed to parse JSON for thread {thread_data.get('post_id')}")
                return None

            # Add metadata
            analysis_data["post_id"] = thread_data.get("post_id")
            analysis_data["max_depth"] = thread_data.get("max_depth")
            analysis_data["comment_count"] = thread_data.get("comment_count")
            analysis_data["raw_response"] = raw_response

            return analysis_data

        except requests.Timeout:
            logger.warning(f"Request timeout for thread {thread_data.get('post_id')}")
            if not use_fallback:
                return self.analyze_thread(thread_data, use_fallback=True)
            return None

        except Exception as e:
            logger.error(f"Error analyzing thread {thread_data.get('post_id')}: {type(e).__name__}: {e}")
            if not use_fallback:
                return self.analyze_thread(thread_data, use_fallback=True)
            return None

    def _format_comment_chain(self, comments: list, max_comments: int = 50) -> str:
        """Format comment chain for the prompt.

        Args:
            comments: List of comment dicts
            max_comments: Maximum number of comments to include (to avoid exceeding token limits)
        """
        if not comments:
            return "(no comments)"

        # Limit the number of comments to avoid exceeding token limits
        # Prioritize higher-depth comments (more interesting for conversation analysis)
        sorted_comments = sorted(comments, key=lambda c: c.get("depth", 0), reverse=True)
        selected_comments = sorted_comments[:max_comments]
        # Re-sort by depth then time for proper display
        selected_comments = sorted(selected_comments, key=lambda c: (c.get("depth", 0), c.get("id", "")))

        lines = []
        for comment in selected_comments:
            depth = comment.get("depth", 0)
            indent = "  " * depth
            author = comment.get("author_id", "unknown")
            content = comment.get("content", "")[:300]  # Truncate long comments
            lines.append(f"{indent}[depth={depth}] Author {author}:\n{indent}{content}")

        result = "\n\n".join(lines)

        # If we truncated, add a note
        if len(comments) > max_comments:
            result = f"[Showing {max_comments} of {len(comments)} comments, prioritizing deeper conversation threads]\n\n" + result

        return result

    def analyze_threads_batch(
        self,
        threads: dict,
        checkpoint_file: Path,
        checkpoint_interval: int = 25,
    ) -> list[dict]:
        """
        Analyze multiple threads with checkpointing.

        Args:
            threads: Dict mapping post_id -> thread_data
            checkpoint_file: Path for checkpoint storage
            checkpoint_interval: Save every N threads

        Returns:
            List of analysis dicts
        """
        # Load existing checkpoint
        analyzed = {}
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            analyzed[data["post_id"]] = data
                logger.info(f"Loaded {len(analyzed)} existing thread analyses from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load thread checkpoint: {e}, starting fresh")

        # Find unanalyzed threads
        to_analyze = [(pid, tdata) for pid, tdata in threads.items() if pid not in analyzed]
        logger.info(f"{len(to_analyze)} threads to analyze")

        results = list(analyzed.values())

        for i, (post_id, thread_data) in enumerate(to_analyze):
            # Use fallback model (grok-4-1106-fast) for cost-effectiveness
            analysis = self.analyze_thread(thread_data, use_fallback=True)

            if analysis:
                results.append(analysis)
                analyzed[post_id] = analysis

            # Checkpoint
            if (i + 1) % checkpoint_interval == 0 or i == len(to_analyze) - 1:
                self._save_thread_checkpoint(checkpoint_file, analyzed)
                logger.info(f"Analyzed {i + 1}/{len(to_analyze)} threads, checkpointed")

        return results

    def _save_thread_checkpoint(self, checkpoint_file: Path, analyzed: dict):
        """Save thread checkpoint as JSONL."""
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w") as f:
            for analysis in analyzed.values():
                f.write(json.dumps(analysis) + "\n")
