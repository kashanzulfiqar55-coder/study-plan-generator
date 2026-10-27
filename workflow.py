"""Multi-stage AI workflow for personalized study-pack generation.

Stages:
1. Plan: design learning objectives and pack outline.
2. Generate: create the first draft from the plan and source notes.
3. Assess: check coverage, clarity, factual caution, and completeness.
4. Review: produce a structured editorial review with actionable fixes.
5. Refine: revise the draft using the assessment and review feedback.

Context is explicitly passed between stages. API calls use bounded retries and
stage-level errors so the UI can report where the workflow failed.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from groq import Groq


class WorkflowError(RuntimeError):
    """Raised when a workflow stage cannot complete."""


@dataclass
class WorkflowConfig:
    model: str = "openai/gpt-oss-20b"
    max_retries: int = 2
    retry_delay_seconds: float = 1.5
    max_tokens: int = 7000
    temperature: float = 0.3


@dataclass
class StageResult:
    stage: str
    content: str
    parsed_json: Any = None
    elapsed_seconds: float = 0.0


@dataclass
class WorkflowRun:
    context: dict[str, Any] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)
    final_pack: str = ""
    errors: list[str] = field(default_factory=list)


def _extract_json(text: str) -> Any:
    """Parse JSON even if the model wraps it in Markdown fences or prose."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_candidates = [p for p in (cleaned.find("{"), cleaned.find("[")) if p >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end < start:
            raise
        return json.loads(cleaned[start:end + 1])


class StudyPackWorkflow:
    def __init__(
        self,
        api_key: str,
        config: WorkflowConfig | None = None,
        on_stage: Callable[[str, str], None] | None = None,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("A Groq API key is required.")
        self.client = Groq(api_key=api_key.strip())
        self.config = config or WorkflowConfig()
        self.on_stage = on_stage

    def _notify(self, stage: str, status: str) -> None:
        if self.on_stage:
            self.on_stage(stage, status)

    def _call(self, stage: str, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> StageResult:
        last_error: Exception | None = None
        start = time.monotonic()
        for attempt in range(self.config.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                # JSON mode is not supported by every provider/model combination.
                # Prompts still request valid JSON; avoid hard-failing on JSON mode.
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                parsed = None
                if json_mode:
                    try:
                        parsed = _extract_json(content)
                    except (json.JSONDecodeError, ValueError):
                        if attempt < self.config.max_retries:
                            last_error = ValueError("Model returned invalid JSON.")
                            time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                            continue
                        raise WorkflowError(f"{stage} stage returned invalid JSON after retries.")
                self._notify(stage, "complete")
                return StageResult(stage, content, parsed, time.monotonic() - start)
            except WorkflowError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    self._notify(stage, f"retrying ({attempt + 1}/{self.config.max_retries})")
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    break
        self._notify(stage, "failed")
        message = str(last_error) if last_error else "Unknown API error"
        raise WorkflowError(f"{stage} stage failed: {message}") from last_error

    @staticmethod
    def _json_string(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)

    def run(
        self,
        *,
        subject: str,
        topic: str,
        learner_level: str,
        language: str,
        learning_goals: str,
        requested_sections: list[str],
        difficulty: str,
        question_count: int,
        source_notes: str,
    ) -> WorkflowRun:
        run = WorkflowRun()
        run.context = {
            "learner_profile": {
                "subject": subject.strip() or "Not specified",
                "topic": topic.strip() or "Use supplied notes",
                "level": learner_level,
                "language": language,
                "learning_goals": learning_goals.strip() or "Understand the topic and prepare for revision.",
                "difficulty": difficulty,
                "question_count": question_count,
                "requested_sections": requested_sections,
            },
            "source_notes": (source_notes or "").strip()[:45000],
        }
        p = run.context["learner_profile"]
        notes = run.context["source_notes"] or "(No source notes supplied. Use general knowledge cautiously and flag syllabus-dependent points.)"

        # Stage 1: Plan
        self._notify("Plan", "running")
        plan_prompt = f"""
Create a personalized study-pack plan. Return ONLY valid JSON with this schema:
{{
  "title": "string",
  "learner_needs": ["string"],
  "learning_objectives": ["measurable objective"],
  "outline": [{{"section": "string", "purpose": "string", "key_points": ["string"]}}],
  "misconceptions_to_check": ["string"],
  "assessment_blueprint": ["question skill or concept"],
  "source_coverage_notes": ["string"]
}}
Learner profile:
{self._json_string(p)}
Source notes:
<source_notes>
{notes}
</source_notes>
Make the objectives suitable for the stated level and goals. Include only requested sections where possible.
Do not claim that unsupported details appear in the source notes.
""".strip()
        plan = self._call(
            "Plan",
            "You are an instructional designer. Produce concise, practical plans and valid JSON only.",
            plan_prompt,
            json_mode=True,
        )
        run.stages.append(plan)
        run.context["plan"] = plan.parsed_json

        # Stage 2: Generate
        self._notify("Generate", "running")
        generate_prompt = f"""
Create the first draft of the study pack in Markdown. Follow the plan and learner profile.
Learner profile:
{self._json_string(p)}
Instructional plan:
{self._json_string(run.context['plan'])}
Source notes:
<source_notes>
{notes}
</source_notes>
Requirements:
- Write in {p['language']} and explain at {p['level']} level.
- Include these requested sections: {', '.join(requested_sections)}.
- Create {question_count} quiz questions at {difficulty} difficulty when quiz questions are requested.
- Keep answers in a separate answer-key section so learners can attempt questions first.
- Include worked examples where useful, and explain unfamiliar terms.
- Prefer source notes when provided. Do not invent citations, page numbers, or source claims.
- Flag uncertainty and syllabus-dependent facts briefly.
- Use clear headings, concise paragraphs, bullets, and tables only when useful.
""".strip()
        draft = self._call(
            "Generate",
            "You are an expert, careful tutor. Create accurate, accessible study materials. Return Markdown only.",
            generate_prompt,
        )
        run.stages.append(draft)
        run.context["draft"] = draft.content

        # Stage 3: Assess
        self._notify("Assess", "running")
        assess_prompt = f"""
Assess the draft against the learner profile and plan. Return ONLY valid JSON:
{{
 "overall_score": 0,
 "pass": true,
 "checks": [{{"criterion":"coverage|clarity|level_fit|assessment_quality|source_fidelity|organization","score":0,"finding":"string"}}],
 "issues": [{{"severity":"high|medium|low","location":"string","problem":"string","recommended_fix":"string"}}],
 "missing_objectives": ["string"],
 "question_quality_notes": ["string"]
}}
Scores must be integers from 1 to 5. Be specific and constructive. Check that requested sections and learning objectives are covered, quiz questions are answerable, answer key aligns, and unsupported facts are not presented as source-derived.
Learner profile:
{self._json_string(p)}
Plan:
{self._json_string(run.context['plan'])}
Draft:
<draft>
{draft.content}
</draft>
""".strip()
        assessment = self._call(
            "Assess",
            "You are a rigorous educational quality assessor. Return valid JSON only; do not rewrite the full pack.",
            assess_prompt,
            json_mode=True,
        )
        run.stages.append(assessment)
        run.context["assessment"] = assessment.parsed_json

        # Stage 4: Review
        self._notify("Review", "running")
        review_prompt = f"""
Review the draft as an editor and learner advocate. Return ONLY valid JSON:
{{
 "editorial_summary":"string",
 "strengths":["string"],
 "priority_fixes":["string"],
 "readability_notes":["string"],
 "safety_or_accuracy_cautions":["string"],
 "keep_unchanged":["string"]
}}
Focus on actionable improvements, readability, learner motivation, inclusivity, and accuracy. Do not add unsupported facts.
Profile:
{self._json_string(p)}
Assessment:
{self._json_string(run.context['assessment'])}
Draft:
<draft>
{draft.content}
</draft>
""".strip()
        review = self._call(
            "Review",
            "You are a supportive academic editor. Return valid JSON only.",
            review_prompt,
            json_mode=True,
        )
        run.stages.append(review)
        run.context["review"] = review.parsed_json

        # Stage 5: Refine
        self._notify("Refine", "running")
        refine_prompt = f"""
Revise the study pack using the plan, assessment, and review. Return the complete final pack in Markdown.
Preserve useful content and the intended learning level. Fix high- and medium-severity issues,
cover missing objectives, improve unclear explanations, and verify that quiz questions match the
answer key. Do not add unsupported citations or fabricated facts. If an issue cannot be resolved
from available information, state the limitation clearly.
Profile:
{self._json_string(p)}
Plan:
{self._json_string(run.context['plan'])}
Assessment:
{self._json_string(run.context['assessment'])}
Editorial review:
{self._json_string(run.context['review'])}
Original draft:
<draft>
{draft.content}
</draft>
Source notes:
<source_notes>
{notes}
</source_notes>
""".strip()
        final = self._call(
            "Refine",
            "You are a meticulous tutor revising educational content. Return the full final Markdown study pack only.",
            refine_prompt,
        )
        run.stages.append(final)
        run.final_pack = final.content
        run.context["final_pack"] = final.content
        return run
