# AI Study Pack Generator — Workflow Design

## Multi-stage workflow

1. **Plan** — Converts learner profile, goals, requested sections, and source notes into a structured learning plan.
2. **Generate** — Produces the initial Markdown pack using the plan and source notes.
3. **Assess** — Evaluates objective coverage, clarity, level fit, quiz quality, source fidelity, and organization.
4. **Review** — Produces editorial feedback, readability notes, and accuracy cautions.
5. **Refine** — Uses the plan, initial draft, assessment, review, and source notes to produce the final pack.

## Context passing

The workflow keeps a run context with:
- learner profile and preferences
- source notes
- structured plan
- initial draft
- assessment JSON
- editorial review JSON
- final refined pack

The context is passed forward explicitly. Uploaded notes are treated as untrusted source data, not instructions to the AI.

## Error handling

- Each stage has bounded retries with a small backoff.
- Invalid JSON from structured stages is retried and then reported as a stage-specific error.
- API failures identify the stage that failed.
- The Streamlit interface shows workflow progress and a user-friendly troubleshooting message.
- API keys should be stored in Streamlit Secrets for deployment, not committed to source control.

## Deployment

Upload `app.py`, `workflow.py`, `prompts.py`, and `requirements.txt` to a GitHub repository. In Streamlit Community Cloud, create an app from that repository and set the main file path to `app.py`.

In Streamlit app settings, add a secret:

```toml
GROQ_API_KEY = "your-real-key"
```

Do not commit the secret file to GitHub. The app also permits entering the key in the sidebar for local testing.

## Known limitations

- Scanned PDFs require OCR before their contents can be extracted.
- This starter uses one LLM provider and a text-based final output.
- Quality assessment is model-based, not a guarantee of factual correctness. Verify high-stakes or syllabus-specific content.
