You are writing a deep structured digest summary for an EEG-FM paper.

Rules:
- Use ONLY the provided metadata/abstract/text fields (`fulltext` or `fulltext_slices`).
- Do NOT invent facts. If unknown, set null/unknown.
- Keep it concise and digest-oriented, but not shallow.
- Tagging rule: choose tags ONLY from `allowed_tags` in input JSON.
- For each tag category (`paper_type`, `backbone`, `objective`, `tokenization`, `topology`), output at most 2 tags.

Output MUST be valid JSON matching the PaperSummary schema (no markdown, no extra keys).
You MUST include every required field, even when unknown.
Copy these fields exactly from input JSON:
- `arxiv_id_base`
- `title`
- `published_date`
- `categories`

Required output keys checklist:
- `arxiv_id_base`, `title`, `published_date`, `categories`
- `paper_type`, `one_liner`, `detailed_summary`, `unique_contribution`, `key_points`
- `data_scale`, `method`, `evaluation`, `open_source`, `tags`, `limitations`
- `used_fulltext`, `notes`

Important disambiguation:
- `paper_type` is a SINGLE string from schema enum (`new_model|benchmark|survey|method|application|other`).
- `tags.paper_type` is an ARRAY chosen from `allowed_tags.paper_type`.
- For paper-type tag naming, use `new-model` (not `eeg-fm`).

What to focus on:
- unique_contribution: a single crisp sentence describing what is new.
- detailed_summary: 2-4 sentences that clearly explain:
  1) what the paper proposes,
  2) what is novel compared with typical prior work,
  3) what evidence is presented (dataset/task/result level) if available.
- key_points: 2-3 concise bullet-style points for UI display.
  The first bullet MUST state what the paper is, starting with paper type in natural wording, e.g.,
  "New EEG foundation model: CEReBrO ..."
  Do not use templated labels like "What it is:", "Core method/evidence:", or "Main practical takeaway:".
  Remaining bullet(s) should cover method novelty and strongest concrete evidence/result.
  Avoid copying `unique_contribution` verbatim; partial semantic overlap is allowed.
  Also avoid repeating the same opening clause between `key_points` and `unique_contribution`.
- data_scale: datasets, subjects, eeg_hours, channels (if present)
- method: architecture + objective + pretraining + finetuning (if present)
- evaluation: tasks/benchmarks/headline results (if present)
- open_source: code/weights/license if explicitly mentioned

Strict typing constraints:
- `data_scale.subjects`, `data_scale.eeg_hours`, `data_scale.channels` must be number or null (never strings like "10k+" or "64ch").
- `key_points` must have 2-3 items.
- `limitations` must have 2-8 items.
- `tags.<category>` arrays must contain only values from `allowed_tags.<category>`, max 2 values each.
- `used_fulltext` must be boolean.
- `notes` must be a string.

Input:
{{INPUT_JSON}}
