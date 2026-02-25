You are writing a deep structured digest summary for a paper selected by this profile.

Rules:
- Use ONLY the provided metadata/abstract/text fields (`fulltext` or `fulltext_slices`).
- Do NOT invent facts. If unknown, set null/unknown.
- Keep it concise and digest-oriented.
- If `allowed_tags` is provided, choose tags ONLY from that object.

Output MUST be valid JSON matching the provided PaperSummary schema (no markdown, no extra keys).
You MUST include every required field from the schema, even when unknown.
Copy identity fields exactly from input JSON:
- `arxiv_id_base`
- `title`
- `published_date`
- `categories`

Focus:
- detailed_summary: what the paper proposes, what appears novel, and what evidence is shown
- unique_contribution: one concise sentence
- key_points: 2-3 concise points

Input:
{{INPUT_JSON}}
