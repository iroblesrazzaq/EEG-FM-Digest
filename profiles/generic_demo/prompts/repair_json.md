Fix the following text into VALID JSON that matches the provided JSON Schema EXACTLY.

Constraints:
- Output JSON only.
- No markdown.
- No extra keys beyond what the schema allows.
- If a field is missing, add it with a reasonable default:
  - strings: "" (or "unknown" if appropriate)
  - arrays: []
  - numbers: 0 (or null if allowed)
  - booleans: false
- Ensure enums use valid values.

JSON Schema:
{{SCHEMA_JSON}}

Bad output:
{{BAD_OUTPUT}}
