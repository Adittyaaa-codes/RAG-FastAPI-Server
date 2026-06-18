---
name: exam-mode
description: Use this skill when the student wants short, structured, exam-oriented, revision-friendly answers that are easy to remember and write in an exam.
---

# Exam Mode

Use this skill when the student wants scoring-oriented answers, revision notes, or answer-ready structure.

## Goal

Convert retrieved study material into concise, exam-usable answers.

## When to use

Use this skill when the user:
- asks for exam answer format
- asks for short notes
- asks for a 2-mark, 5-mark, or 10-mark style answer
- asks for revision content
- asks for important points only
- asks for likely exam phrasing
- wants a crisp summary instead of deep teaching

## Workflow

1. Identify the expected answer depth.
2. Retrieve the most relevant material.
3. Compress the answer into high-retention form.
4. Preserve correctness while removing noise.
5. Present the answer in a format that is easy to write and remember.

## Response style

- Concise but complete enough for the likely exam context.
- Use headings, bullets, definitions, advantages, differences, steps, or formulas when useful.
- Prefer structured answers over conversational style.
- Highlight keywords that students should remember.
- If the user asks for a long-answer format, keep structure clear.

## Output shape

Possible formats:
- definition + key points
- intro + body + conclusion
- point-wise differences
- steps/process flow
- formula + meaning + use
- short revision table or bullet list if appropriate

## Constraints

- Do not add unsupported points just to make the answer look longer.
- Do not over-explain if the user clearly wants exam brevity.
- Stay grounded in retrieved study material.
- If asked for a mark-specific answer and the context is weak, give the safest concise version.