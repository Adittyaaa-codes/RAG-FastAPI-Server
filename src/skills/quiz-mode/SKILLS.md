---
name: quiz-mode
description: Use this skill when the student wants practice questions, self-testing, active recall, or interactive learning based on retrieved study material.
---

# Quiz Mode

Use this skill when the student should learn by testing recall instead of only reading explanations.

## Goal

Turn study material into active recall and guided practice.

## When to use

Use this skill when the user:
- asks for quiz questions
- asks to test their knowledge
- asks for MCQs, short questions, or viva-style questions
- wants practice after studying a concept
- wants flashcard-like interaction
- says they want to revise through questions

## Workflow

1. Identify the topic and difficulty level from the query and retrieved material.
2. Generate questions only from supported material.
3. Prefer one question at a time for interactive learning unless the user asks for a set.
4. After the student answers, evaluate correctness clearly.
5. Explain the correct answer in a short but educational way.
6. Adapt the next question difficulty based on the student's response if interaction continues.

## Question styles

Use one of these depending on context:
- one-line recall question
- MCQ
- short answer
- true/false
- fill in the blank
- viva/oral style conceptual question
- application-based question

## Evaluation style

When checking student answers:
- say whether the answer is correct, partially correct, or incorrect
- explain why
- correct misunderstandings directly
- keep the feedback educational, not judgmental

## Constraints

- Do not ask questions from unsupported content.
- Do not generate trick questions unless explicitly requested.
- Keep questions aligned with the student's level.
- Do not ask too many questions at once unless requested.