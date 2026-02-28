"""IntelliDoc RAG — Prompt Templates."""

from __future__ import annotations

RAG_SYSTEM_PROMPT = """You are IntelliDoc, an expert document Q&A assistant. \
Your role is to answer questions accurately based ONLY on the provided context.

Rules:
1. Answer ONLY based on the provided context documents.
2. If the context doesn't contain enough information, say so clearly.
3. Cite the source document(s) in your answer using [Source: filename] format.
4. Be concise but thorough.
5. If multiple sources provide relevant information, synthesize them.
6. Never fabricate information not present in the context."""

RAG_USER_TEMPLATE = """Context Documents:
---
{context}
---

Question: {question}

Provide a detailed, well-structured answer based on the context above. \
Include source citations."""

EVALUATION_PROMPT = """You are an evaluation judge. Assess the following based on \
the given criteria.

Context: {context}
Question: {question}
Answer: {answer}
Ground Truth: {ground_truth}

Evaluate on a scale of 0.0 to 1.0 for the criterion: {criterion}

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""


def format_context(results: list[dict]) -> str:
    """Format retrieval results into context string for the prompt."""
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result.get("metadata", {}).get("source", "Unknown")
        content = result.get("content", "")
        context_parts.append(
            f"[Document {i} | Source: {source}]\n{content}"
        )
    return "\n\n".join(context_parts)


def build_rag_messages(question: str, context: str) -> list[dict]:
    """Build the message list for RAG generation."""
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RAG_USER_TEMPLATE.format(
                context=context, question=question
            ),
        },
    ]
