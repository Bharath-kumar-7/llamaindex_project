CHUNK_SIZE = 400
SIMILARITY_TOP_K = 5

INITIAL_FACTS_TEMPLATE = """
Use the following LinkedIn profile information.

{context_str}

Generate three interesting facts about this person's career or education.
"""

USER_QUESTION_TEMPLATE = """
Use the following profile information.

{context_str}

Question: {query_str}

Answer only from the context.
If the answer is not available say "I don't know".
"""
