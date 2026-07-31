"""Krok 4 of PLAN_SILNIKA.md: deterministic Polish sentence templates over insights.Fact.

No LLM, by explicit, non-negotiable project rule -- a language model can
confidently fabricate a date or a score, and that risk is unacceptable for
content published under a journalist's byline. Every sentence here is a
plain string template filled from a `Fact`'s own `data` dict, so any
sentence this package produces is exactly as reproducible/verifiable as the
`Fact` it came from.
"""
