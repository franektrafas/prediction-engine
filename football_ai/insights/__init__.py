"""Provenance-carrying facts and records module (Krok 2 of PLAN_SILNIKA.md).

The core of the product per that plan: nothing here is available for free
anywhere else. Every `Fact` this package produces must carry the `match_id`s
it was derived from (`schemas.Fact.provenance`) -- a fact a journalist
publishes under their own name is only as good as its worst error, so
"trust me" is not an acceptable provenance story anywhere in this package.
"""

from __future__ import annotations
