"""SearXNG Search backend."""

import os
from typing import Any

import httpx

from .base import WebSearchBackend
from .registry import register_backend


@register_backend
class SearXNGBackend(WebSearchBackend):
    """SearXNG self-hosted search backend."""

    name = "searxng"

    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url or os.environ.get("SEARXNG_BASE_URL", "")
        ).rstrip("/")

    @property
    def is_available(self) -> bool:
        return bool(self.base_url)

    async def search(self, query: str, count: int, **kwargs: Any) -> str:
        if not self.base_url:
            return "Error: SEARXNG_BASE_URL not configured"

        try:
            n = min(max(count, 1), 20)
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{self.base_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "pageno": 1,
                        "categories": "general",
                    },
                    timeout=15.0,
                )
                r.raise_for_status()

            results = r.json().get("results", [])
            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if content := item.get("content"):
                    snippet = content[:500]
                    suffix = "..." if len(content) > 500 else ""
                    lines.append(f"   {snippet}{suffix}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"
