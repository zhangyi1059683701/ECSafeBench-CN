from __future__ import annotations


def authorization_value(api_key: str, style: str = "bearer") -> str:
    normalized = style.strip().lower()
    if normalized == "raw":
        return api_key
    if normalized == "bearer":
        return f"Bearer {api_key}"
    raise ValueError(f"Unsupported API auth style: {style}. Use 'bearer' or 'raw'.")
