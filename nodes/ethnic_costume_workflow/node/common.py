from comfy_api.latest import IO


def validate_string(value: str, *, strip_whitespace: bool = True) -> None:
    if value is None:
        raise ValueError("Text input is required")
    if strip_whitespace and not str(value).strip():
        raise ValueError("Text input cannot be empty")


def price_badge() -> IO.PriceBadge:
    return IO.PriceBadge(
        depends_on=IO.PriceBadgeDepends(widgets=["quality"]),
        expr="""
        (
          $ranges := {
            "low":    [0.0048, 0.019],
            "medium": [0.041, 0.168],
            "high":   [0.165, 0.67]
          };
          $range := $lookup($ranges, widgets.quality);
          {"type":"range_usd","min_usd": $range[0], "max_usd": $range[1], "format": {"approximate": true}}
        )
        """,
    )
