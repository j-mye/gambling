"""Playing-card HTML for PyScript DOM (no Streamlit)."""

from __future__ import annotations

SUIT_SYMBOLS = {"S": "\u2660", "H": "\u2665", "D": "\u2666", "C": "\u2663"}
SUIT_RED = "#E74C3C"
SUIT_BLACK = "#2C3E50"
CARD_BG = "#FFFFFF"
CARD_SHADOW = "0 3px 10px rgba(0, 0, 0, 0.25)"
CARD_RADIUS = "12px"
CARD_BACK_PRIMARY = "#1F6FEB"
CARD_BACK_SECONDARY = "#DC2626"


def _suit_color(suit: str) -> str:
    return SUIT_RED if suit in {"H", "D"} else SUIT_BLACK


def _display_rank(rank: str) -> str:
    return "10" if (rank or "").strip().upper() == "T" else rank


def _card_shell(inner_html: str, extra_style: str = "", classes: str = "") -> str:
    return (
        f"<div class='{classes}' style='width:var(--card-w,72px);height:var(--card-h,104px);"
        f"max-width:100%;background:{CARD_BG};border-radius:{CARD_RADIUS};"
        f"box-shadow:{CARD_SHADOW};border:1px solid #D9E2EC;display:flex;align-items:center;"
        f"justify-content:center;position:relative;{extra_style}'>{inner_html}</div>"
    )


def render_face_up_card(rank: str, suit: str, classes: str = "") -> str:
    label = _display_rank(rank)
    sym = SUIT_SYMBOLS.get(suit, "?")
    color = _suit_color(suit)
    inner = (
        f"<div style='position:absolute;top:clamp(3px,0.8vmin,8px);left:clamp(4px,1vmin,10px);"
        f"color:{color};font-weight:700;font-size:clamp(10px,2.1vmin,15px);'>{label}{sym}</div>"
        f"<div style='font-size:clamp(18px,4.5vmin,32px);color:{color};'>{sym}</div>"
        f"<div style='position:absolute;bottom:clamp(3px,0.8vmin,8px);right:clamp(4px,1vmin,10px);"
        f"color:{color};font-weight:700;font-size:clamp(10px,2.1vmin,15px);"
        f"transform:rotate(180deg);'>{label}{sym}</div>"
    )
    return _card_shell(inner, classes=classes)


def render_face_down_card(classes: str = "") -> str:
    pattern = (
        f"background:repeating-linear-gradient(45deg,{CARD_BACK_PRIMARY},"
        f"{CARD_BACK_PRIMARY} 8px,{CARD_BACK_SECONDARY} 8px,{CARD_BACK_SECONDARY} 16px);"
    )
    inner = (
        "<div style='width:78%;height:82%;max-width:56px;max-height:86px;border:2px solid "
        "rgba(255,255,255,0.75);border-radius:8px;'></div>"
    )
    return _card_shell(inner, extra_style=pattern, classes=classes)


def hero_card_cell(rank: str, suit: str) -> str:
    """Compact hero / seat hole card; size follows :root --card-w/--card-h."""
    label = _display_rank(rank)
    sym = SUIT_SYMBOLS.get(suit, "?")
    color = "text-red-600" if suit in {"H", "D"} else "text-slate-900"
    return (
        f"<div class='hero-card-cell'>"
        f"<span class='hc-rank {color}'>{label}</span>"
        f"<span class='hc-suit {color}'>{sym}</span></div>"
    )
