"""HTML/CSS playing card renderer for Streamlit markdown."""

from __future__ import annotations

from scripts.ui import theme

SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}


def poker_table_css() -> str:
    return """
<style>
.player-seat { border: 1px solid #334E68; border-radius: 10px; padding: 8px; background: #102A43; color: #F0F4F8; }
.player-seat { position: relative; }
.seat-folded { filter: grayscale(100%); opacity: 0.5; }
.winner { border: 2px solid #2ECC71 !important; box-shadow: 0 0 12px rgba(46, 204, 113, 0.65) !important; }
.loser { border: 2px solid #E74C3C !important; box-shadow: 0 0 12px rgba(231, 76, 60, 0.55) !important; }
.seat-metrics { margin-top: 6px; font-size: 12px; color: #BCCCDC; }
.active-turn { border: 2px solid #F1C40F; box-shadow: 0 0 0 rgba(241,196,15,0.3); animation: pulseTurn 1.1s infinite; }
.action-badge { position: absolute; top: -10px; right: -6px; padding: 4px 8px; border-radius: 999px; background: #F1C40F; color: #1F2933; font-size: 11px; font-weight: 700; z-index: 3; }
.board-row-fixed { display: flex; gap: 10px; justify-content: center; width: 100%; min-height: 104px; align-items: center; }
.poker-board-wrap {
  margin: 0 !important;
  padding: 24px 0 !important;
  box-sizing: border-box;
}
.board-slot { width: 72px; height: 104px; display: inline-flex; align-items: center; justify-content: center; }
.board-slot.placeholder { border: 1px dashed rgba(130, 154, 177, 0.85); border-radius: 12px; background: rgba(15, 23, 42, 0.25); box-shadow: inset 0 0 0 1px rgba(188, 204, 220, 0.2); }
.community-card.deal-animate { animation: slideIn 0.4s ease-out both; }
.deal-delay-0 { animation-delay: 0s; }
.deal-delay-1 { animation-delay: 0.2s; }
.deal-delay-2 { animation-delay: 0.4s; }
.deal-delay-3 { animation-delay: 0.15s; }
.deal-delay-4 { animation-delay: 0.15s; }
@keyframes slideIn { from { opacity: 0; transform: translateY(-12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulseTurn {
  0% { box-shadow: 0 0 0 0 rgba(241,196,15,0.45); }
  70% { box-shadow: 0 0 0 8px rgba(241,196,15,0.0); }
  100% { box-shadow: 0 0 0 0 rgba(241,196,15,0.0); }
}
/* Table: content height only; keep modest gap to hero (tighter when hand-over bar is shown). */
div[data-testid="stVerticalBlock"]:has(.poker-table-anchor) {
  min-height: unset !important;
  margin-bottom: 12px !important;
  padding-bottom: 0 !important;
  gap: 0.35rem !important;
}
div[data-testid="stVerticalBlock"]:has(.poker-hand-over-bar-marker) {
  margin-bottom: 10px !important;
  padding-bottom: 0 !important;
  gap: 0.25rem !important;
}
div[data-testid="stVerticalBlock"]:has(.poker-hand-over-bar-marker) div[data-testid="element-container"] {
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}
/* Hero block: keep top flush; spacing comes from table margin-bottom above */
div[data-testid="stVerticalBlock"]:has(.poker-hero-anchor) {
  min-height: unset !important;
  margin-top: 0 !important;
  padding-top: 0 !important;
  gap: 0.35rem !important;
}
/* Divider between table placeholder and hero (specificity beats generic stMarkdownContainer hr) */
div[data-testid="stMarkdownContainer"] hr.poker-tab-divider {
  margin: 0.2rem 0 0.12rem 0 !important;
  border: none;
  border-top: 1px solid rgba(130, 154, 177, 0.35);
}
div[data-testid="stMarkdownContainer"] hr:not(.poker-tab-divider) {
  margin: 0.35rem 0 !important;
}
.hero-right-spacer {
  min-height: 280px;
  width: 100%;
  box-sizing: border-box;
}
.hero-right-spacer.hero-right-spacer--terminal {
  min-height: 5rem;
}
.poker-hand-over-bar-marker {
  height: 0;
  margin: 0;
  padding: 0;
  overflow: hidden;
}
.hero-action-error-slot--empty {
  min-height: 2.75rem;
  margin: 0;
  padding: 0;
}
.hero-action-error-slot--filled {
  box-sizing: border-box;
  min-height: 2.75rem;
  padding: 0.75rem 1rem;
  border-radius: 0.35rem;
  background: rgba(255, 75, 75, 0.12);
  border: 1px solid rgba(255, 75, 75, 0.55);
  color: #ffb4b4;
  font-size: 0.875rem;
  line-height: 1.4;
}
</style>
"""


def _suit_color(suit: str) -> str:
    """Hearts/Diamonds red; Spades/Clubs black."""
    return theme.SUIT_RED if suit in {"H", "D"} else theme.SUIT_BLACK


def _display_rank(rank: str) -> str:
    """Show ten as '10' (PokerKit and some paths use 'T')."""
    return "10" if (rank or "").strip().upper() == "T" else rank


def _card_shell(inner_html: str, extra_style: str = "", classes: str = "") -> str:
    return (
        f"<div class='{classes}' style='width:72px;height:104px;background:{theme.CARD_BG};border-radius:{theme.CARD_RADIUS};"
        f"box-shadow:{theme.CARD_SHADOW};border:1px solid #D9E2EC;display:flex;align-items:center;"
        f"justify-content:center;position:relative;{extra_style}'>{inner_html}</div>"
    )


def render_face_up_card(rank: str, suit: str, classes: str = "") -> str:
    """Render one face-up card."""
    # Card corners show rank+suit while center shows large suit glyph.
    label = _display_rank(rank)
    sym = SUIT_SYMBOLS.get(suit, "?")
    color = _suit_color(suit)
    inner = (
        f"<div style='position:absolute;top:6px;left:8px;color:{color};font-weight:700;font-size:15px;'>{label}{sym}</div>"
        f"<div style='font-size:30px;color:{color};'>{sym}</div>"
        f"<div style='position:absolute;bottom:6px;right:8px;color:{color};font-weight:700;font-size:15px;"
        f"transform:rotate(180deg);'>{label}{sym}</div>"
    )
    return _card_shell(inner, classes=classes)


def render_face_down_card(classes: str = "") -> str:
    """Render one face-down card with simple pattern."""
    # Patterned back avoids external image assets and stays lightweight.
    pattern = (
        f"background:repeating-linear-gradient(45deg,{theme.CARD_BACK_PRIMARY},"
        f"{theme.CARD_BACK_PRIMARY} 8px,{theme.CARD_BACK_SECONDARY} 8px,{theme.CARD_BACK_SECONDARY} 16px);"
    )
    inner = "<div style='width:56px;height:86px;border:2px solid rgba(255,255,255,0.75);border-radius:8px;'></div>"
    return _card_shell(inner, extra_style=pattern, classes=classes)


def render_placeholder_card(classes: str = "") -> str:
    """Render undealt street placeholder."""
    # Dashed shell keeps board spacing stable before cards are dealt.
    return _card_shell("", extra_style="background:#E9EFF5;border:1px dashed #829AB1;", classes=classes)


def render_card_row(
    cards: list[tuple[str, str]] | None = None,
    mode: str = "face_up",
    slots: int | None = None,
    folded: bool = False,
    animate_indexes: set[int] | None = None,
) -> str:
    """Render horizontal strip of cards."""
    cards = cards or []
    animate_indexes = animate_indexes or set()
    folded_class = " seat-folded" if folded else ""
    card_html_parts: list[str] = []
    if mode == "face_down":
        total = slots or max(2, len(cards))
        card_html_parts = [render_face_down_card(classes=f"community-card{folded_class}") for _ in range(total)]
    elif mode == "placeholder":
        total = slots or 5
        card_html_parts = [render_placeholder_card(classes="community-card") for _ in range(total)]
    else:
        for idx, (rank, suit) in enumerate(cards):
            anim = f" deal-animate deal-delay-{idx}" if idx in animate_indexes else ""
            classes = f"community-card{folded_class}{anim}"
            card_html_parts.append(render_face_up_card(rank, suit, classes=classes))
    return (
        "<div style='display:flex;gap:8px;justify-content:center;align-items:center;flex-wrap:wrap;'>"
        + "".join(card_html_parts)
        + "</div>"
    )


def render_single_face_up_card(rank: str, suit: str, folded: bool = False, animate: bool = False, delay_idx: int = 0) -> str:
    folded_class = " seat-folded" if folded else ""
    anim = f" deal-animate deal-delay-{delay_idx}" if animate else ""
    classes = f"community-card{folded_class}{anim}"
    return render_face_up_card(rank, suit, classes=classes)
