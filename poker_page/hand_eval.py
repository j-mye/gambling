"""7-card NLHE best-hand evaluation for the browser (no scripts/ dependency)."""

from __future__ import annotations

from collections import Counter

RANK_MAP = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

COMBINATION_ORDER = [
    "unknown",
    "high card",
    "pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
]
COMBINATION_MAP = {comb: i for i, comb in enumerate(COMBINATION_ORDER)}


def _card_rank(card: str) -> int | None:
    if not card:
        return None
    if len(card) >= 3 and card[:2].upper() == "10":
        return 10
    if len(card) < 2:
        return None
    return RANK_MAP.get(card[0].upper())


def _card_suit(card: str) -> str | None:
    if not card:
        return None
    if len(card) >= 3 and card[:2].upper() == "10":
        s = card[2].lower()
    elif len(card) >= 2:
        s = card[1].lower()
    else:
        return None
    return s if s in {"c", "d", "h", "s"} else None


def _is_straight(ranks: list[int]) -> bool:
    uniq = sorted(set(ranks))
    if len(uniq) < 5:
        return False
    if 14 in uniq:
        uniq = [1] + uniq
    for i in range(len(uniq) - 4):
        window = uniq[i : i + 5]
        if window == list(range(window[0], window[0] + 5)):
            return True
    return False


def tuples_to_tokens(cards: list[tuple[str, str]]) -> list[str]:
    """Convert (rank, suit) from poker_core view to compact tokens e.g. Ah, 10c."""
    out: list[str] = []
    for rank, suit in cards:
        r = str(rank).strip().upper()
        su = str(suit).strip()
        if len(su) != 1:
            continue
        c = su.upper()
        if c in ("C", "D", "H", "S"):
            out.append(f"{r}{c.lower()}")
    return out


def best_combination_from_tokens(cards: list[str]) -> str:
    card_ranks = [_card_rank(c) for c in cards if _card_rank(c) is not None]
    card_suits = [_card_suit(c) for c in cards if _card_suit(c) is not None]
    counts = Counter(card_ranks)
    counts_values = sorted(counts.values(), reverse=True)

    if not card_ranks:
        return "unknown"
    if len(card_ranks) < 5:
        if 3 in counts_values:
            return "three of a kind"
        if counts_values.count(2) >= 2:
            return "two pair"
        if 2 in counts_values:
            return "pair"
        return "high card"

    suits = Counter(card_suits)
    flush_suit = next((s for s, cnt in suits.items() if cnt >= 5), None)
    flush_cards = [card for card in cards if _card_suit(card) == flush_suit] if flush_suit else []
    flush_ranks = [_card_rank(card) for card in flush_cards if _card_rank(card) is not None]

    straight = _is_straight(card_ranks)
    straight_flush = flush_suit is not None and _is_straight(flush_ranks)

    if straight_flush:
        return "straight flush"
    if 4 in counts_values:
        return "four of a kind"
    if 3 in counts_values and 2 in counts_values:
        return "full house"
    if flush_suit is not None:
        return "flush"
    if straight:
        return "straight"
    if 3 in counts_values:
        return "three of a kind"
    if counts_values.count(2) >= 2:
        return "two pair"
    if 2 in counts_values:
        return "pair"
    return "high card"


def hand_strength_index(cards: list[str]) -> int:
    comb = best_combination_from_tokens(cards)
    return int(COMBINATION_MAP.get(comb, 0))


def board_texture_risk(board_tokens: list[str]) -> float:
    """0..~1 rough danger that the board hits many ranges (paired / wet)."""
    if len(board_tokens) < 3:
        return 0.0
    ranks = [_card_rank(c) for c in board_tokens if _card_rank(c) is not None]
    suits = [_card_suit(c) for c in board_tokens if _card_suit(c) is not None]
    if not ranks:
        return 0.0
    rc = Counter(ranks)
    sc = Counter(suits)
    max_suit = max(sc.values()) if sc else 0
    pairish = sum(1 for v in rc.values() if v >= 2)
    trips = any(v >= 3 for v in rc.values())
    risk = 0.0
    risk += 0.08 * min(2, pairish)
    risk += 0.12 if trips else 0.0
    risk += 0.1 if max_suit >= 3 else 0.0
    risk += 0.08 if max_suit >= 4 else 0.0
    sr = sorted(set(ranks))
    if 14 in sr:
        sr = [1] + sr
    span = max(sr) - min(sr) if sr else 0
    if len(sr) >= 3 and span <= 4:
        risk += 0.1
    return min(0.45, risk)
