"""Shared poker hand and feature helpers for training and inference."""

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
POSITION_MAP = {"SB": 0.0, "BB": 0.25, "UTG": 0.5, "MP": 0.65, "CO": 0.8, "BTN": 1.0}


def _card_rank(card: str) -> int | None:
    if len(card) < 2:
        return None
    return RANK_MAP.get(card[0].upper())


def _card_suit(card: str) -> str | None:
    if len(card) < 2:
        return None
    return card[1].lower()


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


def hand_strength_from_tokens(cards: list[str]) -> int:
    comb = best_combination_from_tokens(cards)
    return COMBINATION_MAP.get(comb, 9)


def parse_cards(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if text in {"", "0", "--", "nan", "None"}:
        return []
    out: list[str] = []
    for token in text.split():
        t = token.strip()
        if len(t) < 2:
            continue
        rank = t[0].upper()
        suit = t[1].lower()
        if rank in RANK_MAP and suit in {"c", "d", "h", "s"}:
            out.append(f"{rank}{suit}")
    return out


def _ranks(cards: list[str]) -> list[int]:
    return [RANK_MAP[c[0]] for c in cards if len(c) >= 2 and c[0] in RANK_MAP]


def _suits(cards: list[str]) -> list[str]:
    return [c[1].lower() for c in cards if len(c) >= 2]


def _straight_draw_flags(cards: list[str]) -> tuple[int, int]:
    ranks = sorted(set(_ranks(cards)))
    if 14 in ranks:
        ranks = [1] + ranks
    gutshot = 0
    open_ended = 0
    for start in range(1, 11):
        target = set(range(start, start + 5))
        have = len(target.intersection(set(ranks)))
        if have >= 4:
            missing = sorted(target.difference(set(ranks)))
            if len(missing) == 1:
                if missing[0] in {start, start + 4}:
                    open_ended = 1
                else:
                    gutshot = 1
    return open_ended, gutshot


def _board_straight_flags(board_cards: list[str]) -> tuple[float, float]:
    ranks = sorted(set(_ranks(board_cards)))
    if not ranks:
        return 0.0, 0.0
    if 14 in ranks:
        ranks = [1] + ranks
    rank_set = set(ranks)
    straight_present = 0.0
    four_liner = 0.0
    for start in range(1, 11):
        target = set(range(start, start + 5))
        have = len(target.intersection(rank_set))
        if have == 5:
            straight_present = 1.0
        elif have >= 4:
            four_liner = 1.0
    return straight_present, four_liner


def _board_pair_profile(board_cards: list[str]) -> tuple[float, float, float]:
    board_counts = Counter(_ranks(board_cards))
    pair_count = float(sum(1 for c in board_counts.values() if c >= 2))
    trips_present = float(any(c >= 3 for c in board_counts.values()))
    full_house = float(
        any(c >= 3 for c in board_counts.values())
        and (sum(1 for c in board_counts.values() if c >= 2) >= 2)
    )
    return pair_count, trips_present, full_house


def _hole_card_features(hole_cards: list[str]) -> dict[str, float]:
    if len(hole_cards) < 2:
        return {
            "hole_rank_high": 0.0,
            "hole_rank_low": 0.0,
            "is_pair": 0.0,
            "is_suited": 0.0,
            "rank_gap": 0.0,
            "has_ace": 0.0,
            "has_broadway": 0.0,
            "preflop_strength": 0.0,
        }
    r1 = RANK_MAP.get(hole_cards[0][0], 0)
    r2 = RANK_MAP.get(hole_cards[1][0], 0)
    high = float(max(r1, r2))
    low = float(min(r1, r2))
    is_pair = float(r1 == r2)
    is_suited = float(hole_cards[0][1].lower() == hole_cards[1][1].lower())
    gap = float(abs(r1 - r2))
    has_ace = float(14 in {r1, r2})
    broadway = float(r1 >= 10 or r2 >= 10)
    # Lightweight preflop strength proxy to prioritize cards over chips.
    strength = high + (0.6 * low) + (2.5 * is_pair) + (0.8 * is_suited) - (0.15 * gap)
    return {
        "hole_rank_high": high,
        "hole_rank_low": low,
        "is_pair": is_pair,
        "is_suited": is_suited,
        "rank_gap": gap,
        "has_ace": has_ace,
        "has_broadway": broadway,
        "preflop_strength": float(strength),
    }


def build_stage_feature_payload(
    stage: str,
    hole_cards: list[str],
    board_cards: list[str],
    *,
    total_bet: float = 0.0,
    current_pot: float = 0.0,
    position: str = "",
    hero_stack: float = 0.0,
    table_stacks: list[float] | None = None,
    big_blind: float = 2.0,
) -> dict[str, float]:
    stage_key = stage.lower()
    cards = list(hole_cards) + list(board_cards)
    ranks = _ranks(cards)
    suits = _suits(cards)
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    max_suit = max(suit_counts.values()) if suit_counts else 0
    pair_count = sum(1 for c in rank_counts.values() if c == 2)
    trips_count = sum(1 for c in rank_counts.values() if c == 3)

    open_ended, gutshot = _straight_draw_flags(cards)
    has_flush_draw = float(max_suit == 4)
    board_ranks = _ranks(board_cards)
    board_high = float(max(board_ranks) if board_ranks else 0.0)
    board_unique = len(set(board_ranks))
    board_connected = 0.0
    if len(board_ranks) >= 3:
        sorted_board = sorted(set(board_ranks))
        board_connected = float((max(sorted_board) - min(sorted_board)) <= 4)
    board_straight_present, board_straight_4liner = _board_straight_flags(board_cards)
    board_pair_count, board_trips_present, board_full_house_present = _board_pair_profile(board_cards)
    board_suits = Counter(_suits(board_cards))
    board_max_suit = max(board_suits.values()) if board_suits else 0
    board_four_flush = float(board_max_suit >= 4)
    board_only_strength = float(hand_strength_from_tokens(board_cards))
    shared_strength_gap = float(hand_strength_from_tokens(cards) - board_only_strength)
    hero_uses_hole_for_best = float(shared_strength_gap > 0.0)

    bb = float(big_blind) if float(big_blind) > 0 else 2.0
    hero_stack_float = max(0.0, float(hero_stack))
    stack_pool = [max(0.0, float(s)) for s in (table_stacks or [])]
    if not stack_pool:
        stack_pool = [hero_stack_float]
    effective_stack = min(hero_stack_float, max(stack_pool))
    sorted_pool = sorted(stack_pool)
    hero_less_equal = sum(1 for s in sorted_pool if s <= hero_stack_float)
    hero_stack_percentile = float(hero_less_equal / max(len(sorted_pool), 1))
    hero_stack_bb = hero_stack_float / bb
    effective_stack_bb = effective_stack / bb
    spr = effective_stack / max(float(current_pot), 1.0)
    is_short_stack = float(hero_stack_bb <= 20.0)

    board_shared_strength_risk = min(
        1.0,
        0.3 * board_four_flush
        + 0.25 * board_straight_present
        + 0.15 * board_straight_4liner
        + 0.2 * float(board_pair_count >= 1)
        + 0.15 * float(board_pair_count >= 2)
        + 0.25 * board_trips_present
        + 0.35 * board_full_house_present
        + 0.2 * float(shared_strength_gap <= 1.0)
        + 0.2 * float(hero_uses_hole_for_best == 0.0),
    )

    payload: dict[str, float] = {
        "hand_strength": float(hand_strength_from_tokens(cards)),
        "total_bet": float(total_bet),
        "current_pot": float(current_pot),
        "position_value": float(POSITION_MAP.get(str(position).upper(), 0.5)),
        "hole_rank_high": 0.0,
        "hole_rank_low": 0.0,
        "is_pair": 0.0,
        "is_suited": 0.0,
        "rank_gap": 0.0,
        "has_ace": 0.0,
        "has_broadway": 0.0,
        "preflop_strength": 0.0,
        "pair_count": float(pair_count),
        "trips_count": float(trips_count),
        "max_suit_count": float(max_suit),
        "flush_draw": float(has_flush_draw),
        "open_ended_straight_draw": float(open_ended),
        "gutshot_draw": float(gutshot),
        "board_high_rank": board_high,
        "board_is_paired": float(board_unique < len(board_ranks)) if board_ranks else 0.0,
        "board_connected": board_connected,
        "hero_stack_bb": hero_stack_bb,
        "effective_stack_bb": effective_stack_bb,
        "spr": float(spr),
        "hero_stack_percentile": hero_stack_percentile,
        "is_short_stack": is_short_stack,
        "board_four_flush": board_four_flush,
        "board_straight_present": board_straight_present,
        "board_straight_4liner": board_straight_4liner,
        "board_pair_count": board_pair_count,
        "board_trips_present": board_trips_present,
        "board_full_house_present": board_full_house_present,
        "hero_uses_hole_for_best": hero_uses_hole_for_best,
        "shared_strength_gap": shared_strength_gap,
        "board_only_strength": board_only_strength,
        "board_shared_strength_risk": board_shared_strength_risk,
    }
    payload.update(_hole_card_features(hole_cards))
    if stage_key == "preflop":
        payload["flush_draw"] = 0.0
        payload["open_ended_straight_draw"] = 0.0
        payload["gutshot_draw"] = 0.0
        payload["board_high_rank"] = 0.0
        payload["board_is_paired"] = 0.0
        payload["board_connected"] = 0.0
    return payload
