"""Multinomial logistic action advisor (fold / call / raise) + confidence from bundled JSON."""

from __future__ import annotations

import json
import math
import os
from typing import Any

_MODEL: dict[str, Any] | None = None

_FEATURE_DIM = 16
_CLASS_ORDER = ("fold", "call", "raise")


def _fallback_model() -> dict[str, Any]:
    """Uniform logits when JSON is missing, empty, or corrupt (avoids JSONDecodeError in UI)."""
    z = [0.0] * _FEATURE_DIM
    return {
        "version": 0,
        "feature_dim": _FEATURE_DIM,
        "class_order": list(_CLASS_ORDER),
        "coef": [list(z), list(z), list(z)],
        "intercept": [0.0, 0.0, 0.0],
        "scaler_mean": [0.0] * _FEATURE_DIM,
        "scaler_scale": [1.0] * _FEATURE_DIM,
    }


def _model_ok(m: Any) -> bool:
    if not isinstance(m, dict):
        return False
    try:
        co = m["coef"]
        mean = m["scaler_mean"]
        scale = m["scaler_scale"]
        ice = m["intercept"]
        classes = m["class_order"]
        if len(co) != 3 or len(ice) != 3:
            return False
        for row in co:
            if len(row) != _FEATURE_DIM:
                return False
        if len(mean) != _FEATURE_DIM or len(scale) != _FEATURE_DIM:
            return False
        if len(classes) != 3:
            return False
    except (KeyError, TypeError):
        return False
    return True


def _model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_model.json")


def _load_model() -> dict[str, Any]:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = _model_path()
    raw = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        _MODEL = _fallback_model()
        return _MODEL

    text = (raw or "").strip()
    if not text:
        _MODEL = _fallback_model()
        return _MODEL

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        _MODEL = _fallback_model()
        return _MODEL

    if not _model_ok(parsed):
        _MODEL = _fallback_model()
        return _MODEL

    _MODEL = parsed
    return _MODEL


def feature_vector_from_view(view: dict[str, Any]) -> list[float]:
    """Must match `notebooks/_train_action_advisor_export.py` synthetic column order (dim=16)."""
    hero = int(view.get("hero", 0)) % 6
    wp = view.get("win_probability")
    if wp is None:
        wp = 0.35
    wp = float(min(0.99, max(0.01, float(wp))))

    stacks = view.get("stacks") or [200] * 6
    stack = float(stacks[hero]) if hero < len(stacks) else 200.0
    stack = max(stack, 1.0)

    call_amt = float(view.get("call_amount") or 0)
    pot = float(view.get("pot_total") or 0)
    facing = 1.0 if view.get("facing_bet") else 0.0
    min_r = float(view.get("min_raise") or 2)
    max_r = float(view.get("max_raise") or stack)
    max_r = max(max_r, 1.0)

    board = view.get("board_cards") or []
    board_n = len(board) / 5.0

    folded = view.get("folded") or [False] * 6
    villains = sum(
        1 for s in range(6) if s != hero and s < len(folded) and not bool(folded[s])
    )
    nv_frac = villains / 5.0

    hero_street = float(view.get("hero_street_bet") or 0)
    hero_total = float(view.get("hero_total_bet") or 0)

    bluffs = view.get("bluff_prob_by_seat") or []
    vb_sum = 0.0
    vb_n = 0
    for s in range(6):
        if s == hero or s >= len(bluffs):
            continue
        b = bluffs[s]
        if b is not None:
            vb_sum += float(b)
            vb_n += 1
    vb = (vb_sum / vb_n) if vb_n else 0.35

    pk = str(view.get("prediction_stage") or "preflop").lower()
    ph = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(pk, 0)
    onehot = [1.0 if ph == i else 0.0 for i in range(4)]

    return [
        wp,
        call_amt / stack,
        pot / stack,
        facing,
        min_r / stack,
        board_n,
        nv_frac,
        stack / 200.0,
        hero_street / stack,
        hero_total / stack,
        vb,
        min_r / max_r,
    ] + onehot


def _hero_best_hand_strength_index(view: dict[str, Any]) -> int:
    """hand_eval category index for hero's best 5 (0=unknown … 9=straight flush). 0 if not evaluable."""
    hero = int(view.get("hero", 0)) % 6
    holes = view.get("hole_by_seat") or []
    if hero >= len(holes) or len(holes[hero]) < 2:
        return 0
    board = view.get("board_cards") or []
    from hand_eval import hand_strength_index, tuples_to_tokens

    all_cards = list(holes[hero]) + list(board)
    toks = tuples_to_tokens(all_cards)
    if len(toks) < 5:
        return 0
    return int(hand_strength_index(toks))


def _card_rank_tuple(rank: str) -> int:
    r = str(rank).upper()
    m = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    return int(m.get(r, 0))


def _hole_ranks_suited(view: dict[str, Any]) -> tuple[int, int, bool] | None:
    """Return (hi_rank, lo_rank, suited) for hero hole cards, or None."""
    if view.get("board_cards"):
        return None
    hero = int(view.get("hero", 0)) % 6
    holes = view.get("hole_by_seat") or []
    if hero >= len(holes) or len(holes[hero]) < 2:
        return None
    c0, c1 = holes[hero][0], holes[hero][1]
    r0, r1 = _card_rank_tuple(c0[0]), _card_rank_tuple(c1[0])
    if r0 <= 0 or r1 <= 0:
        return None
    suited = len(c0) > 1 and len(c1) > 1 and str(c0[1]).lower() == str(c1[1]).lower()
    hi, lo = max(r0, r1), min(r0, r1)
    return hi, lo, suited


def _is_garbage_preflop(view: dict[str, Any]) -> bool:
    """Very weak starting hands — should not defend or open without a strong proxy."""
    h = _hole_ranks_suited(view)
    if h is None:
        return False
    hi, lo, _suited = h
    if hi == lo:
        return hi <= 6  # 66-
    if hi <= 9:
        return True  # T-high or lower unpaired
    if hi == 10 and lo <= 7:
        return True
    if hi == 11 and lo <= 5:
        return True
    return False


def _preflop_premium_for_raise(view: dict[str, Any]) -> bool:
    """Strong opens / 3-bets: 88+, suited broadway, AK/AQ/AJ, KQs."""
    h = _hole_ranks_suited(view)
    if h is None:
        return False
    hi, lo, suited = h
    if hi == lo:
        return hi >= 8  # 88+
    if hi == 14 and lo == 13:
        return True  # AK
    if hi == 14 and lo == 12:
        return True  # AQ any
    if hi == 14 and lo == 11:
        return True  # AJ any
    if hi == 13 and lo == 12:
        return True  # KQ any (KQ suited was already implied; offsuit included)
    if hi == 14 and lo >= 9:
        return suited  # ATs+
    return False


def _can_complete_value_raise(view: dict[str, Any]) -> bool:
    """True if the engine allows a raise/bet beyond check/call (min completion affordable)."""
    call_amt = float(view.get("call_amount") or 0)
    min_r = float(view.get("min_raise") or 0)
    max_r = float(view.get("max_raise") or 0)
    facing = bool(view.get("facing_bet"))
    if max_r + 1e-6 < min_r:
        return False
    if not facing:
        return max_r > call_amt + 1e-6
    return min_r > call_amt + 1e-6


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [x / s for x in ex]


def _predict_logistic(view: dict[str, Any]) -> tuple[str, float]:
    model = _load_model()
    x = feature_vector_from_view(view)
    mean = model["scaler_mean"]
    scale = model["scaler_scale"]
    z: list[float] = []
    for i in range(len(x)):
        sig = float(scale[i]) if i < len(scale) else 1.0
        mu = float(mean[i]) if i < len(mean) else 0.0
        z.append((float(x[i]) - mu) / sig if sig > 1e-12 else 0.0)

    coef = model["coef"]
    intercept = model["intercept"]
    logits: list[float] = []
    for k in range(len(intercept)):
        logits.append(
            sum(float(coef[k][i]) * z[i] for i in range(min(len(z), len(coef[k]))))
            + float(intercept[k])
        )
    probs = _softmax(logits)
    k_best = max(range(len(probs)), key=lambda i: probs[i])
    classes: list[str] = list(model["class_order"])
    return classes[k_best], float(probs[k_best] * 100.0)


def _should_force_fold_weak_hand(
    view: dict[str, Any], wp: float, facing: bool, call_frac: float, tier: int
) -> bool:
    """Fold trash that the softmax often labels as 'call' (proxy miscalibration + class collapse)."""
    if not facing or float(view.get("call_amount") or 0) <= 0:
        return False
    call_amt = float(view.get("call_amount") or 0)
    # Preflop trash: fold real defense prices, but allow tiny completions (e.g. last chip to post BB).
    if _is_garbage_preflop(view) and call_amt > 0 and (call_frac >= 0.008 or call_amt >= 2):
        return True
    # Global: terrible price vs proxy
    if wp <= 0.22 and call_frac >= 0.04:
        return True
    if wp <= 0.30 and call_frac >= 0.12:
        return True
    if wp <= 0.36 and call_frac >= 0.22:
        return True
    # Postflop: one pair or worse, facing meaningful heat
    if view.get("board_cards") and tier <= 2 and call_frac >= 0.07 and wp <= 0.36:
        return True
    if view.get("board_cards") and tier <= 1 and call_frac >= 0.05 and wp <= 0.40:
        return True
    return False


def predict_decision_advisor(view: dict[str, Any]) -> tuple[str, float]:
    """Return (action one of fold/call/raise, confidence 0..100).

    Multinomial logits are trained on a synthetic oracle; live views drift. We apply
    **fold sanity** (weak hand + real price) and **raise nudges** (premiums, strong made
    hands, thin value) so the UI is not stuck on marginal **call** for garbage.
    """
    wp_raw = view.get("win_probability")
    wp = float(min(0.99, max(0.01, float(wp_raw)))) if wp_raw is not None else 0.35
    facing = bool(view.get("facing_bet"))
    stack = float((view.get("stacks") or [200])[int(view.get("hero", 0)) % 6] or 200)
    stack = max(stack, 1.0)
    call_frac = float(view.get("call_amount") or 0) / stack
    tier = _hero_best_hand_strength_index(view)

    action, conf = _predict_logistic(view)

    # Do not fold obvious premiums / monsters when the head mis-ranks.
    monster = tier >= 7 or (tier >= 6 and wp >= 0.55) or _preflop_premium_for_raise(view)
    if not monster and _should_force_fold_weak_hand(view, wp, facing, call_frac, tier):
        return "fold", max(62.0, min(88.0, 100.0 * (0.45 + 0.5 * min(1.0, call_frac / 0.25))))

    # COMBINATION_MAP: 7=full house, 8=quads, 9=straight flush
    if tier >= 7:
        if _can_complete_value_raise(view):
            return "raise", max(86.0, conf)
        if action == "fold":
            return "call", max(78.0, conf)

    if _can_complete_value_raise(view):
        # Straight or better — value / protection (linear head under-raises these).
        if tier >= 5 and wp >= 0.44 and action != "fold":
            return "raise", max(78.0, conf)
        # Premium preflop
        if _preflop_premium_for_raise(view) and not facing and action != "fold":
            return "raise", max(83.0, conf)
        if _preflop_premium_for_raise(view) and facing and call_frac <= 0.12 and wp >= 0.34 and action != "fold":
            return "raise", max(80.0, conf)
        # Open: any non-garbage hand with a typical heuristic — stop limp-only behavior
        if (
            not facing
            and not _is_garbage_preflop(view)
            and (wp >= 0.42 or _preflop_premium_for_raise(view))
            and action == "call"
        ):
            return "raise", max(71.0, conf)
        # Facing a small-to-mid price with playability
        if facing and wp >= 0.52 and call_frac <= 0.20 and action == "call" and not _is_garbage_preflop(view):
            return "raise", max(73.0, conf)
        if facing and wp >= 0.56 and call_frac <= 0.28 and action == "call":
            return "raise", max(74.0, conf)

    return action, conf
