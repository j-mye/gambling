"""Poker engine adapter seam with strict PokerKit path."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import random
from typing import Any, Protocol


MoveType = str


@dataclass
class MoveResult:
    ok: bool
    message: str


class PokerEngineAdapter(Protocol):
    def legal_snapshot(self, state: dict, actor: int | None = None) -> dict[str, Any]: ...
    def legal_actions(self, state: dict) -> set[MoveType]: ...
    def apply_hero_move(self, state: dict, move_type: MoveType, amount: float | None = None) -> MoveResult: ...
    def run_bots_until_hero_or_terminal(self, state: dict) -> list[dict]: ...
    def extract_view_model(self, state: dict) -> dict: ...


class StubNlheAdapter:
    """Deterministic UI-first adapter with simplified hand progression."""

    def _apply_contribution(self, state: dict, seat: int, amount: float) -> None:
        amount = max(0.0, min(amount, state["seat_stacks"][seat]))
        state["seat_stacks"][seat] -= amount
        state["seat_street_bet"][seat] += amount
        state["seat_total_bet"][seat] += amount
        state["pot_total"] += amount

    def _reset_street_bets(self, state: dict) -> None:
        for seat in state["seat_street_bet"]:
            state["seat_street_bet"][seat] = 0.0

    def _clear_action_badges(self, state: dict) -> None:
        for seat in state["last_action_by_seat"]:
            state["last_action_by_seat"][seat] = ""

    def _persist_terminal(self, state: dict, reason: str) -> None:
        if state.get("showdown_persisted"):
            return
        # In this stub model, payoff is inferred from stack delta vs initial 200 baseline.
        baseline = 200.0
        payoffs = {seat: round(stack - baseline, 2) for seat, stack in state["seat_stacks"].items()}
        state["payoffs_by_seat"] = payoffs
        state["final_stacks_by_seat"] = dict(state["seat_stacks"])
        state["hand_end_reason"] = reason
        state["showdown_persisted"] = True

    def legal_actions(self, state: dict) -> set[MoveType]:
        if state["hand_complete"]:
            return set()
        return {"fold", "check_call", "bet_raise"}

    def _advance_street(self, state: dict) -> None:
        streets = ["Preflop", "Flop", "Turn", "River", "Showdown"]
        current_idx = streets.index(state["active_street"])
        state["newly_dealt_count"] = 0
        if state["active_street"] == "Preflop" and len(state["board_cards"]) == 0:
            state["board_cards"].extend([("A", "S"), ("K", "D"), ("7", "H")])
            state["newly_dealt_count"] = 3
            state["board_deal_stage"] = "flop"
        elif state["active_street"] == "Flop" and len(state["board_cards"]) == 3:
            state["board_cards"].append(("2", "C"))
            state["newly_dealt_count"] = 1
            state["board_deal_stage"] = "turn"
        elif state["active_street"] == "Turn" and len(state["board_cards"]) == 4:
            state["board_cards"].append(("Q", "S"))
            state["newly_dealt_count"] = 1
            state["board_deal_stage"] = "river"
        if current_idx < len(streets) - 1:
            state["active_street"] = streets[current_idx + 1]
            if state["active_street"] in {"Flop", "Turn", "River"}:
                self._reset_street_bets(state)
                self._clear_action_badges(state)
        if state["active_street"] == "Showdown":
            state["hand_complete"] = True
            state["last_action"] = "Hand ended at showdown"
            state["board_deal_stage"] = "showdown"
            self._persist_terminal(state, "showdown")

    def apply_hero_move(self, state: dict, move_type: MoveType, amount: float | None = None) -> MoveResult:
        if state["hand_complete"]:
            return MoveResult(False, "Hand already complete.")
        if move_type == "fold":
            state["seat_folded"][state["hero_seat"]] = True
            state["hand_complete"] = True
            state["last_action_by_seat"][state["hero_seat"]] = "Fold"
            state["last_action"] = "Hero folded"
            state["action_log"].append("Hero: Fold")
            self._persist_terminal(state, "fold")
            return MoveResult(True, "You folded.")
        if move_type == "check_call":
            self._apply_contribution(state, state["hero_seat"], 2.0)
            state["last_action_by_seat"][state["hero_seat"]] = "Check/Call"
            state["last_action"] = "Hero check/call"
            state["action_log"].append("Hero: Check/Call")
            return MoveResult(True, "Check/Call applied.")
        if move_type == "bet_raise":
            amount = float(amount or 0.0)
            if amount <= 0:
                return MoveResult(False, "Raise amount must be positive.")
            max_allowed = state["seat_stacks"][state["hero_seat"]]
            if amount > max_allowed:
                return MoveResult(False, f"Raise exceeds stack ({max_allowed:.2f}).")
            self._apply_contribution(state, state["hero_seat"], amount)
            state["last_action_by_seat"][state["hero_seat"]] = f"Raises ${amount:.2f}"
            state["last_action"] = f"Hero raises to {amount:.2f}"
            state["action_log"].append(f"Hero: Bet/Raise {amount:.2f}")
            return MoveResult(True, "Bet/Raise applied.")
        return MoveResult(False, "Unknown move type.")

    def run_bots_until_hero_or_terminal(self, state: dict) -> list[dict]:
        if state["hand_complete"]:
            return []
        max_steps = 20
        frames: list[dict] = []
        seats = [s for s in range(state["seat_count"]) if s != state["hero_seat"]]
        for _ in range(max_steps):
            for seat in seats:
                if state["hand_complete"]:
                    return frames
                if state["seat_folded"][seat]:
                    continue
                state["turn_index"] = seat
                act = "Check/Call" if random.random() < 0.8 else "Fold"
                if act == "Check/Call":
                    call_amt = min(2.0, state["seat_stacks"][seat])
                    self._apply_contribution(state, seat, call_amt)
                    event = f"Bot Seat {seat + 1} checks/calls ${call_amt:.2f}"
                    state["last_action_by_seat"][seat] = f"Calls ${call_amt:.2f}"
                else:
                    state["seat_folded"][seat] = True
                    event = f"Bot Seat {seat + 1} folds"
                    state["last_action_by_seat"][seat] = "Fold"
                state["action_log"].append(f"Seat {seat + 1}: {act}")
                state["last_action"] = f"Seat {seat + 1} {act.lower()}"
                state["animation_tick"] += 1
                frames.append({"message": event, "state": copy.deepcopy(state)})
            self._advance_street(state)
            if state["newly_dealt_count"]:
                state["animation_tick"] += 1
                frames.append({"message": f"Board dealt for {state['active_street']}", "state": copy.deepcopy(state)})
            if state["hand_complete"]:
                return frames
            # Return turn to hero after each street cycle.
            state["pending_actor"] = state["hero_seat"]
            state["turn_index"] = state["hero_seat"]
            break
        return frames

    def extract_view_model(self, state: dict) -> dict:
        return state


class PokerKitNlheAdapter:
    """Strict PokerKit adapter with legal-action gating."""

    def _pk_state(self, state: dict):
        payload = state.get("engine_payload", {})
        return payload.get("pk_state")

    def _method_ok(self, obj: Any, method_name: str) -> bool:
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
        return False

    def _call_if_exists(self, obj: Any, method_name: str, *args):
        method = getattr(obj, method_name, None)
        if callable(method):
            return method(*args)
        raise AttributeError(f"PokerKit state missing method {method_name}")

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _estimate_call_amount(self, pk_state: Any, actor: int) -> float:
        if hasattr(pk_state, "amount_to_call"):
            try:
                return max(0.0, self._to_float(pk_state.amount_to_call))
            except Exception:
                pass
        bets = getattr(pk_state, "bets", None)
        if bets is not None:
            try:
                max_bet = max(float(x) for x in bets)
                return max(0.0, max_bet - float(bets[actor]))
            except Exception:
                pass
        return 0.0

    def _estimate_min_raise_to(self, pk_state: Any, actor: int, call_amount: float) -> float:
        if hasattr(pk_state, "min_bet"):
            return max(self._to_float(getattr(pk_state, "min_bet"), 1.0), 1.0)
        if hasattr(pk_state, "min_completion_betting_or_raising_to_amount"):
            return max(self._to_float(getattr(pk_state, "min_completion_betting_or_raising_to_amount"), 1.0), 1.0)
        stacks = getattr(pk_state, "stacks", None)
        if stacks is not None:
            try:
                return max(1.0, min(float(stacks[actor]), max(1.0, call_amount + 2.0)))
            except Exception:
                pass
        return max(1.0, call_amount + 2.0)

    def _sync_from_pk(self, state: dict) -> None:
        pk_state = self._pk_state(state)
        if pk_state is None:
            return
        if hasattr(pk_state, "turn_index"):
            state["turn_index"] = int(getattr(pk_state, "turn_index"))
        if hasattr(pk_state, "status"):
            status = getattr(pk_state, "status")
            # In PokerKit, terminal can be falsy.
            state["hand_complete"] = not bool(status)
        if hasattr(pk_state, "stacks"):
            try:
                stacks = list(getattr(pk_state, "stacks"))
                state["seat_stacks"] = {i: float(v) for i, v in enumerate(stacks)}
            except Exception:
                pass

    def _advance_forced_blinds(self, state: dict, max_steps: int = 12) -> tuple[bool, str]:
        """
        Auto-post forced blinds/straddles before any voluntary action UI is shown.
        Returns (ok, message). ok=False means blind posting did not settle safely.
        """
        pk_state = self._pk_state(state)
        if pk_state is None:
            return False, "PokerKit state unavailable."
        steps = 0
        while self._method_ok(pk_state, "can_post_blind_or_straddle"):
            steps += 1
            if steps > max_steps:
                return False, "Blind posting did not resolve (safety stop)."
            try:
                self._call_if_exists(pk_state, "post_blind_or_straddle")
            except Exception as exc:
                return False, f"Blind posting failed: {exc}"
            self._sync_from_pk(state)
        return True, ""
        if hasattr(pk_state, "payoffs") and state.get("hand_complete"):
            try:
                payoffs = list(getattr(pk_state, "payoffs"))
                state["payoffs_by_seat"] = {i: float(v) for i, v in enumerate(payoffs)}
                state["final_stacks_by_seat"] = dict(state["seat_stacks"])
                state["showdown_persisted"] = True
                if not state.get("hand_end_reason"):
                    state["hand_end_reason"] = "showdown"
            except Exception:
                pass

    def legal_snapshot(self, state: dict, actor: int | None = None) -> dict[str, Any]:
        pk_state = self._pk_state(state)
        if pk_state is None:
            return {
                "ok": False,
                "reason": "PokerKit state unavailable",
                "can_fold": False,
                "can_check_or_call": False,
                "can_raise": False,
                "facing_bet": False,
                "call_amount": 0.0,
                "min_raise_to": 1.0,
                "max_raise_to": 1.0,
                "actor": state.get("turn_index", state.get("hero_seat", 0)),
                "blind_pending": False,
            }

        # Resolve mandatory blinds first so UI only offers voluntary actions.
        ok, msg = self._advance_forced_blinds(state)
        if not ok:
            return {
                "ok": False,
                "reason": msg,
                "can_fold": False,
                "can_check_or_call": False,
                "can_raise": False,
                "facing_bet": False,
                "call_amount": 0.0,
                "min_raise_to": 1.0,
                "max_raise_to": 1.0,
                "actor": state.get("turn_index", state.get("hero_seat", 0)),
                "blind_pending": True,
            }
        self._sync_from_pk(state)
        actor_idx = state.get("turn_index", state.get("hero_seat", 0)) if actor is None else actor
        call_amount = self._estimate_call_amount(pk_state, actor_idx)
        can_fold = self._method_ok(pk_state, "can_fold")
        can_check_or_call = self._method_ok(pk_state, "can_check_or_call")
        can_raise = (
            self._method_ok(pk_state, "can_complete_bet_or_raise_to")
            or self._method_ok(pk_state, "can_bet")
            or self._method_ok(pk_state, "can_raise")
        )
        min_raise_to = self._estimate_min_raise_to(pk_state, actor_idx, call_amount)
        max_raise_to = min_raise_to
        if hasattr(pk_state, "stacks"):
            try:
                max_raise_to = max(min_raise_to, float(getattr(pk_state, "stacks")[actor_idx]))
            except Exception:
                pass
        return {
            "ok": True,
            "reason": "",
            "actor": actor_idx,
            "can_fold": can_fold,
            "can_check_or_call": can_check_or_call,
            "can_raise": can_raise,
            "facing_bet": call_amount > 0.0,
            "call_amount": call_amount,
            "min_raise_to": min_raise_to,
            "max_raise_to": max_raise_to,
            "blind_pending": self._method_ok(pk_state, "can_post_blind_or_straddle"),
        }

    def legal_actions(self, state: dict) -> set[MoveType]:
        snap = self.legal_snapshot(state)
        actions: set[MoveType] = set()
        if snap["can_fold"]:
            actions.add("fold")
        if snap["can_check_or_call"]:
            actions.add("check_call")
        if snap["can_raise"]:
            actions.add("bet_raise")
        return actions

    def apply_hero_move(self, state: dict, move_type: MoveType, amount: float | None = None) -> MoveResult:
        pk_state = self._pk_state(state)
        if pk_state is None:
            return MoveResult(False, "PokerKit state missing from session payload.")
        ok, msg = self._advance_forced_blinds(state)
        if not ok:
            return MoveResult(False, msg)
        snap = self.legal_snapshot(state, actor=state.get("hero_seat"))
        state["legal_snapshot"] = snap
        try:
            if move_type == "fold":
                if not snap["can_fold"]:
                    return MoveResult(False, "Fold not legal.")
                self._call_if_exists(pk_state, "fold")
                state["last_action_by_seat"][state["hero_seat"]] = "Fold"
            elif move_type == "check_call":
                if not snap["can_check_or_call"]:
                    return MoveResult(False, "Check/Call not legal.")
                self._call_if_exists(pk_state, "check_or_call")
                label = f"Call ${snap['call_amount']:.2f}" if snap["facing_bet"] else "Check"
                state["last_action_by_seat"][state["hero_seat"]] = label
            elif move_type == "bet_raise":
                if not snap["can_raise"]:
                    return MoveResult(False, "Bet/Raise not legal.")
                target = self._to_float(amount, snap["min_raise_to"])
                target = max(snap["min_raise_to"], min(target, snap["max_raise_to"]))
                can_for_amount = self._method_ok(pk_state, "can_complete_bet_or_raise_to")
                if can_for_amount:
                    try:
                        if not bool(getattr(pk_state, "can_complete_bet_or_raise_to")(target)):
                            return MoveResult(False, "Invalid raise amount.")
                    except Exception:
                        return MoveResult(False, "Invalid raise amount.")
                self._call_if_exists(pk_state, "complete_bet_or_raise_to", target)
                action_word = "Raise To" if snap["facing_bet"] else "Bet"
                state["last_action_by_seat"][state["hero_seat"]] = f"{action_word} ${target:.2f}"
            else:
                return MoveResult(False, "Unknown move.")
        except Exception as exc:
            return MoveResult(False, f"PokerKit action failed: {exc}")

        self._sync_from_pk(state)
        return MoveResult(True, "Hero action applied.")

    def run_bots_until_hero_or_terminal(self, state: dict) -> list[dict]:
        pk_state = self._pk_state(state)
        if pk_state is None:
            state["last_action"] = "PokerKit state unavailable."
            return [{"message": "PokerKit state unavailable.", "state": copy.deepcopy(state)}]

        frames: list[dict] = []
        max_steps = 40
        hero = state.get("hero_seat", 0)

        for _ in range(max_steps):
            blind_ok, blind_msg = self._advance_forced_blinds(state)
            if not blind_ok:
                frames.append({"message": blind_msg, "state": copy.deepcopy(state)})
                return frames
            self._sync_from_pk(state)
            if state.get("hand_complete"):
                break
            actor = state.get("turn_index", hero)
            if actor == hero:
                break
            snap = self.legal_snapshot(state, actor=actor)
            state["legal_snapshot"] = snap
            msg = f"Seat {actor + 1}: "
            try:
                if snap["facing_bet"]:
                    # Facing bet: 80% call, 20% fold.
                    if random.random() < 0.8 and snap["can_check_or_call"]:
                        self._call_if_exists(pk_state, "check_or_call")
                        msg += f"Call ${snap['call_amount']:.2f}"
                        state["last_action_by_seat"][actor] = f"Call ${snap['call_amount']:.2f}"
                    elif snap["can_fold"]:
                        self._call_if_exists(pk_state, "fold")
                        msg += "Fold"
                        state["seat_folded"][actor] = True
                        state["last_action_by_seat"][actor] = "Fold"
                    elif snap["can_check_or_call"]:
                        self._call_if_exists(pk_state, "check_or_call")
                        msg += f"Call ${snap['call_amount']:.2f}"
                        state["last_action_by_seat"][actor] = f"Call ${snap['call_amount']:.2f}"
                    else:
                        msg += "No legal action"
                else:
                    # Not facing bet: mostly check, occasional legal open-bet.
                    open_bet = random.random() < 0.2 and snap["can_raise"]
                    if open_bet:
                        target = min(snap["max_raise_to"], max(snap["min_raise_to"], snap["min_raise_to"]))
                        can_bet_target = False
                        try:
                            can_bet_target = bool(getattr(pk_state, "can_complete_bet_or_raise_to")(target))
                        except Exception:
                            can_bet_target = False
                        if can_bet_target:
                            self._call_if_exists(pk_state, "complete_bet_or_raise_to", target)
                            msg += f"Bet ${target:.2f}"
                            state["last_action_by_seat"][actor] = f"Bet ${target:.2f}"
                        elif snap["can_check_or_call"]:
                            self._call_if_exists(pk_state, "check_or_call")
                            msg += "Check"
                            state["last_action_by_seat"][actor] = "Check"
                        else:
                            msg += "No legal action"
                    elif snap["can_check_or_call"]:
                        self._call_if_exists(pk_state, "check_or_call")
                        msg += "Check"
                        state["last_action_by_seat"][actor] = "Check"
                    elif snap["can_fold"]:
                        self._call_if_exists(pk_state, "fold")
                        msg += "Fold"
                        state["seat_folded"][actor] = True
                        state["last_action_by_seat"][actor] = "Fold"
                    else:
                        msg += "No legal action"
            except Exception as exc:
                msg += f"Action error: {exc}"

            state["last_action"] = msg
            self._sync_from_pk(state)
            if state.get("hand_complete"):
                # Trigger showdown data only after true terminal status.
                self._sync_from_pk(state)
            frames.append({"message": msg, "state": copy.deepcopy(state)})
            if state.get("turn_index", hero) == hero:
                break

        return frames

    def extract_view_model(self, state: dict) -> dict:
        self._sync_from_pk(state)
        return state
