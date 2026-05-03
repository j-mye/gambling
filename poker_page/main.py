"""PyScript entry: DOM bridge, hero controls, async bot loop."""

from __future__ import annotations

import asyncio
import html
from typing import Any

from pyscript import document

import coach_engine
import poker_core
from cards_html import hero_card_cell
from game_state import GameState

state = GameState()


def _el(eid: str) -> Any:
    return document.getElementById(eid)


def _sync_coach_toggle_ui() -> None:
    """Match header switch visuals to ``state.coach_mode``."""
    btn = _el("coach-toggle")
    if btn is not None:
        btn.setAttribute("aria-pressed", "true" if state.coach_mode else "false")
    track = _el("coach-toggle-track")
    knob = _el("coach-toggle-knob")
    if track is None or knob is None:
        return
    if state.coach_mode:
        track.className = (
            "w-8 h-4 rounded-full relative transition-colors bg-emerald-500"
        )
        knob.className = (
            "absolute right-0.5 top-0.5 w-3 h-3 rounded-full shadow transition-all "
            "bg-white"
        )
    else:
        track.className = (
            "w-8 h-4 rounded-full relative transition-colors bg-slate-600"
        )
        knob.className = (
            "absolute left-0.5 top-0.5 w-3 h-3 rounded-full shadow transition-all "
            "bg-slate-200"
        )


def _engines_ui_order(hero: int) -> list[int]:
    """Map engine seat index → fixed UI shell 0..5.

    Slot 5 is always the hero (bottom center). Opponents use shells clockwise around
    the oval from the hero's perspective: left center → top-left → top-center →
    top-right → right center (first clockwise neighbor at left center).

    Engine (hero+1) is the next seat in deal order and maps to that first shell.
    """
    h = int(hero) % 6
    # UI slots in clockwise order around the felt (hero fixed at slot 5).
    clockwise_shells = (4, 0, 1, 2, 3)
    order = [0] * 6
    for k, ui_slot in enumerate(clockwise_shells):
        order[ui_slot] = (h + 1 + k) % 6
    order[5] = h
    return order


def _cards_render_signature(view: dict[str, Any], hero: int) -> tuple:
    """Stable signature for board + what each seat's hole area should show."""
    hero = int(hero)
    board = tuple(tuple(c) for c in view.get("board_cards", ()))
    per_seat: list[tuple] = []
    for es in range(6):
        hole = view["hole_by_seat"][es] if es < len(view["hole_by_seat"]) else []
        folded = view["folded"][es] if es < len(view["folded"]) else False
        hc = bool(view.get("hand_complete"))
        if es == hero:
            per_seat.append(("H", tuple((r, s) for r, s in hole)))
        elif hc and not folded:
            per_seat.append(("S", tuple((r, s) for r, s in hole)))
        else:
            per_seat.append(("D", len(hole) or 2))
    return (board, tuple(per_seat))


def _set_controls_disabled(disabled: bool) -> None:
    for eid in ("btn-fold", "btn-call", "btn-raise", "raise-slider", "raise-value"):
        node = _el(eid)
        if node is None:
            continue
        try:
            node.disabled = disabled
        except Exception:
            if disabled:
                node.setAttribute("disabled", "disabled")
            else:
                node.removeAttribute("disabled")


def _sync_raise_slider(view: dict[str, Any]) -> None:
    slider = _el("raise-slider")
    if slider is None:
        return
    mn = int(view.get("min_raise", poker_core._MIN_BET * 2))
    mx = int(view.get("max_raise", poker_core._STARTING_STACK))
    if mx < mn:
        mx = mn
    slider.min = str(mn)
    slider.max = str(mx)
    if state.raise_amount < mn or state.raise_amount > mx:
        state.raise_amount = mn
    slider.value = str(state.raise_amount)
    rv = _el("raise-value")
    if rv is not None:
        try:
            rv.min = str(mn)
            rv.max = str(mx)
            rv.value = str(state.raise_amount)
        except Exception:
            pass


def _sync_raise_button_label(view: dict[str, Any]) -> None:
    facing = view.get("facing_bet", False)
    submit = _el("btn-raise")
    if submit is not None:
        bet_label = "Raise To" if facing else "Bet"
        submit.innerText = f"{bet_label} ${int(state.raise_amount)}"


def update_ui() -> None:
    poker_core.ensure_initialized(state)
    view = poker_core.build_view(state)
    hero = int(state.hero)

    hand_pyid = id(state.poker_state) if state.poker_state is not None else None
    if hand_pyid != state.ui_hand_pyid:
        state.ui_cards_sig = None
        state.ui_hand_pyid = hand_pyid

    new_sig = _cards_render_signature(view, hero)
    cards_changed = new_sig != state.ui_cards_sig

    pot = _el("total-pot")
    if pot is not None:
        pot.innerText = f"${float(view.get('pot_total', 0.0)):,.2f}"

    if cards_changed:
        board = _el("board-cards")
        if board is not None:
            board.innerHTML = poker_core._board_html(view)

    eng_order = _engines_ui_order(hero)
    ti_raw = view.get("turn_index")
    ti_int: int | None = int(ti_raw) if ti_raw is not None else None
    for ui_slot, engine_seat in enumerate(eng_order):
        shell = _el(f"seat-shell-{ui_slot}")
        if shell is None:
            continue
        is_active = (ti_int == int(engine_seat)) and not view["hand_complete"]
        cls = shell.className or ""
        if is_active and "active-turn" not in cls:
            shell.className = (cls + " active-turn").strip()
        elif not is_active and "active-turn" in cls:
            shell.className = " ".join(c for c in cls.split() if c != "active-turn").strip()

        nm = _el(f"seat-name-{ui_slot}")
        if nm is not None:
            if engine_seat == hero:
                nm.innerText = f"YOU (Seat {int(hero) + 1})"
            else:
                nm.innerText = f"Seat {int(engine_seat) + 1}"

        st = _el(f"seat-stack-{ui_slot}")
        if st is not None:
            stack = view["stacks"][engine_seat] if engine_seat < len(view["stacks"]) else 0
            st.innerText = f"${int(stack):,}"

        folded_now = (
            view["folded"][engine_seat]
            if engine_seat < len(view["folded"])
            else False
        )
        inner = _el(f"seat-inner-{ui_slot}")
        if inner is not None:
            icls = inner.className or ""
            if folded_now:
                if "seat-folded" not in icls:
                    inner.className = (icls + " seat-folded").strip()
            else:
                inner.className = " ".join(
                    c for c in icls.split() if c != "seat-folded"
                ).strip()

        blf = _el(f"seat-bluff-{ui_slot}")
        if blf is not None:
            if folded_now:
                blf.innerText = "Bluff —"
            else:
                bps = view.get("bluff_prob_by_seat") or []
                bp = bps[engine_seat] if engine_seat < len(bps) else None
                blf.innerText = (
                    f"Bluff {float(bp) * 100:.0f}%"
                    if bp is not None
                    else "Bluff —"
                )

        role_el = _el(f"seat-role-{ui_slot}")
        if role_el is not None:
            role = view.get("seat_roles", {}).get(engine_seat, "")
            role_el.innerText = role or "\u00a0"

        act = _el(f"seat-action-{ui_slot}")
        if act is not None:
            la = view["last_action"].get(engine_seat, "")
            act.innerText = la or "\u00a0"

        if cards_changed:
            holes = _el(f"seat-holes-{ui_slot}")
            if holes is not None:
                folded_hole = (
                    view["folded"][engine_seat]
                    if engine_seat < len(view["folded"])
                    else False
                )
                hole = view["hole_by_seat"][engine_seat] if engine_seat < len(view["hole_by_seat"]) else []
                if engine_seat == hero:
                    hole_html = "".join(hero_card_cell(r, s) for r, s in hole)
                else:
                    reveal = bool(view["hand_complete"] and not folded_hole)
                    if reveal:
                        hole_html = "".join(hero_card_cell(r, s) for r, s in hole)
                    else:
                        hole_html = "".join(
                            "<div class='w-8 h-11 rounded bg-slate-800 border border-slate-600'></div>"
                            for _ in range(len(hole) or 2)
                        )
                holes.innerHTML = hole_html

    if cards_changed:
        hh = _el("hero-hole-cards")
        if hh is not None:
            hero_hole = view["hole_by_seat"][hero] if hero < len(view["hole_by_seat"]) else []
            hh.innerHTML = "".join(hero_card_cell(r, s) for r, s in hero_hole)
        state.ui_cards_sig = new_sig

    hs = _el("hero-stack")
    if hs is not None:
        stack = view["stacks"][hero] if hero < len(view["stacks"]) else 0
        hs.innerText = f"${float(stack):,.2f}"

    tc = _el("hero-to-call")
    if tc is not None:
        ca = float(view.get("call_amount", 0) or 0)
        tc.innerText = f"${ca:,.2f}"

    wp_el = _el("stat-win-prob")
    if wp_el is not None:
        wp = view.get("win_probability")
        wp_el.innerText = f"{float(wp) * 100:.1f}%" if wp is not None else "—"

    gb = _el("gto-bar")
    if gb is not None:
        gb.innerHTML = coach_engine.gto_bar_html(view)

    ct = _el("coach-text")
    if ct is not None:
        if state.coach_mode:
            ct.innerHTML = (
                '<span class="font-bold text-emerald-500 uppercase mr-1">Coach:</span>'
                + html.escape(coach_engine.coach_message(view))
            )
        else:
            ct.innerText = "Coach mode off."

    _sync_coach_toggle_ui()

    ae = _el("action-error")
    if ae is not None:
        ae.innerText = view.get("action_error", "") or ""

    me = _el("model-error")
    if me is not None:
        parts = [view.get("bot_error", ""), view.get("prediction_error", ""), view.get("bluff_prediction_error", "")]
        me.innerText = " | ".join(p for p in parts if p) or ""

    call_btn = _el("btn-call")
    if call_btn is not None:
        call_btn.innerText = poker_core.stable_call_button_label(view)

    _sync_raise_slider(view)
    _sync_raise_button_label(view)

    hand_over = view.get("hand_complete", False)
    banner = _el("hand-over-banner")
    nxt = _el("btn-next-hand")
    betting = _el("betting-actions")
    if betting is not None:
        betting.classList.toggle("hidden", bool(hand_over))

    if banner is not None:
        if hand_over:
            hp = poker_core.hero_payoff_chips(view)
            msg = "You won!" if hp > 0 else ("You lost." if hp < 0 else "Chopped pot.")
            banner.innerText = f"Hand over — {msg} ({hp:+d} chips)"
            banner.classList.remove("hidden")
            if hp > 0:
                banner.className = (
                    "w-full max-w-md px-3 py-2 rounded-lg border text-sm font-bold text-center "
                    "border-emerald-500/60 bg-emerald-950/40 text-emerald-300"
                )
            elif hp < 0:
                banner.className = (
                    "w-full max-w-md px-3 py-2 rounded-lg border text-sm font-bold text-center "
                    "border-red-500/50 bg-red-950/30 text-red-300"
                )
            else:
                banner.className = (
                    "w-full max-w-md px-3 py-2 rounded-lg border text-sm font-bold text-center "
                    "border-slate-500/50 bg-slate-900/90 text-slate-200"
                )
        else:
            banner.classList.add("hidden")
            banner.innerText = ""
    if nxt is not None:
        nxt.classList.toggle("hidden", not hand_over)

    controls_disabled = (not view.get("is_hero_turn")) or hand_over or state.bots_running
    _set_controls_disabled(controls_disabled)


def _schedule_run_bots() -> None:
    asyncio.ensure_future(run_bots())


async def run_bots() -> None:
    if state.bots_running:
        return
    state.bots_running = True
    _set_controls_disabled(True)
    try:
        for _ in range(120):
            poker_core.run_dealer(state)
            ps = state.poker_state
            if ps is None:
                break
            if not ps.status:
                poker_core.capture_terminal_if_needed(state)
                break
            ti = ps.turn_index
            if ti is None:
                break
            if ti == int(state.hero):
                break
            update_ui()
            await asyncio.sleep(1.5)
            poker_core.run_one_bot_turn(state)
            update_ui()
            await asyncio.sleep(0.5)
    finally:
        state.bots_running = False
        _set_controls_disabled(False)
        update_ui()


def handle_fold(event):  # noqa: ARG001
    if state.bots_running:
        return
    poker_core.ensure_initialized(state)
    view = poker_core.build_view(state)
    if not view.get("is_hero_turn") or view.get("hand_complete"):
        return
    poker_core.apply_hero_action(state, {"type": "fold"})
    poker_core.run_dealer(state)
    poker_core.capture_terminal_if_needed(state)
    update_ui()
    _schedule_run_bots()


def handle_call(event):  # noqa: ARG001
    if state.bots_running:
        return
    poker_core.ensure_initialized(state)
    view = poker_core.build_view(state)
    if not view.get("is_hero_turn") or view.get("hand_complete"):
        return
    poker_core.apply_hero_action(state, {"type": "call"})
    poker_core.run_dealer(state)
    poker_core.capture_terminal_if_needed(state)
    update_ui()
    _schedule_run_bots()


def handle_raise(event):  # noqa: ARG001
    if state.bots_running:
        return
    poker_core.ensure_initialized(state)
    view = poker_core.build_view(state)
    if not view.get("is_hero_turn") or view.get("hand_complete"):
        return
    poker_core.apply_hero_action(state, {"type": "raise", "amount": int(state.raise_amount)})
    poker_core.run_dealer(state)
    poker_core.capture_terminal_if_needed(state)
    update_ui()
    _schedule_run_bots()


def handle_raise_slider(event):  # noqa: ARG001
    slider = _el("raise-slider")
    if slider is None:
        return
    try:
        state.raise_amount = int(float(slider.value))
    except Exception:
        pass
    rv = _el("raise-value")
    if rv is not None:
        try:
            rv.value = str(state.raise_amount)
        except Exception:
            pass
    view = poker_core.build_view(state)
    _sync_raise_button_label(view)


def handle_raise_value_input(event):  # noqa: ARG001
    """Live sync when the typed amount is already within legal min/max."""
    if state.bots_running:
        return
    poker_core.ensure_initialized(state)
    tgt = getattr(event, "target", None)
    if tgt is None:
        return
    raw = (getattr(tgt, "value", None) or "").strip()
    if raw == "":
        return
    try:
        v = int(float(raw))
    except (ValueError, TypeError):
        return
    view = poker_core.build_view(state)
    mn = int(view.get("min_raise", poker_core._MIN_BET * 2))
    mx = int(view.get("max_raise", poker_core._STARTING_STACK))
    if v < mn or v > mx:
        return
    state.raise_amount = v
    sl = _el("raise-slider")
    if sl is not None:
        sl.value = str(v)
    _sync_raise_button_label(view)


def handle_raise_value_change(event):  # noqa: ARG001
    """Clamp and normalize on commit (blur / change)."""
    if state.bots_running:
        return
    poker_core.ensure_initialized(state)
    tgt = getattr(event, "target", None)
    if tgt is None:
        return
    raw = (getattr(tgt, "value", None) or "").strip()
    view = poker_core.build_view(state)
    mn = int(view.get("min_raise", poker_core._MIN_BET * 2))
    mx = int(view.get("max_raise", poker_core._STARTING_STACK))
    if mx < mn:
        mx = mn
    try:
        v = int(float(raw)) if raw else mn
    except (ValueError, TypeError):
        v = mn
    state.raise_amount = max(mn, min(mx, v))
    try:
        tgt.value = str(state.raise_amount)
    except Exception:
        pass
    sl = _el("raise-slider")
    if sl is not None:
        sl.value = str(state.raise_amount)
    _sync_raise_button_label(view)


def handle_next_hand(event):  # noqa: ARG001
    if state.bots_running:
        return
    poker_core.ensure_initialized(state)
    poker_core.new_hand(state, advance_button=True)
    update_ui()
    _schedule_run_bots()


def handle_coach_toggle(event):  # noqa: ARG001
    state.coach_mode = not state.coach_mode
    update_ui()


def py_init():
    poker_core.ensure_initialized(state)
    v = poker_core.build_view(state)
    mn = int(v.get("min_raise", poker_core._MIN_BET * 2))
    state.raise_amount = mn
    update_ui()
    _schedule_run_bots()


py_init()
