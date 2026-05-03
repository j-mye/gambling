"""Famous names for the six seats — reshuffled on restart; busted seats get a new arrival."""

from __future__ import annotations

import random
from typing import Any

# Only these names appear at the table; keep `avatars.py` `_AVATAR_FILES` in sync.
GUEST_POOL: tuple[str, ...] = (
    "Mahatma Gandhi",
    "Madonna",
    "Cleopatra",
    "Napoleon",
    "Taylor Swift",
    "Albert Einstein",
    "Genghis Khan",
    "Julius Caesar",
    "Abraham Lincoln",
    "Freddie Mercury",
    "Elvis Presley",
    "Winston Churchill",
    "Socrates",
    "Mr. Bean",
    "Dr. Bukowy"
)


def assign_full_table(gs: Any) -> None:
    """Give every engine seat a fresh random name (no duplicates at deal time)."""
    names = list(GUEST_POOL)
    random.shuffle(names)
    if len(names) < 6:
        gs.seat_names = list(names) + names[: 6 - len(names)]
    else:
        gs.seat_names = names[:6]


def replace_guest_at_seat(gs: Any, seat: int) -> None:
    """Someone busted out — a new celebrity sits down. Prefer names not already at the table."""
    seat = int(seat) % 6
    while len(gs.seat_names) < 6:
        gs.seat_names.append(random.choice(GUEST_POOL))
    used = {gs.seat_names[i] for i in range(6) if i != seat}
    candidates = [n for n in GUEST_POOL if n not in used]
    if not candidates:
        candidates = list(GUEST_POOL)
    gs.seat_names[seat] = random.choice(candidates)


def ensure_seat_names(gs: Any) -> None:
    """Guarantee six display names exist before UI/engine rely on them."""
    if len(gs.seat_names) != 6 or not any(str(x).strip() for x in gs.seat_names):
        assign_full_table(gs)
