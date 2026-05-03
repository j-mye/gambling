"""Bundled PNG headshots for `table_guests.GUEST_POOL` only — no generated defaults."""

from __future__ import annotations

from table_guests import GUEST_POOL

# 1×1 transparent GIF — used when the seated label is not in GUEST_POOL (e.g. empty seat).
NO_AVATAR_SRC = (
    "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)

# Keys must match `GUEST_POOL` exactly (validated at import).
_AVATAR_FILES: dict[str, str] = {
    "Mahatma Gandhi": "mahatma-gandhi.png",
    "Madonna": "madonna.png",
    "Cleopatra": "cleopatra.png",
    "Napoleon": "napoleon.png",
    "Taylor Swift": "taylor-swift.png",
    "Albert Einstein": "albert-einstein.png",
    "Genghis Khan": "genghis-khan.png",
    "Julius Caesar": "julius-caesar.png",
    "Abraham Lincoln": "abraham-lincoln.png",
    "Freddie Mercury": "freddie-mercury.png",
    "Elvis Presley": "elvis-presley.png",
    "Winston Churchill": "winston-churchill.png",
    "Socrates": "socrates.png",
    "Mr. Bean": "mr-bean.png",
    "Dr. Bukowy": "dr-bukowy.png",
}

_POOL = set(GUEST_POOL)
_AV_KEYS = set(_AVATAR_FILES)
if _POOL != _AV_KEYS:
    raise ValueError(
        "poker_page/avatars.py `_AVATAR_FILES` must match `table_guests.GUEST_POOL` "
        f"exactly; only in pool: {_POOL - _AV_KEYS}, only in avatars: {_AV_KEYS - _POOL}"
    )

_OBJECT_POS: dict[str, str] = {
    "Freddie Mercury": "center 12%",
    "Napoleon": "center 14%",
    "Elvis Presley": "center 24%",
    "Taylor Swift": "center 26%",
    "Cleopatra": "42% 28%",
    "Genghis Khan": "center 22%",
    "Socrates": "center 20%",
    "Mr. Bean": "center 30%",
    "Julius Caesar": "center 25%",
    "Abraham Lincoln": "center 28%",
    "Albert Einstein": "center 30%",
    "Winston Churchill": "center 22%",
    "Mahatma Gandhi": "center 30%",
    "Madonna": "center 28%",
    "Dr. Bukowy": "center 30%",
}


def avatar_src_for_guest(name: str) -> str:
    key = str(name).strip()
    fn = _AVATAR_FILES.get(key)
    if fn:
        return f"./avatar_img/{fn}"
    return NO_AVATAR_SRC


def avatar_object_position_for_guest(name: str) -> str:
    key = str(name).strip()
    return _OBJECT_POS.get(key, "center 26%")
