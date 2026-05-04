# **POKER RAAAAAAAHHHH**

## Environment (PokerKit)

The table engine uses **[PokerKit](https://pypi.org/project/pokerkit/)**. Install project deps, then pick a **Jupyter kernel** that uses the same interpreter:

```bash
pip install -e .
# or
pip install "pokerkit>=0.7.3"
```

If notebooks say `No module named 'pokerkit'`, the notebook kernel is a different Python than where you ran `pip install`. In VS Code / Cursor: **Kernel → Change Kernel** and choose the env where PokerKit is installed, or run `python -m pip install pokerkit` using that kernel’s `python` path.

**PyScript (browser):** `poker_page/pyscript.toml` lists `pokerkit`; PyScript loads it via its package installer when you open the page online.