# CS336 (self-audit)

Working through [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) on my own.

## Layout

Each assignment is vendored from the official `stanford-cs336` repo with `git subtree`, so my solutions live alongside the starter code and upstream fixes can still be pulled in.

| Dir | Topic | Upstream |
|---|---|---|
| `assignment1-basics/` | Tokenizer (BPE), Transformer LM, training loop | [repo](https://github.com/stanford-cs336/assignment1-basics) |
| `assignment2-systems/` | Profiling, FlashAttention (Triton), distributed training | [repo](https://github.com/stanford-cs336/assignment2-systems) |
| `assignment3-scaling/` | Scaling laws | [repo](https://github.com/stanford-cs336/assignment3-scaling) |
| `assignment4-data/` | Data filtering / dedup pipeline | [repo](https://github.com/stanford-cs336/assignment4-data) |
| `assignment5-alignment/` | SFT, RL (GRPO) for reasoning | [repo](https://github.com/stanford-cs336/assignment5-alignment) |
| `notes/` | Lecture notes | [lectures](https://github.com/stanford-cs336/lectures) · [YouTube](https://www.youtube.com/@stanfordonline) |

## Setup

Each assignment uses [uv](https://docs.astral.sh/uv/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
cd assignment1-basics
uv run pytest        # runs the provided tests against your adapters
```

Large datasets go in each assignment's `data/` dir (git-ignored) — see the assignment README for download commands.

## Pulling upstream updates

```sh
git subtree pull --prefix assignment1-basics https://github.com/stanford-cs336/assignment1-basics main --squash
```

## Progress

- [ ] Assignment 1 — basics
- [ ] Assignment 2 — systems
- [ ] Assignment 3 — scaling
- [ ] Assignment 4 — data
- [ ] Assignment 5 — alignment
