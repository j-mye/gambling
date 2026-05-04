# Summary Information

This card documents the dataset used in this project for stage win prediction, visible bluff estimation, and action recommendation support.

| Field             | Description |
| ----------------- | ----------- |
| Name              | Online Poker Games (Kaggle) |
| Curation Date     | 2026-05-04 (card curation date) |
| Sensitivity Level | Internal academic use; low direct PII, moderate re-identification risk via player IDs and timestamps |
| Summary           | Hand-history style online poker data with actions, board cards, stack/pot context, and outcomes. Used to build cleaned features and train models for win probability and bluff-related behavior. |
| Source            | https://www.kaggle.com/datasets/murilogmamaral/online-poker-games |

# Card Authors

| Name | Email |
| ---- | ---- |
| Garrido-Lestachebeli | garrido-lestachebeli@msoe.edu |
| Joshua Myers | myersjr@msoe.edu |

# Known Sensitive Features

| Feature Name | Sensitive Type | Risk Reduction Method |
| ---- | ---- | ---- |
| `name` | Persistent user identifier (pseudonymous) | Do not publish row-level joins; aggregate in reports |
| `date`, `time`, `tourn_id`, `hand_id` | Event trace metadata | Keep outputs aggregated; avoid disclosing individual trajectories |
| `cards` | Potentially game-sensitive private cards | Used only for modeling; do not expose row-level records publicly |

# Data Overview

Computed from local file used by project: `data/gambling.csv`.

| Field | Value |
| ---- | ---- |
| Storage Size | 17,217,794 bytes (~16.42 MB) |
| Number of Rows | 102,615 |
| Number of Features | 35 |
| Data Format | CSV |

# Numerical Features Summary

| Feature | Count | Mean | Std Dev | Min | 25% | Median | 75% | Max |
| ------- | ----- | ---- | ------- | --- | --- | ------ | --- | --- |
| `tourn_id` | 102615 | 3006495280.146 | 47666727.922 | 2929450288 | 2956452405 | 3019666279 | 3028067701 | 3162349514 |
| `table` | 102615 | 1.000 | 0.000 | 1 | 1 | 1 | 1 | 1 |
| `hand_id` | 102615 | 218471515232.680 | 2112092902.834 | 215051507105 | 216240153151 | 219060141327 | 219440848238.5 | 225348949941 |
| `table_size` | 102615 | 3.000 | 0.000 | 3 | 3 | 3 | 3 | 3 |
| `level` | 102615 | 2.182 | 1.278 | 1 | 1 | 2 | 3 | 9 |
| `playing` | 102615 | 2.585 | 0.493 | 2 | 2 | 3 | 3 | 3 |
| `seat` | 102615 | 1.998 | 0.817 | 1 | 1 | 2 | 3 | 3 |
| `stack` | 102615 | 603.757 | 254.730 | 1 | 450 | 530 | 750 | 1499 |
| `pot_pre` | 102615 | 210.564 | 301.203 | 19 | 60 | 90 | 180 | 1500 |
| `pot_flop` | 102615 | 265.936 | 331.052 | 19 | 60 | 120 | 300 | 1500 |
| `pot_turn` | 102615 | 292.680 | 342.572 | 19 | 70 | 140 | 370 | 1500 |
| `pot_river` | 102615 | 314.961 | 353.285 | 19 | 80 | 155 | 440 | 1500 |
| `ante` | 102615 | 0.000 | 0.000 | 0 | 0 | 0 | 0 | 0 |
| `blinds` | 102615 | 21.683 | 19.095 | 0 | 10 | 20 | 30 | 180 |
| `bet_pre` | 102615 | 87.431 | 176.117 | 0 | 20 | 30 | 60 | 1452 |
| `bet_flop` | 102615 | 22.370 | 85.275 | 0 | 0 | 0 | 0 | 1390 |
| `bet_turn` | 102615 | 10.509 | 50.661 | 0 | 0 | 0 | 0 | 1230 |
| `bet_river` | 102615 | 8.802 | 49.821 | 0 | 0 | 0 | 0 | 1180 |
| `balance` | 102615 | 0.000 | 151.179 | -750 | -30 | 0 | 30 | 1000 |

# Categorical Features Summary

| Feature | Unique Values | Most Common Value |
| ------- | ------------- | ----------------- |
| `buyin` | 1 | `$0.92+$0.08` |
| `date` | 54 | `2020-11-06` |
| `time` | 31,320 | `13:53:34` |
| `name` | 3,410 | `fa538846` |
| `position` | 3 | `SB` |
| `action_pre` | 29 | `folds` |
| `action_flop` | 39 | `x` |
| `action_turn` | 43 | `x` |
| `action_river` | 40 | `x` |
| `all_in` | 2 | `False` |
| `cards` | 2,664 | `--` |
| `board_flop` | 20,752 | `0` |
| `board_turn` | 53 | `0` |
| `board_river` | 53 | `0` |
| `combination` | 303 | `<NA>` |
| `result` | 4 | `gave up` |

# Field Information

| Feature Name | Data Type | Statistical Type | Description |
| ---- | ---- | ---- | ---- |
| `buyin` | string | Nominal | Tournament buy-in text |
| `tourn_id` | int | Identifier | Tournament ID |
| `table` | int | Nominal | Table index within tournament |
| `hand_id` | int | Identifier | Hand identifier |
| `date` | string/date-like | Temporal | Hand date |
| `time` | string/time-like | Temporal | Hand timestamp |
| `table_size` | int | Discrete | Number of seats at table |
| `level` | int | Ordinal | Blind/level progression |
| `playing` | int | Discrete | Players still in game snapshot |
| `seat` | int | Nominal | Seat index |
| `name` | string | Nominal | Player pseudonymous ID |
| `stack` | int | Ratio | Player starting/available chips |
| `position` | string | Nominal | Positional role (e.g., SB, BB, BTN-like) |
| `action_pre` | string | Nominal | Preflop action sequence summary |
| `action_flop` | string | Nominal | Flop action sequence summary |
| `action_turn` | string | Nominal | Turn action sequence summary |
| `action_river` | string | Nominal | River action sequence summary |
| `all_in` | bool/string | Binary | Whether player went all-in |
| `cards` | string | Nominal | Hole cards text (when available) |
| `board_flop` | string | Nominal | Flop board cards text |
| `board_turn` | string | Nominal | Turn board card text |
| `board_river` | string | Nominal | River board card text |
| `combination` | string | Nominal | Hand combination label |
| `pot_pre` | int | Ratio | Pot size preflop |
| `pot_flop` | int | Ratio | Pot size on flop |
| `pot_turn` | int | Ratio | Pot size on turn |
| `pot_river` | int | Ratio | Pot size on river |
| `ante` | int | Ratio | Ante amount |
| `blinds` | int | Ratio | Blind level magnitude |
| `bet_pre` | int | Ratio | Chips committed preflop |
| `bet_flop` | int | Ratio | Chips committed on flop |
| `bet_turn` | int | Ratio | Chips committed on turn |
| `bet_river` | int | Ratio | Chips committed on river |
| `result` | string | Nominal | Outcome class text |
| `balance` | int | Interval/Ratio | Net chip outcome for row context |

# Example Entry

- Representative row pattern in this dataset: 3-max table, preflop fold/check dominant, many rows with no flop/turn/river action (`x` / `0`) due to hand ending early.
- Rare/extreme rows: large positive `balance` near +1000 and negative near -750.

# Exploratory Charts

Charts used in project notebooks include:
- Distribution plots and summary statistics for stack/pot/bets.
- Correlation heatmaps.
- Stage-model feature importance charts.
- Residual scatter for regression add-on.

(See `notebooks/00_mega_final_project.ipynb` for rendered figures.)

# Notable Feature Processing

- Built cleaned dataset (`data/cleanedGambling.csv`) from raw CSV with type coercion, missing handling, and stage normalization.
- Engineered stage-aware features (board texture, stack pressure, SPR, hand strength indices).
- Group-aware train/test split by `hand_id` to reduce leakage.
- Target engineering for:
  - `won_flag` (stage win modeling),
  - `is_bluffing` (visible behavior bluff proxy).

# Notes

- Kaggle page metadata was unavailable via automated fetch at card time due a frontend crash; source URL is still canonical.
- This card describes the dataset actually used in the project pipeline and local file stats at curation time.
