# Data Science Final Project: Online Poker Behavioral Analysis

**Team:** Joshua Myers, German Garrido-Lestache Belinchon

## Data Preparation & Parsing (`scripts/data/` & `DataCleaning.ipynb`)
- [ ] **Load and Inspect Data:** Import the dataset of online poker games into a pandas DataFrame inside a notebook.
- [ ] **Clean Core Columns (`cleaner.py`):** Standardize structural columns like buyin, tournament id, hand id, date/time, and table size.
- [ ] **Clean Betting Data (`cleaner.py`):** Parse and clean the street-by-street action columns (pot size, blinds, bets on pre-flop, flop, turn, river).
- [ ] **Initialize PokerKit (`pokerkit_parser.py`):** Write a Python function that reads a single row/hand and translates the dataset's text format into a `PokerKit` game state object.
- [ ] **Validate States:** Run a subset of hands through PokerKit's state engine to ensure the programmatic actions (raises, calls, folds) match the final pot and balances recorded in the dataset.

## Phase 2: Feature Engineering (`scripts/features/` & `EdaSndStreaks.ipynb`)
- [ ] **Calculate Objective Hand Strength (`hand_eval.py`):** Use PokerKit to evaluate the mathematical strength of each player's hand at every street.
- [ ] **Develop a "Bluffing" Target Variable (`hand_eval.py`):** Compare a player's betting behavior to their PokerKit-evaluated hand strength to create a label indicating if a player is bluffing.
- [ ] **Track Stack Sizes (`streaks.py`):** Create a dynamic feature representing the amount of money a player has at the start of each hand (to analyze how it dictates behavior).
- [ ] **Identify Streaks (`streaks.py`):** Track individual recurring players across chronological hands to flag active win streaks and loss streaks.
- [ ] **Isolate All-Ins:** Create a boolean column that flags if a hand resulted in an 'All-In'.

## Phase 3: Simulation & Probabilities (`scripts/simulation/` & `Simulations.ipynb`)
- [ ] **Pre-Flop Equity (`equity.py`):** Pass every starting hand through a PokerKit evaluator to calculate raw pre-flop win probability (to predict win/loss from the starting hand).
- [ ] **Simulate Folded Hands (`monte_carlo.py`):** Filter the dataset for hands where a player folded.
- [ ] **Calculate Folded Win Rates (`monte_carlo.py`):** For each folded hand, use PokerKit to simulate the remaining runout 1,000+ times to calculate how often the folded player would have won on average.
- [ ] **Compute Expected Value (EV):** Combine the folded win probabilities with the pot sizes to determine the expected value the player sacrificed by folding.

## Phase 4: Statistical Analysis & Modeling (`scripts/models/` & `BluffModeling.ipynb`)
- [ ] **Train Bluff Predictor (`bluff_predictor.py`):** Train a classification model (e.g., Random Forest, Logistic Regression) using engineered behavior/bet features to predict whether a player is bluffing.
- [ ] **Train Win Predictor (`win_predictor.py`):** Train a classification/probability model to predict a player's win percentage based on their starting hand, table position, and stack size.
- [ ] **Train Money Predictor (`money_predictor.py`):** Train a regression model to predict the expected amount of money won (or lost) in a hand based on betting patterns, board texture, and player history.
- [ ] **Analyze Streak Impact (`stats_tests.py`):** Run statistical tests (t-tests/ANOVA) to see if bet sizing or fold frequency significantly changes based on win/loss streaks.
- [ ] **Analyze Bankroll Impact (`stats_tests.py`):** Calculate the correlation between stack size and aggression metrics to see how money dictates behavior.
- [ ] **Evaluate All-In Profiles (`stats_tests.py`):** Group the 'All-In' data to compare the specific hand strengths and board textures of 'All-In' losses vs. wins.

## Phase 5: Dashboard Integration (`main.py`)
- [ ] **Initialize Streamlit App:** Create `main.py` with the base multi-tab layout.
- [ ] **Build Tab 1 (Poker Simulator):** Use streamlit session stats and python random to make a user able to play a robot in texas hold em poker. Hook the opponents actions to the bluff predictor and user actions to the `win_predictor` and `money_predictor`.
- [ ] **Build Tab 2 (EDA & Streaks):** Import cleaned data and `plotly` charts showing stack size/streak impacts.
- [ ] **Build Tab 3 (Bluff Predictor):** Import `predict_bluff` function and hook it up to interactive sliders for bet size, position, and streaks.
- [ ] **Build Tab 4 (Fold Simulator):** Import `simulate_folded_hand` function and connect it to user inputs for folded cards, board texture, and runout count.
- [ ] **Final Polish:** Ensure all UI elements work cohesively and the app runs smoothly via `streamlit run main.py`.