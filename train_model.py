import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from collections import Counter

# Load the cleaned data
df = pd.read_csv('files/one_dollar_spin_and_go.csv')

# Sample for faster training
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# Create basic features
df['won_flag'] = df['result'].isin(['won', 'took chips']).astype(int)

# Functions from notebook
rank_map = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

def parse_cards(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text in ('', '0', '--', 'nan', 'None'):
        return []
    return [card for card in text.split() if card not in ('0', '--')]

def card_rank(card):
    if len(card) < 2:
        return None
    return rank_map.get(card[0].upper())

def card_suit(card):
    if len(card) < 2:
        return None
    return card[1].lower()

def is_straight(ranks):
    ranks = sorted(set(ranks))
    if len(ranks) < 5:
        return False
    if 14 in ranks:
        ranks = [1] + ranks
    for i in range(len(ranks) - 4):
        window = ranks[i:i + 5]
        if window == list(range(window[0], window[0] + 5)):
            return True
    return False

def best_combination_from_cards(cards):
    card_ranks = [card_rank(c) for c in cards if card_rank(c) is not None]
    card_suits = [card_suit(c) for c in cards if card_suit(c) is not None]
    counts = Counter(card_ranks)
    counts_values = sorted(counts.values(), reverse=True)

    if len(card_ranks) == 0:
        return 'unknown'
    if len(card_ranks) < 5:
        if 3 in counts_values:
            return 'three of a kind'
        if counts_values.count(2) >= 2:
            return 'two pair'
        if 2 in counts_values:
            return 'pair'
        return 'high card'

    suits = Counter(card_suits)
    flush_suit = next((s for s, cnt in suits.items() if cnt >= 5), None)
    flush_cards = [card for card in cards if card_suit(card) == flush_suit] if flush_suit else []
    flush_ranks = [card_rank(card) for card in flush_cards]

    straight = is_straight(card_ranks)
    straight_flush = flush_suit is not None and is_straight(flush_ranks)

    if straight_flush:
        return 'straight flush'
    if 4 in counts_values:
        return 'four of a kind'
    if 3 in counts_values and 2 in counts_values:
        return 'full house'
    if flush_suit is not None:
        return 'flush'
    if straight:
        return 'straight'
    if 3 in counts_values:
        return 'three of a kind'
    if counts_values.count(2) >= 2:
        return 'two pair'
    if 2 in counts_values:
        return 'pair'
    return 'high card'

# Encode combinations
combination_order = ['unknown', 'high card', 'pair', 'two pair', 'three of a kind', 'straight', 'flush', 'full house', 'four of a kind', 'straight flush']
combination_map = {comb: i for i, comb in enumerate(combination_order)}

# Function to get hand strength at a stage
def get_hand_strength(row, stage):
    cards = parse_cards(row['cards'])  # hole cards
    if stage >= 1:  # flop
        cards += parse_cards(row['board_flop'])
    if stage >= 2:  # turn
        cards += parse_cards(row['board_turn'])
    if stage >= 3:  # river
        cards += parse_cards(row['board_river'])
    comb = best_combination_from_cards(cards)
    return combination_map.get(comb, 9)

# Function to get features up to stage
def get_features_up_to_stage(row, stage):
    features = {}
    features['hand_strength'] = get_hand_strength(row, stage)
    
    # Bets up to stage
    bet_cols = ['bet_pre', 'bet_flop', 'bet_turn', 'bet_river'][:stage+1]
    features['total_bet'] = row[bet_cols].fillna(0).sum()
    
    pot_cols = ['pot_pre', 'pot_flop', 'pot_turn', 'pot_river'][:stage+1]
    pot_values = [row[col] for col in pot_cols if not pd.isna(row[col]) and row[col] > 0]
    features['current_pot'] = pot_values[-1] if pot_values else 0
    
    features['bet_to_pot_ratio'] = features['total_bet'] / features['current_pot'] if features['current_pot'] > 0 else 0
    
    # Actions up to stage
    action_cols = ['action_pre', 'action_flop', 'action_turn', 'action_river'][:stage+1]
    actions = [row[col] for col in action_cols if not pd.isna(row[col])]
    features['raise_count'] = sum('raise' in str(action).lower() for action in actions)
    features['has_raise'] = int(features['raise_count'] > 0)
    
    # Street bets
    features['street_bets'] = sum(row[bet_cols].gt(0))
    
    return features

# Add features for each stage
stages = ['preflop', 'flop', 'turn', 'river']
stage_configs = [
    {'board': [], 'bet': ['bet_pre'], 'pot': ['pot_pre'], 'action': ['action_pre']},
    {'board': ['board_flop'], 'bet': ['bet_pre', 'bet_flop'], 'pot': ['pot_pre', 'pot_flop'], 'action': ['action_pre', 'action_flop']},
    {'board': ['board_flop', 'board_turn'], 'bet': ['bet_pre', 'bet_flop', 'bet_turn'], 'pot': ['pot_pre', 'pot_flop', 'pot_turn'], 'action': ['action_pre', 'action_flop', 'action_turn']},
    {'board': ['board_flop', 'board_turn', 'board_river'], 'bet': ['bet_pre', 'bet_flop', 'bet_turn', 'bet_river'], 'pot': ['pot_pre', 'pot_flop', 'pot_turn', 'pot_river'], 'action': ['action_pre', 'action_flop', 'action_turn', 'action_river']}
]

for stage_name, config in zip(stages, stage_configs):
    # Hand strength
    def get_strength(row):
        cards = parse_cards(row['cards'])
        for b in config['board']:
            cards += parse_cards(row[b])
        return combination_map.get(best_combination_from_cards(cards), 9)
    
    df[f'hand_strength_{stage_name}'] = df.apply(get_strength, axis=1)
    
    # Total bet
    df[f'total_bet_{stage_name}'] = df[config['bet']].fillna(0).sum(axis=1)
    
    # Current pot
    def get_pot(row):
        for p in reversed(config['pot']):
            if not pd.isna(row[p]) and row[p] > 0:
                return row[p]
        return 0
    df[f'current_pot_{stage_name}'] = df.apply(get_pot, axis=1)
    
    # Bet to pot ratio
    df[f'bet_to_pot_ratio_{stage_name}'] = df[f'total_bet_{stage_name}'] / df[f'current_pot_{stage_name}'].replace(0, 1)
    
    # Raise count
    def get_raises(row):
        count = 0
        for a in config['action']:
            if not pd.isna(row[a]):
                count += 'raise' in str(row[a]).lower()
        return count
    df[f'raise_count_{stage_name}'] = df.apply(get_raises, axis=1)
    
    df[f'has_raise_{stage_name}'] = (df[f'raise_count_{stage_name}'] > 0).astype(int)
    
    # Street bets
    df[f'street_bets_{stage_name}'] = df[config['bet']].gt(0).sum(axis=1)

# Now train models
models = {}
feature_cols = ['hand_strength', 'total_bet', 'current_pot', 'bet_to_pot_ratio', 'raise_count', 'has_raise', 'street_bets']

for stage_name in stages:
    print(f"Training model for {stage_name}")
    cols = [f'{col}_{stage_name}' for col in feature_cols]
    X = df[cols]
    y = df['won_flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    base_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    base_model.fit(X_train, y_train)  # Fit for importance
    model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy for {stage_name}: {accuracy_score(y_test, y_pred)}")
    
    # Feature importance from base model
    base_importances = base_model.feature_importances_
    feature_importance = pd.DataFrame({'feature': cols, 'importance': base_importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(f"Top features for {stage_name}:")
    print(feature_importance.head(3).to_string(index=False))
    print()
    
    models[stage_name] = model

# Save models
joblib.dump(models, 'poker_models.pkl')
joblib.dump(feature_cols, 'feature_names.pkl')

# Prediction function that computes derived features internally
def predict_win_probability(stage, hand_strength, total_bet, current_pot, raise_count, has_raise, street_bets):
    """
    Predict win probability for a poker hand.
    
    Args:
        stage: 'preflop', 'flop', 'turn', or 'river'
        hand_strength: 0-9 scale of hand strength
        total_bet: total amount bet so far
        current_pot: current pot size
        raise_count: number of raises
        has_raise: 1 if raised, 0 otherwise
        street_bets: number of betting rounds played
    
    Returns:
        float: probability of winning (0-1)
    """
    # Compute derived features
    bet_to_pot_ratio = total_bet / current_pot if current_pot > 0 else 0
    
    # Create feature vector in correct order
    features = [
        hand_strength,
        total_bet,
        current_pot,
        bet_to_pot_ratio,
        raise_count,
        has_raise,
        street_bets
    ]
    
    model = models[stage]
    prob = model.predict_proba([features])[0][1]
    return prob

# Example usage
if __name__ == "__main__":
    # Test prediction
    prob = predict_win_probability('preflop', 5, 50, 100, 1, 1, 1)
    print(f"Example prediction: {prob:.2%}")