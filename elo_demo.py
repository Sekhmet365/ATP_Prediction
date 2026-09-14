def win_probability(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, a_won, k=32):
    probability_a = win_probability(rating_a, rating_b)

    actual_a = 1 if a_won else 0
    change = k * (actual_a - probability_a)

    return rating_a + change, rating_b - change


# Example: the stronger player faces an underdog
rating_a = 1800
rating_b = 1500

probability = win_probability(rating_a, rating_b)
print(f"A's win probability: {probability:.1%}")

# Scenario 1: the favourite wins
new_a, new_b = update_elo(rating_a, rating_b, a_won=True)
print(f"If A wins: A = {new_a:.1f}, B = {new_b:.1f}")

# Scenario 2: the underdog wins
new_a, new_b = update_elo(rating_a, rating_b, a_won=False)
print(f"If B wins: A = {new_a:.1f}, B = {new_b:.1f}")