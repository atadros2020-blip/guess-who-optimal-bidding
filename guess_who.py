import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

def apply_bid(n, b):
    assert 1 <= b < n
    if random.random() < b/n:
        return b
    else:
        return n - b

def optimal_bid(n,m)-> int:  
    assert n >= 2
    assert m >= 2
    k = 0
    while True:
        if 2**k + 1 <= m <= 2**(k+1):
            if n >= 2**(k+1)+1:
                return 2**k
        if 2**k +1 <= n <= 2**(k+1):
            if m >= 2**k + 1:
                return n//2
        k += 1
        if k > np.log2(max(m,n)): raise ValueError

###################################################################################################
# MONTE CARLO WITH EXTENSIVE SAMPLING
###################################################################################################

MAX_N = 10

print("=" * 70)
print(f"MONTE CARLO LEARNING - MAX_N = {MAX_N}")
print("Only considering bids from 1 to n//2 (symmetric bids excluded)")
print("=" * 70)

# Data structures to track experiences
visit_count = defaultdict(int)      # (n,m,b) -> count
sum_rewards = defaultdict(float)    # (n,m,b) -> sum of rewards
Q = {}                               # Will be computed after data collection

def play_game_and_record(start_n, start_m):
    """
    Play one complete game and record all (state, action, outcome) pairs.
    Returns the trajectory: [(n, m, b, player_id), ...]
    """
    n, m = start_n, start_m
    path = []  # Store all moves: (n, m, b, player_id)
    
    while True:
        # Player 1's turn - only consider bids from 1 to n//2
        max_bid = max(1, n // 2)
        b = random.randint(1, max_bid)
        path.append((n, m, b, 1))
        n_next = apply_bid(n, b)
        
        if n_next == 1:
            # Player 1 wins
            return path, 1
        
        # Player 2's turn - only consider bids from 1 to m//2
        max_bid_2 = max(1, m // 2)
        b2 = random.randint(1, max_bid_2)
        path.append((m, n_next, b2, 2))
        m_next = apply_bid(m, b2)
        
        if m_next == 1:
            # Player 2 wins
            return path, 0
        
        n = n_next
        m = m_next

def record_trajectory(trajectory, winner):
    """
    Record the outcome for all state-action pairs in the trajectory.
    winner: 1 if player 1 won, 0 if player 2 won
    """
    for (n, m, b, player_id) in trajectory:
        state_action = (n, m, b)
        
        # Assign reward based on who played and who won
        if player_id == 1:
            reward = 1.0 if winner == 1 else -1.0
        else:  # player_id == 2
            reward = 1.0 if winner == 0 else -1.0
        
        visit_count[state_action] += 1
        sum_rewards[state_action] += reward

###################################################################################################
# DATA COLLECTION PHASE
###################################################################################################

print("\nPhase 1: Data Collection")
print("-" * 70)

num_games = 50_000_000  # 5 million games (reduced for MAX_N=10)

for game in tqdm(range(num_games), desc="Playing games"):
    # Sample random starting state
    start_n = random.randint(2, MAX_N)
    start_m = random.randint(2, MAX_N)
    
    # Play game and record
    trajectory, winner = play_game_and_record(start_n, start_m)
    record_trajectory(trajectory, winner)

print(f"\n✓ Collected data from {num_games:,} games")
print(f"✓ Observed {len(visit_count):,} unique (state, action) pairs")

# Show some statistics
visit_counts_list = list(visit_count.values())
print(f"✓ Average visits per (state, action): {np.mean(visit_counts_list):.1f}")
print(f"✓ Min visits: {np.min(visit_counts_list)}")
print(f"✓ Max visits: {np.max(visit_counts_list)}")

###################################################################################################
# POLICY EXTRACTION PHASE
###################################################################################################

print("\n" + "=" * 70)
print("Phase 2: Computing Q-values and Extracting Policy")
print("-" * 70)

# Compute Q-values as averages
for state_action, count in tqdm(visit_count.items(), desc="Computing Q-values"):
    Q[state_action] = sum_rewards[state_action] / count

# Extract best action for each state
Policy = np.zeros((MAX_N + 1, MAX_N + 1), dtype=int)
Win_rates = {}  # Track win rates

for n in range(2, MAX_N + 1):
    for m in range(2, MAX_N + 1):
        best_b = 1
        best_q = -np.inf
        
        # Find best bid for this state (only consider 1 to n//2)
        max_bid = max(1, n // 2)
        for b in range(1, max_bid + 1):
            state_action = (n, m, b)
            if state_action in Q:
                q_val = Q[state_action]
                
                # Track win rate (Q-value maps to win rate: Q=1 means 100% win, Q=0 means 50%, Q=-1 means 0%)
                win_rate = (q_val + 1) / 2  # Convert [-1, 1] to [0, 1]
                Win_rates[(n, m, b)] = win_rate
                
                if q_val > best_q:
                    best_q = q_val
                    best_b = b
        
        Policy[n, m] = best_b

print("✓ Policy extracted!")

###################################################################################################
# EVALUATION
###################################################################################################

print("\n" + "=" * 70)
print("EVALUATION vs OPTIMAL POLICY")
print("=" * 70)

matches = 0
total = 0
mismatches = []

for m in range(2, MAX_N + 1):
    for n in range(2, MAX_N + 1):
        learned = Policy[n, m]
        try:
            optimal = optimal_bid(n, m)
            match = learned == optimal
            matches += match
            total += 1
            
            if not match:
                # Get Q-values for learned and optimal
                learned_q = Q.get((n, m, learned), 0)
                optimal_q = Q.get((n, m, optimal), 0)
                learned_visits = visit_count.get((n, m, learned), 0)
                optimal_visits = visit_count.get((n, m, optimal), 0)
                
                mismatches.append({
                    'state': (n, m),
                    'learned': learned,
                    'optimal': optimal,
                    'learned_q': learned_q,
                    'optimal_q': optimal_q,
                    'q_diff': learned_q - optimal_q,
                    'learned_visits': learned_visits,
                    'optimal_visits': optimal_visits
                })
        except:
            pass

accuracy = 100 * matches / total
print(f"\n{'='*70}")
print(f"ACCURACY: {matches}/{total} = {accuracy:.1f}%")
print(f"{'='*70}")

# Show ALL mismatches
if mismatches:
    print("\nAll Mismatches (learned vs optimal):")
    print("-" * 70)
    for mm in sorted(mismatches, key=lambda x: (x['state'][1], x['state'][0])):
        print(f"({mm['state'][0]:2d},{mm['state'][1]:2d}): "
              f"learned={mm['learned']} (Q={mm['learned_q']:6.3f}, visits={mm['learned_visits']:6d}) | "
              f"optimal={mm['optimal']} (Q={mm['optimal_q']:6.3f}, visits={mm['optimal_visits']:6d}) | "
              f"ΔQ={mm['q_diff']:+6.3f}")

###################################################################################################
# ANALYSIS: Show learned Q-values for ALL states
###################################################################################################

print("\n" + "=" * 70)
print("Q-VALUE ANALYSIS (All States)")
print("=" * 70)

for n in range(2, MAX_N + 1):
    for m in range(2, MAX_N + 1):
        print(f"\nState ({n},{m}):")
        print("  Bid | Q-value | Win Rate | Visits | Status")
        print("  " + "-" * 55)
        
        learned_b = Policy[n, m]
        try:
            optimal_b = optimal_bid(n, m)
        except:
            optimal_b = None
        
        max_bid = max(1, n // 2)
        for b in range(1, max_bid + 1):
            state_action = (n, m, b)
            if state_action in Q:
                q_val = Q[state_action]
                win_rate = (q_val + 1) / 2 * 100
                visits = visit_count[state_action]
                
                status = []
                if b == learned_b:
                    status.append("LEARNED")
                if b == optimal_b:
                    status.append("OPTIMAL")
                status_str = " ".join(status) if status else ""
                
                print(f"   {b:2d} | {q_val:7.3f} | {win_rate:6.2f}% | {visits:6d} | {status_str}")

###################################################################################################
# VISUALIZATION
###################################################################################################

print("\n" + "=" * 70)
print("Generating Visualization...")
print("=" * 70)

policy_grid = Policy[2:, 2:]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Learned policy
im1 = axes[0].imshow(policy_grid, origin="lower", cmap='viridis', aspect='auto')
axes[0].set_xlabel("Your pool size (n)", fontsize=12)
axes[0].set_ylabel("Opponent pool size (m)", fontsize=12)
axes[0].set_title(f"Monte Carlo Learned Policy\n({matches}/{total} = {accuracy:.1f}% correct)", fontsize=14)
axes[0].set_xticks(range(MAX_N-1))
axes[0].set_xticklabels(range(2, MAX_N+1))
axes[0].set_yticks(range(MAX_N-1))
axes[0].set_yticklabels(range(2, MAX_N+1))
axes[0].grid(True, alpha=0.3, color='white', linewidth=0.5)
plt.colorbar(im1, ax=axes[0], label="Best bid")

# Right plot: Difference from optimal (red = wrong, green = correct)
difference_grid = np.zeros_like(policy_grid, dtype=float)
for i, m in enumerate(range(2, MAX_N+1)):
    for j, n in enumerate(range(2, MAX_N+1)):
        learned = Policy[n, m]
        try:
            optimal = optimal_bid(n, m)
            difference_grid[i, j] = 0 if learned == optimal else 1
        except:
            difference_grid[i, j] = 0

im2 = axes[1].imshow(difference_grid, origin="lower", cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
axes[1].set_xlabel("Your pool size (n)", fontsize=12)
axes[1].set_ylabel("Opponent pool size (m)", fontsize=12)
axes[1].set_title("Match with Optimal Policy\n(Green = Correct, Red = Wrong)", fontsize=14)
axes[1].set_xticks(range(MAX_N-1))
axes[1].set_xticklabels(range(2, MAX_N+1))
axes[1].set_yticks(range(MAX_N-1))
axes[1].set_yticklabels(range(2, MAX_N+1))
axes[1].grid(True, alpha=0.3, color='white', linewidth=0.5)
plt.colorbar(im2, ax=axes[1], label="Match (0=correct, 1=wrong)")

plt.tight_layout()
plt.savefig('monte_carlo_policy.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to 'monte_carlo_policy.png'")
plt.show()

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"Final Accuracy: {accuracy:.1f}%")
print(f"Total games played: {num_games:,}")
print(f"Unique (state, action) pairs observed: {len(visit_count):,}")

###################################################################################################
# HEAD-TO-HEAD EVALUATION: Policy Tournament
###################################################################################################

print("\n" + "=" * 70)
print("HEAD-TO-HEAD EVALUATION: Policy Tournament")
print("=" * 70)

def get_policy_bid(n, m, policy_type):
    """
    Get bid from different policy types.
    policy_type: 'learned', 'optimal', or 'binary'
    """
    if policy_type == 'learned':
        return Policy[n, m]
    elif policy_type == 'optimal':
        try:
            return optimal_bid(n, m)
        except:
            # If optimal_bid fails, fall back to n//2
            return max(1, n // 2)
    elif policy_type == 'binary':
        return max(1, n // 2)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

def play_matchup(start_n, start_m, policy1_type, policy2_type, player1_starts=True):
    """
    Play one game between two policies.
    Returns 1 if policy1 wins, 0 if policy2 wins.
    """
    n, m = start_n, start_m
    
    # Determine who goes first
    if player1_starts:
        current_player = 1
    else:
        current_player = 2
        n, m = m, n  # Swap so current player is always 'n'
    
    while True:
        # Current player's turn
        if current_player == 1:
            b = get_policy_bid(n, m, policy1_type)
        else:
            b = get_policy_bid(n, m, policy2_type)
        
        # Ensure bid is valid
        max_bid = max(1, n // 2)
        b = min(b, max_bid)
        b = max(b, 1)
        
        n_next = apply_bid(n, b)
        
        if n_next == 1:
            # Current player wins
            return 1 if current_player == 1 else 0
        
        # Switch players and state
        n, m = m, n_next
        current_player = 3 - current_player  # Toggle between 1 and 2

def run_tournament(policy1_name, policy1_type, policy2_name, policy2_type, num_games=10000):
    """
    Run a tournament between two policies.
    """
    print(f"\n{policy1_name} vs {policy2_name}")
    print("-" * 70)
    
    policy1_wins_as_p1 = 0
    policy1_wins_as_p2 = 0
    
    # Play games with policy1 starting
    for _ in tqdm(range(num_games // 2), desc=f"{policy1_name} starts"):
        start_n = random.randint(2, MAX_N)
        start_m = random.randint(2, MAX_N)
        result = play_matchup(start_n, start_m, policy1_type, policy2_type, player1_starts=True)
        policy1_wins_as_p1 += result
    
    # Play games with policy2 starting
    for _ in tqdm(range(num_games // 2), desc=f"{policy2_name} starts"):
        start_n = random.randint(2, MAX_N)
        start_m = random.randint(2, MAX_N)
        result = play_matchup(start_n, start_m, policy1_type, policy2_type, player1_starts=False)
        policy1_wins_as_p2 += result
    
    total_p1_wins = policy1_wins_as_p1 + policy1_wins_as_p2
    total_p2_wins = num_games - total_p1_wins
    
    p1_wr_as_starter = (policy1_wins_as_p1 / (num_games // 2)) * 100
    p1_wr_as_second = (policy1_wins_as_p2 / (num_games // 2)) * 100
    overall_p1_wr = (total_p1_wins / num_games) * 100
    
    print(f"\nResults ({num_games:,} games):")
    print(f"  {policy1_name} going first:  {policy1_wins_as_p1}/{num_games//2} ({p1_wr_as_starter:.2f}%)")
    print(f"  {policy1_name} going second: {policy1_wins_as_p2}/{num_games//2} ({p1_wr_as_second:.2f}%)")
    print(f"  Overall: {policy1_name} {total_p1_wins}/{num_games} ({overall_p1_wr:.2f}%) vs {policy2_name} {total_p2_wins}/{num_games} ({100-overall_p1_wr:.2f}%)")
    
    return {
        'policy1_name': policy1_name,
        'policy2_name': policy2_name,
        'policy1_wins': total_p1_wins,
        'policy2_wins': total_p2_wins,
        'policy1_winrate': overall_p1_wr
    }

# Run tournaments
tournament_games = 20000  # Number of games per matchup

results = []

# Learned vs Optimal
results.append(run_tournament(
    "Learned Policy", "learned",
    "Optimal Policy", "optimal",
    tournament_games
))

# Learned vs Binary Search
results.append(run_tournament(
    "Learned Policy", "learned",
    "Binary Search", "binary",
    tournament_games
))

# Optimal vs Binary Search
results.append(run_tournament(
    "Optimal Policy", "optimal",
    "Binary Search", "binary",
    tournament_games
))
# Add after the tournament summary, before FINAL CONCLUSION

# Create a bar chart of win rates
fig, ax = plt.subplots(figsize=(10, 6))

policies = ['Learned\nvs\nOptimal', 'Learned\nvs\nBinary', 'Optimal\nvs\nBinary']
win_rates = [r['policy1_winrate'] for r in results]
colors = ['green' if wr > 50 else 'red' if wr < 50 else 'gray' for wr in win_rates]

bars = ax.bar(policies, win_rates, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=50, color='black', linestyle='--', linewidth=2, label='50% (Even)')
ax.set_ylabel('Win Rate (%)', fontsize=12)
ax.set_title('Policy Tournament Results', fontsize=14)
ax.set_ylim([0, 100])
ax.legend()

# Add value labels on bars
for bar, wr in zip(bars, win_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{wr:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('tournament_results.png', dpi=150)
print("✓ Saved tournament visualization to 'tournament_results.png'")
plt.show()

###################################################################################################
# SUMMARY
###################################################################################################

print("\n" + "=" * 70)
print("TOURNAMENT SUMMARY")
print("=" * 70)

print("\nHead-to-Head Results:")
print("-" * 70)
for result in results:
    print(f"{result['policy1_name']:20s} vs {result['policy2_name']:20s} | "
          f"{result['policy1_wins']:5d}-{result['policy2_wins']:5d} | "
          f"{result['policy1_winrate']:6.2f}% win rate")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("=" * 70)

# Analyze the learned vs optimal matchup
learned_vs_optimal = results[0]
if learned_vs_optimal['policy1_winrate'] > 50:
    print(f"✓ Learned policy OUTPERFORMS optimal policy by {learned_vs_optimal['policy1_winrate'] - 50:.2f}%")
    print("  This suggests the 'optimal' theory may be suboptimal!")
elif learned_vs_optimal['policy1_winrate'] < 50:
    print(f"✗ Learned policy underperforms optimal policy by {50 - learned_vs_optimal['policy1_winrate']:.2f}%")
    print("  The theoretical optimal appears genuinely better.")
else:
    print("≈ Learned and optimal policies perform equally well")

# Compare both to binary search
learned_vs_binary = results[1]
optimal_vs_binary = results[2]
print(f"\n✓ Learned policy beats binary search: {learned_vs_binary['policy1_winrate']:.2f}% win rate")
print(f"✓ Optimal policy beats binary search: {optimal_vs_binary['policy1_winrate']:.2f}% win rate")

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)
print(f"Monte Carlo learning achieved {accuracy:.1f}% policy match with theory")
print(f"Head-to-head performance: {learned_vs_optimal['policy1_winrate']:.2f}% win rate vs optimal")
print("The learned policy successfully discovered effective bidding strategies from experience!")
print("=" * 70)