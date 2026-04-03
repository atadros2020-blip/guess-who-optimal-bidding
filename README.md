# Auditing "Guess Who?" Optimal Strategy via Monte Carlo Simulation

Will reinforcement learning independently learn the optimal bidding strategy for Guess Who proven by Nica (2015)? Not really, but Monte Carlo simulation can.

## Background

Nica (2015) models the board game "Guess Who?" as a simple stochastic game and proves that binary search (always bidding b = n/2) is not always optimal. When a player is trailing, they should make bold, risky bids to catch up. When ahead, they should play conservatively with binary search.

The optimal strategy (Theorem 1.1) depends on both players' pool sizes (n, m):

- **Ahead** (your pool size is in a lower power-of-2 bracket than your opponent's): Play it safe. Bid b = floor(n/2).
- **Behind** (your pool size is in a higher bracket): Play bold. Bid a smaller, riskier amount to try to leapfrog your opponent.

**Paper:** Nica, M. (2015). *Optimal Strategy in "Guess Who?": Beyond Binary Search.* Probability in the Engineering and Informational Sciences, 30(4). [arXiv:1509.03327](https://arxiv.org/abs/1509.03327)

## What I learned

I started this project trying to use Q-learning to discover the optimal policy. It didn't work. The stochastic, zero-sum structure of Guess Who creates enormous noise. The same state-action pair can lead to a win or a loss depending on a coin flip and what the opponent does. Q-learning's bootstrapped value estimates couldn't stabilize through that.

Monte Carlo simulation worked because it sidesteps bootstrapping entirely. Instead of learning value estimates incrementally, it plays millions of complete games with random bidding, records every (state, action, outcome) triple, and averages the results. Brute force, but it converges.

The Monte Carlo learned policy matches the paper's optimal strategy across the tested state space.

## How the simulation works

1. Both players start with pool sizes (n, m) drawn uniformly from [2, N_MAX].
2. Each turn, a player chooses a bid b in [1, n/2] uniformly at random.
3. A bid of size b on pool size n succeeds (b remaining) with probability b/n, and fails (n-b remaining) with probability (n-b)/n.
4. First player to reach pool size 1 wins.
5. Every state-action pair visited in the game receives +1 (win) or -1 (loss).
6. After 50 million games, Q-values are computed as average reward per state-action pair, and the greedy policy is extracted.

## Files

- `guess_who.py` - Monte Carlo simulation with full Q-value analysis and policy visualization
- `failed Q value model.py` - Q-learning attempt (didn't converge due to noise)
- `failed statistical model.py` - Direct expected value computation attempt (got recursive dependencies wrong)
- `one_player_game.py` - Single-player version testing fixed bid ratios to build intuition
- `two_player_game.py` - Q-learning against a binary search opponent

## Usage

```bash
pip install numpy matplotlib tqdm
python guess_who.py
```

Note: The default runs 50M games, which takes a while. Reduce `num_games` for a faster but less precise run.
