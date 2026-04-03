import numpy as np
import random
import matplotlib.pyplot as plt

#apply a bid b on pool size n, return new pool size
def apply_bid(n, b):
    assert 1 <= b < n

    if random.random() < b/n:
        return b
    else:
        return n - b

#The optimal search in a vacuum: binary search
def binary_search(n):
    return n//2

#optimal strategy as proposed by paper
def optimal_bid(n,m)-> int:  
    #takes n (player 1's pool size) and m (player 2's pool size) and returns b (the optimal bid size)
    assert n >= 2
    assert m >= 2

    # k is (0,1,2,3...)
    #if n >= 2^(k+1) + 1 while 2^k +1 <= m <= 2^(k+1) then optimal b = 2^(log2(m-1))
    #if 2^k + 1 <= n <= 2^(k+1) while m >= 2^k + 1  while m >= 2^k +1 then optimal b = n//2
    k = 0
    while True:
        #are we in the weeds?
        if 2**k + 1 <= m <= 2**(k+1):
            if n >= 2**(k+1)+1:
                return 2**k
                

        #are we ahead?
        if 2**k +1 <= n <= 2**(k+1):
            if m >= 2**k + 1:
                return n//2
                
        k += 1
        if k > np.log2(max(m,n)): raise ValueError

###################################################################################################



#creating the learning environment: a 3D matrix (n,m,b) where values Q are the expected value
MAX_N = 20
Q = np.zeros((MAX_N + 1, MAX_N + 1, MAX_N +1))

#since 0 < b < n, set Q very negative where the moves are invalid
for n in range(MAX_N + 1):
    for b in range(MAX_N + 1):
        if b==0 or b>=n:
            Q[n,:,b] = -1e9

#returns the best bid (evaluated by Q) given n and m
def best_bid(n,m):
    return np.argmax(Q[n,m]) #argmax returns the indice containing the highest Q value along the [n,m] axis (b)

#epsilon will start at 1 and steadily decrease to 0.05
#higher epsilons will make exploration more likely
def choose_bid(n,m, epsilon):
    if random.random() < epsilon:
        return random.randint(1,n-1) #random valid bid
    else:
        return best_bid(n,m)
    
#updates Q (learning)
def update_Q(n, #Player 1's pool size
             m, #Player 2's pool size
             b, #bid size
             reward, #Q
             n_next, m_next, 
             alpha=0.1, #learning rate (higher values change Q faster, but are less stable)
             gamma=1.0 #future discounting (higher values mean future wins matter as much as current ones)
             ):
    best_future = np.max(Q[n_next,m_next]) if n_next > 1 and m_next > 1 else 0
    Q[n,m,b] += alpha * (reward + gamma * best_future - Q[n,m,b])

def one_game(n, epsilon):
    assert n > 1

    turn_count = 0
    #have opponent sometimes start ahead to train model to play while behind
    if random.random() > 0.5:
        m = n
    else:
        m = random.randint(2,n-1)
    
    #play rounds until someone wins
    while True:
        turn_count += 1
        b = choose_bid(n,m,epsilon)
        n_next = apply_bid(n,b)
        m_next = apply_bid(m,binary_search(m))
        
        #check if terminal
        if n_next == 1:
            #update Q, then end the game
            update_Q(n,m,b,1,n_next,m_next)
            return [1,turn_count]

        if m_next == 1:
            #update Q, then end the game
            update_Q(n,m,b,-1,n_next,m_next)
            return [0,turn_count]

        #update q,n,m
        update_Q(n,m,b,0,n_next,m_next)
        n = n_next
        m = m_next
        turn_count += 1

for i in range (100):
    wins = 0
    epsilon = 1
    for i in range(5000):
        wins += one_game(20,epsilon)[0]
        epsilon = epsilon - 0.95/5000
        if i == 4999: print(wins/5000)