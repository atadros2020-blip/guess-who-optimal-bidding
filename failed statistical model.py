import numpy as np
import random
import matplotlib.pyplot as plt

#in this approach I'm going to start with the Qube and for each b value in each state find the expected value
#this isnt really RL, but it should lead to the right answer
#in this version we are actually going to iterate through the Qube from b=1 upwards.

#Q is the learning environment: a 3D matrix (n,m,b) where values are the expected value
MAX_N = 10 #initial pool size
Q = np.zeros((MAX_N + 1, MAX_N + 1, MAX_N//2 +1)) #initialize all expected values at 0


for n in range(2,MAX_N):
    for m in range(2, MAX_N):
        for b in range(1, n//2):
            #if b == 1 what is the chance you win the game (if n ==2  then chance is 1)
            if b ==1 or n ==2:
                Q[n,m,b] = 1 if n ==2 else b/n 

            #if b > 1 then the game wont end. multiply the chance it pays off by Q of the resulting state
            #include the 
            else:
                #probability hit/miss
                p_hit = b/n
                p_miss = (n-b)/n

                #resulting n and m value if hit/miss
                n_hit = b
                n_miss = n-b

                m_hit = np.argmax(Q[m,n_hit])
                m_miss = np.argmax(Q[m,n_miss])

                #expected value if hit or miss
                ev_hit = np.max(Q[n_hit,m_hit])
                ev_miss = np.max(Q[n_miss,m_miss])

                expected_value = ev_hit * p_hit + ev_miss * p_miss

                #update Q
                Q[n,m,b] = expected_value

policy = []
for m in range(2,MAX_N+1):
    row = []
    for n in range(2,MAX_N+1):
        row.append(np.argmax(Q[n,m]))
    policy.append(row)

plt.imshow(policy, origin="lower")
plt.xlabel("Your pool size (n)")
plt.ylabel("Opponent pool size (m)")
plt.colorbar(label="Best bid")
plt.title("Learned Policy")
plt.show()