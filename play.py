
from chopsticks import train, play


trained = train(250)
print("explored", len(trained.q), "keys")
print("*"*30, "\n")
play(trained)
