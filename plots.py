from chopsticks import train
import matplotlib.pyplot as plt
from timeit import default_timer as timer

games_played = [10, 20, 50, 75, 100, 125, 150, 200, 400, 700, 900, 1100, 1400, 1600, 1800, 2100, 4000, 7500, 10000]
keys_explored = []
times = []
for i in games_played:
    print(f"Computing for {i} games")
    start = timer()
    avg = sum([len(train(i).q) for j in range(3)])/3
    t = (timer() - start)/3
    times.append(t)
    keys_explored.append(avg)

# games_played.append(10000)
# keys_explored.append(sum([train(10000)] for i in range(2))/2)
print(games_played)
print(keys_explored)
plt.scatter(games_played, keys_explored)
plt.title("Games Trained v/s Keys Explored")

plt.savefig("plot1.png")
plt.close()

print(games_played)
print(times)
plt.scatter(games_played, times)
plt.title("Games Trained v/s Time taken")
plt.savefig("time.png")



