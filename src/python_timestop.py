import time
import math

y1 = time.time()
for i in range(500):
    x = math.sqrt(i)
y2 = time.time()
dt = y2 - y1
print(dt)
