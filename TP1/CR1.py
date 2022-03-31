
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

# Create a normal distribution with mean as 5 and standard deviation as 10
#
mu = 1
std = 0.75
snd = stats.norm(mu, std)
#
# Generate 1000 random values between -100, 100
#
x = np.linspace(-100, 100, 1000)
#
# Plot the standard normal distribution for different values of random variable
# falling in the range -100, 100
#
snd1=stats.norm(-1,0.75)
plt.figure(figsize=(7.5,7.5))
plt.plot(x, snd.pdf(x))
plt.plot(x, snd1.pdf(x))
plt.xlim(-60, 60)
plt.title('Normal Distribution', fontsize='15')
plt.xlabel('Values of Random Variable X', fontsize='15')
plt.ylabel('Probability', fontsize='15')
plt.show()