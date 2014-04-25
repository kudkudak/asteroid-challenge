import matplotlib.pylab as plt

N=10.
x_plt, y_plt, clr_plt = [0]*int(N), [0]*int(N), [0]*int(N)
for i in xrange(int(N)):
    act = [i/N,  i/N]
    x_plt[i] = act[0]
    y_plt[i] = act[1]
    clr_plt[i] = i % 2 + 10

print x_plt
print y_plt

plt.scatter(x_plt, y_plt,s=90, c=clr_plt)
plt.show()