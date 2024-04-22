import matplotlib.pyplot as plt

data_frac = ['1%','5%','10%','30%','50%','100%']
# mark = ['o','v','^','s','p','*']
overall_loss = [2.228 , 1.241, 0.671, 0.381, 0.343, 0.270]
func_loss = [0.725, 0.443, 0.358, 0.248, 0.222, 0.175]
struc_loss = [1.503, 0.798, 0.313, 0.133, 0.121, 0.095]



plt.plot(data_frac, func_loss, marker = 'o',label = 'function loss',markersize=10)

plt.plot(data_frac, struc_loss, marker='v',label = 'structure loss',markersize=10)

plt.plot((data_frac), overall_loss, marker='*',label = 'overall loss',markersize=10)

plt.legend()
plt.xticks(data_frac)
plt.savefig(f'./DeepGate3-Transformer/plot/overall/loss_fraction.png',dpi=300)