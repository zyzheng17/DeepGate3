import matplotlib.pyplot as plt

# data_frac = ['1%','5%','10%','30%','50%','100%']
# mark = ['o','v','^','s','p','*']
data_frac = ['5%','10%','30%','50%','100%']
mark = ['v','^','s','p','*']
overall_loss_dg3 = [1.9057 ,1.6452 ,1.1972 ,1.0694 ,0.8553 , ]
overall_loss_dg2 = [3.0906 ,2.6250 ,2.4031 ,2.1491 ,2.2215 ,]

plt.plot(data_frac, overall_loss_dg3, marker = 's',markersize=10,label = 'DeepGate3')
plt.plot(data_frac, overall_loss_dg2, marker = 'v',markersize=10,label = 'DeepGate2')
plt.legend(fontsize=12)
plt.xticks(data_frac)
plt.savefig(f'./DeepGate3-Transformer/plot/overall/overall_loss.png',dpi=300,bbox_inches='tight', pad_inches=0.2)
plt.close()
con_pre_dg3 = [89.31,89.99,91.94,92.99,93.32,]
con_pre_dg2 = [85.62,84.91,85.42,84.95,85.58,]

plt.plot(data_frac, con_pre_dg3, marker = 's',markersize=10,label = 'DeepGate3')
plt.plot(data_frac, con_pre_dg2, marker = 'v',markersize=10,label = 'DeepGate2')
plt.legend(fontsize=12)
plt.xticks(data_frac)
plt.savefig(f'./DeepGate3-Transformer/plot/overall/con_pre.png',dpi=300,bbox_inches='tight', pad_inches=0.2)
plt.close()
ham_dg3 = [0.0931 ,0.0710 ,0.0459 ,0.0348 ,0.0237 ,]
ham_dg2 = [0.1057,0.1072 ,0.0970 ,0.0914 ,0.0898 ,]

plt.plot(data_frac, ham_dg3, marker = 's',markersize=10,label = 'DeepGate3')
plt.plot(data_frac, ham_dg2, marker = 'v',markersize=10,label = 'DeepGate2')
plt.legend(fontsize=12)
plt.xticks(data_frac)
plt.savefig(f'./DeepGate3-Transformer/plot/overall/ham_dist.png',dpi=300,bbox_inches='tight', pad_inches=0.2)
plt.close()
on_hop_dg3 = [83.79,85.04,88.45,94.21,95.18,]
on_hop_dg2 = [74.13,74.47,75.81,78.52,78.57,]

plt.plot(data_frac, on_hop_dg3, marker = 's',markersize=10,label = 'DeepGate3')
plt.plot(data_frac, on_hop_dg2, marker = 'v',markersize=10,label = 'DeepGate2')
plt.legend(fontsize=12)
plt.xticks(data_frac)
plt.savefig(f'./DeepGate3-Transformer/plot/overall/on_hop.png',dpi=300,bbox_inches='tight', pad_inches=0.2)
plt.close()