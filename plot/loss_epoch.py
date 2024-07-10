import matplotlib.pyplot as plt
import numpy as np

# data_frac = [1,5,10,30,50,100]
data_frac = [5]
mark = ['o','v','^','s','p','*']
# mark = ['v','^','s','p','*']

overall_loss = {}
func_loss = {}
struc_loss = {}

for frac in data_frac:
    overall_loss[frac] = []
    func_loss[frac] = []
    struc_loss[frac] = []


# frac = 1
# path = f'/home/zyzheng23/project/DeepGate3-Transformer/plot/exp_log/plain_workload_{frac}p_balanced_unfixed.log'
# with open(path,'r') as f:
#     lines = f.readlines()
#     for i,line in enumerate(lines):
#         if '(val)' in line and 'Iter: 5' in line:
#             l1 = float(lines[i+24].split(':')[-1])
#             overall_loss[frac].append(l1)

#             l2 = float(lines[i+25].split(':')[-1])
#             func_loss[frac].append(l2)

#             l3 = float(lines[i+26].split(':')[-1])
#             struc_loss[frac].append(l3)


for frac in data_frac:
    path = f'/home/zyzheng23/project/DeepGate3-Transformer/exp/0504_plain_workload_100p_wo_PT.log'
    with open(path,'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if '(val)' in line:
                l1 = float(lines[i+24].split(':')[-1])
                overall_loss[frac].append(l1)

                l2 = float(lines[i+25].split(':')[-1])
                func_loss[frac].append(l2)

                l3 = float(lines[i+26].split(':')[-1])
                struc_loss[frac].append(l3)


for i,frac in enumerate(data_frac):
    plt.plot(range(len(overall_loss[frac])),overall_loss[frac], label=f'{frac}%')
plt.legend()
plt.savefig(f'/home/zyzheng23/project/DeepGate3-Transformer/plot/loss_epoch/overall_loss.png',dpi=300)
plt.close()

for i,frac in enumerate(data_frac):
    plt.plot(range(len(func_loss[frac])),func_loss[frac], label=f'{frac}%')
plt.legend()
plt.savefig(f'/home/zyzheng23/project/DeepGate3-Transformer/plot/loss_epoch/func_loss.png',dpi=300)
plt.close()

for i,frac in enumerate(data_frac):
    plt.plot(range(len(struc_loss[frac])),struc_loss[frac], label=f'{frac}%')
# plt.legend()
plt.savefig(f'/home/zyzheng23/project/DeepGate3-Transformer/plot/loss_epoch/struc_loss.png',dpi=300)
plt.close()


for i,frac in enumerate(data_frac):
    overall_loss[frac].append(overall_loss[frac][-1])
    plt.plot(np.array(list(range(len(overall_loss[frac][::20]))))*20,overall_loss[frac][::20], label=f'{frac}%',marker = mark[i])
plt.legend()

plt.savefig(f'/home/zyzheng23/project/DeepGate3-Transformer/plot/loss_epoch/overall_loss_with_marker.png',dpi=300)
plt.close()

for i,frac in enumerate(data_frac):
    func_loss[frac].append(func_loss[frac][-1])
    plt.plot(np.array(list(range(len(func_loss[frac][::20]))))*20,func_loss[frac][::20], label=f'{frac}%',marker = mark[i])
plt.legend()
plt.savefig(f'/home/zyzheng23/project/DeepGate3-Transformer/plot/loss_epoch/func_loss_with_marker.png',dpi=300)
plt.close()

for i,frac in enumerate(data_frac):
    struc_loss[frac].append(struc_loss[frac][-1])
    plt.plot(np.array(list(range(len(struc_loss[frac][::20]))))*20,struc_loss[frac][::20], label=f'{frac}%',marker = mark[i])
# plt.legend()
plt.savefig(f'/home/zyzheng23/project/DeepGate3-Transformer/plot/loss_epoch/struc_loss_with_marker.png',dpi=300)
plt.close()