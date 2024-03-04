import matplotlib.pyplot as plt

path = '/uac/gds/zyzheng23/projects/DeepGate3-Transformer/exp/train_nomask_sepc_pooling_nntrm_alltask_corrmask.log'
name = path.split('/')[-1].split('.')[0]
hams = []
probs = []
with open(path,'r') as f:
    for line in f.readlines():
        if 'overall hamming distance' in line:
            ham = float(line.split(':')[-1])
            hams.append(ham)
        if 'overall probability loss' in line:
            prob = float(line.split(':')[-1])
            probs.append(prob)

plt.plot(range(len(hams[1::2])),hams[1::2])
plt.plot(range(len(probs[1::2])),probs[1::2])
plt.legend(['hamining dist','prob loss'])
plt.savefig(f'/uac/gds/zyzheng23/projects/DeepGate3-Transformer/exp/plot/{name}.png')
