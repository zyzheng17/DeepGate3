import matplotlib.pyplot as plt

path = '/uac/gds/zyzheng23/projects/DeepGate3-Transformer/exp/structure_train.log'
name = path.split('/')[-1].split('.')[0]
hams = []
probs = []
accs = []
with open(path,'r') as f:
    for line in f.readlines():
        if 'overall hamming distance' in line:
            ham = float(line.split(':')[-1])
            hams.append(ham)

        if 'overall connect acc' in line:
            acc = float(line.split(':')[-1])
            accs.append(acc)

        # if 'overall probability loss' or 'overall prob loss' in line:
        #     prob = float(line.split(':')[-1])
        #     probs.append(prob)

# plt.plot(range(len(hams[1::2])),hams[1::2])
plt.plot(range(len(accs[1::2])),accs[1::2])
# plt.plot(range(len(probs[1::2])),probs[1::2])
# plt.legend(['hamining dist'])
plt.legend(['connect acc'])
plt.savefig(f'/uac/gds/zyzheng23/projects/DeepGate3-Transformer/exp/plot/{name}.png')
