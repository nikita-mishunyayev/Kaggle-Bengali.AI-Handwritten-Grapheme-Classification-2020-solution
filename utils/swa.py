import torch
from collections import OrderedDict


config = {}
config['FOLD_NUMBER'] = 0
config['MODEL_NAME'] = 'SEResnext50-128x128-1ch-v5.1.4-{}fold'.format(config['FOLD_NUMBER'])
config['train_stage'] = 'stage1'


def swa(paths):
    state_dicts = []
    for path in paths:
        state_dicts.append(torch.load(path, map_location=torch.device('cpu'))['model'])

    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(state_dicts)

    return average_dict

if __name__ == "__main__":
    path = '../NN_WEIGHTS/{0}/models/{0}_{1}_{2}.pth'
    best_epochs = [30, 34, 39]

    torch.save(swa([
        path.format(config['MODEL_NAME'], config['train_stage'], epoch) for epoch in best_epochs
    ]), '{0}_avg{1}.pth'.format(config['MODEL_NAME'], best_epochs))
