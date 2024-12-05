import os
import sys
import time
import yaml
import pprint
import numpy as np
import pandas as pd

from torch_geometric.data import DataLoader

from dataset_htp import HTPMolDataset, HTPMolDatasetWrapper
from model_htp import HTPFineTune


def main(config):
    #config['dataset']['task'] = 'classification'
    dataset = HTPMolDatasetWrapper(config['batch_size'], **config['dataset'])
    dataset.get_dataset()
    test_loader = DataLoader(
            dataset=dataset.dataset,
            batch_size=dataset.batch_size,
            shuffle=False
    )

    fine_tune = HTPFineTune(dataset, config)
    if fine_tune.config['model_type'] == 'gin':
        from MolCLR.models.ginet_finetune import GINet
        model = GINet('classification', **fine_tune.config["model"]).to(fine_tune.device)
    elif fine_tune.config['model_type'] == 'gcn':
        from MolCLR.models.gcn_finetune import GCN
        model = GCN('classification', **fine_tune.config["model"]).to(fine_tune.device)
    
    if config['grad']:
        predictions, grads = fine_tune.htp_pred(model, config["model_path"], test_loader, return_grad=True)
    else:
        predictions = fine_tune.htp_pred(model, config["model_path"], test_loader)

    dataset.dataset.data["prediction"] = predictions
    return dataset.dataset.data


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    config = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
    pprint.pprint(config)

    df = main(config)
    print(df.drop(['index'], axis=1))

    os.makedirs('predictions', exist_ok=True)
    name = os.path.basename(config['dataset']['data_path']).split('.')[0]
    df.to_csv(
        'predictions/{}_prediction.csv'.format(name), 
        mode='a', index=False
    )
