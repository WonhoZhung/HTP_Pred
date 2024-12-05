import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

from torch_geometric.data import DataLoader

from dataset_htp import HTPMolDataset, HTPMolDatasetWrapper
from model_htp import HTPFineTune


def main(config):
    dataset = HTPMolDatasetWrapper(config['batch_size'], **config['dataset'])
    dataset.get_dataset()
    test_loader = DataLoader(
            dataset=dataset.dataset,
            batch_size=dataset.batch_size,
    )

    fine_tune = HTPFineTune(dataset, config)
    if fine_tune.config['model_type'] == 'gin':
        from MolCLR.models.ginet_finetune import GINet
        model = GINet(fine_tune.config['dataset']['task'], **fine_tune.config["model"]).to(fine_tune.device)
        model = fine_tune._load_pre_trained_weights(model)
    elif fine_tune.config['model_type'] == 'gcn':
        from MolCLR.models.gcn_finetune import GCN
        model = GCN(fine_tune.config['dataset']['task'], **fine_tune.config["model"]).to(fine_tune.device)
        model = fine_tune._load_pre_trained_weights(model)
    
    if config['grad']:
        predictions, grads = fine_tune.predict(model, config["model_path"], test_loader, return_grad=True)
    else:
        predictions = fine_tune.predict(model, config["model_path"], test_loader)

    print(predictions)



if __name__ == "__main__":
    yaml_path = sys.argv[1]
    config = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
    print(config)

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        result = main(config)
        results_list.append([target, result])

    os.makedirs('predictions', exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(
        'predictions/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']), 
        mode='a', index=False, header=False
    )
