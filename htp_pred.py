import os
import sys
import time
import yaml
import pprint
import pickle
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.SimilarityMaps import GetSimilarityMapFromWeights

from torch_geometric.data import DataLoader

from dataset_htp import HTPMolDataset, HTPMolDatasetWrapper
from model_htp import HTPFineTune


def main(config):
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
        dataset.dataset.data["prediction"] = predictions
        return dataset.dataset.data, grads
    else:
        predictions, _ = fine_tune.htp_pred(model, config["model_path"], test_loader)
        dataset.dataset.data["prediction"] = predictions
        return dataset.dataset.data, None


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    config = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
    pprint.pprint(config)

    df, grads = main(config)
    print(df.drop(['index'], axis=1))

    os.makedirs('predictions', exist_ok=True)
    name = os.path.basename(config['dataset']['data_path']).split('.')[0]
    df.to_csv(
        f'predictions/{name}_prediction.csv', 
        mode='a', index=False
    )

    if config['grad'] and len(grads) > 0:
        with open(f'predictions/{name}_grad.pkl', 'wb') as w:
            pickle.dump(grads, w)

    if config['grad'] and config['visualize']:
        for i, (smi, grad) in enumerate(zip(df['smiles'], grads)):
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            img = Draw.MolDraw2DCairo(800, 800)
            GetSimilarityMapFromWeights(mol, grad.tolist(), draw2d=img, contourLines=0)
            img.FinishDrawing()
            img.WriteDrawingText(f'predictions/{name}_grad_{i}.png')