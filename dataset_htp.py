import pandas as pd
from MolCLR.dataset.dataset_test import MolTestDataset, MolTestDatasetWrapper


class HTPMolDataset(MolTestDataset):
    def __init__(self, data_path):
        super().__init__(data_path, target='index', task='classification')
        self.data = pd.read_csv(data_path)
        #self.data.drop(["index"], axis=1, inplace=True)
        self.smiles_data = self.read_smiles(data_path)
        self.labels = [0] * len(self.smiles_data)
    
    def read_smiles(self, data_path):
        from rdkit import Chem
        import csv
        smiles_data = []
        with open(data_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                smiles = row['smiles']
                mol = Chem.MolFromSmiles(smiles)
                if mol != None:
                    smiles_data.append(smiles)                        
        return smiles_data
    

class HTPMolDatasetWrapper(MolTestDatasetWrapper):
    def __init__(
        self,
        batch_size, 
        num_workers, 
        data_path
    ):
        super().__init__(
            batch_size=batch_size, 
            num_workers=num_workers, 
            valid_size=0.,
            test_size=0.,
            data_path=data_path,
            target='index',
            task='classification',
            splitting='random'
        )

    def get_dataset(self):
        dataset = HTPMolDataset(data_path=self.data_path)
        self.dataset = dataset
        return dataset
