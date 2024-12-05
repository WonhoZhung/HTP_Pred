from MolCLR.dataset.dataset_test import MolTestDataset, MolTestDatasetWrapper


class HTPMolDataset(MolTestDataset):
    def __init__(self, data_path):
        super().__init__(self, data_path, target='index', task='classification')

class HTPMolDatasetWrapper(MolTestDatasetWrapper):
    def __init__(
        self,
        batch_size, 
        num_workers, 
        data_path
    ):
        super().__init__(
            self,
            batch_size=batch_size, 
            num_workers=num_workers, 
            valid_size=0.,
            test_size=0.,
            data_path=data_path,
            target='index',
            task='classification'
        )

    def get_dataset(self):
        dataset = HTPMolDataset(data_path=self.data_path)
        self.dataset = dataset
        return dataset
