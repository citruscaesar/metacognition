from pandas import DataFrame
from torch.utils.data import Dataset 
from sklearn.preprocessing import LabelEncoder

from data.imageloaders import ImageLoader

class DataframeClassificationDataset(Dataset):
    """
    DataFrame Columns = (index[int], image_path[str|Path], label[int|str], split[optional], name[str])
    """
    def __init__(self, dataframe: DataFrame, **kwargs):
        self.dataframe = dataframe

    def __len__(self):
        pass

    def __getitem__(self, index) -> tuple:

        """
        Returns a tuple = (image[Image], label[Tensor], name[str], path[str])
        """
        return ()
        
class DataframeSegmentationDataset(Dataset):
    """
    DataFrame Columns = (index[int], image_path[str|Path], mask_path[str|Path], split[optional], name[str])
    """
    def __init__(self, dataframe: DataFrame, **kwargs):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index) -> tuple:
        """
        Returns a tuple = (image[Image], mask[Image], name[optional[str]], path[str])
        """
        return ()