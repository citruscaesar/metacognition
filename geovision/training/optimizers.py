#type: ignore

from torch.optim import (
    Optimizer,
    SGD,
    Adam
)

class OptimizerFactory:
    def __init__(self, optim_string:str):
        if optim_string == "sgd":
            self.optimizer = self.__init_sgd

        elif optim_string == "adam":
            self.optimizer = self.__init_adam
        
    def __init_sgd(self, **kwargs):
        return SGD(params = kwargs.get("params"), 
                   lr = kwargs.get("lr"),
                   momentum = kwargs.get("momentum"),
                   weight_decay = kwargs.get("weight_decay"))

    def __init_adam(self, **kwargs):
        return Adam(params = kwargs.get("params"), 
                    lr = kwargs.get("lr"))