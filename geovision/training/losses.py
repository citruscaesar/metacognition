from torch.nn import (
    CrossEntropyLoss,
    MSELoss
)

#Type Hints
from typing import Any, Union
from torch.nn import Module

out_type = Union[CrossEntropyLoss, MSELoss]

class LossFactory:
    losses = {
        "cross_entropy": CrossEntropyLoss(),
        "mean_squared_error": MSELoss(),
    }

    @classmethod
    def get(cls, loss_string: str) -> Module:
        assert loss_string in cls.losses, "invalid string"
        return cls.losses.get(loss_string) # type : ignore