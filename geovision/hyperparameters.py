from dataclasses import dataclass, asdict

@dataclass(frozen = True, repr = True)
class Hyperparameters:
    task: str
    num_classes: int
    metrics: list[str]

    learning_rate: float
    batch_size: int
    num_workers: int
    optimizer: callable  # type: ignore
    criterion: callable # type: ignore

    def get_dict(self) -> dict:
        return asdict(self)

@dataclass
class TrainerConfig:
    pass