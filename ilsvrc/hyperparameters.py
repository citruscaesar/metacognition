from dataclasses import dataclass, asdict

@dataclass(frozen = True, repr = True)
class Hyperparameters:
    task: str
    random_seed: int

    num_classes: int
    test_split: float
    metrics: list[str]

    learning_rate: float
    batch_size: int
    num_workers: int
    optimizer: callable  # type: ignore
    criterion: callable # type: ignore

    local_cache_limit: int | str

    def get_dict(self) -> dict:
        return asdict(self)

@dataclass
class TrainerConfig:
    pass