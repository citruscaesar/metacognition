from pathlib import Path
import pandas as pd

IMAGENETTE_PATH = Path.home() / "datasets" / "imagenette"

imagenette_class_labels = { 
  'n03000684': 'chain saw',
  'n03888257': 'parachute',
  'n02102040': 'English springer',
  'n02979186': 'cassette player',
  'n01440764': 'tench',
  'n03028079': 'church',
  'n03417042': 'garbage truck',
  'n03394916': 'French horn',
  'n03425413': 'gas pump',
  'n03445777': 'golf ball'
}

def imagenette_dataframe(root: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        data = {"path": list(root.rglob("*.JPEG"))}
    )
    df["path"] = df["path"].apply(lambda x: Path(x.parents[1].stem) / x.parents[0].stem / x.name)
    df["label"] = df["path"].apply(lambda x: x.parents[0].stem)
    df["split"] = df["path"].apply(lambda x: x.parents[1].stem)
    df["name"] = df["label"].apply(lambda x: imagenette_class_labels[x])
    return df

def save_imagenette_metadata() -> None:
    df = imagenette_dataframe(IMAGENETTE_PATH)
    names = pd.read_table(IMAGENETTE_PATH/"LOC_synset_mapping.txt", header=None)
    names = names.iloc[:, 0].str.split(',', expand=True, n=1)
    names = names.iloc[:, 0].str.split(' ', expand=True, n=1)
    names.columns = ["id", "name"] #type: ignore
    names = names.set_index("id")

    df = df.join(names, on = "label")
    df.to_csv(Path.cwd() / "metadata" / "imagenette-dataframe.csv")

    label_names = df[["label", "name"]].drop_duplicates(keep = "first")
    label_names = label_names.set_index("label")
    label_names.to_csv(Path.cwd() / "metadata" / "imagenette-class-labels.csv")