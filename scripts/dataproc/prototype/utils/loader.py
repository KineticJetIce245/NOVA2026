from pathlib import Path
from dataclasses import dataclass

import copy

from nova2026.config import PROJECT_ROOT

ROOT = PROJECT_ROOT / "datasets"
dataset_list = ["COG-BCI"]
trial_filename_list = ["PVT.set"]
resting_filename_list = ["RS_Beg_EC.set", "RS_Beg_EO.set"]

@dataclass
class Loader:
  
  def __init__(
    self, 
    dataset_list: list[str],
    trial_filename_list: list[str], 
    resting_filename_list: list[str],
    root: str | Path
    ):
    
    self.dataset_list = dataset_list
    self.trial_filename_list = trial_filename_list
    self.resting_filename_list = resting_filename_list
    self.root = Path(root)
    
    self.path_dict: dict[tuple[str, str, str], list[str | Path]] = {}
    
    self._build_dict()
  
  def _recurse(self, path: Path, data_list: list[str | Path], filename: str) -> None:
    for item_path in path.iterdir():
      if item_path.is_dir(): 
        self._recurse(item_path, data_list, filename)
      else:
        if item_path.name.lower() == filename.lower():
          data_list.append(path)
    return
  
  def _build_dict(self) -> None:
    for dataset in self.dataset_list:
      dataset_path = self.root / dataset
      for trial_file in self.trial_filename_list:
        key = (dataset, "trial", trial_file)
        self.path_dict[key] = []
        self._recurse(dataset_path, self.path_dict[key], trial_file)
      for resting_file in self.resting_filename_list:
        key = (dataset, "resting", resting_file)
        self.path_dict[key] = []
        self._recurse(dataset_path, self.path_dict[key], resting_file)
  
  def get_path_dict(self):
    return copy.deepcopy(self.path_dict)
    

if __name__ == "__main__":
  my_loader = Loader(dataset_list, trial_filename_list, resting_filename_list, ROOT)
  print(my_loader.path_dict)
  #print(my_loader.path_dict["COG-BCI", "resting", "RS_Beg_EC.set"])
  # for key in my_loader.path_dict.keys():
  #   print(key)