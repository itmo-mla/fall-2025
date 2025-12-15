import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    dataset_name: str = 'wine'
    target_col: str = 'target'
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.25
    
    k_range: List[int] = field(default_factory=lambda: list(range(1, 21)))

    loo_k_range: Optional[List[int]] = None
    
    save_plots: bool = True
    plots_dir: str = 'plots'
    
    @classmethod
    def from_yaml(cls, yaml_path: str = 'config.yaml') -> 'Config':
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            print(f"Файл {yaml_path} не найден. Используются значения по умолчанию.")
            return cls()
        
        with open(yaml_path, 'r', encoding='utf-8') as f: config_dict = yaml.safe_load(f)
        
        dataset_config = config_dict.get('dataset', {})
        knn_config = config_dict.get('knn', {})
        loo_config = config_dict.get('loo', {})
        prototype_config = config_dict.get('prototype_selection', {})
        viz_config = config_dict.get('visualization', {})
        
        return cls(
            dataset_name=dataset_config.get('name', 'iris'),
            target_col=dataset_config.get('target_col', 'target'),
            seed=dataset_config.get('seed', 42),
            test_size=dataset_config.get('test_size', 0.2),
            val_size=dataset_config.get('val_size', 0.25),
            
            k_range=knn_config.get('k_range', list(range(1, 21))),
            loo_k_range=loo_config.get('k_range', None),
            
            save_plots=viz_config.get('save_plots', True),
            plots_dir=viz_config.get('plots_dir', 'plots'),
        )
