from typing import Dict


class Config:
    __slots__ = ""

    def __init__(self, config_dict: Dict[str, type]):
        for key, item in config_dict.items():
            setattr(self, key, item)
