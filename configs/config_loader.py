import yaml
import os

class ConfigLoader:
    _config = None

    @classmethod
    def load_config(cls, config_path="configs/config.yaml"):
        if cls._config is None:
            # Tìm đường dẫn tuyệt đối dựa trên vị trí của file này (thư mục configs)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(base_dir, config_path)
            
            with open(full_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
        return cls._config

    @classmethod
    def get(cls, key_path, default=None):
        """
        Lấy giá trị từ file config bằng chuỗi đường dẫn.
        Ví dụ: ConfigLoader.get("environment.max_power_kw")
        """
        config = cls.load_config()
        keys = key_path.split('.')
        val = config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default
