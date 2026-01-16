# src/config/incremental_settings.py
# standard libs
import json

# 3rd libs
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class IncrementalSettings(BaseSettings):
    MIN_INTERVAL_SEC: int = Field (
        default = 6 * 3600,
        ge = 10* 60,
        description= "Thời gian mỗi lần thực hiện IL"
        )
    IL_DATA_DIR : Path = Path("./train_data")
    IL_MODEL_DIR: Path = Path("./src/models/pre_models")
    IL_LOGS: Path = Path("./il_logs/")
    IL_STATE_FILE: Path = Path("./src/job/il_state.json")
    IL_LOCK_FILE: Path = Path("./src/job/il.lock")
    IL_LABEL_FILE: Path = Path("./src/config/Label.json")
    IL_FIXED_MEM_BUDGET: int = 1200000
    
    def IL_LABEL(self) -> dict:
        if not self.IL_LABEL_FILE.exists():
            default_label = {"Benign": 400000, "DDoS": 400000, "DoS": 400000}
            self.IL_LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.IL_LABEL_FILE.write_text(json.dumps(default_label, indent=2))
            return default_label
        
        try:
            return json.loads(self.IL_LABEL_FILE.read_text())
        except json.JSONDecodeError:
            # Nếu file lỗi → dùng default và ghi đè
            default = {"Benign": 400000, "DDoS": 400000, "DoS": 400000}
            self.IL_LABEL_FILE.write_text(json.dumps(default, indent=2))
            return default

    def IL_LABEL_SAVE(self, new_label: dict):
        self.IL_LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.IL_LABEL_FILE.write_text(json.dumps(new_label, indent=2, ensure_ascii=False))
        print(f"[IL Settings] Đã cập nhật IL_LABEL: {new_label}")
    
    herding_replay_ratio: float = Field(
        default=0.2,
        ge=0.1, le=0.8,
        description="Tỷ lệ dữ liệu gần trung bình được lấy để train"
    )
    
    # # Dataset
    # initial_train_ratio: float = Field(
    #     default=0.5,
    #     ge=0.1, le=0.8,
    #     description="Tỷ lệ dữ liệu dùng để train ban đầu"
    # )
    # increment_batch_size: int = Field(
    #     default=500,
    #     ge=100, le=50000, multiple_of=100,
    #     description="Kích thước batch cho mỗi bước incremental"
    # )

    # # Model (XGBoost)
    # learning_rate: float = Field(
    #     default=0.05,
    #     ge=0.001, le=0.2,
    #     description="Learning rate cho XGBoost"
    # )
    # max_depth: int = Field(
    #     default=6,
    #     ge=3, le=12,
    #     description="Độ sâu tối đa của cây"
    # )
    # trees_per_step: int = Field(
    #     default=20,
    #     ge=1, le=100,
    #     description="Số cây thêm vào mỗi bước incremental"
    # )

    # # Anti-forgetting
    # replay_buffer_size: int = Field(
    #     default=5000,
    #     ge=0, le=100000, multiple_of=500,
    #     description="Kích thước buffer replay để chống catastrophic forgetting"
    # )
    # replay_ratio: float = Field(
    #     default=0.3,
    #     ge=0.0, le=1.0,
    #     description="Tỷ lệ replay trong mỗi batch"
    # )

    # Control
    enable_training: bool = Field(
        default=True,
        description="Bật/tắt training incremental"
    )

    model_config = SettingsConfigDict(
        env_file="src/config/.config_incremental",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore" # Có thể thêm để tránh lỗi khi dư thừa biến trong .env
    )

# Singleton instance
incremental_settings = IncrementalSettings()