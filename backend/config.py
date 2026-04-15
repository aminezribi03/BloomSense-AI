

import os
from pathlib import Path


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data_v2"

        
        default_db_path = data_dir / "app.db"
        self.DATABASE_URL: str = os.getenv(
            "DATABASE_URL", f"sqlite:///{default_db_path}")

     
        self.MODEL_PATH: Path = Path(
            os.getenv(
                "MODEL_PATH", str(data_dir / "models" / "iris_rf.joblib")
            )
        )
        self.SCALER_PATH: Path = Path(
            os.getenv(
                "SCALER_PATH", str(data_dir / "models" / "scaler.joblib")
            )
        )
        self.METRICS_PATH: Path = Path(
            os.getenv(
                "METRICS_PATH", str(data_dir / "models" / "metrics.json")
            )
        )

       
        self.LOG_DIR: Path = Path(
            os.getenv("LOG_DIR", str(project_root / "logs"))
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of settings (useful for debug)."""
        return {
            "DATABASE_URL": self.DATABASE_URL,
            "MODEL_PATH": str(self.MODEL_PATH),
            "SCALER_PATH": str(self.SCALER_PATH),
            "METRICS_PATH": str(self.METRICS_PATH),
            "LOG_DIR": str(self.LOG_DIR),
        }


settings = Settings()
