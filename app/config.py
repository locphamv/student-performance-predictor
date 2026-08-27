from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

class Settings(BaseSettings):
    app_environment: str = "development"
    model_filename: str = (
        "student-pass-pipeline.joblib"
    )

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
