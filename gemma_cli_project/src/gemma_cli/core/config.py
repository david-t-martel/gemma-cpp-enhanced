from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "gemma-2b-it"
    temperature: float = 0.7

    class Config:
        env_file = ".env"
