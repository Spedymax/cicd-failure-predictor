from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_secret_key: str = Field(min_length=32)
    app_log_level: str = "INFO"

    # Fly Postgres attach sets DATABASE_URL — when present it wins over the
    # raw POSTGRES_* parts so the same image works locally (compose) and in
    # managed environments without code changes.
    database_url_override: str | None = Field(default=None, alias="DATABASE_URL")

    postgres_user: str = "cicd_predictor"
    postgres_password: str = "changeme"
    postgres_db: str = "cicd_predictor"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    redis_url: str = "redis://localhost:6379/0"

    github_webhook_secret: str
    github_api_token: str

    ml_model_dir: str = "../data/artifacts/v26_5class"
    ml_inference_timeout_sec: float = 5.0

    # Public URL of the dashboard; used as ``target_url`` for GitHub commit
    # statuses so reviewers can click through to the prediction detail page.
    app_public_url: str = "http://localhost:3000"
    # When true (default), the system posts commit statuses back to GitHub
    # so BLOCK actually gates merge via branch protection. Set to false to
    # run in pure observation mode (e.g. for synthetic / acme/* fixtures).
    github_post_status: bool = True

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            url = self.database_url_override
            # Fly hands out ``postgres://...``; psycopg3 wants the explicit driver tag.
            if url.startswith("postgres://"):
                url = "postgresql+psycopg://" + url[len("postgres://"):]
            elif url.startswith("postgresql://"):
                url = "postgresql+psycopg://" + url[len("postgresql://"):]
            return url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
