from fastapi import APIRouter

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.policies import router as policies_router
from app.api.predictions import router as predictions_router
from app.api.repositories import router as repositories_router
from app.api.stats import router as stats_router
from app.api.webhook import router as webhook_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(admin_router)
api_router.include_router(webhook_router)
api_router.include_router(predictions_router)
api_router.include_router(stats_router)
api_router.include_router(policies_router)
api_router.include_router(repositories_router)


@api_router.get("/ping")
def ping() -> dict[str, str]:
    return {"pong": "ok"}
