import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import postgres_db
from routers import plagiarism_checker,zip_checker,update_database

postgres_db.Base.metadata.create_all(bind=postgres_db.engine)
app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "/docs để vào swagger"}

app.include_router(plagiarism_checker.router)
app.include_router(zip_checker.router)
app.include_router(update_database.router)

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8688, proxy_headers=True, forwarded_allow_ips="*"
    )