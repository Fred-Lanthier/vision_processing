from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
import uuid
from datetime import datetime, timedelta

from . import models, schemas, auth, database
from .utils import calculate_nutriscore
from .routers import users, coach, meals

# Initialisation DB
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Nutrition AI API")
app.include_router(users.router)
app.include_router(coach.router)
app.include_router(meals.router)

# CORS pour React
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173", 
        "http://132.207.24.13:5173",  # Votre IP Mobile
        "*" # Fallback pour le dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage fichiers statiques (images uploadées)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Montage des assets du build React (s'ils existent)
import os
from fastapi.responses import FileResponse
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=f"{frontend_dist}/assets"), name="assets")

# --- AUTH ROUTES ---
@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- SERVE REACT APP (CATCH-ALL) ---
# Doit être défini en DERNIER pour ne pas bloquer les routes API
if os.path.exists(frontend_dist):
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # On vérifie si le fichier existe physiquement (ex: favicon.ico, manifest.json)
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # Sinon, on renvoie index.html (SPA Routing)
        return FileResponse(f"{frontend_dist}/index.html")
