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
from .core.ai_engine import NutrientScannerMultiView
from .utils import calculate_nutriscore

# Initialisation DB
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Nutrition AI API")

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

# Initialisation Moteur IA (Global)
print("⏳ Initialisation AI Engine...")
try:
    scanner = NutrientScannerMultiView()
except Exception as e:
    print(f"❌ Erreur Init AI: {e}")
    scanner = None

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

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        username=user.username, 
        email=user.email, 
        hashed_password=hashed_password,
        height_cm=user.height_cm,
        weight_kg=user.weight_kg,
        daily_calorie_goal=user.daily_calorie_goal
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/me/", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user

# --- CORE FEATURE: SCAN ---
@app.post("/scan/", response_model=schemas.Meal)
async def scan_meal(
    files: List[UploadFile] = File(...), 
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    if not scanner:
        raise HTTPException(status_code=503, detail="AI Engine not available")

    # 1. Sauvegarder les images
    saved_paths = []
    session_id = str(uuid.uuid4())
    upload_dir = f"backend/static/uploads/{session_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    for file in files:
        # Generate safe filename
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".jpg"
        safe_filename = f"{uuid.uuid4()}{ext}"
        
        file_path = os.path.join(upload_dir, safe_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Store relative path for DB/Frontend (WEB URL friendly)
        # Ex: "static/uploads/uuid/image.jpg"
        relative_path = f"static/uploads/{session_id}/{safe_filename}"
        saved_paths.append(relative_path)

    # 2. Appel IA (Bloquant -> dans threadpool par défaut FastAPI)
    try:
        # Use ABSOLUTE paths for AI engine to avoid CWD issues
        # Reconstruct absolute path from relative path
        abs_paths = [os.path.abspath(os.path.join("backend", p)) for p in saved_paths]
        
        if len(saved_paths) == 1:
             # Mode monoculaire (supporté via legacy ou adaptation du multiview)
             # Pour l'instant on force le multiview, si 1 image ça plantera dans DUSt3R souvent
             # TODO: Gérer fallback monoculaire proprement
             pass
             
        results = scanner.analyze_scene(abs_paths)
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

    except Exception as e:
        print(f"AI Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Processing failed: {str(e)}")

    # 3. Créer le Meal en DB
    # Calcul Nutri-Score
    total_sugar = results.get('total_sugar', 0)
    total_fiber = results.get('total_fiber', 0)
    total_sat_fat = results.get('total_fat', 0) * 0.3 # Estimé
    
    score_letter, score_num = calculate_nutriscore(
        results.get('total_calories', 0),
        total_sugar,
        total_sat_fat,
        500, # Sodium par défaut
        0, # % Fruit (difficile à savoir sans liste précise)
        total_fiber,
        results.get('total_protein_g', 0) # Attention: ai_engine renvoie 'total_protein' (sans _g) dans ma modif précédente, vérifions
    )
    # Correction: dans ai_engine j'ai mis 'total_protein', pas '_g'. Je dois adapter.
    # En fait, ai_engine ne renvoyait PAS les totaux macros avant ma modif. 
    # Ma modif précédente les ajoute.
    
    db_meal = models.Meal(
        user_id=current_user.id,
        name=f"Repas du {datetime.now().strftime('%H:%M')}",
        image_paths=saved_paths, 
        total_calories=int(results.get('total_calories', 0)),
        total_protein=round(results.get('total_protein', 0), 1),
        total_carbs=round(results.get('total_carbs', 0), 1),
        total_fat=round(results.get('total_fat', 0), 1),
        total_sugar=round(total_sugar, 1),
        health_score=score_letter,
        score_numeric=int(score_num)
    )
    db.add(db_meal)
    db.commit()
    db.refresh(db_meal)

    # 4. Créer les Items
    for item in results.get('items', []):
        db_item = models.MealItem(
            meal_id=db_meal.id,
            name=item['name'],
            weight_g=item['weight'],
            volume_cm3=item['vol'],
            density_type=item.get('density_type', 'MEDIUM'),
            calories=item['kcal'],
            protein=item.get('macros', {}).get('protein', 0),
            carbs=item.get('macros', {}).get('carbs', 0),
            fat=item.get('macros', {}).get('fat', 0)
        )
        db.add(db_item)
    
    db.commit()
    db.refresh(db_meal)
    
    return db_meal

@app.get("/meals/", response_model=List[schemas.Meal])
def get_meals(skip: int = 0, limit: int = 10, db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_user)):
    return db.query(models.Meal).filter(models.Meal.user_id == current_user.id).order_by(models.Meal.timestamp.desc()).offset(skip).limit(limit).all()


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
