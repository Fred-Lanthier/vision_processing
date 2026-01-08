from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import shutil
import os
import uuid
from datetime import datetime

from .. import models, schemas, auth, database
from ..core.ai_engine import NutrientScannerMultiView
from ..utils import calculate_nutriscore

router = APIRouter(
    prefix="/meals",
    tags=["meals"],
    responses={404: {"description": "Not found"}},
)

# Reference to the AI engine (injected or imported)
# Ideally, we should use a dependency injection pattern or singleton.
# For now, we'll try to import `scanner` from main, but that causes circular imports.
# Better: Initialize a local reference or move `scanner` to a shared module.
# Let's assume we re-instantiate or share.
# Actually, `scanner` initialization is heavy. Let's create a singleton in `core/ai_engine.py` or `dependencies.py`.
# For now, to keep it simple, I will assume the `scanner` is available via a dependency or global variable.
# BUT, `main.py` initializes it.
# Solution: Create `backend/dependencies.py` to hold the scanner instance.

@router.post("/scan/", response_model=schemas.Meal)
async def scan_meal(
    files: List[UploadFile] = File(...), 
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    # Get scanner from dependencies
    from ..dependencies import get_scanner
    scanner = get_scanner()
    
    if not scanner:
        raise HTTPException(status_code=503, detail="AI Engine not available")

    # 1. Sauvegarder les images
    saved_paths = []
    session_id = str(uuid.uuid4())
    upload_dir = f"backend/static/uploads/{session_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    for file in files:
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".jpg"
        safe_filename = f"{uuid.uuid4()}{ext}"
        
        file_path = os.path.join(upload_dir, safe_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        relative_path = f"static/uploads/{session_id}/{safe_filename}"
        saved_paths.append(relative_path)

    # 2. Appel IA
    try:
        abs_paths = [os.path.abspath(os.path.join("backend", p)) for p in saved_paths]
        results = scanner.analyze_scene(abs_paths)
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

    except Exception as e:
        print(f"AI Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Processing failed: {str(e)}")

    # 3. Créer le Meal en DB
    total_sugar = results.get('total_sugar', 0)
    total_fiber = results.get('total_fiber', 0)
    total_sat_fat = results.get('total_fat', 0) * 0.3 
    
    score_letter, score_num = calculate_nutriscore(
        results.get('total_calories', 0),
        total_sugar,
        total_sat_fat,
        500, 
        0, 
        total_fiber,
        results.get('total_protein_g', 0) 
    )
    
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

@router.get("/", response_model=List[schemas.Meal])
def get_meals(
    date: Optional[str] = None, # Format YYYY-MM-DD
    skip: int = 0, 
    limit: int = 50, 
    db: Session = Depends(database.get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    query = db.query(models.Meal).filter(models.Meal.user_id == current_user.id)
    
    if date:
        # Filter by specific date
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            # SQLite specific or generic date filtering
            # We want filtering by DAY, ignoring time
            query = query.filter(func.date(models.Meal.timestamp) == target_date)
        except ValueError:
            pass # Ignore invalid date format
            
    return query.order_by(models.Meal.timestamp.desc()).offset(skip).limit(limit).all()
