from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import datetime

from .. import models, schemas, auth, database

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=schemas.User)
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
        daily_calorie_goal=user.daily_calorie_goal,
        objective=user.objective,
        dietary_likes=user.dietary_likes,
        dietary_dislikes=user.dietary_dislikes
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/me/", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user

@router.put("/me/", response_model=schemas.User)
def update_user_profile(user_update: schemas.UserUpdate, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(database.get_db)):
    # Update fields if provided
    for field, value in user_update.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    
    db.commit()
    db.refresh(current_user)
    return current_user

@router.post("/me/weight", response_model=schemas.WeightEntryBase)
def add_weight_entry(entry: schemas.WeightEntryBase, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(database.get_db)):
    # Check if entry exists for this day
    target_date = entry.date.date()
    existing_entry = db.query(models.WeightEntry).filter(
        models.WeightEntry.user_id == current_user.id,
        func.date(models.WeightEntry.date) == target_date
    ).first()

    if existing_entry:
        existing_entry.weight_kg = entry.weight_kg
        existing_entry.date = entry.date # Update time as well
        db_entry = existing_entry
    else:
        db_entry = models.WeightEntry(
            user_id=current_user.id,
            weight_kg=entry.weight_kg,
            date=entry.date
        )
        db.add(db_entry)
    
    # Also update current weight profile
    current_user.weight_kg = entry.weight_kg
    
    db.commit()
    db.refresh(db_entry)
    return db_entry

@router.get("/me/weight", response_model=List[schemas.WeightEntryBase])
def get_weight_history(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(database.get_db)):
    return db.query(models.WeightEntry).filter(models.WeightEntry.user_id == current_user.id).order_by(models.WeightEntry.date.asc()).all()
