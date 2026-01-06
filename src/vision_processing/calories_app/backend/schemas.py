from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# --- USER SCHEMAS ---
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str
    height_cm: float = 175.0
    weight_kg: float = 70.0
    daily_calorie_goal: int = 2500

class User(UserBase):
    id: int
    current_streak: int
    xp_points: int
    daily_calorie_goal: int
    
    class Config:
        from_attributes = True

# --- MEAL SCHEMAS ---
class MealItemBase(BaseModel):
    name: str
    weight_g: int
    volume_cm3: float
    density_type: str
    calories: int
    protein: float
    carbs: float
    fat: float

class MealCreate(BaseModel):
    name: str

class Meal(BaseModel):
    id: int
    name: str
    timestamp: datetime
    image_paths: List[str]
    total_calories: int
    total_protein: float
    total_carbs: float
    total_fat: float
    total_sugar: float = 0.0
    health_score: str
    items: List[MealItemBase]

    class Config:
        from_attributes = True

# --- AUTH SCHEMAS ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
