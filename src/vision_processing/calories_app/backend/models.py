from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    # Objectifs & Profil
    height_cm = Column(Float, default=175.0)
    weight_kg = Column(Float, default=70.0)
    age = Column(Integer, default=30)
    gender = Column(String, default="M") # M/F
    activity_level = Column(String, default="moderate") # sedentary, light, moderate, active, extreme
    daily_calorie_goal = Column(Integer, default=2500)
    
    # Gamification
    current_streak = Column(Integer, default=0)
    last_login = Column(DateTime, default=datetime.utcnow)
    xp_points = Column(Integer, default=0) # Pour monter de niveau
    
    meals = relationship("Meal", back_populates="owner")

class Meal(Base):
    __tablename__ = "meals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, default="Repas") # "Déjeuner", "Dîner"...
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Images (Stockées comme liste JSON de chemins)
    image_paths = Column(JSON) 
    
    # Totaux (Cache pour éviter de recalculer)
    total_calories = Column(Integer)
    total_protein = Column(Float)
    total_carbs = Column(Float)
    total_fat = Column(Float)
    total_sugar = Column(Float, default=0.0)
    
    # Score IA
    health_score = Column(String, default="B") # A, B, C, D, E
    score_numeric = Column(Integer, default=50) # 0-100 (pour les moyennes)
    
    owner = relationship("User", back_populates="meals")
    items = relationship("MealItem", back_populates="meal")

class MealItem(Base):
    __tablename__ = "meal_items"

    id = Column(Integer, primary_key=True, index=True)
    meal_id = Column(Integer, ForeignKey("meals.id"))
    
    name = Column(String) # ex: "Pepperoni Pizza"
    weight_g = Column(Integer)
    volume_cm3 = Column(Float)
    density_type = Column(String) # LOW, MEDIUM, HIGH
    
    calories = Column(Integer)
    protein = Column(Float)
    carbs = Column(Float)
    fat = Column(Float)
    
    meal = relationship("Meal", back_populates="items")
