from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from .. import models, auth, database
from ..core.ai_coach import AICoach

router = APIRouter(
    prefix="/coach",
    tags=["coach"],
    responses={404: {"description": "Not found"}},
)

coach = AICoach()

@router.get("/suggest", response_model=List[Dict[str, Any]])
def get_meal_suggestions(current_user: models.User = Depends(auth.get_current_user)):
    """
    Generates meal suggestions based on the user's profile using Gemini.
    """
    if not coach.client:
        raise HTTPException(status_code=503, detail="AI Service unavailable")
    
    # Calculate target per meal (roughly)
    target_per_meal = current_user.daily_calorie_goal // 3
    
    recipes = coach.generate_recipes(
        objective=current_user.objective,
        calorie_target=target_per_meal,
        likes=current_user.dietary_likes,
        dislikes=current_user.dietary_dislikes
    )
    
    return recipes
