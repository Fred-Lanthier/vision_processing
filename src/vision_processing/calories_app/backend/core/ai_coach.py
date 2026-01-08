import os
import json
from google import genai
from typing import List, Dict

class AICoach:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("⚠️ GOOGLE_API_KEY not found. AI Coach will not work.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def generate_recipes(self, objective: str, calorie_target: int, likes: List[str], dislikes: List[str]) -> List[Dict]:
        if not self.client:
            return []

        prompt = f"""
        You are an expert nutritionist coach. The user has the following profile:
        - Objective: {objective} (e.g., lose weight, gain muscle, maintain)
        - Daily Calorie Target: {calorie_target} kcal
        - Likes: {', '.join(likes) if likes else 'No specific preferences'}
        - Dislikes: {', '.join(dislikes) if dislikes else 'No specific dislikes'}

        Generate 3 distinct, healthy, and delicious meal recipes (breakfast, lunch, or dinner options) that fit this profile.
        
        RETURN ONLY VALID JSON. Structure:
        [
            {{
                "name": "Recipe Name",
                "calories": 500,
                "protein": 30,
                "carbs": 40,
                "fat": 15,
                "prep_time_mins": 15,
                "ingredients": ["item 1", "item 2"],
                "instructions": ["step 1", "step 2"]
            }},
            ...
        ]
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=prompt
            )
            text = response.text.replace("```json", "").replace("```", "").strip()
            recipes = json.loads(text)
            return recipes
        except Exception as e:
            print(f"Error generating recipes: {e}")
            return []
