import React, { useState, useEffect } from 'react';
import { Card, Button } from '../components/UI';
import { coachService } from '../services/api';
import { Sparkles, Clock, ChefHat, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function CoachPage() {
    const [recipes, setRecipes] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        loadSuggestions();
    }, []);

    const loadSuggestions = async () => {
        try {
            setLoading(true);
            const data = await coachService.getSuggestions();
            setRecipes(data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-6 pb-24 space-y-6">
            <header className="flex items-center gap-3 mb-6">
                <button onClick={() => navigate(-1)} className="bg-white p-2 rounded-full shadow-sm">
                    <ArrowRight className="rotate-180" size={20} />
                </button>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    AI Kitchen <Sparkles className="text-yellow-400 fill-yellow-400" size={20}/>
                </h1>
            </header>

            <div className="bg-gradient-to-r from-primary to-orange-400 p-6 rounded-3xl text-white shadow-lg shadow-orange-200">
                <h2 className="font-black text-xl mb-1">Your Daily Menu</h2>
                <p className="text-orange-100 text-sm font-medium">Curated based on your goals & taste.</p>
                <Button 
                    variant="secondary" 
                    size="sm" 
                    className="mt-4 bg-white/20 text-white border-none hover:bg-white/30 backdrop-blur-sm"
                    onClick={loadSuggestions}
                >
                    {loading ? 'Thinking...' : 'Regenerate'}
                </Button>
            </div>

            <div className="space-y-4">
                {loading ? (
                    <div className="text-center py-12 text-gray-400 animate-pulse">
                        <ChefHat size={48} className="mx-auto mb-4 opacity-50" />
                        <p>Cooking up ideas...</p>
                    </div>
                ) : (
                    recipes.map((recipe, i) => (
                        <Card key={i} className="group hover:shadow-lg transition-all duration-300 border-l-4 border-l-transparent hover:border-l-primary">
                            <div className="flex justify-between items-start mb-2">
                                <h3 className="font-bold text-lg text-gray-800 group-hover:text-primary transition-colors">{recipe.name}</h3>
                                <span className="bg-green-100 text-green-700 text-xs font-bold px-2 py-1 rounded-full">
                                    {recipe.calories} kcal
                                </span>
                            </div>
                            
                            <div className="flex gap-4 text-xs text-gray-400 font-bold uppercase tracking-wider mb-4">
                                <div className="flex items-center gap-1">
                                    <Clock size={12} /> {recipe.prep_time_mins} min
                                </div>
                                <div>Prot: {recipe.protein}g</div>
                                <div>Carbs: {recipe.carbs}g</div>
                                <div>Fat: {recipe.fat}g</div>
                            </div>

                            <div className="bg-gray-50 rounded-xl p-3 mb-3">
                                <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase">Ingredients</h4>
                                <div className="flex flex-wrap gap-2">
                                    {recipe.ingredients.map((ing: string, j: number) => (
                                        <span key={j} className="text-xs bg-white border border-gray-100 px-2 py-1 rounded-md text-gray-600">
                                            {ing}
                                        </span>
                                    ))}
                                </div>
                            </div>
                            
                            <Button className="w-full" size="sm" variant="secondary">View Recipe</Button>
                        </Card>
                    ))
                )}
            </div>
        </div>
    );
}
