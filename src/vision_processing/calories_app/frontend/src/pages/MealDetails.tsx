import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ChevronLeft, Share2, MoreHorizontal } from 'lucide-react';
import { Card } from '../components/UI';

export default function MealDetailsPage() {
    const navigate = useNavigate();
    const { state } = useLocation();
    const meal = state?.meal;

    if (!meal) {
        navigate('/');
        return null;
    }

    const getScoreColor = (score: string) => {
        const map: any = { 'A': 'bg-green-500', 'B': 'bg-lime-500', 'C': 'bg-yellow-500', 'D': 'bg-orange-500', 'E': 'bg-red-500' };
        return map[score] || 'bg-gray-500';
    };

    return (
        <div className="flex flex-col h-full bg-white">
            {/* Header Image */}
            <div className="relative h-72 w-full bg-gray-200">
                <img 
                    src={meal.image_paths[0] ? meal.image_paths[0] : ''} 
                    className="w-full h-full object-cover"
                />
                <button onClick={() => navigate(-1)} className="absolute top-6 left-6 bg-white/50 backdrop-blur-md p-2 rounded-full">
                    <ChevronLeft size={24} />
                </button>
                
                {/* Score Badge */}
                <div className={`absolute bottom-6 right-6 ${getScoreColor(meal.health_score)} text-white font-black text-4xl w-16 h-16 rounded-xl flex items-center justify-center shadow-lg border-4 border-white`}>
                    {meal.health_score}
                </div>
            </div>

            <div className="flex-1 -mt-6 bg-white rounded-t-3xl p-6 relative z-10">
                <div className="flex justify-between items-start mb-2">
                    <h1 className="text-2xl font-bold text-gray-900">{meal.name}</h1>
                    <span className="text-gray-400 text-sm">{new Date(meal.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                </div>

                {/* Big Macros */}
                <div className="grid grid-cols-4 gap-2 mb-8">
                    <MacroBox label="Calories" val={meal.total_calories} unit="kcal" />
                    <MacroBox label="Protein" val={meal.total_protein} unit="g" />
                    <MacroBox label="Carbs" val={meal.total_carbs} unit="g" />
                    <MacroBox label="Fat" val={meal.total_fat} unit="g" />
                </div>
                
                {/* Sugar & Score Details */}
                <div className="bg-gray-50 p-4 rounded-xl mb-6">
                    <div className="flex justify-between items-center mb-2">
                        <span className="font-bold text-gray-600">Sugar</span>
                        <span className="font-bold text-gray-900">{meal.total_sugar || 0}g</span>
                    </div>
                     <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-pink-400 h-2 rounded-full" style={{width: `${Math.min(meal.total_sugar * 2, 100)}%`}}></div>
                    </div>
                </div>

                <h3 className="font-bold text-lg mb-4">Items Detected</h3>
                <div className="space-y-3">
                    {meal.items.map((item: any, i: number) => (
                        <div key={i} className="flex justify-between items-center border-b border-gray-100 pb-3 last:border-0">
                            <div>
                                <p className="font-bold text-gray-800">{item.name}</p>
                                <p className="text-xs text-gray-400">{item.weight_g}g</p>
                            </div>
                            <span className="font-bold text-gray-600">{item.calories} kcal</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

function MacroBox({ label, val, unit }: any) {
    return (
        <div className="flex flex-col items-center bg-gray-50 p-3 rounded-2xl">
            <span className="text-lg font-black text-gray-900">{Math.round(val)}</span>
            <span className="text-[10px] text-gray-400 uppercase font-bold">{label}</span>
        </div>
    );
}
