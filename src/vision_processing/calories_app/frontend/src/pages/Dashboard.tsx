import React, { useEffect, useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { mealService } from '../services/api';
import { Card } from '../components/UI';
import { Flame, Droplet, Wheat, Dumbbell, ChevronRight } from 'lucide-react';
import { format } from 'date-fns';
import clsx from 'clsx';
import { useNavigate } from 'react-router-dom';

export default function Dashboard() {
    const { user } = useAuthStore();
    const navigate = useNavigate();
    const [todayMeals, setTodayMeals] = useState<any[]>([]);
    const [stats, setStats] = useState({ kcal: 0, pro: 0, carb: 0, fat: 0 });
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const meals = await mealService.getHistory(0, 10);
            if (Array.isArray(meals)) {
                setTodayMeals(meals);
                
                // Calculate today's stats
                const today = meals.reduce((acc: any, meal: any) => ({
                    kcal: acc.kcal + (Number(meal.total_calories) || 0),
                    pro: acc.pro + (Number(meal.total_protein) || 0),
                    carb: acc.carb + (Number(meal.total_carbs) || 0),
                    fat: acc.fat + (Number(meal.total_fat) || 0),
                }), { kcal: 0, pro: 0, carb: 0, fat: 0 });
                setStats(today);
            }
        } catch (e) { 
            console.error("Error loading dashboard data:", e); 
        } finally {
            setIsLoading(false);
        }
    };

    if (isLoading) {
        return <div className="p-6 flex justify-center pt-20"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div></div>;
    }

    const goal = user?.daily_calorie_goal || 2500;
    const progress = Math.min((stats.kcal / goal) * 100, 100);

    const getScoreColor = (score: string) => {
        const map: any = { 'A': 'bg-green-100 text-green-700', 'B': 'bg-lime-100 text-lime-700', 'C': 'bg-yellow-100 text-yellow-700', 'D': 'bg-orange-100 text-orange-700', 'E': 'bg-red-100 text-red-700' };
        return map[score] || 'bg-gray-100 text-gray-700';
    };

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Today</h1>
                    <p className="text-gray-500 font-medium">{format(new Date(), 'EEEE, MMM d')}</p>
                </div>
                <div className="bg-orange-100 text-primary font-bold px-3 py-1 rounded-full text-sm">
                    🔥 {user?.current_streak || 0} Day Streak
                </div>
            </div>

            {/* Main Stats Card */}
            <Card className="bg-gray-900 text-white border-none shadow-xl relative overflow-hidden">
                <div className="absolute top-0 right-0 -mr-10 -mt-10 w-40 h-40 bg-primary opacity-20 rounded-full blur-3xl"></div>
                
                <div className="relative z-10">
                    <div className="flex justify-between items-end mb-4">
                        <div>
                            <p className="text-gray-400 text-sm font-medium uppercase tracking-wider">Calories</p>
                            <div className="text-5xl font-black tracking-tighter">
                                {Math.round(stats.kcal)} <span className="text-xl text-gray-500 font-normal">/ {user?.daily_calorie_goal}</span>
                            </div>
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="h-4 bg-gray-700 rounded-full overflow-hidden mb-6">
                        <div 
                            className="h-full bg-gradient-to-r from-orange-500 to-red-500 transition-all duration-1000 ease-out"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>

                    {/* Macros */}
                    <div className="grid grid-cols-3 gap-4">
                        <MacroItem icon={Dumbbell} label="Protein" val={stats.pro} color="text-blue-400" />
                        <MacroItem icon={Wheat} label="Carbs" val={stats.carb} color="text-yellow-400" />
                        <MacroItem icon={Droplet} label="Fat" val={stats.fat} color="text-pink-400" />
                    </div>
                </div>
            </Card>

            {/* Recent Activity Feed */}
            <div>
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-lg font-bold">Recent Meals</h2>
                    <button className="text-primary text-sm font-bold hover:underline">View All</button>
                </div>

                <div className="space-y-3">
                    {todayMeals.length === 0 ? (
                        <div className="text-center py-10 text-gray-400 bg-white rounded-2xl border border-dashed border-gray-300">
                            <p>No meals logged yet today.</p>
                            <p className="text-sm">Tap the + button to scan!</p>
                        </div>
                    ) : (
                        todayMeals.map((meal) => (
                            <div 
                                key={meal.id} 
                                onClick={() => navigate(`/meal/${meal.id}`, { state: { meal } })}
                                className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm flex gap-4 items-center active:scale-95 transition-transform cursor-pointer"
                            >
                                {/* Thumbnail */}
                                <div className="w-16 h-16 bg-gray-200 rounded-xl overflow-hidden flex-shrink-0">
                                    {meal.image_paths && meal.image_paths[0] ? (
                                        <img src={meal.image_paths[0]} className="w-full h-full object-cover" onError={(e) => (e.target as HTMLImageElement).src = 'https://placehold.co/100?text=Meal'} />
                                    ) : (
                                        <div className="w-full h-full flex items-center justify-center text-gray-400 text-xs">No Img</div>
                                    )}
                                </div>
                                
                                <div className="flex-1">
                                    <div className="flex justify-between items-start">
                                        <h3 className="font-bold text-gray-900">{meal.name}</h3>
                                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${getScoreColor(meal.health_score)}`}>
                                            Score {meal.health_score || '?'}
                                        </span>
                                    </div>
                                    <p className="text-sm text-gray-500 mt-1">
                                        {Math.round(meal.total_calories)} kcal • {Math.round(meal.total_protein)}g Pro
                                    </p>
                                </div>
                                <ChevronRight className="text-gray-300" size={20} />
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}

function MacroItem({ icon: Icon, label, val, color }: any) {
    return (
        <div className="flex flex-col items-center p-2 bg-gray-800 rounded-xl">
            <Icon size={18} className={clsx("mb-1", color)} />
            <span className="text-lg font-bold">{Math.round(val)}g</span>
            <span className="text-[10px] text-gray-400 uppercase tracking-wide">{label}</span>
        </div>
    );
}
