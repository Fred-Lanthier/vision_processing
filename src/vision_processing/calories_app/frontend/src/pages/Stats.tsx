import React, { useState, useEffect } from 'react';
import { Card, Button } from '../components/UI';
import { useAuthStore } from '../store/authStore';
import { mealService, userService } from '../services/api';
import { 
    AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
    BarChart, Bar, ReferenceLine, CartesianGrid
} from 'recharts';
import { TrendingUp, Scale, ChevronLeft, ChevronRight, Calendar } from 'lucide-react';
import { format, subDays, startOfWeek, endOfWeek, eachDayOfInterval, isSameDay } from 'date-fns';
import { animate } from 'framer-motion';

export default function StatsPage() {
    const { user } = useAuthStore();
    const [weightHistory, setWeightHistory] = useState<any[]>([]);
    const [calorieHistory, setCalorieHistory] = useState<any[]>([]);
    const [newWeight, setNewWeight] = useState('');
    const [showWeightInput, setShowWeightInput] = useState(false);
    const [viewMode, setViewMode] = useState<'week' | 'month'>('week');

    const [calDomain, setCalDomain] = useState<[number, number]>([0, 3000]);
    const [weightDomain, setWeightDomain] = useState<[number, number]>([50, 100]);

    useEffect(() => {
        loadData();
    }, [viewMode]);

    const loadData = async () => {
        const today = new Date();
        const goal = user?.daily_calorie_goal || 2500;
        const targetWeight = user?.target_weight_kg || 65;
        
        // --- 1. Calorie Range ---
        let calStart, calEnd;
        if (viewMode === 'week') {
            calStart = startOfWeek(today, { weekStartsOn: 0 }); 
            calEnd = endOfWeek(today, { weekStartsOn: 0 });
        } else {
            calStart = subDays(today, 29);
            calEnd = today;
        }
        const calInterval = eachDayOfInterval({ start: calStart, end: calEnd });

        // --- 2. Weight Range ---
        const weightStart = subDays(today, viewMode === 'week' ? 6 : 29);
        const weightEnd = today;
        const weightInterval = eachDayOfInterval({ start: weightStart, end: weightEnd });

        // --- 3. Load Weight Data ---
        try {
            const weights = await userService.getWeightHistory();
            const weightMap: Record<string, number> = {};
            let minW = 1000, maxW = 0;
            
            weights.forEach((w: any) => {
                const d = new Date(w.date).toDateString();
                weightMap[d] = w.weight_kg;
            });

            const weightData = weightInterval.map(date => {
                const dKey = date.toDateString();
                const val = weightMap[dKey];
                if (val !== undefined) {
                    if (val < minW) minW = val;
                    if (val > maxW) maxW = val;
                }
                return {
                    name: format(date, 'd MMM'),
                    fullDate: format(date, 'MMM d, yyyy'),
                    weight: val || null
                };
            });

            // Calculate Target Domain
            let targetMin = 1000, targetMax = 0;
            const recorded = weightData.filter(w => w.weight !== null);
            if (recorded.length === 0) {
                targetMin = targetWeight - 5;
                targetMax = targetWeight + 5;
            } else {
                targetMin = Math.min(...recorded.map(r => r.weight), targetWeight);
                targetMax = Math.max(...recorded.map(r => r.weight), targetWeight);
            }
            const finalMin = Math.floor(targetMin - 2);
            const finalMax = Math.ceil(targetMax + 2);

            // Animate Weight Domain
            animate(weightDomain[0], finalMin, {
                duration: 1.2,
                ease: [0.4, 0, 0.2, 1],
                onUpdate: (latestMin) => {
                    setWeightDomain(prev => [latestMin, prev[1]]);
                }
            });
            animate(weightDomain[1], finalMax, {
                duration: 1.2,
                ease: [0.4, 0, 0.2, 1],
                onUpdate: (latestMax) => {
                    setWeightDomain(prev => [prev[0], latestMax]);
                }
            });

            setWeightHistory(weightData);
        } catch (e) {}

        // --- 4. Load Calorie Data ---
        try {
            const meals = await mealService.getHistory(0, 100);
            const daysMap: Record<string, number> = {};
            let maxCal = 0;
            
            meals.forEach((m: any) => {
                const d = new Date(m.timestamp).toDateString();
                const total = (daysMap[d] || 0) + m.total_calories;
                daysMap[d] = total;
                if (total > maxCal) maxCal = total;
            });

            const finalMaxCal = Math.max(maxCal, goal) + 500;

            // Animate Calorie Domain
            animate(calDomain[1], finalMaxCal, {
                duration: 1.2,
                ease: [0.4, 0, 0.2, 1],
                onUpdate: (latest) => setCalDomain([0, latest])
            });

            const calorieData = calInterval.map(date => ({
                name: format(date, viewMode === 'week' ? 'EEE' : 'd'),
                fullDate: format(date, 'MMM d, yyyy'),
                calories: daysMap[date.toDateString()] || 0,
                goal: goal
            }));
            
            setCalorieHistory(calorieData);
        } catch (e) {}
    };

    const handleAddWeight = async () => {
        if (!newWeight) return;
        await userService.addWeight(Number(newWeight));
        setNewWeight('');
        setShowWeightInput(false);
        loadData();
    };

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-gray-900 text-white text-[10px] p-2 rounded-lg shadow-xl border border-gray-700">
                    <p className="font-bold mb-1 opacity-60 uppercase tracking-tighter">{payload[0].payload.fullDate || label}</p>
                    <p className="text-sm font-black">
                        {Math.round(payload[0].value)} <span className="text-gray-400 font-normal">{payload[0].dataKey === 'weight' ? 'kg' : 'kcal'}</span>
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="p-6 space-y-8 pb-24 bg-gray-50 min-h-screen">
            <header className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-black text-gray-900 tracking-tighter">Stats</h1>
                    <p className="text-gray-400 text-xs font-bold uppercase tracking-widest mt-1">Personal Training</p>
                </div>
                <div className="flex bg-gray-200 p-1 rounded-xl">
                    <button 
                        onClick={() => setViewMode('week')}
                        className={`px-4 py-1.5 text-[10px] font-black uppercase rounded-lg transition-all ${viewMode === 'week' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500'}`}
                    >
                        Week
                    </button>
                    <button 
                        onClick={() => setViewMode('month')}
                        className={`px-4 py-1.5 text-[10px] font-black uppercase rounded-lg transition-all ${viewMode === 'month' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500'}`}
                    >
                        Month
                    </button>
                </div>
            </header>

            {/* Calorie Trend Chart */}
            <section>
                <div className="flex items-center gap-2 mb-4 text-gray-400 uppercase text-[10px] font-black tracking-widest">
                    <TrendingUp size={12} className="text-orange-500" />
                    Daily Calories
                </div>
                <Card className="h-64 p-4 pt-6 shadow-sm border-none bg-white overflow-visible">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={calorieHistory} margin={{ left: -20, right: 10 }}>
                            <defs>
                                <linearGradient id="colorCal" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#F97316" stopOpacity={0.15}/>
                                    <stop offset="95%" stopColor="#F97316" stopOpacity={0}/>
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F3F4F6" />
                            <XAxis 
                                dataKey="name" 
                                tick={{fontSize: 10, fontWeight: 'bold', fill: '#9CA3AF'}} 
                                axisLine={false} 
                                tickLine={false} 
                                padding={{ left: 10, right: 10 }}
                            />
                            <YAxis 
                                orientation="right" 
                                domain={calDomain}
                                tickFormatter={(value) => Math.round(value).toString()}
                                tick={{fontSize: 9, fontWeight: 'bold', fill: '#D1D5DB'}} 
                                axisLine={false} 
                                tickLine={false}
                                tickCount={5}
                            />
                            <Tooltip cursor={{stroke: '#F97316', strokeWidth: 1, strokeDasharray: '4 4'}} content={<CustomTooltip />} />
                            <ReferenceLine y={user?.daily_calorie_goal || 2500} stroke="#3B82F6" strokeDasharray="5 5" strokeWidth={1.5} label={{ value: 'Goal', position: 'insideTopLeft', fontSize: 10, fill: '#3B82F6' }} />
                            <Area 
                                type="monotone" 
                                dataKey="calories" 
                                stroke="#F97316" 
                                strokeWidth={3}
                                fillOpacity={1} 
                                fill="url(#colorCal)" 
                                dot={{ r: 3, fill: '#fff', stroke: '#F97316', strokeWidth: 2 }}
                                activeDot={{ r: 5, strokeWidth: 0, fill: '#F97316' }}
                                animationDuration={1500}
                                animationEasing="cubic-bezier(0.4, 0, 0.2, 1)"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </Card>
            </section>

            {/* Weight Progress */}
            <section>
                <div className="flex justify-between items-center mb-4">
                    <div className="flex items-center gap-2 text-gray-400 uppercase text-[10px] font-black tracking-widest">
                        <Scale size={12} className="text-emerald-500" />
                        Weight tracker
                    </div>
                    <button 
                        onClick={() => setShowWeightInput(!showWeightInput)}
                        className="text-primary text-[10px] font-black uppercase bg-primary/10 hover:bg-primary/20 px-3 py-1.5 rounded-lg transition-colors"
                    >
                        + Update
                    </button>
                </div>
                
                {showWeightInput && (
                    <div className="flex gap-2 mb-4 bg-white p-3 rounded-2xl shadow-xl border border-gray-100 animate-in fade-in slide-in-from-top-2">
                        <input 
                            type="number" 
                            value={newWeight}
                            onChange={(e) => setNewWeight(e.target.value)}
                            placeholder="kg"
                            className="w-full p-2 bg-gray-50 rounded-xl text-sm font-bold outline-none"
                            autoFocus
                        />
                        <Button size="sm" onClick={handleAddWeight}>Save</Button>
                    </div>
                )}

                <Card className="h-64 p-4 pt-8 shadow-sm border-none bg-white relative overflow-visible">
                    {weightHistory.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={weightHistory} margin={{ left: -20, right: 10 }}>
                                <defs>
                                    <linearGradient id="colorWeight" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.15}/>
                                        <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F3F4F6" />
                                <XAxis 
                                    dataKey="name" 
                                    tick={{fontSize: 9, fontWeight: 'bold', fill: '#9CA3AF'}} 
                                    axisLine={false} 
                                    tickLine={false}
                                                                    padding={{ left: 10, right: 10 }}
                                                                    />
                                                                                                    <YAxis 
                                                                                                        domain={weightDomain} 
                                                                                                        orientation="right" 
                                                                                                        tickFormatter={(value) => Math.round(value).toString()}
                                                                                                        tick={{fontSize: 9, fontWeight: 'bold', fill: '#D1D5DB'}} 
                                                                                                        axisLine={false} 
                                                                                                        tickLine={false}
                                                                                                        tickCount={5}
                                                                                                    />                                                                <Tooltip cursor={{stroke: '#10B981', strokeWidth: 1, strokeDasharray: '4 4'}} content={<CustomTooltip />} />
                                                                <ReferenceLine y={user?.target_weight_kg || 65} stroke="#EF4444" strokeDasharray="5 5" strokeWidth={1.5} label={{ value: 'Target', position: 'insideTopLeft', fontSize: 10, fill: '#EF4444' }} />
                                                                <Area 
                                                                    type="monotone" 
                                                                    dataKey="weight"                                    stroke="#10B981" 
                                    strokeWidth={3}
                                    fillOpacity={1} 
                                    fill="url(#colorWeight)" 
                                    dot={{ r: 3, fill: '#fff', stroke: '#10B981', strokeWidth: 2 }}
                                    activeDot={{ r: 5, strokeWidth: 0, fill: '#10B981' }}
                                    animationDuration={1500}
                                    animationEasing="cubic-bezier(0.4, 0, 0.2, 1)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="flex items-center justify-center h-full text-gray-300 text-xs font-bold uppercase tracking-widest">
                            No data
                        </div>
                    )}
                </Card>
            </section>
        </div>
    );
}
