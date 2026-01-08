import React, { useEffect, useState } from 'react';
import { Card } from '../components/UI';
import { ChevronLeft, ChevronRight, Utensils } from 'lucide-react';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, isSameDay, isToday, startOfWeek, endOfWeek, addDays } from 'date-fns';
import { mealService } from '../services/api';
import { useAuthStore } from '../store/authStore';
import { useNavigate } from 'react-router-dom';

export default function CalendarPage() {
    const { user } = useAuthStore();
    const navigate = useNavigate();
    const [currentDate, setCurrentDate] = useState(new Date());
    const [selectedDate, setSelectedDate] = useState(new Date());
    const [calendarData, setCalendarData] = useState<Record<string, number>>({});
    const [dailyMeals, setDailyMeals] = useState<any[]>([]);
    
    // Generate days including padding for grid alignment
    const monthStart = startOfMonth(currentDate);
    const monthEnd = endOfMonth(monthStart);
    const startDate = startOfWeek(monthStart);
    const endDate = endOfWeek(monthEnd);

    const dateFormat = "d";
    const rows = [];
    let days = [];
    let day = startDate;
    let formattedDate = "";

    const loadCalendarData = async () => {
        try {
            // Fetch month's data (mocked by fetching history limit)
            const meals = await mealService.getHistory(0, 100);
            const data: Record<string, number> = {};
            meals.forEach((m: any) => {
                const dateKey = new Date(m.timestamp).toDateString();
                data[dateKey] = (data[dateKey] || 0) + m.total_calories;
            });
            setCalendarData(data);
        } catch (e) {}
    };

    const loadDailyMeals = async () => {
        try {
            const dateStr = format(selectedDate, "yyyy-MM-dd");
            const meals = await mealService.getMealsByDate(dateStr);
            setDailyMeals(meals);
        } catch (e) {
            setDailyMeals([]);
        }
    };

    useEffect(() => {
        loadCalendarData();
    }, [currentDate]);

    useEffect(() => {
        loadDailyMeals();
    }, [selectedDate]);

    // Status Ring Logic
    const getStatusColor = (consumed: number) => {
        const target = user?.daily_calorie_goal || 2500;
        const ratio = consumed / target;
        if (ratio > 1.1) return 'border-red-500 text-red-500'; 
        if (ratio > 0.9) return 'border-orange-400 text-orange-500'; 
        return 'border-green-500 text-green-600'; 
    };

    const prevMonth = () => setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
    const nextMonth = () => setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));

    // Calendar Grid Generation
    while (day <= endDate) {
        for (let i = 0; i < 7; i++) {
            formattedDate = format(day, dateFormat);
            const cloneDay = day;
            
            const dateKey = cloneDay.toDateString();
            const calories = calendarData[dateKey] || 0;
            const goal = user?.daily_calorie_goal || 2500;
            const percentage = Math.min((calories / goal) * 100, 100);
            const isOver = calories > goal;
            
            // Color selection
            const ringColor = isOver ? '#EF4444' : '#10B981'; // Red if over, Green if under
            const trackColor = '#F3F4F6'; // Light gray background
            
            const statusClass = calories > 0 ? '' : 'text-gray-700';
            const isSelected = isSameDay(day, selectedDate);
            const isCurrentMonth = day.getMonth() === monthStart.getMonth();

            days.push(
                <div key={day.toString()} className="flex flex-col items-center justify-center p-1">
                    <button 
                        onClick={() => setSelectedDate(cloneDay)}
                        className={`
                            w-10 h-10 rounded-full flex items-center justify-center transition-all relative
                            ${isSelected ? 'shadow-lg scale-110 z-10' : ''}
                            ${!isCurrentMonth ? 'opacity-30' : ''}
                        `}
                        style={{
                            background: calories > 0 
                                ? `conic-gradient(${ringColor} ${percentage}%, ${trackColor} ${percentage}%)` 
                                : 'transparent'
                        }}
                    >
                        {/* Inner Circle to create Ring effect */}
                        <div className={`
                            w-[80%] h-[80%] rounded-full flex items-center justify-center
                            ${isSelected ? 'bg-primary text-white' : 'bg-white text-gray-800'}
                        `}>
                            <span className="text-xs font-black">{formattedDate}</span>
                        </div>
                    </button>
                    {calories > 0 && isCurrentMonth && (
                        <span className="text-[8px] font-bold text-gray-400 mt-1">{Math.round(calories)}</span>
                    )}
                </div>
            );
            day = addDays(day, 1);
        }
        rows.push(<div className="grid grid-cols-7 gap-1" key={day.toString()}>{days}</div>);
        days = [];
    }

    return (
        <div className="p-6 pb-24">
            <h1 className="text-2xl font-bold mb-6">Calendar</h1>

            <Card className="mb-6 overflow-hidden">
                <div className="flex justify-between items-center mb-6 p-2">
                    <button onClick={prevMonth} className="p-2 hover:bg-gray-100 rounded-full"><ChevronLeft /></button>
                    <h2 className="font-bold text-lg">{format(currentDate, 'MMMM yyyy')}</h2>
                    <button onClick={nextMonth} className="p-2 hover:bg-gray-100 rounded-full"><ChevronRight /></button>
                </div>

                <div className="grid grid-cols-7 gap-1 text-center text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                    {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(d => <div key={d}>{d}</div>)}
                </div>

                <div className="space-y-2">
                    {rows}
                </div>
            </Card>

            <div>
                <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
                    {isToday(selectedDate) ? "Today's Summary" : format(selectedDate, "MMMM d 'Summary'")}
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                        (calendarData[selectedDate.toDateString()] || 0) > (user?.daily_calorie_goal || 2500) 
                        ? 'bg-red-100 text-red-600' 
                        : 'bg-green-100 text-green-600'
                    }`}>
                        {calendarData[selectedDate.toDateString()] || 0} kcal
                    </span>
                </h3>
                
                <div className="space-y-3">
                    {dailyMeals.length > 0 ? (
                        dailyMeals.map((meal) => (
                            <div 
                                key={meal.id} 
                                onClick={() => navigate(`/meal/${meal.id}`, { state: { meal } })}
                                className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm flex gap-4 items-center active:scale-95 transition-transform cursor-pointer"
                            >
                                <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center flex-shrink-0 overflow-hidden">
                                    {meal.image_paths && meal.image_paths[0] ? (
                                        <img src={meal.image_paths[0]} className="w-full h-full object-cover" />
                                    ) : (
                                        <Utensils size={20} className="text-gray-400" />
                                    )}
                                </div>
                                <div className="flex-1">
                                    <div className="flex justify-between">
                                        <h4 className="font-bold text-gray-900 text-sm">{meal.name}</h4>
                                        <span className="text-xs font-bold text-primary">{Math.round(meal.total_calories)} kcal</span>
                                    </div>
                                    <div className="text-xs text-gray-400 mt-1">
                                        {format(new Date(meal.timestamp), 'h:mm a')} • {Math.round(meal.total_protein)}g Pro
                                    </div>
                                </div>
                                <ChevronRight size={16} className="text-gray-300" />
                            </div>
                        ))
                    ) : (
                        <div className="text-center py-8 text-gray-400 bg-white rounded-2xl border border-dashed border-gray-200">
                            <p className="text-sm">No meals recorded for this day.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
