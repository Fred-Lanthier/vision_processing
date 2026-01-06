import React from 'react';
import { Card } from '../components/UI';
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight } from 'lucide-react';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, isSameMonth, isToday, isSameDay } from 'date-fns';

export default function CalendarPage() {
    const [currentDate, setCurrentDate] = React.useState(new Date());
    const [selectedDate, setSelectedDate] = React.useState(new Date());
    
    // Mock Data for "Activity Dots"
    const activeDays = [1, 3, 4, 7, 8, 12, 15, 20]; // Days of current month with data

    const days = eachDayOfInterval({
        start: startOfMonth(currentDate),
        end: endOfMonth(currentDate),
    });

    const prevMonth = () => {
        setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
    };
    
    const nextMonth = () => {
        setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));
    };

    return (
        <div className="p-6">
            <h1 className="text-2xl font-bold mb-6">Activity Log</h1>

            <Card className="mb-6">
                <div className="flex justify-between items-center mb-6">
                    <button onClick={prevMonth} className="p-2 hover:bg-gray-100 rounded-full"><ChevronLeft /></button>
                    <h2 className="font-bold text-lg">{format(currentDate, 'MMMM yyyy')}</h2>
                    <button onClick={nextMonth} className="p-2 hover:bg-gray-100 rounded-full"><ChevronRight /></button>
                </div>

                <div className="grid grid-cols-7 gap-2 text-center text-sm font-medium text-gray-400 mb-2">
                    {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map(d => <div key={d}>{d}</div>)}
                </div>

                <div className="grid grid-cols-7 gap-2">
                    {days.map((day, i) => {
                        const hasActivity = activeDays.includes(day.getDate()); // Mock check
                        const isSelected = isSameDay(day, selectedDate);
                        
                        return (
                            <button 
                                key={i}
                                onClick={() => setSelectedDate(day)}
                                className={`
                                    aspect-square rounded-xl flex flex-col items-center justify-center relative transition-all
                                    ${isSelected ? 'bg-primary text-white shadow-lg shadow-orange-200' : 'hover:bg-gray-50 text-gray-700'}
                                    ${isToday(day) && !isSelected ? 'text-primary font-bold' : ''}
                                `}
                            >
                                <span className="text-sm">{day.getDate()}</span>
                                {hasActivity && !isSelected && (
                                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-1"></div>
                                )}
                            </button>
                        );
                    })}
                </div>
            </Card>

            <div>
                <h3 className="font-bold text-gray-900 mb-4">
                    {isToday(selectedDate) ? "Today's Log" : format(selectedDate, "MMMM d 'Log'")}
                </h3>
                {/* Reuse Meal List Component or Mock */}
                <div className="text-center py-8 text-gray-400 bg-white rounded-2xl border border-dashed border-gray-200">
                    <p>No activity recorded for this day.</p>
                </div>
            </div>
        </div>
    );
}
