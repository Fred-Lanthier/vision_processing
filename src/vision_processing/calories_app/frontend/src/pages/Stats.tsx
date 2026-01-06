import React from 'react';
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Card } from '../components/UI';
import { TrendingUp, Award } from 'lucide-react';

const data = [
  { name: 'M', kcal: 2100 },
  { name: 'T', kcal: 2300 },
  { name: 'W', kcal: 1950 },
  { name: 'T', kcal: 2600 },
  { name: 'F', kcal: 2400 },
  { name: 'S', kcal: 2800 },
  { name: 'S', kcal: 2200 },
];

export default function StatsPage() {
    return (
        <div className="p-6 space-y-6">
            <h1 className="text-2xl font-bold">Your Progress</h1>

            <div className="grid grid-cols-2 gap-4">
                 <Card className="flex flex-col items-center justify-center py-6">
                    <div className="bg-green-100 p-3 rounded-full mb-3 text-green-600">
                        <TrendingUp size={24} />
                    </div>
                    <span className="text-2xl font-black text-gray-900">2,350</span>
                    <span className="text-xs text-gray-400 uppercase font-bold">Avg. Calories</span>
                 </Card>
                 <Card className="flex flex-col items-center justify-center py-6">
                    <div className="bg-yellow-100 p-3 rounded-full mb-3 text-yellow-600">
                        <Award size={24} />
                    </div>
                    <span className="text-2xl font-black text-gray-900">12</span>
                    <span className="text-xs text-gray-400 uppercase font-bold">Best Streak</span>
                 </Card>
            </div>

            <Card>
                <h3 className="font-bold mb-6">Weekly Calorie Trend</h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data}>
                            <XAxis 
                                dataKey="name" 
                                axisLine={false} 
                                tickLine={false} 
                                tick={{fill: '#9ca3af', fontSize: 12}} 
                                dy={10}
                            />
                            <Tooltip 
                                cursor={{fill: 'transparent'}}
                                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                            />
                            <Bar 
                                dataKey="kcal" 
                                fill="#ff4500" 
                                radius={[6, 6, 6, 6]} 
                                barSize={12}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </Card>
        </div>
    );
}
