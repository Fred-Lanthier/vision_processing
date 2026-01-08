import React, { useState, useEffect } from 'react';
import { useAuthStore } from '../store/authStore';
import { Button, Card } from '../components/UI';
import { userService } from '../services/api';
import { User, LogOut, Target, ThumbsUp, ThumbsDown, Save, ChevronRight } from 'lucide-react';

export default function ProfilePage() {
    const { user, setUser, logout } = useAuthStore();
    const [isEditing, setIsEditing] = useState(false);
    
    // Form State
    const [formData, setFormData] = useState({
        weight_kg: user?.weight_kg || 70,
        target_weight_kg: user?.target_weight_kg || 65,
        height_cm: user?.height_cm || 175,
        daily_calorie_goal: user?.daily_calorie_goal || 2500,
        objective: user?.objective || 'maintain',
        likes: (user?.dietary_likes || []).join(', '),
        dislikes: (user?.dietary_dislikes || []).join(', ')
    });

    useEffect(() => {
        if (user) {
            setFormData({
                weight_kg: user.weight_kg,
                target_weight_kg: user.target_weight_kg || 65,
                height_cm: user.height_cm,
                daily_calorie_goal: user.daily_calorie_goal,
                objective: user.objective || 'maintain',
                likes: (user.dietary_likes || []).join(', '),
                dislikes: (user.dietary_dislikes || []).join(', ')
            });
        }
    }, [user]);

    const handleSave = async () => {
        try {
            const updatedUser = await userService.updateProfile({
                weight_kg: Number(formData.weight_kg),
                target_weight_kg: Number(formData.target_weight_kg),
                height_cm: Number(formData.height_cm),
                daily_calorie_goal: Number(formData.daily_calorie_goal),
                objective: formData.objective,
                dietary_likes: formData.likes.split(',').map(s => s.trim()).filter(Boolean),
                dietary_dislikes: formData.dislikes.split(',').map(s => s.trim()).filter(Boolean)
            });
            setUser(updatedUser);
            setIsEditing(false);
        } catch (error) {
            console.error("Failed to update profile", error);
        }
    };

    return (
        <div className="p-6 space-y-6 pb-24">
            <header className="flex justify-between items-center">
                <h1 className="text-2xl font-bold">Your Coach</h1>
                <Button 
                    size="sm" 
                    variant={isEditing ? "primary" : "secondary"}
                    onClick={() => isEditing ? handleSave() : setIsEditing(true)}
                >
                    {isEditing ? <Save size={16} className="mr-2"/> : null}
                    {isEditing ? "Save" : "Edit"}
                </Button>
            </header>

            {/* Profile Header */}
            <div className="flex items-center gap-4 py-2">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center text-primary border-2 border-white shadow-sm">
                    <User size={32} />
                </div>
                <div>
                    <h2 className="text-xl font-bold text-gray-900">{user?.username}</h2>
                    <p className="text-gray-500 text-xs uppercase font-bold tracking-wider">{formData.objective.replace('_', ' ')}</p>
                </div>
            </div>

            {/* Objectives Section */}
            <section className="space-y-3">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider ml-1">Objective</h3>
                <Card className="p-4 space-y-4">
                    <div className="grid grid-cols-3 gap-2">
                        {['lose_weight', 'maintain', 'gain_muscle'].map((obj) => (
                            <button
                                key={obj}
                                disabled={!isEditing}
                                onClick={() => setFormData({...formData, objective: obj})}
                                className={`p-2 rounded-xl text-xs font-bold border transition-all ${
                                    formData.objective === obj 
                                    ? 'bg-primary text-white border-primary shadow-md transform scale-105' 
                                    : 'bg-gray-50 text-gray-500 border-gray-100'
                                }`}
                            >
                                {obj.replace('_', ' ').toUpperCase()}
                            </button>
                        ))}
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 pt-2">
                        <InputGroup label="Weight (kg)" value={formData.weight_kg} 
                            onChange={v => setFormData({...formData, weight_kg: v})} disabled={!isEditing} />
                        <InputGroup label="Target Weight" value={formData.target_weight_kg} 
                            onChange={v => setFormData({...formData, target_weight_kg: v})} disabled={!isEditing} />
                        <InputGroup label="Calorie Target" value={formData.daily_calorie_goal} 
                            onChange={v => setFormData({...formData, daily_calorie_goal: v})} disabled={!isEditing} />
                    </div>
                </Card>
            </section>

            {/* Preferences Section */}
            <section className="space-y-3">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider ml-1">Dietary Preferences</h3>
                <Card className="p-0 overflow-hidden">
                    <div className="p-4 border-b border-gray-100">
                        <div className="flex items-center gap-2 mb-2 text-green-600 font-bold text-sm">
                            <ThumbsUp size={16} /> Likes
                        </div>
                        {isEditing ? (
                            <textarea 
                                className="w-full bg-gray-50 rounded-lg p-2 text-sm"
                                rows={2}
                                value={formData.likes}
                                onChange={(e) => setFormData({...formData, likes: e.target.value})}
                                placeholder="Italian, Spicy, Chicken..."
                            />
                        ) : (
                            <div className="flex flex-wrap gap-2">
                                {formData.likes ? formData.likes.split(',').map((tag, i) => (
                                    <span key={i} className="px-2 py-1 bg-green-50 text-green-700 rounded-lg text-xs font-medium">{tag.trim()}</span>
                                )) : <span className="text-gray-400 text-sm">None specified</span>}
                            </div>
                        )}
                    </div>
                    <div className="p-4">
                        <div className="flex items-center gap-2 mb-2 text-red-500 font-bold text-sm">
                            <ThumbsDown size={16} /> Dislikes
                        </div>
                        {isEditing ? (
                            <textarea 
                                className="w-full bg-gray-50 rounded-lg p-2 text-sm"
                                rows={2}
                                value={formData.dislikes}
                                onChange={(e) => setFormData({...formData, dislikes: e.target.value})}
                                placeholder="Mushrooms, Fish..."
                            />
                        ) : (
                            <div className="flex flex-wrap gap-2">
                                {formData.dislikes ? formData.dislikes.split(',').map((tag, i) => (
                                    <span key={i} className="px-2 py-1 bg-red-50 text-red-700 rounded-lg text-xs font-medium">{tag.trim()}</span>
                                )) : <span className="text-gray-400 text-sm">None specified</span>}
                            </div>
                        )}
                    </div>
                </Card>
            </section>

            <Button 
                variant="secondary" 
                className="w-full mt-6 text-red-500 bg-red-50 hover:bg-red-100 border-none"
                onClick={logout}
            >
                <LogOut size={20} />
                Log Out
            </Button>
        </div>
    );
}

function InputGroup({ label, value, onChange, disabled }: { label: string, value: any, onChange: (v: any) => void, disabled: boolean }) {
    return (
        <div>
            <label className="block text-[10px] text-gray-400 font-bold uppercase mb-1">{label}</label>
            <input 
                type="number" 
                value={value} 
                disabled={disabled}
                onChange={(e) => onChange(e.target.value)}
                className="w-full p-2 bg-gray-50 rounded-xl font-bold text-gray-800 disabled:bg-white disabled:text-gray-900 border border-gray-100 focus:border-primary outline-none transition-colors"
            />
        </div>
    );
}