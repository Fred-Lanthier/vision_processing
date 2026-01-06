import React, { useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { useNavigate } from 'react-router-dom';
import { Button, Input, Card } from '../components/UI';
import { Target, Lock, Mail, User } from 'lucide-react';

export default function LoginPage() {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({ username: '', password: '', email: '', height: 175, weight: 70, goal: 2500 });
    const { login, register, isLoading } = useAuthStore();
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            if (isLogin) {
                await login(formData.username, formData.password);
            } else {
                await register({
                    username: formData.username,
                    email: formData.email,
                    password: formData.password,
                    height_cm: formData.height,
                    weight_kg: formData.weight,
                    daily_calorie_goal: formData.goal
                });
                await login(formData.username, formData.password);
            }
            navigate('/');
        } catch (error: any) {
            console.error(error);
            const msg = error.response?.data?.detail || "Connection to server failed. Is backend running?";
            alert(`Auth Failed: ${msg}`);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col justify-center items-center p-6 text-center max-w-md mx-auto">
            <div className="mb-8 animate-slide-up">
                <div className="bg-primary/10 p-4 rounded-full inline-block mb-4">
                    <Target size={48} className="text-primary" />
                </div>
                <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">NutritionAI</h1>
                <p className="text-gray-500 font-medium">Fuel your ambition.</p>
            </div>

            <Card className="w-full animate-fade-in">
                <form onSubmit={handleSubmit} className="flex flex-col gap-4 text-left">
                    <h2 className="text-xl font-bold mb-2">{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
                    
                    <Input 
                        placeholder="Username" 
                        value={formData.username}
                        onChange={(e: any) => setFormData({...formData, username: e.target.value})}
                    />
                    
                    {!isLogin && (
                        <Input 
                            placeholder="Email" 
                            type="email"
                            value={formData.email}
                            onChange={(e: any) => setFormData({...formData, email: e.target.value})}
                        />
                    )}
                    
                    <Input 
                        placeholder="Password" 
                        type="password" 
                        value={formData.password}
                        onChange={(e: any) => setFormData({...formData, password: e.target.value})}
                    />

                    {!isLogin && (
                        <div className="grid grid-cols-2 gap-2">
                             <Input 
                                placeholder="Height (cm)" 
                                type="number"
                                value={formData.height}
                                onChange={(e: any) => setFormData({...formData, height: Number(e.target.value)})}
                            />
                             <Input 
                                placeholder="Weight (kg)" 
                                type="number"
                                value={formData.weight}
                                onChange={(e: any) => setFormData({...formData, weight: Number(e.target.value)})}
                            />
                        </div>
                    )}

                    <Button isLoading={isLoading} type="submit" className="mt-2">
                        {isLogin ? 'Log In' : 'Join Now'}
                    </Button>
                </form>
            </Card>

            <p className="mt-6 text-sm text-gray-500">
                {isLogin ? "No account?" : "Already have an account?"} {' '}
                <button 
                    onClick={() => setIsLogin(!isLogin)}
                    className="text-primary font-bold hover:underline"
                >
                    {isLogin ? 'Sign up' : 'Log in'}
                </button>
            </p>
        </div>
    );
}
