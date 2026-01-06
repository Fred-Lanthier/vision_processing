import React from 'react';
import { useAuthStore } from '../store/authStore';
import { Button, Card } from '../components/UI';
import { User, Settings, LogOut, Shield, Bell, HelpCircle } from 'lucide-react';

export default function ProfilePage() {
    const { user, logout } = useAuthStore();

    return (
        <div className="p-6 space-y-6">
            <h1 className="text-2xl font-bold">Profile</h1>

            {/* Profile Header */}
            <div className="flex flex-col items-center py-4">
                <div className="w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center text-primary mb-4 border-4 border-white shadow-lg">
                    <User size={48} />
                </div>
                <h2 className="text-xl font-bold text-gray-900">{user?.username}</h2>
                <p className="text-gray-500 text-sm">Member since Jan 2026</p>
            </div>

            {/* Stats Card */}
            <div className="grid grid-cols-2 gap-4">
                <Card className="text-center">
                    <span className="block text-2xl font-black text-primary">{user?.current_streak || 0}</span>
                    <span className="text-[10px] text-gray-400 uppercase font-bold">Day Streak</span>
                </Card>
                <Card className="text-center">
                    <span className="block text-2xl font-black text-secondary">{user?.daily_calorie_goal || 2500}</span>
                    <span className="text-[10px] text-gray-400 uppercase font-bold">Calorie Goal</span>
                </Card>
            </div>

            {/* Settings List */}
            <div className="space-y-2">
                <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider ml-1 mb-2">Settings</h3>
                
                <SettingsItem icon={Settings} label="Account Settings" />
                <SettingsItem icon={Bell} label="Notifications" />
                <SettingsItem icon={Shield} label="Privacy & Security" />
                <SettingsItem icon={HelpCircle} label="Help & Support" />
            </div>

            <Button 
                variant="secondary" 
                className="w-full mt-6 text-red-500 bg-red-50 hover:bg-red-100 border-none"
                onClick={logout}
            >
                <LogOut size={20} />
                Log Out
            </Button>
            
            <p className="text-center text-[10px] text-gray-300 uppercase tracking-widest pt-4">
                NutritionAI v1.0.0
            </p>
        </div>
    );
}

function SettingsItem({ icon: Icon, label }: { icon: any, label: string }) {
    return (
        <button className="w-full flex items-center gap-4 p-4 bg-white rounded-2xl border border-gray-100 hover:bg-gray-50 transition-colors">
            <div className="p-2 bg-gray-50 rounded-lg text-gray-500">
                <Icon size={20} />
            </div>
            <span className="font-bold text-gray-700">{label}</span>
            <div className="ml-auto text-gray-300">
                <Settings size={16} className="rotate-90 opacity-0" /> {/* Spacer/Chevron replacement */}
                <span className="text-xl">›</span>
            </div>
        </button>
    );
}
