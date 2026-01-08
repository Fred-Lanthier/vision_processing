import { create } from 'zustand';
import { authService } from '../services/api';

interface User {
    id: number;
    username: string;
    daily_calorie_goal: number;
    current_streak: number;
    weight_kg: number;
    target_weight_kg: number;
    height_cm: number;
    objective: string;
    dietary_likes: string[];
    dietary_dislikes: string[];
}

interface AuthState {
    user: User | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    login: (u: string, p: string) => Promise<void>;
    register: (data: any) => Promise<void>;
    logout: () => void;
    checkAuth: () => Promise<void>;
    setUser: (user: User) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
    user: null,
    isAuthenticated: false,
    isLoading: true,
    setUser: (user) => set({ user }),
    login: async (u, p) => {
        await authService.login(u, p);
        const user = await authService.getMe();
        set({ user, isAuthenticated: true });
    },
    register: async (data) => {
        await authService.register(data);
        // Auto login after register? Or redirect.
    },
    logout: () => {
        authService.logout();
        set({ user: null, isAuthenticated: false });
    },
    checkAuth: async () => {
        try {
            const token = localStorage.getItem('token');
            if (token) {
                const user = await authService.getMe();
                set({ user, isAuthenticated: true });
            }
        } catch (e) {
            localStorage.removeItem('token');
            set({ user: null, isAuthenticated: false });
        } finally {
            set({ isLoading: false });
        }
    }
}));
