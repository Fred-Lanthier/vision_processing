/** 
 * Mock API Service - To be replaced with real calls to backend
 */
import axios from 'axios';

// En mode Production (servi par Backend), on utilise des chemins relatifs
const API_URL = ''; 

const api = axios.create({
    baseURL: API_URL,
});


// Add token to requests
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

export const authService = {
    login: async (username, password) => {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);
        const res = await api.post('/token', formData);
        if (res.data.access_token) {
            localStorage.setItem('token', res.data.access_token);
        }
        return res.data;
    },
    register: async (userData) => {
        const res = await api.post('/users/', userData);
        return res.data;
    },
    getMe: async () => {
        const res = await api.get('/users/me/');
        return res.data;
    },
    logout: () => {
        localStorage.removeItem('token');
    }
};

export const mealService = {
    scan: async (files) => {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        const res = await api.post('/meals/scan/', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return res.data;
    },
    getHistory: async (skip = 0, limit = 50) => {
        const res = await api.get(`/meals/?skip=${skip}&limit=${limit}`);
        return res.data;
    },
    getMealsByDate: async (date) => {
        const res = await api.get(`/meals/?date=${date}`);
        return res.data;
    }
};

export const userService = {
    updateProfile: async (data) => {
        const res = await api.put('/users/me/', data);
        return res.data;
    },
    addWeight: async (weight_kg, date = null) => {
        const payload = { 
            weight_kg, 
            date: date || new Date().toISOString() 
        };
        const res = await api.post('/users/me/weight', payload);
        return res.data;
    },
    getWeightHistory: async () => {
        const res = await api.get('/users/me/weight');
        return res.data;
    }
};

export const coachService = {
    getSuggestions: async () => {
        const res = await api.get('/coach/suggest');
        return res.data;
    }
};

export default api;