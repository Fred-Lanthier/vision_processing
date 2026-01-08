import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import Layout from './components/Layout';
import LoginPage from './pages/Auth';
import Dashboard from './pages/Dashboard';
import ScanPage from './pages/Scan';
import CalendarPage from './pages/Calendar';
import StatsPage from './pages/Stats';
import ProfilePage from './pages/Profile';
import MealDetailsPage from './pages/MealDetails';
import CoachPage from './pages/Coach';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
    const { isAuthenticated, isLoading } = useAuthStore();
    
    if (isLoading) return <div className="flex items-center justify-center h-screen">Loading...</div>;
    
    if (!isAuthenticated) {
        return <Navigate to="/login" />;
    }
    
    return <Layout>{children}</Layout>;
}

class ErrorBoundary extends React.Component<{children: React.ReactNode}, {hasError: boolean, error: any}> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }

  componentDidCatch(error: any, errorInfo: any) {
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <div className="p-10 text-red-600"><h1>Something went wrong.</h1><pre>{JSON.stringify(this.state.error?.message, null, 2)}</pre></div>;
    }

    return this.props.children;
  }
}

export default function App() {
    const checkAuth = useAuthStore(state => state.checkAuth);

    useEffect(() => {
        checkAuth();
    }, []);

    return (
        <ErrorBoundary>
            <BrowserRouter>
                <Routes>
                    <Route path="/login" element={<LoginPage />} />
                    
                    <Route path="/" element={
                        <ProtectedRoute>
                            <Dashboard />
                        </ProtectedRoute>
                    } />
                    
                    <Route path="/scan" element={
                        <ProtectedRoute>
                            <ScanPage />
                        </ProtectedRoute>
                    } />
                    
                    <Route path="/coach" element={
                        <ProtectedRoute>
                            <CoachPage />
                        </ProtectedRoute>
                    } />

                    <Route path="/calendar" element={
                        <ProtectedRoute>
                            <CalendarPage />
                        </ProtectedRoute>
                    } />

                    <Route path="/stats" element={
                        <ProtectedRoute>
                            <StatsPage />
                        </ProtectedRoute>
                    } />
                    
                    <Route path="/meal/:id" element={
                        <ProtectedRoute>
                            <MealDetailsPage />
                        </ProtectedRoute>
                    } />
                    
                    <Route path="/profile" element={
                        <ProtectedRoute>
                            <ProfilePage />
                        </ProtectedRoute>
                    } />
                </Routes>
            </BrowserRouter>
        </ErrorBoundary>
    );
}