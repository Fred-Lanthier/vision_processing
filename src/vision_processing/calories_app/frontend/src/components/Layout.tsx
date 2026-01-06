import { Link, useLocation } from 'react-router-dom';
import { Home, Calendar, PlusCircle, BarChart2, User } from 'lucide-react';
import clsx from 'clsx';

export default function Layout({ children }: { children: React.ReactNode }) {
    return (
        <div className="flex flex-col h-screen bg-gray-50 max-w-md mx-auto shadow-2xl overflow-hidden relative">
            {/* Main Content Area */}
            <main className="flex-1 overflow-y-auto pb-20 scrollbar-hide">
                {children}
            </main>

            {/* Bottom Navigation Bar */}
            <NavBar />
        </div>
    );
}

function NavBar() {
    const location = useLocation();
    
    const navItems = [
        { icon: Home, label: 'Home', path: '/' },
        { icon: Calendar, label: 'Log', path: '/calendar' },
        { icon: PlusCircle, label: 'Scan', path: '/scan', primary: true },
        { icon: BarChart2, label: 'Stats', path: '/stats' },
        { icon: User, label: 'Profile', path: '/profile' },
    ];

    return (
        <nav className="absolute bottom-0 w-full bg-white border-t border-gray-100 px-6 py-3 flex justify-between items-center z-50">
            {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                const Icon = item.icon;
                
                if (item.primary) {
                    return (
                        <Link key={item.path} to={item.path} className="relative -top-5">
                            <div className="bg-primary hover:bg-orange-600 text-white p-4 rounded-full shadow-lg transition-transform hover:scale-105 active:scale-95 ring-4 ring-gray-50">
                                <Icon size={28} strokeWidth={2.5} />
                            </div>
                        </Link>
                    );
                }

                return (
                    <Link 
                        key={item.path} 
                        to={item.path}
                        className={clsx(
                            "flex flex-col items-center gap-1 transition-colors duration-200",
                            isActive ? "text-primary" : "text-gray-400 hover:text-gray-600"
                        )}
                    >
                        <Icon size={24} strokeWidth={isActive ? 2.5 : 2} />
                        <span className="text-[10px] font-medium">{item.label}</span>
                    </Link>
                );
            })}
        </nav>
    );
}
