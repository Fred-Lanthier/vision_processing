import { Link, useLocation } from 'react-router-dom';
import { Home, Calendar, PlusCircle, BarChart2, User } from 'lucide-react';
import clsx from 'clsx';
import { AnimatePresence, motion } from 'framer-motion';
import { useRef, useEffect } from 'react';

const navOrder = ['/', '/calendar', '/scan', '/stats', '/profile'];

export default function Layout({ children }: { children: React.ReactNode }) {
    const location = useLocation();
    const prevIndexRef = useRef(0);
    const currentIndex = navOrder.indexOf(location.pathname);
    
    // Fallback for pages not in nav (like /meal/123 or /coach) -> treat as deeper level
    const effectiveIndex = currentIndex === -1 ? 100 : currentIndex;
    const direction = effectiveIndex > prevIndexRef.current ? 1 : -1;

    useEffect(() => {
        prevIndexRef.current = effectiveIndex;
    }, [location.pathname]);

    const variants = {
        enter: (dir: number) => ({
            x: dir > 0 ? "100%" : "-100%",
            opacity: 0
        }),
        center: {
            x: 0,
            opacity: 1
        },
        exit: (dir: number) => ({
            x: dir > 0 ? "-100%" : "100%",
            opacity: 0
        })
    };

    return (
        <div className="flex flex-col h-screen bg-gray-50 max-w-md mx-auto shadow-2xl overflow-hidden relative">
            {/* Main Content Area */}
            <main className="flex-1 overflow-y-auto pb-20 scrollbar-hide overflow-x-hidden">
                <AnimatePresence mode="popLayout" custom={direction}>
                    <motion.div
                        key={location.pathname}
                        custom={direction}
                        variants={variants}
                        initial="enter"
                        animate="center"
                        exit="exit"
                        transition={{ type: "spring", stiffness: 260, damping: 20 }}
                        className="h-full"
                    >
                        {children}
                    </motion.div>
                </AnimatePresence>
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
