import React from 'react';
import clsx from 'clsx';
import { Loader2 } from 'lucide-react';

export function Button({ 
    children, 
    onClick, 
    variant = 'primary', 
    className, 
    isLoading,
    type = 'button'
}: any) {
    const variants = {
        primary: "bg-primary text-white hover:bg-orange-600 shadow-md",
        secondary: "bg-gray-100 text-gray-800 hover:bg-gray-200",
        outline: "border-2 border-primary text-primary hover:bg-orange-50",
        ghost: "text-gray-500 hover:bg-gray-100"
    };

    return (
        <button 
            type={type}
            disabled={isLoading}
            onClick={onClick}
            className={clsx(
                "px-4 py-3 rounded-xl font-bold transition-all active:scale-95 flex items-center justify-center gap-2",
                variants[variant],
                isLoading && "opacity-70 cursor-not-allowed",
                className
            )}
        >
            {isLoading && <Loader2 className="animate-spin" size={20} />}
            {children}
        </button>
    );
}

export function Card({ children, className }: any) {
    return (
        <div className={clsx("bg-white rounded-2xl shadow-sm border border-gray-100 p-5", className)}>
            {children}
        </div>
    );
}

export function Input({ label, ...props }: any) {
    return (
        <div className="flex flex-col gap-2">
            {label && <label className="text-sm font-semibold text-gray-700 ml-1">{label}</label>}
            <input 
                {...props}
                className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all font-medium text-gray-800"
            />
        </div>
    );
}
