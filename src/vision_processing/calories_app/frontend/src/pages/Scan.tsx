import React, { useState, useRef } from 'react';
import { Camera, Upload, X, Check } from 'lucide-react';
import { Button, Card } from '../components/UI';
import { useNavigate } from 'react-router-dom';
import { mealService } from '../services/api';
import heic2any from 'heic2any';

export default function ScanPage() {
    const navigate = useNavigate();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
    const [previews, setPreviews] = useState<any[]>([]);
    const [isScanning, setIsScanning] = useState(false);
    const [isConverting, setIsConverting] = useState(false);

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            const rawFiles = Array.from(e.target.files);
            setIsConverting(true);
            
            const processedFiles: File[] = [];

            try {
                for (const file of rawFiles) {
                    // Check if HEIC
                    if (file.type === "image/heic" || file.type === "image/heif" || file.name.toLowerCase().endsWith('.heic')) {
                        console.log(`Converting HEIC: ${file.name}`);
                        try {
                            const convertedBlob = await heic2any({
                                blob: file,
                                toType: "image/jpeg",
                                quality: 0.8
                            });
                            
                            const conversionResult = Array.isArray(convertedBlob) ? convertedBlob[0] : convertedBlob;
                            const newFile = new File([conversionResult], file.name.replace(/\.heic$/i, ".jpg"), {
                                type: "image/jpeg",
                                lastModified: new Date().getTime()
                            });
                            processedFiles.push(newFile);
                        } catch (err) {
                            console.error("HEIC Conversion failed", err);
                            processedFiles.push(file); // Fallback to original
                        }
                    } else {
                        processedFiles.push(file);
                    }
                }

                setSelectedFiles(prev => [...prev, ...processedFiles]);
                
                const newPreviews = processedFiles.map(file => ({
                    url: URL.createObjectURL(file),
                    name: file.name,
                    size: (file.size / 1024 / 1024).toFixed(2) + ' MB',
                    type: file.type
                }));
                
                // @ts-ignore
                setPreviews(prev => [...prev, ...newPreviews]);
            } finally {
                setIsConverting(false);
            }
        }
    };

    const handleScan = async () => {
        if (selectedFiles.length === 0) return;
        
        setIsScanning(true);
        try {
            await mealService.scan(selectedFiles);
            // Success
            navigate('/');
        } catch (e) {
            alert("Scan failed. Check backend.");
        } finally {
            setIsScanning(false);
        }
    };

    const removeImage = (index: number) => {
        const newFiles = [...selectedFiles];
        newFiles.splice(index, 1);
        setSelectedFiles(newFiles);
        
        const newPreviews = [...previews];
        newPreviews.splice(index, 1);
        setPreviews(newPreviews);
    };

    return (
        <div className="flex flex-col h-full relative">
             <div className="flex justify-between items-center p-6 pb-2">
                <h1 className="text-2xl font-bold">Scan Meal</h1>
                <button onClick={() => navigate('/')} className="p-2 bg-gray-100 rounded-full hover:bg-gray-200 transition-colors">
                    <X size={20} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 pt-2 pb-32">
                <p className="text-sm text-gray-500 mb-6">
                    Take multiple photos from different angles for better 3D volume estimation.
                </p>

                {/* Image Grid */}
                <div className="grid grid-cols-2 gap-4">
                    {/* Add Button */}
                    <button 
                        onClick={() => fileInputRef.current?.click()}
                        className="aspect-square rounded-2xl border-2 border-dashed border-gray-300 flex flex-col items-center justify-center text-gray-400 hover:border-primary hover:text-primary transition-colors bg-gray-50 active:scale-95"
                    >
                        <Camera size={32} />
                        <span className="text-sm font-bold mt-2">Add Photo</span>
                    </button>

                    {previews.map((file: any, i) => (
                        <div key={i} className="relative aspect-square rounded-2xl overflow-hidden shadow-sm group animate-fade-in bg-gray-100 border border-gray-200">
                            <img 
                                src={file.url} 
                                className="w-full h-full object-cover" 
                                alt={`Scan ${i}`}
                                onError={(e) => {
                                    // Fallback visual
                                    (e.target as HTMLImageElement).style.opacity = '0.1';
                                }} 
                            />
                            
                            {/* Overlay Info (Debug) */}
                            <div className="absolute bottom-0 left-0 w-full bg-black/50 text-white text-[10px] p-1 truncate">
                                {file.size} - {file.type.split('/')[1] || 'unk'}
                            </div>

                            {/* Error Indicator (if img hidden/broken) */}
                            <div className="absolute inset-0 flex flex-col items-center justify-center -z-10 text-gray-400 p-2 text-center">
                                <span className="text-xs font-mono">{file.name}</span>
                            </div>

                            <button 
                                onClick={() => removeImage(i)}
                                className="absolute top-2 right-2 bg-black/60 text-white p-1.5 rounded-full hover:bg-red-500 transition-colors"
                            >
                                <X size={14} />
                            </button>
                        </div>
                    ))}
                </div>
            </div>

            {/* Bottom Sticky Action Area */}
            <div className="absolute bottom-0 left-0 w-full p-6 bg-gradient-to-t from-white via-white to-transparent pb-24 z-10">
                <Button 
                    onClick={handleScan} 
                    isLoading={isScanning || isConverting}
                    className="w-full py-4 text-lg shadow-xl shadow-orange-200"
                    disabled={selectedFiles.length === 0 || isConverting}
                >
                    {isConverting ? 'Processing Images...' : (selectedFiles.length > 0 ? `Analyze ${selectedFiles.length} Photos` : 'Select Photos')}
                </Button>
            </div>

            <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileSelect} 
                className="hidden" 
                multiple 
                accept="image/*, .heic" 
            />
        </div>
    );
}
