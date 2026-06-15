import React, { createContext, useContext, useState } from 'react';

interface AppState {
    theme: 'dark' | 'light';
    toggleTheme: () => void;
    dataset: string;
    setDataset: (d: string) => void;
    selectedModels: string[];
    setSelectedModels: (m: string[]) => void;
    predictionHistory: any[];
    addPrediction: (p: any) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
    const [theme, setTheme] = useState<'dark' | 'light'>('dark');
    const [dataset, setDataset] = useState('breast_cancer');
    const [selectedModels, setSelectedModels] = useState<string[]>(['classical_dnn', 'hqnn']);
    const [predictionHistory, setPredictionHistory] = useState<any[]>([]);

    const toggleTheme = () => {
        const next = theme === 'dark' ? 'light' : 'dark';
        setTheme(next);
        document.documentElement.setAttribute('data-theme', next);
    };

    const addPrediction = (p: any) => {
        setPredictionHistory(prev => [p, ...prev.slice(0, 49)]);
    };

    return (
        <AppContext.Provider value={{
            theme, toggleTheme,
            dataset, setDataset,
            selectedModels, setSelectedModels,
            predictionHistory, addPrediction,
        }}>
            {children}
        </AppContext.Provider>
    );
}

export function useApp() {
    const ctx = useContext(AppContext);
    if (!ctx) throw new Error('useApp must be inside AppProvider');
    return ctx;
}
