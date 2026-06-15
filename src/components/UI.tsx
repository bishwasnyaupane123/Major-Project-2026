import { useState, useEffect, useRef } from 'react';
import { useApp } from '../context';

interface AnimatedNumberProps {
    value: number;
    decimals?: number;
    suffix?: string;
    duration?: number;
}

export function AnimatedNumber({ value, decimals = 1, suffix = '', duration = 800 }: AnimatedNumberProps) {
    const [display, setDisplay] = useState(0);
    const raf = useRef<number | null>(null);

    useEffect(() => {
        const start = display;
        const end = value;
        const startTime = performance.now();
        const animate = (now: number) => {
            const progress = Math.min((now - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setDisplay(start + (end - start) * eased);
            if (progress < 1) raf.current = requestAnimationFrame(animate);
        };
        raf.current = requestAnimationFrame(animate);
        return () => { if (raf.current) cancelAnimationFrame(raf.current); };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [value]);

    return <span>{display.toFixed(decimals)}{suffix}</span>;
}

interface ConfidenceBarProps {
    value: number; // 0-1
    color?: string;
}

export function ConfidenceBar({ value, color = 'linear-gradient(90deg,#8b5cf6,#3b82f6)' }: ConfidenceBarProps) {
    return (
        <div className="confidence-bar-wrap">
            <div
                className="confidence-bar"
                style={{ width: `${value * 100}%`, background: color }}
            />
        </div>
    );
}

export function ThemeToggle() {
    const { theme, toggleTheme } = useApp();
    return (
        <button
            onClick={toggleTheme}
            className="btn btn-outline btn-sm"
            title="Toggle theme"
            id="theme-toggle-btn"
        >
            {theme === 'dark' ? '☀️' : '🌙'} {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
    );
}

export function TypeBadge({ type }: { type: string }) {
    return (
        <span className={`badge badge-${type}`}>
            {type === 'quantum' ? '⚛️' : '🔷'} {type}
        </span>
    );
}

export function Skeleton({ width = '100%', height = 20, className = '' }: { width?: string | number; height?: number; className?: string }) {
    return (
        <div
            className={`skeleton ${className}`}
            style={{ width, height: `${height}px` }}
        />
    );
}

export function LoadingSpinner({ size = 24, color = 'var(--accent-quantum)' }: { size?: number; color?: string }) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            style={{ animation: 'rotate-slow 1s linear infinite' }}
        >
            <circle cx="12" cy="12" r="10" fill="none" stroke={color} strokeWidth="2.5" strokeDasharray="40 20" />
        </svg>
    );
}

export function MetricChip({ label, value, color }: { label: string; value: string | number; color?: string }) {
    return (
        <div className="metric-chip animate-fade-up">
            <span className="metric-value" style={{ color: color || 'var(--accent-quantum-2)' }}>
                {value}
            </span>
            <span className="metric-label">{label}</span>
        </div>
    );
}

export function WinnerBadge({ modelName }: { modelName: string }) {
    return (
        <div style={{ textAlign: 'center', marginTop: 12 }}>
            <span className="badge badge-winner">
                🏆 Winner: {modelName}
            </span>
        </div>
    );
}

export function CircuitVisualizer({ circuitInfo }: { circuitInfo: any }) {
    if (!circuitInfo) return null;
    const gates = circuitInfo.gates || [];
    const wires = circuitInfo.wires || 4;
    const W = 600, H = Math.max(120, wires * 28 + 40);

    return (
        <div style={{ overflowX: 'auto' }}>
            <svg width={W} height={H} style={{ display: 'block', margin: '0 auto' }}>
                {Array.from({ length: wires }).map((_, i) => (
                    <line key={i} x1={20} y1={30 + i * 28} x2={W - 20} y2={30 + i * 28}
                        stroke="rgba(139,92,246,0.25)" strokeWidth={1.5} />
                ))}
                {gates.map((gate: string, gi: number) => (
                    <g key={gi} style={{ animation: `fadeIn 0.3s ease ${gi * 0.1}s both` }}>
                        <rect
                            x={50 + gi * 90} y={15} width={70} height={H - 30}
                            fill="rgba(139,92,246,0.08)" stroke="rgba(139,92,246,0.3)"
                            strokeWidth={1} rx={6}
                        />
                        <text
                            x={85 + gi * 90} y={H / 2}
                            fill="var(--accent-quantum-2)"
                            fontSize="10"
                            fontFamily="JetBrains Mono, monospace"
                            textAnchor="middle"
                            dominantBaseline="middle"
                        >{gate}</text>
                    </g>
                ))}
                {Array.from({ length: wires }).map((_, i) => (
                    <text key={i} x={12} y={34 + i * 28}
                        fill="var(--text-muted)" fontSize="9"
                        fontFamily="JetBrains Mono, monospace">q{i}</text>
                ))}
            </svg>
            <p style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.78rem', marginTop: 8 }}>
                {circuitInfo.description}
            </p>
        </div>
    );
}
