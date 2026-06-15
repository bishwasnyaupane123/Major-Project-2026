
import {
    Chart as ChartJS,
    CategoryScale, LinearScale, BarElement, LineElement,
    PointElement, ArcElement, Title, Tooltip, Legend, Filler,
    RadialLinearScale,
} from 'chart.js';
import { Bar, Line, Doughnut } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale, LinearScale, BarElement, LineElement,
    PointElement, ArcElement, Title, Tooltip, Legend, Filler,
    RadialLinearScale,
);

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } },
        tooltip: {
            backgroundColor: 'rgba(15,15,35,0.95)',
            borderColor: 'rgba(139,92,246,0.3)',
            borderWidth: 1,
            titleColor: '#f1f5f9',
            bodyColor: '#94a3b8',
        },
    },
    scales: {
        x: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
        },
        y: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
        },
    },
    animation: { duration: 800, easing: 'easeInOutQuart' as const },
};

interface ComparisonBarProps {
    classicalResult?: any;
    quantumResult?: any;
}

export function ComparisonBarChart({ classicalResult, quantumResult }: ComparisonBarProps) {
    if (!classicalResult && !quantumResult) return null;
    const cm = classicalResult?.metrics || {};
    const qm = quantumResult?.metrics || {};
    const labels = ['Accuracy', 'Confidence', 'F1 Score', 'AUC'];
    const getData = (m: any, result: any) => [
        (m.accuracy || 0) * 100,
        ((result?.confidence || 0)) * 100,
        (m.f1 || m.silhouette_score || 0) * 100,
        (m.auc || 0) * 100,
    ];

    const data = {
        labels,
        datasets: [
            {
                label: classicalResult?.model_name || 'Classical',
                data: getData(cm, classicalResult),
                backgroundColor: 'rgba(59,130,246,0.7)',
                borderColor: '#3b82f6',
                borderWidth: 2,
                borderRadius: 6,
            },
            {
                label: quantumResult?.model_name || 'Quantum',
                data: getData(qm, quantumResult),
                backgroundColor: 'rgba(139,92,246,0.7)',
                borderColor: '#8b5cf6',
                borderWidth: 2,
                borderRadius: 6,
            },
        ],
    };

    return (
        <div className="chart-container" style={{ height: 220 }}>
            <Bar data={data} options={{ ...CHART_DEFAULTS, plugins: { ...CHART_DEFAULTS.plugins, title: { display: true, text: 'Model Comparison', color: '#f1f5f9', font: { size: 13, family: 'Space Grotesk, sans-serif' } } } }} />
        </div>
    );
}

interface InferenceLineProps {
    history: { time: number; classical_ms: number; quantum_ms: number }[];
}

export function InferenceLineChart({ history }: InferenceLineProps) {
    if (!history.length) return null;
    const labels = history.map((_, i) => `#${i + 1}`);
    const data = {
        labels,
        datasets: [
            {
                label: 'Classical (ms)',
                data: history.map(h => h.classical_ms),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59,130,246,0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: '#3b82f6',
            },
            {
                label: 'Quantum (ms)',
                data: history.map(h => h.quantum_ms),
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139,92,246,0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: '#8b5cf6',
            },
        ],
    };

    return (
        <div className="chart-container" style={{ height: 200 }}>
            <Line
                data={data}
                options={{ ...CHART_DEFAULTS, plugins: { ...CHART_DEFAULTS.plugins, title: { display: true, text: 'Inference Time History', color: '#f1f5f9', font: { size: 13, family: 'Space Grotesk, sans-serif' } } } }}
            />
        </div>
    );
}

interface PredictionPieProps {
    result: any;
    title?: string;
    color?: string;
}

export function PredictionPie({ result, title, color = '#8b5cf6' }: PredictionPieProps) {
    if (!result?.confidence) return null;
    const conf = result.confidence;
    const data = {
        labels: [result.prediction, 'Other'],
        datasets: [{
            data: [conf * 100, (1 - conf) * 100],
            backgroundColor: [color, 'rgba(255,255,255,0.06)'],
            borderColor: [color, 'rgba(255,255,255,0.1)'],
            borderWidth: 2,
        }],
    };

    return (
        <div className="chart-container" style={{ height: 160 }}>
            <Doughnut
                data={data}
                options={{
                    ...CHART_DEFAULTS,
                    cutout: '70%',
                    plugins: {
                        ...CHART_DEFAULTS.plugins,
                        title: { display: !!title, text: title || '', color: '#f1f5f9', font: { size: 12, family: 'Space Grotesk, sans-serif' } },
                    },
                    scales: undefined as any,
                }}
            />
        </div>
    );
}

interface BattleBarProps {
    scores: Record<string, number>;
    modelNames: Record<string, string>;
}

export function BattleBarChart({ scores, modelNames }: BattleBarProps) {
    const ids = Object.keys(scores);
    const data = {
        labels: ids.map(id => modelNames[id] || id),
        datasets: [{
            label: 'Wins',
            data: ids.map(id => scores[id]),
            backgroundColor: ids.map((_, i) => i === 0 ? 'rgba(59,130,246,0.8)' : 'rgba(139,92,246,0.8)'),
            borderColor: ids.map((_, i) => i === 0 ? '#3b82f6' : '#8b5cf6'),
            borderWidth: 2,
            borderRadius: 8,
        }],
    };

    return (
        <div className="chart-container" style={{ height: 180 }}>
            <Bar
                data={data}
                options={{
                    ...CHART_DEFAULTS,
                    indexAxis: 'y' as const,
                    plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } },
                }}
            />
        </div>
    );
}

export function FeatureImportanceChart({ importance }: { importance: Record<string, number> }) {
    const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const data = {
        labels: sorted.map(([k]) => k.replace(/_/g, ' ')),
        datasets: [{
            label: 'Importance',
            data: sorted.map(([, v]) => v * 100),
            backgroundColor: sorted.map((_, i) => `rgba(139,92,246,${0.9 - i * 0.07})`),
            borderColor: '#8b5cf6',
            borderWidth: 1,
            borderRadius: 4,
        }],
    };

    return (
        <div className="chart-container" style={{ height: 220 }}>
            <Bar
                data={data}
                options={{
                    ...CHART_DEFAULTS,
                    indexAxis: 'y' as const,
                    plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false }, title: { display: true, text: 'Feature Importance (HQNN)', color: '#f1f5f9', font: { size: 13, family: 'Space Grotesk, sans-serif' } } },
                }}
            />
        </div>
    );
}
