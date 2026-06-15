import { motion } from 'framer-motion';
import { AnimatedNumber, ConfidenceBar, TypeBadge, CircuitVisualizer } from './UI';
import { PredictionPie, FeatureImportanceChart } from './Charts';

interface ResultCardProps {
    result: any;
    isWinner?: boolean;
    side?: 'left' | 'right';
    circuitInfo?: any;
    showCircuit?: boolean;
}

function formatMetricValue(key: string, val: number): string {
    if (key === 'inference_time_ms') return `${val.toFixed(1)} ms`;
    return `${(val * 100).toFixed(1)}%`;
}

const METRIC_LABELS: Record<string, string> = {
    accuracy: 'Accuracy',
    auc: 'AUC',
    f1: 'F1',
    precision: 'Precision',
    recall: 'Recall',
    silhouette_score: 'Silhouette',
};

export default function ResultCard({
    result,
    isWinner,
    side = 'left',
    circuitInfo,
    showCircuit,
}: ResultCardProps) {
    if (!result) {
        return (
            <motion.div
                className={`card card-${side === 'left' ? 'classical' : 'quantum'}`}
                animate={{ opacity: [0.4, 0.7, 0.4] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                style={{ minHeight: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
                <p className="text-muted text-center">Submit a prediction to see results</p>
            </motion.div>
        );
    }

    if (result.error) {
        return (
            <div className="card" style={{ borderColor: 'rgba(239,68,68,0.3)' }}>
                <p style={{ color: 'var(--accent-red)' }}>⚠️ Error: {result.error}</p>
            </div>
        );
    }

    const cardClass = result.model_type === 'quantum' ? 'card-quantum' : 'card-classical';
    const barColor = result.model_type === 'quantum'
        ? 'linear-gradient(90deg,#7c3aed,#8b5cf6)'
        : 'linear-gradient(90deg,#1d4ed8,#3b82f6)';

    const metrics = result.metrics || {};
    const visibleMetrics = Object.entries(metrics).filter(([k]) => METRIC_LABELS[k]);

    return (
        <motion.div
            className={`card ${cardClass}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
        >
            {/* Header */}
            <div className="flex justify-between items-center" style={{ marginBottom: 20 }}>
                <div>
                    <TypeBadge type={result.model_type} />
                    <h4 style={{ marginTop: 6, color: 'var(--text-primary)' }}>{result.model_name}</h4>
                </div>
                {isWinner && <span style={{ fontSize: '1.8rem' }}>🏆</span>}
            </div>

            {/* Prediction */}
            <div style={{
                background: result.model_type === 'quantum' ? 'rgba(139,92,246,0.08)' : 'rgba(59,130,246,0.08)',
                borderRadius: 'var(--radius-md)',
                padding: '16px 20px',
                marginBottom: 16,
                textAlign: 'center',
                border: `1px solid ${result.model_type === 'quantum' ? 'rgba(139,92,246,0.2)' : 'rgba(59,130,246,0.2)'}`,
            }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Prediction</p>
                <p style={{
                    fontSize: '1.5rem',
                    fontWeight: 800,
                    fontFamily: 'Space Grotesk',
                    color: result.model_type === 'quantum' ? 'var(--accent-quantum-2)' : 'var(--accent-classical-2)',
                }}>
                    {result.prediction}
                </p>
            </div>

            {/* Confidence */}
            <div style={{ marginBottom: 16 }}>
                <div className="flex justify-between" style={{ marginBottom: 6 }}>
                    <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Confidence</span>
                    <span style={{ fontFamily: 'JetBrains Mono', fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                        <AnimatedNumber value={(result.confidence || 0) * 100} decimals={1} suffix="%" />
                    </span>
                </div>
                <ConfidenceBar value={result.confidence || 0} color={barColor} />
            </div>

            {/* Inference time */}
            <div className="flex justify-between items-center" style={{ marginBottom: 16 }}>
                <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>⏱ Inference Time</span>
                <span style={{ fontFamily: 'JetBrains Mono', fontSize: '0.88rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                    <AnimatedNumber value={result.inference_time_ms || 0} decimals={1} suffix=" ms" />
                </span>
            </div>

            {/* Metrics */}
            {visibleMetrics.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                    <p className="text-xs text-muted" style={{ marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        Validation Metrics
                    </p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {visibleMetrics.map(([key, val]) => (
                            <div key={key} style={{
                                background: 'var(--bg-glass)',
                                border: '1px solid var(--border-subtle)',
                                borderRadius: 'var(--radius-sm)',
                                padding: '5px 10px',
                                fontSize: '0.78rem',
                            }}>
                                <span style={{ color: 'var(--text-muted)' }}>{METRIC_LABELS[key]}: </span>
                                <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                                    {formatMetricValue(key, val as number)}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Confidence chart */}
            <PredictionPie
                result={result}
                color={result.model_type === 'quantum' ? '#8b5cf6' : '#3b82f6'}
            />

            {/* Feature importance (HQNN) */}
            {result.feature_importance && (
                <div style={{ marginTop: 16 }}>
                    <FeatureImportanceChart importance={result.feature_importance} />
                </div>
            )}

            {/* Attention weights (Transformer) */}
            {result.attention_weights && Object.keys(result.attention_weights).length > 0 && (
                <div style={{ marginTop: 16 }}>
                    <p className="text-xs text-muted" style={{ marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        Token Attention
                    </p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {Object.entries(result.attention_weights).slice(0, 15).map(([token, weight]) => (
                            <span
                                key={token}
                                className="attention-token"
                                style={{
                                    background: `rgba(139,92,246,${Math.min(weight as number * 3, 0.6)})`,
                                    border: '1px solid rgba(139,92,246,0.2)',
                                    padding: '2px 7px',
                                    borderRadius: 4,
                                    fontSize: '0.78rem',
                                    fontFamily: 'JetBrains Mono',
                                    color: 'var(--text-primary)',
                                }}
                                title={`Weight: ${(weight as number).toFixed(4)}`}
                            >
                                {token}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Circuit info */}
            {result.circuit_info && (
                <div style={{ marginTop: 16, padding: '10px 12px', background: 'rgba(139,92,246,0.06)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(139,92,246,0.2)', overflow: 'hidden', wordBreak: 'break-word' }}>
                    <p className="text-xs" style={{ color: 'var(--accent-quantum-2)', fontWeight: 600, marginBottom: 4 }}>
                        ⚛️ Quantum Circuit: {result.circuit_info.circuit_type}
                    </p>
                    <p className="text-xs text-muted" style={{ overflowWrap: 'break-word' }}>{result.circuit_info.description}</p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)', marginTop: 4 }}>
                        {result.circuit_info.n_qubits} qubits · {result.circuit_info.n_layers} layers
                    </p>
                </div>
            )}

            {/* Circuit visualizer */}
            {showCircuit && circuitInfo && (
                <div style={{ marginTop: 16 }}>
                    <p className="text-xs text-muted" style={{ marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        Circuit Diagram
                    </p>
                    <CircuitVisualizer circuitInfo={circuitInfo} />
                </div>
            )}
        </motion.div>
    );
}
