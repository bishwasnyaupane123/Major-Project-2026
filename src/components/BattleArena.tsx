import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { runBattle } from '../api';
import { BattleBarChart } from './Charts';

const MODEL_NAMES: Record<string, string> = {
    classical_knn: 'Classical KNN',
    quantum_knn: 'Quantum KNN',
    classical_dnn: 'Classical ANN',
    hqnn: 'Hybrid QNN',
    classical_transformer: 'Classical Transformer',
    quantum_transformer: 'Quantum Transformer',
};

interface BattleArenaProps {
    dataset: string;
    classicalModel: string;
    quantumModel: string;
    onClose: () => void;
}

export default function BattleArena({ dataset, classicalModel, quantumModel, onClose }: BattleArenaProps) {
    const [status, setStatus] = useState<'idle' | 'loading' | 'done'>('idle');
    const [battleData, setBattleData] = useState<any>(null);
    const [revealedRounds, setRevealedRounds] = useState(0);
    const [showVictory, setShowVictory] = useState(false);

    const startBattle = async () => {
        setStatus('loading');
        setBattleData(null);
        setRevealedRounds(0);
        setShowVictory(false);
        try {
            const data = await runBattle({
                dataset,
                model_ids: [classicalModel, quantumModel],
                input_data: {},
            });
            setBattleData(data);
            setStatus('done');
            // Reveal rounds one by one
            for (let i = 1; i <= (data.rounds?.length || 0); i++) {
                await new Promise(r => setTimeout(r, 600));
                setRevealedRounds(i);
            }
            await new Promise(r => setTimeout(r, 800));
            setShowVictory(true);
        } catch (e: any) {
            console.error(e);
            setStatus('idle');
            alert('Battle failed: ' + (e.message || String(e)));
        }
    };

    const scores = battleData?.scores || {};
    const winner = battleData?.overall_winner || '';
    const rounds = battleData?.rounds || [];

    const winnerName = winner === 'draw' ? 'DRAW!' :
        (winner === classicalModel ? MODEL_NAMES[classicalModel] : MODEL_NAMES[quantumModel]);

    return (
        <div className="victory-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
            <motion.div
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.85 }}
                style={{
                    background: 'linear-gradient(135deg,#0a0a1f,#12121e)',
                    border: '2px solid rgba(139,92,246,0.3)',
                    borderRadius: 'var(--radius-xl)',
                    padding: '36px',
                    maxWidth: 700,
                    width: '95%',
                    maxHeight: '90vh',
                    overflowY: 'auto',
                    position: 'relative',
                }}
            >
                {/* Close */}
                <button onClick={onClose} style={{ position: 'absolute', top: 16, right: 16, background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '1.4rem', cursor: 'pointer' }}>×</button>

                {/* Header */}
                <div className="text-center" style={{ marginBottom: 28 }}>
                    <div style={{ fontSize: '3rem', marginBottom: 8 }}>⚔️</div>
                    <h2 className="gradient-text" style={{ marginBottom: 4 }}>Live Model Battle Arena</h2>
                    <p className="text-secondary text-sm">10 random samples · winner by confidence</p>
                </div>

                {/* Competitors */}
                <div className="flex justify-between items-center" style={{ marginBottom: 24, gap: 16 }}>
                    <div className="card card-classical" style={{ flex: 1, textAlign: 'center', padding: '14px 16px' }}>
                        <div style={{ fontSize: '1.8rem' }}>🔷</div>
                        <p style={{ fontWeight: 700, marginTop: 4 }}>{MODEL_NAMES[classicalModel]}</p>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Classical</p>
                        {scores[classicalModel] !== undefined && (
                            <div style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--accent-classical-2)', marginTop: 8 }}>
                                {scores[classicalModel]}
                            </div>
                        )}
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 900, color: 'var(--text-muted)' }}>VS</div>
                    <div className="card card-quantum" style={{ flex: 1, textAlign: 'center', padding: '14px 16px' }}>
                        <div style={{ fontSize: '1.8rem' }}>⚛️</div>
                        <p style={{ fontWeight: 700, marginTop: 4 }}>{MODEL_NAMES[quantumModel]}</p>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Quantum</p>
                        {scores[quantumModel] !== undefined && (
                            <div style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--accent-quantum-2)', marginTop: 8 }}>
                                {scores[quantumModel]}
                            </div>
                        )}
                    </div>
                </div>

                {/* Progress */}
                {status === 'done' && (
                    <div style={{ marginBottom: 20 }}>
                        <div className="flex justify-between" style={{ marginBottom: 6, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            <span>Battle Progress</span>
                            <span>{revealedRounds}/{rounds.length} rounds</span>
                        </div>
                        <div className="battle-progress">
                            <div
                                className="battle-progress-fill"
                                style={{ width: `${(revealedRounds / rounds.length) * 100}%` }}
                            />
                        </div>
                    </div>
                )}

                {/* Rounds */}
                {rounds.slice(0, revealedRounds).map((round: any, i: number) => {
                    const cResult = round.results?.[classicalModel] || {};
                    const qResult = round.results?.[quantumModel] || {};
                    const roundWinner = round.round_winner;
                    return (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            style={{
                                display: 'flex',
                                gap: 8,
                                marginBottom: 8,
                                padding: '10px 14px',
                                background: 'var(--bg-glass)',
                                border: '1px solid var(--border-subtle)',
                                borderRadius: 'var(--radius-md)',
                                fontSize: '0.82rem',
                            }}
                        >
                            <span style={{ color: 'var(--text-muted)', minWidth: 30 }}>#{i + 1}</span>
                            <div style={{ flex: 1 }}>
                                <span style={{ color: 'var(--accent-classical-2)' }}>{cResult.prediction}</span>
                                <span style={{ color: 'var(--text-muted)', margin: '0 6px' }}>({((cResult.confidence || 0) * 100).toFixed(0)}%)</span>
                            </div>
                            <div style={{ flex: 1, textAlign: 'right' }}>
                                <span style={{ color: 'var(--text-muted)', margin: '0 6px' }}>({((qResult.confidence || 0) * 100).toFixed(0)}%)</span>
                                <span style={{ color: 'var(--accent-quantum-2)' }}>{qResult.prediction}</span>
                            </div>
                            <span style={{ minWidth: 20, textAlign: 'center' }}>
                                {roundWinner === classicalModel ? '🔷' : roundWinner === quantumModel ? '⚛️' : '🤝'}
                            </span>
                        </motion.div>
                    );
                })}

                {/* Start button */}
                {status === 'idle' && (
                    <button
                        className="btn btn-battle"
                        onClick={startBattle}
                        style={{ width: '100%', justifyContent: 'center', marginTop: 8 }}
                        id="start-battle-btn"
                    >
                        ⚔️ Start Battle!
                    </button>
                )}

                {status === 'loading' && !battleData && (
                    <div className="text-center" style={{ padding: 20 }}>
                        <div style={{ fontSize: '2rem', marginBottom: 8 }}>⚛️</div>
                        <p className="text-secondary">Loading battle data...</p>
                    </div>
                )}

                {/* Victory + Chart */}
                <AnimatePresence>
                    {showVictory && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            style={{ marginTop: 20 }}
                        >
                            <div style={{
                                background: 'linear-gradient(135deg,rgba(245,158,11,0.1),rgba(245,158,11,0.05))',
                                border: '2px solid rgba(245,158,11,0.4)',
                                borderRadius: 'var(--radius-lg)',
                                padding: '20px',
                                textAlign: 'center',
                                marginBottom: 16,
                            }}>
                                <div style={{ fontSize: '3rem' }}>🏆</div>
                                <h3 className="gradient-text-gold" style={{ margin: '8px 0 4px' }}>{winnerName}</h3>
                                <p className="text-secondary text-sm">
                                    Final Score: {MODEL_NAMES[classicalModel]} {scores[classicalModel] || 0} –{' '}
                                    {scores[quantumModel] || 0} {MODEL_NAMES[quantumModel]}
                                </p>
                            </div>
                            <BattleBarChart scores={scores} modelNames={MODEL_NAMES} />
                            <button className="btn btn-outline" onClick={startBattle} style={{ marginTop: 12, width: '100%', justifyContent: 'center' }}>
                                🔄 Battle Again
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </div>
    );
}
