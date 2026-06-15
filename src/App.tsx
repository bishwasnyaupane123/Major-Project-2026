import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster, toast } from 'react-hot-toast';
import { useApp } from './context';
import { predict, getCircuitInfo, trainModel } from './api';
import InputForm from './components/InputForm';
import ResultCard from './components/ResultCard';
import BattleArena from './components/BattleArena';
import { ComparisonBarChart, InferenceLineChart } from './components/Charts';
import { ThemeToggle, MetricChip } from './components/UI';

const DATASET_OPTIONS = [
  { id: 'breast_cancer', name: '🩺 Breast Cancer', icon: '🩺', models: ['classical_dnn', 'hqnn'], task: 'classification' },
  { id: 'crop', name: '🌾 Crop Rec.', icon: '🌾', models: ['classical_knn', 'quantum_knn'], task: 'classification' },
  { id: 'imdb', name: '🎬 IMDB Reviews', icon: '🎬', models: ['classical_transformer', 'quantum_transformer'], task: 'sentiment' },
];

const MODEL_PAIRS: Record<string, [string, string]> = {
  breast_cancer: ['classical_dnn', 'hqnn'],
  crop: ['classical_knn', 'quantum_knn'],
  imdb: ['classical_transformer', 'quantum_transformer'],
};

const MODEL_META: Record<string, { name: string; desc: string; type: string }> = {
  classical_dnn: { name: 'Classical ANN', desc: 'Pre-trained Keras ANN · PCA (30→6) · Accuracy ~95.6%', type: 'classical' },
  hqnn: { name: 'Hybrid QNN', desc: 'Pre-trained 6-qubit HQNN · PCA preprocessing · Accuracy ~96.5%', type: 'quantum' },
  classical_knn: { name: 'Classical KNN', desc: 'LDA features · k-Nearest Neighbors · CV accuracy 97.67%', type: 'classical' },
  quantum_knn: { name: 'Quantum KNN', desc: 'Swap-test quantum distance · LDA features · accuracy 97.50%', type: 'quantum' },
  classical_transformer: { name: 'Classical Transformer (DistilBERT)', desc: 'Fine-tuned DistilBERT · Test accuracy ~87.6%', type: 'classical' },
  quantum_transformer: { name: 'Quantum Transformer', desc: '4-qubit AngleEmbedding · BasicEntanglerLayers', type: 'quantum' },
};

export default function App() {
  const { dataset, setDataset } = useApp();

  const [results, setResults] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);
  const [inferenceHistory, setInferenceHistory] = useState<any[]>([]);
  const [showBattle, setShowBattle] = useState(false);
  const [showCircuit, setShowCircuit] = useState(false);
  const [circuitInfo, setCircuitInfo] = useState<Record<string, any>>({});
  const [trained, setTrained] = useState<Record<string, boolean>>({});
  const [training, setTraining] = useState<Record<string, boolean>>({});
  const [activeTab, setActiveTab] = useState<'compare' | 'charts' | 'history'>('compare');
  const [showTutorial, setShowTutorial] = useState(false);

  const [classicalModel, quantumModel] = MODEL_PAIRS[dataset] || ['classical_dnn', 'hqnn'];

  // Load circuit info for quantum model
  useEffect(() => {
    getCircuitInfo(dataset, quantumModel)
      .then(info => setCircuitInfo(prev => ({ ...prev, [quantumModel]: info })))
      .catch(() => { });
  }, [dataset, quantumModel]);

  const handleTrain = async (modelId: string) => {
    setTraining(prev => ({ ...prev, [modelId]: true }));
    const tid = toast.loading(`Training ${MODEL_META[modelId]?.name}...`);
    try {
      const res = await trainModel(dataset, modelId);
      setTrained(prev => ({ ...prev, [modelId]: true }));
      toast.success(`${MODEL_META[modelId]?.name} ready! Acc: ${res.metrics?.accuracy ? (res.metrics.accuracy * 100).toFixed(1) + '%' : 'done'}`, { id: tid });
    } catch (e: any) {
      toast.error(`Training failed: ${e.message || String(e)}`, { id: tid });
    } finally {
      setTraining(prev => ({ ...prev, [modelId]: false }));
    }
  };

  const handlePredict = useCallback(async (inputData: any) => {
    setLoading(true);
    const toastId = toast.loading('Running inference on both models...');
    try {
      const res = await predict({
        dataset,
        model_ids: [classicalModel, quantumModel],
        input_data: inputData,
      });

      const newResults: Record<string, any> = {};
      res.results.forEach((r: any) => { newResults[r.model_id] = r; });
      setResults(newResults);

      const cTime = newResults[classicalModel]?.inference_time_ms || 0;
      const qTime = newResults[quantumModel]?.inference_time_ms || 0;
      setInferenceHistory(prev => [...prev.slice(-19), { time: Date.now(), classical_ms: cTime, quantum_ms: qTime }]);

      toast.success('Inference complete!', { id: toastId });
    } catch (e: any) {
      toast.error('Inference failed. Is the backend running?', { id: toastId });
    } finally {
      setLoading(false);
    }
  }, [dataset, classicalModel, quantumModel]);

  const cResult = results[classicalModel];
  const qResult = results[quantumModel];
  const cWins = (cResult?.confidence || 0) > (qResult?.confidence || 0);
  const qWins = (qResult?.confidence || 0) > (cResult?.confidence || 0);
  const hasResults = !!(cResult || qResult);

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-primary)' }}>
      <Toaster position="top-right" toastOptions={{ style: { background: '#12121e', color: '#f1f5f9', border: '1px solid rgba(139,92,246,0.3)' } }} />

      {/* ── Navbar ── */}
      <nav className="nav">
        <div className="nav-inner">
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
              style={{ fontSize: '1.6rem' }}
            >⚛️</motion.div>
            <div>
              <span style={{ fontFamily: 'Space Grotesk', fontWeight: 800, fontSize: '1.1rem', color: 'var(--text-primary)' }}>
                Quantum<span className="gradient-text">ML</span> Arena
              </span>
              <span style={{ display: 'block', fontSize: '0.65rem', color: 'var(--text-muted)', letterSpacing: '0.15em', textTransform: 'uppercase' }}>
                Classical vs Quantum · Major Project Demo
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="btn btn-outline btn-sm" onClick={() => setShowTutorial(true)} id="tutorial-btn">
              ❓ Guide
            </button>
            <ThemeToggle />
          </div>
        </div>
      </nav>

      {/* ── Ambient orbs ── */}
      <div className="orb orb-quantum" style={{ width: 500, height: 500, top: -200, left: -150, position: 'fixed' }} />
      <div className="orb orb-classical" style={{ width: 400, height: 400, bottom: -150, right: -100, position: 'fixed' }} />

      {/* ── Hero ── */}
      <section style={{ padding: '52px 24px 32px', textAlign: 'center', maxWidth: 900, margin: '0 auto' }}>
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <h1 style={{ marginBottom: 16, lineHeight: 1.1 }}>
            <span className="gradient-text">Classical</span>{' '}
            <span style={{ color: 'var(--text-muted)' }}>vs</span>{' '}
            <span className="gradient-text">Quantum</span>
            <br />Machine Learning
          </h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: 600, margin: '0 auto 28px', fontSize: '1.05rem' }}>
            Run side-by-side predictions on real datasets. Compare accuracy, confidence, and inference speed.
            Experience the power of hybrid quantum-classical models.
          </p>
        </motion.div>

        {/* Dataset selector */}
        <motion.div
          className="flex justify-center gap-3"
          style={{ flexWrap: 'wrap' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          {DATASET_OPTIONS.map(ds => (
            <button
              key={ds.id}
              onClick={() => { setDataset(ds.id); setResults({}); }}
              id={`ds-btn-${ds.id}`}
              style={{
                padding: '10px 20px',
                borderRadius: 'var(--radius-full)',
                border: dataset === ds.id ? '2px solid var(--accent-quantum)' : '1px solid var(--border-subtle)',
                background: dataset === ds.id ? 'rgba(139,92,246,0.15)' : 'var(--bg-glass)',
                color: dataset === ds.id ? 'var(--accent-quantum-2)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '0.9rem',
                transition: 'all 0.2s ease',
                boxShadow: dataset === ds.id ? 'var(--glow-quantum)' : 'none',
              }}
            >
              {ds.icon} {ds.name}
            </button>
          ))}
        </motion.div>
      </section>

      {/* ── Main content ── */}
      <div className="container" style={{ paddingBottom: 60 }}>

        {/* Model info cards */}
        <motion.div
          className="comparison-grid"
          style={{ marginBottom: 24 }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          {/* Classical model card */}
          <div className="card card-classical">
            <div className="flex justify-between items-center" style={{ marginBottom: 12 }}>
              <span className="badge badge-classical">🔷 Classical</span>
              <button
                className="btn btn-classical btn-sm"
                onClick={() => handleTrain(classicalModel)}
                disabled={training[classicalModel]}
                id={`train-classical-btn`}
              >
                {training[classicalModel] ? '⏳ Training...' : trained[classicalModel] ? '✓ Trained' : '🚀 Pre-train'}
              </button>
            </div>
            <h3 style={{ marginBottom: 6 }}>{MODEL_META[classicalModel]?.name}</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{MODEL_META[classicalModel]?.desc}</p>
          </div>

          {/* VS divider */}
          <div className="vs-divider">
            <motion.div className="vs-text" animate={{ scale: [1, 1.08, 1] }} transition={{ duration: 2, repeat: Infinity }}>VS</motion.div>
            <button
              className="btn btn-battle btn-sm"
              onClick={() => setShowBattle(true)}
              id="battle-btn"
            >
              ⚔️ Battle
            </button>
            <button
              className="btn btn-outline btn-sm"
              onClick={() => setShowCircuit(!showCircuit)}
              id="circuit-btn"
            >
              {showCircuit ? '🙈 Hide' : '⚛️ Circuit'}
            </button>
          </div>

          {/* Quantum model card */}
          <div className="card card-quantum">
            <div className="flex justify-between items-center" style={{ marginBottom: 12 }}>
              <span className="badge badge-quantum">⚛️ Quantum</span>
              <button
                className="btn btn-quantum btn-sm"
                onClick={() => handleTrain(quantumModel)}
                disabled={training[quantumModel]}
                id={`train-quantum-btn`}
              >
                {training[quantumModel] ? '⏳ Training...' : trained[quantumModel] ? '✓ Trained' : '🚀 Pre-train'}
              </button>
            </div>
            <h3 style={{ marginBottom: 6 }}>{MODEL_META[quantumModel]?.name}</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{MODEL_META[quantumModel]?.desc}</p>
          </div>
        </motion.div>

        {/* ── Input form ── */}
        <motion.div
          className="card"
          style={{ marginBottom: 24 }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="section-header">
            <div className="section-icon" style={{ background: 'rgba(139,92,246,0.15)' }}>⚡</div>
            <div>
              <h3>Input Data</h3>
              <p className="text-sm text-muted">Adjust the sliders to set feature values, then run prediction</p>
            </div>
          </div>
          <InputForm
            dataset={dataset}
            onSubmit={handlePredict}
            loading={loading}
            featureImportance={qResult?.feature_importance}
          />
        </motion.div>

        {/* ── Tabs ── */}
        {hasResults && (
          <div className="flex justify-center" style={{ marginBottom: 20 }}>
            <div className="tabs">
              {(['compare', 'charts', 'history'] as const).map(tab => (
                <button
                  key={tab}
                  className={`tab ${activeTab === tab ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab)}
                  id={`tab-${tab}`}
                >
                  {tab === 'compare' ? '🔬 Compare' : tab === 'charts' ? '📊 Charts' : '📈 History'}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Winner banner ── */}
        <AnimatePresence>
          {hasResults && (cWins || qWins) && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{
                textAlign: 'center',
                marginBottom: 16,
                padding: '12px 20px',
                background: 'linear-gradient(135deg,rgba(245,158,11,0.1),rgba(245,158,11,0.05))',
                border: '1px solid rgba(245,158,11,0.3)',
                borderRadius: 'var(--radius-full)',
                display: 'inline-block',
                width: '100%',
              }}
            >
              <span className="badge badge-winner">
                🏆 {cWins ? MODEL_META[classicalModel]?.name : MODEL_META[quantumModel]?.name} wins this round!
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Tab content ── */}
        <AnimatePresence mode="wait">
          {activeTab === 'compare' && (
            <motion.div
              key="compare"
              className="comparison-grid"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
                <ResultCard
                  result={loading ? null : cResult}
                  isWinner={cWins}
                  side="left"
                  circuitInfo={circuitInfo[classicalModel]}
                  showCircuit={showCircuit}
                />
              </motion.div>
              <div className="vs-divider">
                <div className="vs-text">VS</div>
              </div>
              <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
                <ResultCard
                  result={loading ? null : qResult}
                  isWinner={qWins}
                  side="right"
                  circuitInfo={circuitInfo[quantumModel]}
                  showCircuit={showCircuit}
                />
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'charts' && hasResults && (
            <motion.div key="charts" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="grid-2" style={{ marginBottom: 20 }}>
                <ComparisonBarChart classicalResult={cResult} quantumResult={qResult} />
                <InferenceLineChart history={inferenceHistory} />
              </div>
              {/* Quick stats */}
              {cResult && qResult && (
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center' }}>
                  <MetricChip label="Classical Confidence" value={`${((cResult.confidence || 0) * 100).toFixed(1)}%`} color="var(--accent-classical-2)" />
                  <MetricChip label="Quantum Confidence" value={`${((qResult.confidence || 0) * 100).toFixed(1)}%`} color="var(--accent-quantum-2)" />
                  <MetricChip label="Classical Time" value={`${(cResult.inference_time_ms || 0).toFixed(1)}ms`} color="var(--accent-classical-2)" />
                  <MetricChip label="Quantum Time" value={`${(qResult.inference_time_ms || 0).toFixed(1)}ms`} color="var(--accent-quantum-2)" />
                  <MetricChip label="Speed Ratio" value={`${((qResult.inference_time_ms || 1) / (cResult.inference_time_ms || 1)).toFixed(1)}×`} color="var(--accent-gold)" />
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'history' && (
            <motion.div key="history" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <InferenceLineChart history={inferenceHistory} />
              {inferenceHistory.length > 0 ? (
                <div style={{ marginTop: 16 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                        {['#', 'Classical (ms)', 'Quantum (ms)', 'Faster'].map(h => (
                          <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', fontSize: '0.72rem' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {inferenceHistory.map((h, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                          <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>#{i + 1}</td>
                          <td style={{ padding: '8px 12px', fontFamily: 'JetBrains Mono', color: 'var(--accent-classical-2)' }}>{h.classical_ms.toFixed(1)}</td>
                          <td style={{ padding: '8px 12px', fontFamily: 'JetBrains Mono', color: 'var(--accent-quantum-2)' }}>{h.quantum_ms.toFixed(1)}</td>
                          <td style={{ padding: '8px 12px' }}>
                            <span className={`badge badge-${h.classical_ms < h.quantum_ms ? 'classical' : 'quantum'}`}>
                              {h.classical_ms < h.quantum_ms ? '🔷 Classical' : '⚛️ Quantum'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-center text-muted" style={{ padding: 40 }}>Run predictions to see history</p>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Battle Arena modal ── */}
      <AnimatePresence>
        {showBattle && (
          <BattleArena
            dataset={dataset}
            classicalModel={classicalModel}
            quantumModel={quantumModel}
            onClose={() => setShowBattle(false)}
          />
        )}
      </AnimatePresence>

      {/* ── Tutorial overlay ── */}
      <AnimatePresence>
        {showTutorial && (
          <div className="victory-overlay" onClick={() => setShowTutorial(false)}>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              style={{
                background: 'linear-gradient(135deg,#0d0d1a,#12121e)',
                border: '1px solid var(--border-quantum)',
                borderRadius: 'var(--radius-xl)',
                padding: '36px',
                maxWidth: 540,
                width: '90%',
                maxHeight: '85vh',
                overflowY: 'auto',
              }}
              onClick={e => e.stopPropagation()}
            >
              <h3 className="gradient-text" style={{ marginBottom: 20, fontSize: '1.5rem' }}>👋 Quick Guide</h3>
              {[
                ['1️⃣ Select Dataset', 'Choose from Breast Cancer, Crop Recommendation, or IMDB Reviews.'],
                ['2️⃣ Pre-train Models', 'Click "🚀 Pre-train" for each model card — they train automatically on first use.'],
                ['3️⃣ Set Input Values', 'Use the sliders (or text area for IMDB) to set input feature values.'],
                ['4️⃣ Run Prediction', 'Click "⚡ Run Prediction" to get results from both models simultaneously.'],
                ['5️⃣ Compare Results', 'View the split-screen comparison. The winner 🏆 badge highlights the more confident model.'],
                ['6️⃣ Battle Arena', 'Click "⚔️ Battle" for a gamified 10-round match and live scoreboard.'],
                ['7️⃣ Explore Charts', 'The "📊 Charts" tab shows bar charts, pie charts, and inference time comparisons.'],
              ].map(([title, desc]) => (
                <div key={title} style={{ marginBottom: 14, display: 'flex', gap: 12 }}>
                  <span style={{ fontSize: '1rem', flexShrink: 0, marginTop: 2 }}>{title.slice(0, 2)}</span>
                  <div>
                    <p style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: 2 }}>{title.slice(3)}</p>
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{desc}</p>
                  </div>
                </div>
              ))}
              <button className="btn btn-primary" onClick={() => setShowTutorial(false)} style={{ width: '100%', justifyContent: 'center', marginTop: 8 }} id="close-tutorial-btn">
                Got it! Let's go ⚡
              </button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* ── FAB ── */}
      <motion.button
        className="btn btn-primary"
        style={{
          position: 'fixed', bottom: 28, right: 28, zIndex: 40,
          borderRadius: 'var(--radius-full)', width: 56, height: 56, padding: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1.4rem', boxShadow: '0 8px 30px rgba(139,92,246,0.4)',
        }}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setShowTutorial(true)}
        id="fab-help-btn"
        title="Open tutorial"
      >
        ❓
      </motion.button>
    </div>
  );
}
