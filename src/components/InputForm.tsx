import { motion, AnimatePresence } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { useState } from 'react';

// ─── All 30 WDBC breast-cancer features (min/max/default from real dataset) ───
const BC_MEAN = [
    { name: 'radius_mean',              label: 'Radius',             min: 6.981,    max: 28.11,  step: 0.01,   default: 13.37   },
    { name: 'texture_mean',             label: 'Texture',            min: 9.71,     max: 39.28,  step: 0.01,   default: 18.84   },
    { name: 'perimeter_mean',           label: 'Perimeter',          min: 43.79,    max: 188.5,  step: 0.1,    default: 86.24   },
    { name: 'area_mean',                label: 'Area',               min: 143.5,    max: 2501,   step: 1,      default: 551.1   },
    { name: 'smoothness_mean',          label: 'Smoothness',         min: 0.0526,   max: 0.1634, step: 0.0001, default: 0.09587 },
    { name: 'compactness_mean',         label: 'Compactness',        min: 0.0194,   max: 0.3454, step: 0.001,  default: 0.09263 },
    { name: 'concavity_mean',           label: 'Concavity',          min: 0,        max: 0.4268, step: 0.001,  default: 0.06154 },
    { name: 'concave points_mean',      label: 'Concave Points',     min: 0,        max: 0.2012, step: 0.001,  default: 0.0335  },
    { name: 'symmetry_mean',            label: 'Symmetry',           min: 0.106,    max: 0.304,  step: 0.001,  default: 0.1792  },
    { name: 'fractal_dimension_mean',   label: 'Fractal Dimension',  min: 0.04996,  max: 0.09744,step: 0.0001, default: 0.06154 },
];

const BC_SE = [
    { name: 'radius_se',                label: 'Radius SE',          min: 0.1115,   max: 2.873,  step: 0.001,  default: 0.3242  },
    { name: 'texture_se',               label: 'Texture SE',         min: 0.3602,   max: 4.885,  step: 0.01,   default: 1.108   },
    { name: 'perimeter_se',             label: 'Perimeter SE',       min: 0.757,    max: 21.98,  step: 0.01,   default: 2.287   },
    { name: 'area_se',                  label: 'Area SE',            min: 6.802,    max: 542.2,  step: 0.1,    default: 24.53   },
    { name: 'smoothness_se',            label: 'Smoothness SE',      min: 0.001713, max: 0.03113,step: 0.0001, default: 0.00638 },
    { name: 'compactness_se',           label: 'Compactness SE',     min: 0.002252, max: 0.1354, step: 0.001,  default: 0.02045 },
    { name: 'concavity_se',             label: 'Concavity SE',       min: 0,        max: 0.396,  step: 0.001,  default: 0.02589 },
    { name: 'concave points_se',        label: 'Concave Points SE',  min: 0,        max: 0.05279,step: 0.0001, default: 0.01093 },
    { name: 'symmetry_se',              label: 'Symmetry SE',        min: 0.007882, max: 0.07895,step: 0.001,  default: 0.01873 },
    { name: 'fractal_dimension_se',     label: 'Fractal Dim SE',     min: 0.000895, max: 0.02984,step: 0.0001, default: 0.003187},
];

const BC_WORST = [
    { name: 'radius_worst',             label: 'Radius Worst',       min: 7.93,     max: 36.04,  step: 0.01,   default: 14.97   },
    { name: 'texture_worst',            label: 'Texture Worst',      min: 12.02,    max: 49.54,  step: 0.01,   default: 25.41   },
    { name: 'perimeter_worst',          label: 'Perimeter Worst',    min: 50.41,    max: 251.2,  step: 0.1,    default: 97.66   },
    { name: 'area_worst',               label: 'Area Worst',         min: 185.2,    max: 4254,   step: 1,      default: 686.5   },
    { name: 'smoothness_worst',         label: 'Smoothness Worst',   min: 0.07117,  max: 0.2226, step: 0.001,  default: 0.1313  },
    { name: 'compactness_worst',        label: 'Compactness Worst',  min: 0.02729,  max: 1.058,  step: 0.001,  default: 0.2119  },
    { name: 'concavity_worst',          label: 'Concavity Worst',    min: 0,        max: 1.252,  step: 0.001,  default: 0.2267  },
    { name: 'concave points_worst',     label: 'Concave Pts Worst',  min: 0,        max: 0.291,  step: 0.001,  default: 0.09993 },
    { name: 'symmetry_worst',           label: 'Symmetry Worst',     min: 0.1565,   max: 0.6638, step: 0.001,  default: 0.2822  },
    { name: 'fractal_dimension_worst',  label: 'Fractal Dim Worst',  min: 0.05504,  max: 0.2075, step: 0.0001, default: 0.08004 },
];

const ALL_BC_FIELDS = [...BC_MEAN, ...BC_SE, ...BC_WORST];

const DATASETS_META: Record<string, any> = {
    crop: {
        fields: [
            { name: 'N',           label: 'Nitrogen (N)',      min: 0,   max: 200, step: 1,   default: 70  },
            { name: 'P',           label: 'Phosphorus (P)',    min: 0,   max: 200, step: 1,   default: 40  },
            { name: 'K',           label: 'Potassium (K)',     min: 0,   max: 250, step: 1,   default: 40  },
            { name: 'temperature', label: 'Temperature (°C)',  min: -5,  max: 50,  step: 0.1, default: 25  },
            { name: 'humidity',    label: 'Humidity (%)',      min: 0,   max: 100, step: 0.1, default: 70  },
            { name: 'ph',          label: 'Soil pH',           min: 0,   max: 14,  step: 0.01,default: 6.5 },
            { name: 'rainfall',    label: 'Rainfall (mm)',     min: 0,   max: 500, step: 1,   default: 100 },
        ],
        type: 'numeric',
        groups: null,
    },
    breast_cancer: {
        fields: ALL_BC_FIELDS,
        type: 'numeric',
        groups: [
            { label: '📊 Mean Features',            emoji: '📊', fields: BC_MEAN   },
            { label: '📐 Standard Error Features',  emoji: '📐', fields: BC_SE     },
            { label: '⚠️ Worst Features',            emoji: '⚠️', fields: BC_WORST  },
        ],
    },
    imdb: {
        fields: [
            { name: 'review', label: 'Movie Review', type: 'textarea', default: 'This movie was absolutely fantastic! The acting was superb.' },
        ],
        type: 'text',
        groups: null,
    },
};

interface InputFormProps {
    dataset: string;
    onSubmit: (data: any) => void;
    loading?: boolean;
    featureImportance?: Record<string, number>;
}

// ── Single feature slider card ──
function FeatureCard({ field, register, watched, featureImportance }: any) {
    const getImportanceColor = (name: string) => {
        if (!featureImportance) return 'var(--bg-glass)';
        const v = featureImportance[name] || 0;
        return `rgba(139,92,246,${v * 0.35})`;
    };

    const decimals = field.step < 0.001 ? 4 : field.step < 0.01 ? 4 : field.step < 1 ? 2 : 0;

    return (
        <div
            style={{
                background: getImportanceColor(field.name),
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '10px 12px',
                transition: 'all 0.3s ease',
            }}
        >
            <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
                <label className="label" style={{ margin: 0, fontSize: '0.72rem' }}>{field.label}</label>
                {featureImportance && featureImportance[field.name] !== undefined && (
                    <span style={{ fontSize: '0.65rem', color: 'var(--accent-quantum-2)' }}>
                        {(featureImportance[field.name] * 100).toFixed(0)}%
                    </span>
                )}
            </div>
            <input
                type="range"
                {...register(field.name, { valueAsNumber: true })}
                min={field.min}
                max={field.max}
                step={field.step}
                id={`slider-${field.name}`}
                style={{ width: '100%', accentColor: 'var(--accent-quantum)', cursor: 'pointer' }}
            />
            <div className="flex justify-between" style={{ marginTop: 1 }}>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{field.min}</span>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-quantum-2)', fontFamily: 'monospace' }}>
                    {Number(watched[field.name] ?? field.default).toFixed(decimals)}
                </span>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{field.max}</span>
            </div>
        </div>
    );
}

// ── Collapsible group section ──
function FeatureGroup({ group, register, watched, featureImportance, defaultOpen }: any) {
    const [open, setOpen] = useState<boolean>(defaultOpen ?? true);

    return (
        <div style={{ marginBottom: 12 }}>
            {/* Group header */}
            <button
                type="button"
                onClick={() => setOpen(o => !o)}
                style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'var(--bg-glass)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    padding: '8px 14px',
                    cursor: 'pointer',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    marginBottom: open ? 8 : 0,
                    transition: 'all 0.2s',
                }}
            >
                <span>{group.label}</span>
                <span style={{
                    fontSize: '0.7rem',
                    color: 'var(--text-muted)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                }}>
                    {group.fields.length} features
                    <span style={{ transition: 'transform 0.2s', display: 'inline-block', transform: open ? 'rotate(180deg)' : 'none' }}>▼</span>
                </span>
            </button>

            {/* Grid of sliders */}
            <AnimatePresence initial={false}>
                {open && (
                    <motion.div
                        key="content"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.25 }}
                        style={{ overflow: 'hidden' }}
                    >
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                            gap: 8,
                            paddingBottom: 4,
                        }}>
                            {group.fields.map((field: any) => (
                                <FeatureCard
                                    key={field.name}
                                    field={field}
                                    register={register}
                                    watched={watched}
                                    featureImportance={featureImportance}
                                />
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export default function InputForm({ dataset, onSubmit, loading, featureImportance }: InputFormProps) {
    const meta = DATASETS_META[dataset] || { fields: [], type: 'numeric', groups: null };

    const defaultValues: Record<string, any> = {};
    meta.fields.forEach((f: any) => { defaultValues[f.name] = f.default ?? ''; });

    const { register, handleSubmit, watch, reset } = useForm({ defaultValues });
    const watched = watch();

    return (
        <form onSubmit={handleSubmit(onSubmit)} id="prediction-form">

            {/* Grouped layout for breast cancer (30 features in 3 sections) */}
            {meta.groups ? (
                <>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        marginBottom: 12,
                        fontSize: '0.78rem',
                        color: 'var(--text-muted)',
                        fontStyle: 'italic',
                    }}>
                        🔬 All 30 WDBC features — adjust each slider or leave at median values
                    </div>
                    {meta.groups.map((group: any, gi: number) => (
                        <FeatureGroup
                            key={group.label}
                            group={group}
                            register={register}
                            watched={watched}
                            featureImportance={featureImportance}
                            defaultOpen={gi === 0}  // only first group open by default
                        />
                    ))}
                </>
            ) : meta.type === 'text' ? (
                /* IMDB textarea */
                <div>
                    {meta.fields.map((field: any) => (
                        <div key={field.name}>
                            <label className="label" style={{ fontSize: '0.78rem' }}>{field.label}</label>
                            <textarea
                                {...register(field.name)}
                                className="input"
                                rows={5}
                                placeholder="Enter your movie review..."
                                id={`input-${field.name}`}
                                style={{ width: '100%' }}
                            />
                        </div>
                    ))}
                </div>
            ) : (
                /* Crop — flat grid */
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12 }}>
                    {meta.fields.map((field: any) => (
                        <FeatureCard
                            key={field.name}
                            field={field}
                            register={register}
                            watched={watched}
                            featureImportance={featureImportance}
                        />
                    ))}
                </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-3" style={{ marginTop: 20 }}>
                <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={loading}
                    id="run-prediction-btn"
                    style={{ flex: 1, justifyContent: 'center' }}
                >
                    {loading ? (
                        <>
                            <svg width="16" height="16" viewBox="0 0 24 24" style={{ animation: 'rotate-slow 1s linear infinite' }}>
                                <circle cx="12" cy="12" r="10" fill="none" stroke="white" strokeWidth="2.5" strokeDasharray="40 20" />
                            </svg>
                            Running Inference...
                        </>
                    ) : '⚡ Run Prediction'}
                </button>
                <button
                    type="button"
                    className="btn btn-outline btn-sm"
                    onClick={() => reset(defaultValues)}
                    id="reset-form-btn"
                >
                    ↺ Reset
                </button>
            </div>
        </form>
    );
}
