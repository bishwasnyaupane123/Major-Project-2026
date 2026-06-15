import axios from 'axios';

const BASE = 'http://localhost:8000/api';

const api = axios.create({ baseURL: BASE });

export const getDatasets = () => api.get('/datasets').then(r => r.data);
export const getDataset = (id: string) => api.get(`/datasets/${id}`).then(r => r.data);
export const getModels = (dataset?: string) => api.get('/models', { params: { dataset } }).then(r => r.data);
export const predict = (payload: any) => api.post('/predict', payload).then(r => r.data);
export const uploadFile = (file: File, dataset: string) => {
  const fd = new FormData();
  fd.append('file', file);
  return api.post(`/upload?dataset=${dataset}`, fd).then(r => r.data);
};
export const runBattle = (payload: any) => api.post('/battle', payload).then(r => r.data);
export const getCircuitInfo = (dataset: string, modelId: string) =>
  api.get(`/circuit/${dataset}/${modelId}`).then(r => r.data);
export const trainModel = (dataset: string, modelId: string) =>
  api.post(`/train/${dataset}/${modelId}`).then(r => r.data);

export default api;
