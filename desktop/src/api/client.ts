import axios from 'axios';
import { useSearchStore } from '../store/useSearchStore';

export const getBaseUrl = () => {
  const port = useSearchStore.getState().backendPort;
  return `http://127.0.0.1:${port}`;
};

export const getWsUrl = () => {
  const port = useSearchStore.getState().backendPort;
  return `ws://127.0.0.1:${port}`;
};

export const api = axios.create();

api.interceptors.request.use((config) => {
  config.baseURL = getBaseUrl();
  return config;
});
