/** @type {import('tailwindcss').Config} */
// SwissDocAI brand (swissdocai.ch): blue from the shield, Swiss red accent, Inter
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'] },
      colors: {
        brand: {
          50: '#eff6ff', 100: '#dbeafe', 200: '#bfdbfe', 300: '#93c5fd', 400: '#60a5fa',
          500: '#3b82f6', 600: '#2563eb', 700: '#1d4ed8', 800: '#1e40af', 900: '#1e3a8a',
        },
        swiss: { 50: '#fef2f2', 100: '#fee2e2', 500: '#ef4444', 600: '#dc2626', 700: '#b91c1c' },
      },
      boxShadow: { card: '0 1px 2px rgba(28,25,23,.04), 0 4px 16px rgba(28,25,23,.06)' },
    },
  },
  plugins: [],
};
