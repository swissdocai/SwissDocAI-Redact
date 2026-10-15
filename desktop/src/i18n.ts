import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import de from './locales/de.json';
import en from './locales/en.json';
import fr from './locales/fr.json';
import it from './locales/it.json';

export const LANGUAGES = { de: 'Deutsch', fr: 'Français', it: 'Italiano', en: 'English' } as const;
export type Language = keyof typeof LANGUAGES;

i18n.use(initReactI18next).init({
  resources: { de: { translation: de }, fr: { translation: fr }, it: { translation: it }, en: { translation: en } },
  lng: 'de',
  fallbackLng: 'de',
  interpolation: { escapeValue: false },
});

export default i18n;
