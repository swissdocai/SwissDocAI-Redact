import { Calendar, Check, Hand, Hash, Landmark, MapPin, Phone, QrCode, Sparkles, Tag, TriangleAlert, User } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Box, Finding, Findings, QrCode as QrCodeItem } from '../api';

const ICONS: Record<string, typeof User> = {
  person: User, birth_date: Calendar, address: MapPin, contact: Phone, identifier: Hash, financial: Landmark,
  date: Calendar, organization: Landmark, other: Tag,
};

interface Props {
  findings: Findings | null;
  onToggle: (item: Finding, redacted: boolean) => void;
  onToggleQrCode: (code: QrCodeItem, redacted: boolean) => void;
  onJump: (page: number, box: Box) => void;
}

function Tick({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  const { t } = useTranslation();
  return (
    <button
      role="checkbox"
      aria-checked={checked}
      title={t('findings.uncheckHint')}
      onClick={() => onChange(!checked)}
      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors ${
        checked ? 'border-stone-900 bg-stone-900 text-white' : 'border-stone-300 bg-white'
      }`}
    >
      {checked && <Check className="h-3 w-3" strokeWidth={3} />}
    </button>
  );
}

export function FindingsPanel({ findings, onToggle, onToggleQrCode, onJump }: Props) {
  const { t } = useTranslation();
  const count = (findings?.groups.reduce((sum, g) => sum + g.items.length, 0) ?? 0) + (findings?.qrCodes.length ?? 0);
  return (
    <aside className="flex w-72 shrink-0 flex-col border-l border-stone-200 bg-white xl:w-80">
      <div className="flex items-center justify-between border-b border-stone-200 px-5 py-4">
        <h2 className="text-sm font-semibold text-stone-900">{t('findings.title')}</h2>
        {findings && <span className="rounded-full bg-stone-100 px-2 py-0.5 text-xs font-medium text-stone-600">{count}</span>}
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto px-3 py-4">
        {findings && findings.review.length > 0 && (
          <section className="rounded-lg border border-amber-200 bg-amber-50 p-2">
            <div className="flex items-center gap-2 px-2 pb-1 text-xs font-semibold text-amber-800">
              <TriangleAlert className="h-3.5 w-3.5" />
              {t('findings.check')}
            </div>
            {findings.review.map((item, i) => (
              <button
                key={i}
                onClick={() => onJump(item.page, item.box)}
                className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-sm text-amber-900 hover:bg-amber-100"
              >
                <span className="truncate">{item.text}</span>
                <span className="ml-2 shrink-0 text-xs text-amber-700">{t('findings.page', { n: item.page + 1 })}</span>
              </button>
            ))}
          </section>
        )}

        {findings && count === 0 && <p className="px-2 text-sm text-stone-500">{t('findings.none')}</p>}

        {findings?.groups.map((group) => {
          const Icon = ICONS[group.category] ?? Tag;
          return (
            <section key={group.category}>
              <div className="flex items-center gap-2 px-2 pb-1.5">
                <Icon className="h-3.5 w-3.5 text-stone-400" />
                <span className="label">{t(`cat.${group.category}`, { defaultValue: group.category })}</span>
                <span className="text-xs text-stone-400">{group.items.length}</span>
              </div>
              <ul className="space-y-0.5">
                {group.items.map((item) => (
                  <li key={item.index} className="group flex items-center gap-2.5 rounded-md px-2 py-1.5 hover:bg-stone-50">
                    <Tick checked={item.redacted} onChange={(redacted) => onToggle(item, redacted)} />
                    <button
                      onClick={() => onJump(item.page, item.box)}
                      className={`min-w-0 flex-1 truncate text-left text-sm ${item.redacted ? 'text-stone-800' : 'text-stone-400 line-through'}`}
                      title={item.source === 'llm' ? t('findings.byModel') : `${t('findings.byRule')} (${item.source.replace('rule:', '')})`}
                    >
                      {item.text}
                    </button>
                    {item.source !== 'llm' && <Sparkles className="h-3 w-3 shrink-0 text-stone-300" />}
                    <span className="shrink-0 text-xs text-stone-400">{t('findings.places', { count: item.places })}</span>
                  </li>
                ))}
              </ul>
            </section>
          );
        })}

        {findings && findings.qrCodes.length > 0 && (
          <section>
            <div className="flex items-center gap-2 px-2 pb-1.5">
              <QrCode className="h-3.5 w-3.5 text-stone-400" />
              <span className="label">{t('findings.qrCodes')}</span>
              <span className="text-xs text-stone-400">{findings.qrCodes.length}</span>
            </div>
            <ul className="space-y-0.5">
              {findings.qrCodes.map((code, i) => (
                <li key={`${code.page}-${code.index}`} className="flex items-center gap-2.5 rounded-md px-2 py-1.5 hover:bg-stone-50">
                  <Tick checked={code.redacted} onChange={(redacted) => onToggleQrCode(code, redacted)} />
                  <button
                    onClick={() => onJump(code.page, code.box)}
                    className={`min-w-0 flex-1 truncate text-left text-sm ${code.redacted ? 'text-stone-800' : 'text-stone-400 line-through'}`}
                  >
                    {t('findings.qrCode', { n: i + 1 })}
                  </button>
                  <span className="shrink-0 text-xs text-stone-400">{t('findings.page', { n: code.page + 1 })}</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        {findings && findings.manual > 0 && (
          <div className="flex items-center gap-2 px-2 text-xs text-stone-500">
            <Hand className="h-3.5 w-3.5" />
            {t('findings.manual')}: {findings.manual}
          </div>
        )}

        {findings && findings.notFound.length > 0 && (
          <details className="px-2 text-xs text-stone-500">
            <summary className="cursor-pointer">{t('findings.notOnPage')} ({findings.notFound.length})</summary>
            <ul className="mt-1 space-y-0.5 pl-3">
              {findings.notFound.map((text, i) => <li key={i} className="truncate">{text}</li>)}
            </ul>
          </details>
        )}
      </div>
    </aside>
  );
}
