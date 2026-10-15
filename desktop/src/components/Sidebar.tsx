import { CircleAlert, CircleCheck, FilePlus2, FileText, LoaderCircle, Settings, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { DocSummary } from '../api';

interface Props {
  docs: DocSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
  onAdd: () => void;
  onRemove: (id: string) => void;
  onSettings: () => void;
}

export function StatusBadge({ doc }: { doc: DocSummary }) {
  const { t } = useTranslation();
  const busy = doc.status === 'queued' || doc.status === 'analyzing' || doc.status === 'saving';
  const { stage, done, total } = doc.progress;
  if (busy) {
    const detail = doc.status !== 'queued' && stage ? `${t(`stage.${stage}`)}${total > 1 ? ` ${done + 1}/${total}` : ''}` : '';
    return (
      <span className="flex items-center gap-1.5 text-xs text-brand-700">
        <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
        <span className="truncate">{detail || t(`status.${doc.status}`)}</span>
      </span>
    );
  }
  const styles: Record<string, string> = {
    waiting: 'text-stone-500',
    review: 'text-amber-700',
    saved: 'text-emerald-700',
    failed: 'text-swiss-600',
  };
  return (
    <span className={`flex items-center gap-1.5 text-xs ${styles[doc.status] ?? 'text-stone-500'}`}>
      {doc.status === 'saved' && <CircleCheck className="h-3.5 w-3.5" />}
      {doc.status === 'failed' && <CircleAlert className="h-3.5 w-3.5" />}
      {doc.status === 'review' && <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />}
      {t(`status.${doc.status}`)}
    </span>
  );
}

export function Sidebar({ docs, selected, onSelect, onAdd, onRemove, onSettings }: Props) {
  const { t } = useTranslation();
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-stone-200 bg-white xl:w-72">
      <div className="flex items-center gap-3 border-b border-stone-200 px-5 py-4">
        <img src="/logo.png" alt="SwissDocAI" className="h-7 w-auto" draggable={false} />
        <span className="rounded-md bg-stone-100 px-2 py-0.5 text-xs font-semibold text-stone-600">{t('app.name')}</span>
      </div>

      <div className="px-4 pt-4">
        <button className="btn-primary w-full" onClick={onAdd}>
          <FilePlus2 className="h-4 w-4" />
          {t('nav.add')}
        </button>
      </div>

      <div className="label px-5 pb-2 pt-5">{t('nav.documents')}</div>
      <nav className="flex-1 space-y-1 overflow-y-auto px-3 pb-3">
        {docs.map((doc) => (
          <div
            key={doc.id}
            onClick={() => onSelect(doc.id)}
            className={`group relative flex cursor-pointer items-start gap-3 rounded-lg px-3 py-2.5 transition-colors ${
              doc.id === selected ? 'bg-brand-50 ring-1 ring-brand-200' : 'hover:bg-stone-100'
            }`}
          >
            <FileText className={`mt-0.5 h-4 w-4 shrink-0 ${doc.id === selected ? 'text-brand-600' : 'text-stone-400'}`} />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium text-stone-800" title={doc.path}>{doc.name}</div>
              <StatusBadge doc={doc} />
            </div>
            {!['queued', 'analyzing', 'saving'].includes(doc.status) && (
              <button
                className="invisible absolute right-2 top-2 rounded p-0.5 text-stone-400 hover:bg-stone-200 hover:text-stone-700 group-hover:visible"
                title={t('nav.remove')}
                onClick={(e) => { e.stopPropagation(); onRemove(doc.id); }}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        ))}
      </nav>

      <div className="border-t border-stone-200 p-3">
        <button className="btn-ghost w-full justify-start" onClick={onSettings}>
          <Settings className="h-4 w-4" />
          {t('nav.settings')}
        </button>
      </div>
    </aside>
  );
}
