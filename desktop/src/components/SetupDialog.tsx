import { Cpu, Download, HardDrive, LoaderCircle, Zap } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { SetupState } from '../api';

interface Props {
  setup: SetupState;
  onStart: () => void;
  onLater: () => void;
}

export function SetupDialog({ setup, onStart, onLater }: Props) {
  const { t } = useTranslation();
  const gb = setup.needed.reduce((sum, item) => sum + item.gb, 0);
  const share = setup.total ? setup.done / setup.total : 0;
  const hw = setup.hardware;
  // what runs the local models; not shown when only the OCR model is missing (both models in the cloud)
  const localModels = setup.needed.some((item) => item.name.startsWith('llama.cpp') || item.name.endsWith('.gguf'));
  const tooLittleSpace = setup.freeGB !== undefined && setup.freeGB < gb + 0.5;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/30 p-6 backdrop-blur-[2px]">
      <div className="card w-full max-w-lg p-8">
        <img src="/logo.png" alt="SwissDocAI" className="mb-6 h-9 w-auto" draggable={false} />
        <h2 className="text-lg font-semibold">{t('setup.title')}</h2>
        <p className="mt-2 text-sm text-stone-600">{t('setup.text', { gb: gb.toFixed(1) })}</p>
        {hw && localModels && (
          <p className="mt-3 flex gap-2 text-sm text-stone-600">
            {hw.backend === 'cpu' || hw.runsOn === 'CPU'
              ? <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-stone-400" />
              : <Zap className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" />}
            <span>
              {hw.backend === 'cpu' || !hw.gpu
                ? t('setup.cpu')
                : t(hw.runsOn === 'CPU' ? 'setup.gpuSmall' : 'setup.gpu', { gpu: hw.gpu, backend: hw.backendName })}
            </span>
          </p>
        )}
        <ul className="mt-4 space-y-1 rounded-lg bg-stone-50 p-3 text-sm text-stone-600">
          {setup.needed.map((item) => (
            <li key={item.name} className="flex justify-between gap-4">
              <span className="truncate">{item.name}</span>
              <span className="shrink-0 tabular-nums text-stone-400">{item.gb.toFixed(2)} GB</span>
            </li>
          ))}
        </ul>
        {tooLittleSpace && !setup.running && (
          <p className="mt-3 flex gap-2 text-sm text-swiss-600">
            <HardDrive className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{t('setup.disk', { need: gb.toFixed(1), free: setup.freeGB?.toFixed(1) })}</span>
          </p>
        )}
        {setup.running && (
          <div className="mt-5">
            <div className="h-2 overflow-hidden rounded-full bg-stone-100">
              <div className="h-full rounded-full bg-brand-600 transition-all" style={{ width: `${Math.round(share * 100)}%` }} />
            </div>
            <p className="mt-2 truncate text-xs text-stone-500">
              {t('setup.progress', { what: setup.what, done: (setup.done / 1e9).toFixed(2), total: (setup.total / 1e9).toFixed(2) })}
            </p>
          </div>
        )}
        {setup.error && <p className="mt-4 text-sm text-swiss-600">{t('setup.failed', { error: setup.error })}</p>}
        <div className="mt-6 flex justify-end gap-2">
          <button className="btn-outline" onClick={onLater} disabled={setup.running}>{t('setup.later')}</button>
          <button className="btn-primary" onClick={onStart} disabled={setup.running}>
            {setup.running ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            {t(setup.error ? 'setup.resume' : 'setup.download')}
          </button>
        </div>
      </div>
    </div>
  );
}
