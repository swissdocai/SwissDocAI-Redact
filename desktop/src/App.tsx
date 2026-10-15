import {
  CircleAlert, CircleCheck, Eye, FolderOpen, LoaderCircle, MoveHorizontal, Save, ScanSearch, TriangleAlert, Upload,
  X, ZoomIn, ZoomOut,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  call, connect, pageImageUrl, type Box, type DocSummary, type Finding, type Findings, type PageView, type Policy,
  type QrCode, type Report, type Settings, type SetupState,
} from './api';
import { FindingsPanel } from './components/FindingsPanel';
import { DocumentViewer, type Highlight } from './components/PageViewer';
import { SettingsDialog } from './components/SettingsDialog';
import { SetupDialog } from './components/SetupDialog';
import { Sidebar, StatusBadge } from './components/Sidebar';
import i18n from './i18n';
import { engineError, engineInfo, onFilesDropped, pickFiles, pickSavePath, showInFolder } from './platform';

const BUSY = ['queued', 'analyzing', 'saving'];

function Splash({ failed }: { failed: string | null }) {
  const { t } = useTranslation();
  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 bg-white">
      <img src="/logo.png" alt="SwissDocAI" className="h-12 w-auto" draggable={false} />
      {failed ? (
        <div className="max-w-md text-center">
          <p className="flex items-center justify-center gap-2 text-sm font-medium text-swiss-600">
            <CircleAlert className="h-4 w-4" /> {t('engine.failed')}
          </p>
          <p className="mt-2 break-words text-xs text-stone-500">{failed}</p>
          <button className="btn-outline mt-4" onClick={() => window.location.reload()}>{t('engine.retry')}</button>
        </div>
      ) : (
        <p className="flex items-center gap-2 text-sm text-stone-500">
          <LoaderCircle className="h-4 w-4 animate-spin" /> {t('engine.starting')}
        </p>
      )}
    </div>
  );
}

function EmptyState({ onAdd, cloud }: { onAdd: () => void; cloud: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="flex h-full items-center justify-center p-10">
      <button
        onClick={onAdd}
        className="group flex w-full max-w-xl flex-col items-center rounded-2xl border-2 border-dashed border-stone-300 bg-white px-10 py-14 text-center transition-colors hover:border-brand-400 hover:bg-brand-50/40"
      >
        <img src="/logo-icon.png" alt="" className="mb-6 h-16 w-auto opacity-90" draggable={false} />
        <span className="text-lg font-semibold text-stone-900">{t('empty.title')}</span>
        <span className="mt-2 max-w-sm text-sm text-stone-500">{t('empty.text')}</span>
        <span className="btn-primary mt-6"><Upload className="h-4 w-4" /> {t('empty.button')}</span>
        <span className="mt-6 text-xs text-stone-400">{t(cloud ? 'app.taglineCloud' : 'app.tagline')}</span>
      </button>
    </div>
  );
}

/** Review or result: the switch the user needs most, so it is big and coloured. */
function ModeSwitch({ mode, onChange }: { mode: 'review' | 'result'; onChange: (mode: 'review' | 'result') => void }) {
  const { t } = useTranslation();
  const button = 'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-colors';
  return (
    <div className="inline-flex rounded-xl border border-stone-200 bg-stone-100 p-1">
      <button onClick={() => onChange('review')}
        className={`${button} ${mode === 'review' ? 'bg-brand-600 text-white shadow-sm' : 'text-stone-600 hover:bg-white hover:text-stone-900'}`}>
        <ScanSearch className="h-4 w-4" /> {t('toolbar.review')}
      </button>
      <button onClick={() => onChange('result')}
        className={`${button} ${mode === 'result' ? 'bg-stone-900 text-white shadow-sm' : 'text-stone-600 hover:bg-white hover:text-stone-900'}`}>
        <Eye className="h-4 w-4" /> {t('toolbar.result')}
      </button>
    </div>
  );
}

export default function App() {
  const { t } = useTranslation();
  const [connected, setConnected] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [setup, setSetup] = useState<SetupState | null>(null);
  const [showSetup, setShowSetup] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [docs, setDocs] = useState<DocSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [pages, setPages] = useState<(PageView | null)[]>([]);
  const [visiblePage, setVisiblePage] = useState(0);
  const [findings, setFindings] = useState<Findings | null>(null);
  const [mode, setMode] = useState<'review' | 'result'>('review');
  const [zooms, setZooms] = useState<Record<string, number>>({});
  const [lastZoom, setLastZoom] = useState(1);
  const [highlight, setHighlight] = useState<Highlight | null>(null);
  const [dropping, setDropping] = useState(false);
  const [notice, setNotice] = useState<{ name: string; path: string; ok: boolean | null; count: number } | null>(null);
  const selectedRef = useRef<string | null>(null);
  selectedRef.current = selected;

  // the engine: started by the Tauri shell, ready when /api/health answers
  useEffect(() => {
    let stopped = false;
    (async () => {
      for (let i = 0; i < 900 && !stopped; i++) {
        const info = await engineInfo().catch(() => null);
        if (info) {
          connect(info.url, info.token);
          try {
            await call('/api/health');
            setConnected(true);
            return;
          } catch { /* still starting */ }
        }
        const error = await engineError().catch(() => null);
        if (error) return setFailed(error);
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
      if (!stopped) setFailed('timeout');
    })();
    return () => { stopped = true; };
  }, []);

  useEffect(() => {
    if (!connected) return;
    call<Settings>('/api/settings').then((s) => { setSettings(s); i18n.changeLanguage(s.language); });
    call<SetupState>('/api/setup').then((s) => { setSetup(s); if (s.needed.length) setShowSetup(true); });
  }, [connected]);

  useEffect(() => {
    if (!setup?.running) return;
    const timer = setInterval(async () => {
      const state = await call<SetupState>('/api/setup');
      setSetup(state);
      if (!state.running && !state.error && state.needed.length === 0) setShowSetup(false);
    }, 500);
    return () => clearInterval(timer);
  }, [setup?.running]);

  const refreshDocs = useCallback(async () => {
    const list = await call<DocSummary[]>('/api/documents');
    setDocs(list);
    return list;
  }, []);
  const busy = docs.some((d) => BUSY.includes(d.status));
  useEffect(() => {
    if (!connected) return;
    refreshDocs();
    const timer = setInterval(refreshDocs, busy ? 700 : 3000);
    return () => clearInterval(timer);
  }, [connected, busy, refreshDocs]);

  const current = docs.find((d) => d.id === selected) ?? null;
  const pageCount = current?.pages ?? 0;
  const hasJob = pageCount > 0;
  const zoom = selected ? zooms[selected] ?? lastZoom : lastZoom;
  const setZoom = (value: number) => {
    if (selected) setZooms({ ...zooms, [selected]: value });
    setLastZoom(value);
  };

  const loadFindings = useCallback(async (id: string) => setFindings(await call<Findings>(`/api/documents/${id}/findings`)), []);
  const loadPage = useCallback(async (id: string, n: number) => {
    const view = await call<PageView>(`/api/documents/${id}/pages/${n}`);
    setPages((all) => (selectedRef.current === id ? all.map((p, i) => (i === n ? view : p)) : all));
  }, []);
  useEffect(() => {
    if (!current || !hasJob) {
      setPages([]);
      setFindings(null);
      return;
    }
    setPages(Array(pageCount).fill(null));
    for (let n = 0; n < pageCount; n++) loadPage(current.id, n);
    loadFindings(current.id);
  }, [current?.id, hasJob, pageCount, loadPage, loadFindings]);

  // a document that just finished saving: say whether reading it again found anything
  const previous = useRef<Record<string, string>>({});
  useEffect(() => {
    for (const doc of docs) {
      if (previous.current[doc.id] === 'saving' && doc.status === 'saved') {
        call<Report>(`/api/documents/${doc.id}/report`).then((report) => setNotice({
          name: doc.savedTo.split(/[\\/]/).pop() ?? doc.name, path: doc.savedTo,
          ok: report.verification?.ok ?? null, count: report.verification?.stillReadable.length ?? 0,
        }));
      }
      previous.current[doc.id] = doc.status;
    }
  }, [docs]);

  useEffect(() => {
    if (!highlight) return;
    const timer = setTimeout(() => setHighlight(null), 2500);
    return () => clearTimeout(timer);
  }, [highlight]);

  const select = (id: string | null) => {
    setSelected(id);
    setVisiblePage(0);
    setHighlight(null);
    setMode('review');
    setNotice(null);
  };

  const addPaths = useCallback(async (paths: string[]) => {
    if (!paths.length) return;
    const added = await call<DocSummary[]>('/api/documents', 'POST', { paths });
    const list = await refreshDocs();
    const first = added[0] ?? list[0];
    if (!selectedRef.current && first) select(first.id);
  }, [refreshDocs]);

  useEffect(() => {
    if (!connected) return;
    let unlisten: (() => void) | undefined;
    onFilesDropped(addPaths, setDropping).then((fn) => { unlisten = fn; });
    return () => unlisten?.();
  }, [connected, addPaths]);

  const analyze = async (ids?: string[]) => {
    if (setup?.needed.length) return setShowSetup(true);
    await call('/api/analyze', 'POST', ids ? { ids } : {});
    refreshDocs();
  };

  const save = async () => {
    if (!current) return;
    const target = await pickSavePath(`${current.path.replace(/\.[^.\\/]+$/, '')}.redacted.pdf`);
    if (!target) return;
    setNotice(null);
    await call(`/api/documents/${current.id}/save`, 'POST', { path: target });
    refreshDocs();
  };

  const remove = async (id: string) => {
    await call(`/api/documents/${id}`, 'DELETE');
    if (id === selected) select(null);
    refreshDocs();
  };

  const setPage = (n: number, view: PageView) => setPages((all) => all.map((p, i) => (i === n ? view : p)));
  const click = async (n: number, x: number, y: number) => {
    if (!current) return;
    const result = await call<{ changed: boolean; page: PageView }>(`/api/documents/${current.id}/pages/${n}/click`, 'POST', { x, y });
    if (result.changed) {
      setPage(n, result.page);
      loadFindings(current.id);
    }
  };
  const drawBox = async (n: number, box: Box) => {
    if (!current) return;
    setPage(n, await call<PageView>(`/api/documents/${current.id}/pages/${n}/boxes`, 'POST', { box }));
    loadFindings(current.id);
  };
  const toggle = async (item: Finding, redacted: boolean) => {
    if (!current) return;
    setFindings(await call<Findings>(`/api/documents/${current.id}/entities/${item.index}`, 'POST', { redacted }));
    for (let n = 0; n < pageCount; n++) loadPage(current.id, n);
  };
  const toggleQrCode = async (code: QrCode, redacted: boolean) => {
    if (!current) return;
    setFindings(await call<Findings>(`/api/documents/${current.id}/pages/${code.page}/qr-codes/${code.index}`, 'POST', { redacted }));
    loadPage(current.id, code.page);
  };
  const jump = (n: number, box: Box) => {
    setMode('review');
    setHighlight({ page: n, box });
  };

  const saveSettings = async (changes: Omit<Partial<Settings>, 'policy'> & { policy: Policy | null; swissdocai_token?: string }) => {
    const updated = await call<Settings>('/api/settings', 'PUT', changes);
    setSettings(updated);
    i18n.changeLanguage(updated.language);
    setShowSettings(false);
    setSetup(await call<SetupState>('/api/setup'));
  };

  if (!connected) return <Splash failed={failed} />;

  const waiting = docs.filter((d) => d.status === 'waiting' || d.status === 'failed');
  return (
    <div className="flex h-full">
      <Sidebar
        docs={docs}
        selected={selected}
        onSelect={select}
        onAdd={async () => addPaths(await pickFiles())}
        onRemove={remove}
        onSettings={() => setShowSettings(true)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        {!current ? (
          <EmptyState onAdd={async () => addPaths(await pickFiles())}
            cloud={!!settings && (settings.reader === 'swissdocai' || settings.engine === 'swissdocai')} />
        ) : (
          <>
            <header className="flex h-16 shrink-0 items-center gap-4 border-b border-stone-200 bg-white px-5">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-semibold text-stone-900" title={current.path}>{current.name}</div>
                <StatusBadge doc={current} />
              </div>
              {hasJob && (
                <>
                  {pageCount > 1 && (
                    <span className="hidden text-xs tabular-nums text-stone-500 lg:inline">
                      {t('toolbar.page', { n: visiblePage + 1, total: pageCount })}
                    </span>
                  )}
                  <div className="flex items-center">
                    <button className="icon-btn" title={t('toolbar.zoomOut')} onClick={() => setZoom(Math.max(0.4, zoom / 1.2))}><ZoomOut className="h-4 w-4" /></button>
                    <button className="icon-btn" title={t('toolbar.fit')} onClick={() => setZoom(1)}><MoveHorizontal className="h-4 w-4" /></button>
                    <button className="icon-btn" title={t('toolbar.zoomIn')} onClick={() => setZoom(Math.min(4, zoom * 1.2))}><ZoomIn className="h-4 w-4" /></button>
                  </div>
                  <ModeSwitch mode={mode} onChange={setMode} />
                </>
              )}
              {(current.status === 'waiting' || current.status === 'failed') && (
                <>
                  {waiting.length > 1 && <button className="btn-outline" onClick={() => analyze()}>{t('toolbar.findAll')}</button>}
                  <button className="btn-primary" onClick={() => analyze([current.id])}>
                    <ScanSearch className="h-4 w-4" /> {t('toolbar.find')}
                  </button>
                </>
              )}
              {hasJob && (
                <button className="btn-dark" onClick={save} disabled={current.status === 'saving'}>
                  {current.status === 'saving' ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                  {t('toolbar.save')}
                </button>
              )}
            </header>

            {notice && (
              <div className={`flex items-center gap-3 border-b px-5 py-2.5 text-sm ${notice.ok === false ? 'border-amber-200 bg-amber-50 text-amber-900' : 'border-emerald-200 bg-emerald-50 text-emerald-900'}`}>
                {notice.ok === false ? <TriangleAlert className="h-4 w-4 shrink-0" /> : <CircleCheck className="h-4 w-4 shrink-0" />}
                <span className="min-w-0 flex-1 truncate">
                  {notice.ok === null ? t('saved.plain', { name: notice.name })
                    : notice.ok ? t('saved.verified', { name: notice.name })
                      : t('saved.unverified', { name: notice.name, count: notice.count })}
                </span>
                <button className="btn-ghost py-1" onClick={() => showInFolder(notice.path)}><FolderOpen className="h-4 w-4" /> {t('saved.show')}</button>
                <button className="icon-btn" onClick={() => setNotice(null)}><X className="h-4 w-4" /></button>
              </div>
            )}

            <div className="relative min-h-0 flex-1">
              {hasJob ? (
                <>
                  <DocumentViewer
                    key={current.id}
                    pages={pages}
                    imageUrl={(n) => pageImageUrl(current.id, n)}
                    mode={mode}
                    zoom={zoom}
                    highlight={highlight}
                    onClick={click}
                    onBox={drawBox}
                    onVisiblePage={setVisiblePage}
                  />
                  {mode === 'review' && (
                    <div className="pointer-events-none absolute bottom-4 left-1/2 hidden max-w-[calc(100%-2rem)] -translate-x-1/2 truncate whitespace-nowrap rounded-full bg-stone-900/85 px-4 py-1.5 text-xs text-white shadow-lg lg:block">
                      {t('viewer.hint')}
                    </div>
                  )}
                </>
              ) : BUSY.includes(current.status) ? (
                <div className="flex h-full flex-col items-center justify-center gap-3 text-stone-500">
                  <LoaderCircle className="h-8 w-8 animate-spin text-brand-600" />
                  <StatusBadge doc={current} />
                </div>
              ) : current.status === 'failed' ? (
                <div className="flex h-full items-center justify-center p-10">
                  <div className="card max-w-lg p-6">
                    <p className="flex items-center gap-2 text-sm font-semibold text-swiss-600"><CircleAlert className="h-4 w-4" /> {t('viewer.failed')}</p>
                    <p className="mt-2 break-words text-xs text-stone-500">{current.error}</p>
                  </div>
                </div>
              ) : (
                <div className="flex h-full items-center justify-center p-10">
                  <div className="card max-w-md p-8 text-center">
                    <ScanSearch className="mx-auto mb-4 h-10 w-10 text-brand-500" />
                    <p className="text-base font-semibold">{t('viewer.waiting')}</p>
                    <p className="mt-1 text-sm text-stone-500">{t('viewer.waitingText')}</p>
                    <button className="btn-primary mt-5" onClick={() => analyze([current.id])}>
                      <ScanSearch className="h-4 w-4" /> {t('toolbar.find')}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </main>

      {hasJob && <FindingsPanel findings={findings} onToggle={toggle} onToggleQrCode={toggleQrCode} onJump={jump} />}

      {dropping && (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-brand-600/10 ring-4 ring-inset ring-brand-500">
          <div className="card flex items-center gap-3 px-6 py-4 text-sm font-medium text-brand-700">
            <Upload className="h-5 w-5" /> {t('empty.title')}
          </div>
        </div>
      )}
      {showSettings && settings && <SettingsDialog settings={settings} onClose={() => setShowSettings(false)} onSave={saveSettings} />}
      {showSetup && setup && (
        <SetupDialog
          setup={setup}
          onStart={async () => setSetup(await call<SetupState>('/api/setup', 'POST'))}
          onLater={() => setShowSetup(false)}
        />
      )}
    </div>
  );
}
