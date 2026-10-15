import { Ban, Cloud, Cpu, Info, KeyRound, Plus, RotateCcw, ShieldCheck, X } from 'lucide-react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { call, type Policy, type Settings } from '../api';
import { LANGUAGES, type Language } from '../i18n';

interface Props {
  settings: Settings;
  onClose: () => void;
  onSave: (changes: Omit<Partial<Settings>, 'policy'> & { policy: Policy | null; swissdocai_token?: string }) => Promise<void>;
}

function AutoTextarea({ value, onChange, focus }: { value: string; onChange: (v: string) => void; focus: boolean }) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const fit = () => {
    const element = ref.current;
    if (!element || element.clientWidth < 40) return; // not laid out yet
    element.style.height = '0px';
    element.style.height = `${element.scrollHeight + 2}px`;
  };
  useLayoutEffect(fit, [value]);
  useLayoutEffect(() => {
    // the width is known only once the dialog is laid out, and changes with the window
    let width = 0;
    const observer = new ResizeObserver(([entry]) => {
      if (Math.abs(entry.contentRect.width - width) > 1) {
        width = entry.contentRect.width;
        fit();
      }
    });
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (focus) ref.current?.focus();
  }, [focus]);
  return (
    <textarea
      ref={ref}
      value={value}
      rows={1}
      onChange={(e) => onChange(e.target.value.replace(/\n/g, ' '))}
      onKeyDown={(e) => { if (e.key === 'Enter') e.preventDefault(); }}
      className="field resize-none overflow-hidden leading-snug"
    />
  );
}

function ItemList({ items, onChange, accent }: { items: string[]; onChange: (items: string[]) => void; accent: string }) {
  const { t } = useTranslation();
  const [added, setAdded] = useState(-1);
  return (
    <div className="space-y-1.5">
      {items.map((item, i) => (
        <div key={i} className="group flex items-start gap-2">
          <span className={`mt-3.5 h-1.5 w-1.5 shrink-0 rounded-full ${accent}`} />
          <AutoTextarea
            value={item}
            focus={i === added}
            onChange={(value) => onChange(items.map((x, j) => (j === i ? value : x)))}
          />
          <button
            className="icon-btn mt-1 opacity-0 group-hover:opacity-100"
            title={t('settings.removeItem')}
            onClick={() => onChange(items.filter((_, j) => j !== i))}
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      ))}
      <button className="btn-ghost -ml-1 text-brand-700" onClick={() => { setAdded(items.length); onChange([...items, '']); }}>
        <Plus className="h-4 w-4" />
        {t('settings.addItem')}
      </button>
    </div>
  );
}

function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint: string }) {
  return (
    <label className="flex cursor-pointer items-start gap-3">
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors ${checked ? 'bg-brand-600' : 'bg-stone-300'}`}
      >
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${checked ? 'left-[18px]' : 'left-0.5'}`} />
      </button>
      <span>
        <span className="block text-sm font-medium text-stone-800">{label}</span>
        <span className="block text-xs text-stone-500">{hint}</span>
      </span>
    </label>
  );
}

export function SettingsDialog({ settings, onClose, onSave }: Props) {
  const { t } = useTranslation();
  const [tab, setTab] = useState<'policy' | 'general'>('policy');
  const [draft, setDraft] = useState<Settings>(settings);
  const [policyTouched, setPolicyTouched] = useState(false);
  const [saving, setSaving] = useState(false);
  const [token, setToken] = useState('');

  const setPolicy = (policy: Partial<Policy>) => {
    setDraft({ ...draft, policy: { ...draft.policy, ...policy } });
    setPolicyTouched(true);
  };
  const reset = async () => {
    const standard = await call<Policy>(`/api/policy/default?language=${draft.language}`);
    setDraft({ ...draft, policy: standard, policyIsDefault: true });
    setPolicyTouched(false);
  };
  const save = async () => {
    setSaving(true);
    const clean = (items: string[]) => items.map((x) => x.trim()).filter(Boolean);
    const policy = policyTouched
      ? { redact: clean(draft.policy.redact), keep: clean(draft.policy.keep), notes: draft.policy.notes.trim() }
      : draft.policyIsDefault ? null : draft.policy;
    await onSave({ reader: draft.reader, engine: draft.engine, dpi: draft.dpi, language: draft.language,
      always_redact_contacts: draft.always_redact_contacts, always_redact_titled_names: draft.always_redact_titled_names,
      redact_qr_codes: draft.redact_qr_codes, policy, ...(token.trim() ? { swissdocai_token: token.trim() } : {}) });
    setSaving(false);
  };
  const isDefault = draft.policyIsDefault && !policyTouched;
  // pages of SwissDocAI credit per document page: one for reading it, one for finding the personal data
  const cloudPages = (draft.reader === 'swissdocai' ? 1 : 0) + (draft.engine === 'swissdocai' ? 1 : 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/30 p-6 backdrop-blur-[2px]">
      <div className="card flex max-h-full w-full max-w-3xl flex-col">
        <div className="flex items-center justify-between border-b border-stone-200 px-6 py-4">
          <h2 className="text-base font-semibold">{t('settings.title')}</h2>
          <button className="icon-btn" onClick={onClose}><X className="h-4 w-4" /></button>
        </div>
        <div className="flex gap-1 border-b border-stone-200 px-6">
          {(['policy', 'general'] as const).map((key) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`-mb-px border-b-2 px-3 py-2.5 text-sm font-medium ${
                tab === key ? 'border-brand-600 text-brand-700' : 'border-transparent text-stone-500 hover:text-stone-800'
              }`}
            >
              {t(key === 'policy' ? 'settings.tabPolicy' : 'settings.tabGeneral')}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          {tab === 'policy' ? (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${isDefault ? 'bg-stone-100 text-stone-600' : 'bg-brand-50 text-brand-700'}`}>
                  {t(isDefault ? 'settings.isDefault' : 'settings.isCustom')}
                </span>
                <button className="btn-ghost" onClick={reset} disabled={isDefault}>
                  <RotateCcw className="h-4 w-4" />
                  {t('settings.reset')}
                </button>
              </div>
              <section>
                <h3 className="text-sm font-semibold text-stone-900">{t('settings.redact')}</h3>
                <p className="mb-2 text-xs text-stone-500">{t('settings.redactHint')}</p>
                <ItemList items={draft.policy.redact} onChange={(redact) => setPolicy({ redact })} accent="bg-stone-900" />
              </section>
              <section>
                <h3 className="text-sm font-semibold text-stone-900">{t('settings.keep')}</h3>
                <p className="mb-2 text-xs text-stone-500">{t('settings.keepHint')}</p>
                <ItemList items={draft.policy.keep} onChange={(keep) => setPolicy({ keep })} accent="bg-emerald-500" />
              </section>
              <section>
                <h3 className="mb-2 text-sm font-semibold text-stone-900">{t('settings.notes')}</h3>
                <textarea
                  className="field min-h-[72px]"
                  value={draft.policy.notes}
                  placeholder={t('settings.notesPlaceholder')}
                  onChange={(e) => setPolicy({ notes: e.target.value })}
                />
              </section>
              <section className="space-y-4 rounded-lg border border-stone-200 bg-stone-50 p-4">
                <Toggle
                  checked={draft.always_redact_contacts}
                  onChange={(v) => setDraft({ ...draft, always_redact_contacts: v })}
                  label={t('settings.contactsRule')}
                  hint={t('settings.contactsRuleHint')}
                />
                <Toggle
                  checked={draft.always_redact_titled_names}
                  onChange={(v) => setDraft({ ...draft, always_redact_titled_names: v })}
                  label={t('settings.titledRule')}
                  hint={t('settings.titledRuleHint')}
                />
                <Toggle
                  checked={draft.redact_qr_codes}
                  onChange={(v) => setDraft({ ...draft, redact_qr_codes: v })}
                  label={t('settings.qrRule')}
                  hint={t('settings.qrRuleHint')}
                />
              </section>
              <p className="flex items-start gap-2 text-xs text-stone-500">
                <ShieldCheck className="h-4 w-4 shrink-0 text-stone-400" />
                {t('settings.fixedHint')}
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              <section>
                <h3 className="mb-1 text-sm font-semibold text-stone-900">{t('settings.readerTitle')}</h3>
                <p className="mb-3 text-xs text-stone-500">{t('settings.readerIntro')}</p>
                <div className="grid grid-cols-3 gap-3">
                  {([['local', Cpu, 'settings.readerLocal', 'settings.readerLocalHint'],
                     ['swissdocai', Cloud, 'settings.readerCloud', 'settings.readerCloudHint'],
                     ['none', Ban, 'settings.readerNone', 'settings.readerNoneHint']] as const).map(([value, Icon, label, hint]) => (
                    <button
                      key={value}
                      onClick={() => setDraft({ ...draft, reader: value })}
                      className={`rounded-xl border p-4 text-left transition-colors ${
                        draft.reader === value ? 'border-brand-500 bg-brand-50 ring-2 ring-brand-500/20' : 'border-stone-200 hover:bg-stone-50'
                      }`}
                    >
                      <Icon className={`mb-2 h-5 w-5 ${draft.reader === value ? 'text-brand-600' : 'text-stone-400'}`} />
                      <div className="text-sm font-semibold text-stone-900">{t(label)}</div>
                      <div className="text-xs text-stone-500">{t(hint)}</div>
                    </button>
                  ))}
                </div>
                {draft.reader === 'none' && (
                  <p className="mt-3 rounded-lg border border-stone-200 bg-stone-50 p-4 text-xs leading-relaxed text-stone-600">
                    {t('settings.readerNoneDetail')}
                  </p>
                )}
              </section>
              <section>
                <h3 className="mb-1 text-sm font-semibold text-stone-900">{t('settings.piiTitle')}</h3>
                <p className="mb-3 text-xs text-stone-500">{t('settings.piiIntro')}</p>
                <div className="grid grid-cols-2 gap-3">
                  {([['local', Cpu, 'settings.piiLocal', 'settings.piiLocalHint'],
                     ['swissdocai', Cloud, 'settings.piiCloud', 'settings.piiCloudHint']] as const).map(([value, Icon, label, hint]) => (
                    <button
                      key={value}
                      onClick={() => setDraft({ ...draft, engine: value })}
                      className={`rounded-xl border p-4 text-left transition-colors ${
                        draft.engine === value ? 'border-brand-500 bg-brand-50 ring-2 ring-brand-500/20' : 'border-stone-200 hover:bg-stone-50'
                      }`}
                    >
                      <Icon className={`mb-2 h-5 w-5 ${draft.engine === value ? 'text-brand-600' : 'text-stone-400'}`} />
                      <div className="text-sm font-semibold text-stone-900">{t(label)}</div>
                      <div className="text-xs text-stone-500">{t(hint)}</div>
                    </button>
                  ))}
                </div>
              </section>
              {cloudPages > 0 && (
                <section className="rounded-lg border border-stone-200 bg-stone-50 p-4">
                  <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-stone-900">
                    <Cloud className="h-4 w-4 text-brand-600" /> {t('settings.cloudTitle')}
                  </div>
                  <p className="mb-3 text-xs leading-relaxed text-stone-600">
                    {t(cloudPages === 2 ? 'settings.cloudCostBoth' : 'settings.cloudCostOne')}
                  </p>
                  <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-stone-800">
                    <KeyRound className="h-4 w-4 text-stone-400" /> {t('settings.token')}
                  </label>
                  {draft.swissdocaiTokenFromEnvironment ? (
                    <p className="text-xs text-stone-500">{t('settings.tokenFromEnv')}</p>
                  ) : (
                    <>
                      <input
                        type="password"
                        className="field font-mono"
                        placeholder="sd_…"
                        value={token}
                        autoComplete="off"
                        onChange={(e) => setToken(e.target.value)}
                      />
                      <p className={`mt-1.5 text-xs ${draft.swissdocaiToken || token ? 'text-stone-500' : 'text-swiss-600'}`}>
                        {draft.swissdocaiToken ? t('settings.tokenSaved', { hint: draft.swissdocaiToken })
                          : token ? t('settings.tokenHint') : t('settings.tokenMissing')}
                      </p>
                    </>
                  )}
                </section>
              )}
              <section className="grid grid-cols-2 gap-6">
                <div>
                  <h3 className="mb-2 text-sm font-semibold text-stone-900">{t('settings.resolution')}</h3>
                  <div className="inline-flex rounded-lg border border-stone-200 p-0.5">
                    {[150, 200, 300].map((dpi) => (
                      <button
                        key={dpi}
                        onClick={() => setDraft({ ...draft, dpi })}
                        className={`rounded-md px-3 py-1.5 text-sm ${draft.dpi === dpi ? 'bg-stone-900 text-white' : 'text-stone-600 hover:bg-stone-100'}`}
                      >
                        {dpi} dpi
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="mb-2 text-sm font-semibold text-stone-900">{t('settings.language')}</h3>
                  <select
                    className="field"
                    value={draft.language}
                    onChange={(e) => setDraft({ ...draft, language: e.target.value as Language })}
                  >
                    {Object.entries(LANGUAGES).map(([code, name]) => <option key={code} value={code}>{name}</option>)}
                  </select>
                </div>
              </section>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between border-t border-stone-200 px-6 py-4">
          <span className="flex items-center gap-1.5 text-xs text-stone-500">
            <Info className="h-3.5 w-3.5" />
            {t('settings.nextDocument')}
          </span>
          <div className="flex gap-2">
            <button className="btn-outline" onClick={onClose}>{t('settings.cancel')}</button>
            <button className="btn-primary" onClick={save} disabled={saving}>{t('settings.save')}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
