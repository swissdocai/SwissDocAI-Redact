// What the Tauri shell provides: the engine's address, file dialogs, dropped files.
// In a plain browser (development) the engine comes from VITE_ENGINE_URL / VITE_ENGINE_TOKEN
// and files are entered as paths.
import i18n from './i18n';

export const inTauri = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

const FILE_TYPES = ['pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'webp'];

export async function engineInfo(): Promise<{ url: string; token: string } | null> {
  if (!inTauri) {
    const url = import.meta.env.VITE_ENGINE_URL as string | undefined;
    return url ? { url, token: (import.meta.env.VITE_ENGINE_TOKEN as string) ?? '' } : null;
  }
  const { invoke } = await import('@tauri-apps/api/core');
  return invoke<{ url: string; token: string } | null>('engine_info');
}

export async function engineError(): Promise<string | null> {
  if (!inTauri) return null;
  const { invoke } = await import('@tauri-apps/api/core');
  return invoke<string | null>('engine_error');
}

export async function pickFiles(): Promise<string[]> {
  if (!inTauri) {
    const path = window.prompt(i18n.t('empty.pathPrompt'));
    return path ? [path.trim()] : [];
  }
  const { open } = await import('@tauri-apps/plugin-dialog');
  const picked = await open({ multiple: true, filters: [{ name: i18n.t('nav.documents'), extensions: FILE_TYPES }] });
  return picked === null ? [] : Array.isArray(picked) ? picked : [picked];
}

export async function pickSavePath(defaultPath: string): Promise<string | null> {
  if (!inTauri) return window.prompt(i18n.t('empty.pathPrompt'), defaultPath);
  const { save } = await import('@tauri-apps/plugin-dialog');
  return save({ defaultPath, filters: [{ name: 'PDF', extensions: ['pdf'] }] });
}

export async function showInFolder(path: string) {
  if (!inTauri) return;
  const { revealItemInDir } = await import('@tauri-apps/plugin-opener');
  await revealItemInDir(path);
}

export async function onFilesDropped(handler: (paths: string[]) => void, hover: (over: boolean) => void) {
  if (!inTauri) return () => {};
  const { getCurrentWebview } = await import('@tauri-apps/api/webview');
  return getCurrentWebview().onDragDropEvent((event) => {
    const payload = event.payload;
    if (payload.type === 'over' || payload.type === 'enter') hover(true);
    else if (payload.type === 'leave') hover(false);
    else if (payload.type === 'drop') {
      hover(false);
      handler(payload.paths.filter((p) => FILE_TYPES.includes(p.split('.').pop()?.toLowerCase() ?? '')));
    }
  });
}
