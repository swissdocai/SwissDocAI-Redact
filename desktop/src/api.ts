// The local engine (python -m swissdocai.app.api): settings, documents, review edits.

export type Box = [number, number, number, number];
export type Status = 'waiting' | 'queued' | 'analyzing' | 'review' | 'saving' | 'saved' | 'failed';

export interface DocSummary {
  id: string;
  name: string;
  path: string;
  status: Status;
  error: string;
  progress: { stage: string; done: number; total: number };
  pages: number;
  savedTo: string;
  verified: boolean | null | undefined;
}

export interface PageView {
  width: number;
  height: number;
  words: { box: Box; text: string; state: 'redacted' | 'kept' | 'manual' | null }[];
  boxes: { box: Box; category: string }[];
  hidden: { box: Box; category: string }[]; // signatures and QR codes switched off
  review: { box: Box; text: string }[];
}

export interface Finding {
  index: number;
  text: string;
  places: number;
  source: string;
  redacted: boolean;
  page: number;
  box: Box;
}

export interface QrCode {
  page: number;
  index: number; // on its page
  box: Box;
  redacted: boolean;
}

export interface Findings {
  groups: { category: string; items: Finding[] }[];
  review: { text: string; page: number; box: Box }[];
  notFound: string[];
  manual: number;
  qrCodes: QrCode[];
}

export interface Policy {
  redact: string[];
  keep: string[];
  notes: string;
}

export interface Settings {
  engine: 'local' | 'swissdocai';
  reader: 'local' | 'swissdocai' | 'none';
  dpi: number;
  language: 'de' | 'fr' | 'it' | 'en';
  policy: Policy;
  policyIsDefault: boolean;
  always_redact_contacts: boolean;
  always_redact_titled_names: boolean;
  redact_qr_codes: boolean;
  swissdocaiToken: string | null; // the stored token's last characters, never the token
  swissdocaiTokenFromEnvironment: boolean;
}

export interface Hardware {
  gpu: string | null;
  vramGB: number;
  backend: 'cuda' | 'rocm' | 'vulkan' | 'metal' | 'sycl' | 'cpu';
  backendName: string; // "NVIDIA CUDA", "AMD ROCm", "Vulkan", "Apple Metal", ...
  chosen: boolean; // tried on this machine and kept (else the one expected before the download)
  runsOn: 'GPU' | 'CPU'; // where the language model runs
  ocr: string | null; // ONNX Runtime's device: Dml, CUDA or CPU
}

export interface SetupState {
  needed: { name: string; gb: number }[];
  running: boolean;
  what: string;
  done: number;
  total: number;
  error: string;
  hardware?: Hardware;
  freeGB?: number;
}

export interface Report {
  output?: string;
  verification?: { ok: boolean; stillReadable: unknown[] };
}

let engine = { url: '', token: '' };

export function connect(url: string, token: string) {
  engine = { url, token };
}

export async function call<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`${engine.url}${path}`, {
    method,
    headers: { 'X-Token': engine.token, ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

export function pageImageUrl(docId: string, n: number): string {
  return `${engine.url}/api/documents/${docId}/pages/${n}/image?token=${encodeURIComponent(engine.token)}`;
}
