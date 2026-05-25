/**
 * IndexedDB-backed log of per-inference observability records. One record per inference;
 * exportable as NDJSON for offline analysis.
 */
export type LogRecord = {
  ts: number;
  textLength: number;
  spansCount: number;
  latencyMs: number;
  backend: string;
  version: number;
};

const DB_NAME = 'pii-redactor';
const STORE = 'inference-log';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function logInference(rec: LogRecord): Promise<void> {
  const db = await open();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE).add(rec);
  });
  db.close();
}

export async function readAll(): Promise<LogRecord[]> {
  const db = await open();
  return new Promise<LogRecord[]>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => {
      resolve(req.result as LogRecord[]);
      db.close();
    };
    req.onerror = () => reject(req.error);
  });
}

export async function clearLog(): Promise<void> {
  const db = await open();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE).clear();
  });
  db.close();
}

export function downloadNdjson(records: LogRecord[], filename = 'inference-log.ndjson') {
  const ndjson = records.map((r) => JSON.stringify(r)).join('\n');
  const blob = new Blob([ndjson], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
