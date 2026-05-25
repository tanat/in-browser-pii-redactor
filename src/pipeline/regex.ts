import type { EntityType, Span } from '../types/span';

function makeSpan(type: EntityType, start: number, end: number, text: string): Span {
  return { type, start, end, text, source: 'regex', confidence: 1.0 };
}

function runRegex(text: string, type: EntityType, re: RegExp): Span[] {
  const out: Span[] = [];
  for (const m of text.matchAll(re)) {
    if (m.index === undefined) continue;
    out.push(makeSpan(type, m.index, m.index + m[0].length, m[0]));
  }
  return out;
}

export const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;
export function extractEmails(text: string): Span[] {
  return runRegex(text, 'EMAIL', EMAIL_RE);
}

// North-American phone numbers, optional country code, optional parens around area code.
export const PHONE_RE =
  /(?<![\d-])(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)/g;
export function extractPhones(text: string): Span[] {
  return runRegex(text, 'PHONE', PHONE_RE);
}

export const SSN_RE = /\b\d{3}-\d{2}-\d{4}\b/g;
export function extractSSN(text: string): Span[] {
  return runRegex(text, 'SSN', SSN_RE);
}

const MONTH =
  '(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)';
export const DATE_ISO_RE = /\b\d{4}-\d{2}-\d{2}\b/g;
export const DATE_NAMED_RE = new RegExp(`\\b${MONTH}\\s+\\d{1,2}(?:,\\s*\\d{2,4})?\\b`, 'g');
export function extractDates(text: string): Span[] {
  return [...runRegex(text, 'DATE', DATE_ISO_RE), ...runRegex(text, 'DATE', DATE_NAMED_RE)].sort(
    (a, b) => a.start - b.start,
  );
}

export const ADDRESS_RE =
  /\b\d{1,6}\s+[A-Za-z][A-Za-z0-9.\s]*?\s(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Way|Drive|Dr|Court|Ct)\b\.?/g;
export function extractAddresses(text: string): Span[] {
  return runRegex(text, 'ADDRESS', ADDRESS_RE);
}

export function extractAllRegex(text: string): Span[] {
  return [
    ...extractEmails(text),
    ...extractPhones(text),
    ...extractSSN(text),
    ...extractDates(text),
    ...extractAddresses(text),
  ].sort((a, b) => a.start - b.start);
}
