import { describe, it, expect } from 'vitest';
import {
  extractEmails,
  extractPhones,
  extractSSN,
  extractDates,
  extractAddresses,
} from '../regex';

describe('extractEmails', () => {
  it('finds a basic email', () => {
    const spans = extractEmails('Email me at john.smith@example.com please.');
    expect(spans).toHaveLength(1);
    expect(spans[0]).toMatchObject({
      type: 'EMAIL',
      text: 'john.smith@example.com',
    });
  });

  it('returns empty when no email present', () => {
    expect(extractEmails('Just a sentence with no email.')).toEqual([]);
  });

  it('finds multiple emails on one line', () => {
    const spans = extractEmails('Contact a@b.io or c.d+tag@x.co.uk');
    expect(spans).toHaveLength(2);
    expect(spans.map((s) => s.text)).toEqual(['a@b.io', 'c.d+tag@x.co.uk']);
  });
});

describe('extractPhones', () => {
  it('finds a NA phone with parens', () => {
    const spans = extractPhones('Call (555) 123-4567 today.');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('(555) 123-4567');
  });

  it('returns empty when only digits with no separator', () => {
    expect(extractPhones('Account 5551234567.')).toEqual([]);
  });

  it('handles country-code prefix and dotted format', () => {
    const spans = extractPhones('+1 415.555.0123 or 212-555-0199');
    expect(spans).toHaveLength(2);
    expect(spans.map((s) => s.text)).toContain('+1 415.555.0123');
    expect(spans.map((s) => s.text)).toContain('212-555-0199');
  });
});

describe('extractSSN', () => {
  it('finds an SSN', () => {
    const spans = extractSSN('SSN: 123-45-6789 on file.');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('123-45-6789');
  });

  it('rejects non-SSN digit patterns', () => {
    expect(extractSSN('Code 1234-56-789 is not SSN.')).toEqual([]);
  });

  it('finds SSN at start of text', () => {
    const spans = extractSSN('999-00-1111 was the test SSN.');
    expect(spans).toHaveLength(1);
    expect(spans[0].start).toBe(0);
  });
});

describe('extractDates', () => {
  it('finds an ISO date', () => {
    const spans = extractDates('DOB 1962-03-14 confirmed.');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('1962-03-14');
  });

  it('returns empty for non-date numeric tokens', () => {
    expect(extractDates('Order 12345 ships tomorrow.')).toEqual([]);
  });

  it('finds named-month date with comma year', () => {
    const spans = extractDates('Effective January 5, 2025 going forward.');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('January 5, 2025');
  });
});

describe('extractAddresses', () => {
  it('finds a street address with suffix', () => {
    const spans = extractAddresses('Lives at 142 Maple St in Burlington.');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('142 Maple St');
  });

  it('returns empty when no street suffix', () => {
    expect(extractAddresses('Lives at 142 Maple in Burlington.')).toEqual([]);
  });

  it('handles multi-word street names with longer suffix', () => {
    const spans = extractAddresses('Send to 1600 Pennsylvania Avenue, DC.');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('1600 Pennsylvania Avenue');
  });
});
