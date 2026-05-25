export type EntityType =
  | 'EMAIL'
  | 'PHONE'
  | 'SSN'
  | 'DATE'
  | 'ADDRESS'
  | 'PERSON'
  | 'LOCATION';

export type Span = {
  type: EntityType;
  start: number;
  end: number;
  text: string;
  source: 'regex' | 'ml';
  confidence: number;
};

export type MlSpan = Omit<Span, 'source'> & { source: 'ml' };
export type RegexSpan = Omit<Span, 'source'> & { source: 'regex' };
