PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS financials (
  ticker TEXT NOT NULL,
  fiscal_year INTEGER NOT NULL CHECK (fiscal_year BETWEEN 1990 AND 2100),
  fiscal_quarter TEXT NOT NULL CHECK (fiscal_quarter IN ('Q1','Q2','Q3','Q4')),
  revenue REAL,     -- in USD millions (choose and be consistent)
  eps REAL,         -- GAAP EPS
  PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
);

CREATE INDEX IF NOT EXISTS idx_financials_ticker_year
  ON financials (ticker, fiscal_year);
