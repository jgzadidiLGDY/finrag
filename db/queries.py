import re, sqlite3
DB_PATH = "data/artifacts/finrag.db"

TICKER_RE = re.compile(r"^[A-Z]{1,6}$")  #pre-compiles the regex pattern for efficiency
VALID_QUARTERS = {"Q1","Q2","Q3","Q4"}

def _conn():
    return sqlite3.connect(DB_PATH)

def _validate_ticker(ticker: str) -> str:
    """
    it ensures a string meets the requirements of a stock ticker symbol. 
      > The validation is based on "regex" 
      > return the uppercase version of the ticker symbol
    """   
    t = (ticker or "").upper()
    if not TICKER_RE.match(t): 
        raise ValueError("Invalid ticker")
    return t

def _validate_quarters(qs):
    """
    it  validates a list of strings to ensure they are 
    valid financial quarters (Q1, Q2, Q3, or Q4)
    """  
    qs = [q.upper() for q in qs]
    if any(q not in VALID_QUARTERS for q in qs): 
        raise ValueError("Invalid quarter")
    return qs

def _validate_year(y: int) -> int:
    """
    it validates year
    """      
    y = int(y)
    if not (1990 <= y <= 2100):
        raise ValueError("Invalid year")
    return y

def get_eps(ticker: str, fiscal_year: int, quarters: list[str]):
    """
    it retrieves earnings-per-share (EPS) data for a given stock, fiscal year, and list of quarters from a database. 
       > It first validates the inputs using separate validation functions 
       > then executes a SQL query to fetch the requested data
    """ 
    t = _validate_ticker(ticker)
    y = _validate_year(fiscal_year)
    qs = _validate_quarters(quarters)
    qmarks = ",".join("?"*len(qs))
    sql = f"""SELECT fiscal_quarter, eps FROM financials
              WHERE ticker=? AND fiscal_year=? AND fiscal_quarter IN ({qmarks})
              ORDER BY CASE fiscal_quarter WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2
                                           WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END"""
    with _conn() as conn:
        return conn.execute(sql, [t, y, *qs]).fetchall()

def get_revenue(ticker: str, fiscal_year: int, quarters: list[str]):
    """
    it retrieves revenue data for a given stock, fiscal year, and list of quarters from a database. 
       > It first validates the inputs using separate validation functions 
       > then executes a SQL query to fetch the requested data
    """ 
    t = _validate_ticker(ticker) 
    y = _validate_year(fiscal_year) 
    qs = _validate_quarters(quarters)
    qmarks = ",".join("?"*len(qs))
    sql = f"""SELECT fiscal_quarter, revenue FROM financials
              WHERE ticker=? AND fiscal_year=? AND fiscal_quarter IN ({qmarks})
              ORDER BY CASE fiscal_quarter WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2
                                           WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END"""
    with _conn() as conn:
        return conn.execute(sql, [t, y, *qs]).fetchall()

def get_financials_row(ticker: str, fiscal_year: int, quarter: str):
    """
    it retrieves a single row of financial data for a specific stock ticker, fiscal year, and quarter from a database. 
    """ 
    t = _validate_ticker(ticker) 
    y = _validate_year(fiscal_year)
    q = _validate_quarters([quarter])[0]
    sql = """SELECT fiscal_quarter, revenue, eps FROM financials
             WHERE ticker=? AND fiscal_year=? AND fiscal_quarter=?"""
    with _conn() as conn:
        return conn.execute(sql, [t, y, q]).fetchone()
