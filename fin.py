import argparse   # handle command-line arguments
from db.service import get_eps_and_rev

def main():
    #an ArgumentParser object to to parse command-line arguments
    p = argparse.ArgumentParser()
    #required=True flag means the user must provide this argument
    p.add_argument("--ticker", required=True)
    p.add_argument("--year", type=int, required=True)
    #one or more quarters 
    p.add_argument("--quarters", nargs="+", default=["Q1","Q2","Q3","Q4"])
    args = p.parse_args()

    res = get_eps_and_rev(args.ticker, args.year, args.quarters)
    print(f"{res['ticker']}  FY{res['fiscal_year']}")
    print("Quarter  |  EPS   | Revenue (USD mm)")
    print("-------------------------------------")
    for row in res["data"]:
        print(f"{row['quarter']:>7} | {row['eps']!s:>5} | {row['revenue']!s:>8}")

if __name__ == "__main__": main()
