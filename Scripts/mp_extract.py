
def main():

    parser = argparse.ArgumentParser(description='Program for downloading Melting Point data:', epilog="")
    parser.add_argument("--dirs", "-d", help="Directory of csv file for melting point data")

    args = parser.parse_args()
    
if __name__ == "__main__":
    import os
    import sys
    import argparse
    import pandas as pd
    import psycopg2
