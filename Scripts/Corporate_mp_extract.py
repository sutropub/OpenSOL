
def main():
    """
    a. Here we show one example to merge data from a PostgreSQL database to the CSD dataset: ./../Datasets/CSD_melting_point.csv
    b. The melting point data pulled from a database should contain four columns in the following order:
        * CID: compound ID
        * MP: melting point measurement
        * smiles: SMILES structure of a compound
        * unit: C (Celsius) or K (Kelvin), the default is C.
    c. The melting point data will be appended to the CSD dataset, e.g., ./../Datasets/CSD_melting_point.csv
    usage:
    python Corporate_mp_extract.py -n hostname -d database_name -u user_name -p password -f ../Datasets/CSD_melting_point.csv

    """

    parser = argparse.ArgumentParser(description='Program for downloading Corporate Melting Point data:', epilog="")
    parser.add_argument("--hostname", "-n", help="hostname")
    parser.add_argument("--database_name", "-d", help="database name")
    parser.add_argument("--user", "-u", help="user name")
    parser.add_argument("--password", "-p", help="password")
    parser.add_argument("--file", "-f", help="melting point data file downloaded from the CSD")

    args = parser.parse_args()

    cols = ['CID', 'MP', 'smiles', 'unit']

    try:
        conn = psycopg2.connect(
        host=args.hostname,
        database=args.database_name,
        user=args.user,
        password=args.password
        )

        cursor = conn.cursor()

        """
        Complete the following dummy SQL query
        """
        query = "select CID, MP, smiles, unit from *;"
        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()
        df = pd.DataFrame(results, columns=cols)
        df.dropna(axis=0, inplace=True)
        df['MP'] = pd.to_numeric(df.MP)
        df_group = df.groupby('CID').agg({'MP': 'mean', 'smiles': 'max', 'unit': 'max'})
        if df_group.unit.iloc[0] == 'K':
            df['MP'] = df['MP'] - 273.15
        df_group = df_group[['MP', 'smiles']]
        df_group.to_csv(args.file, mode='a', index=True, header=False)
    except:
        pass
    
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import psycopg2
    main()
