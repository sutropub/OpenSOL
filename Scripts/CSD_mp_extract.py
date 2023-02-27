#
# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2021-09-09: created by the Cambridge Crystallographic Data Centre
#

'''
    a. Activate CCDC environment in Windows command prompt
            "C:\Program Files\CCDC\Python_API_2022\miniconda\Scripts\activate"
    b. Download melting point data
    c. Download crystal structure
    
    mp_extract.py    -   return csv file with data for entries with a melting point in the CSD.
'''

def main():

    parser = argparse.ArgumentParser(description='Program for downloading Melting Point data:', epilog="")
    parser.add_argument("--output", "-o", help="Directory of csv file for melting point data")

    args = parser.parse_args()

    if not args.output:
        csv_file = './../Datasets/CSD_melting_point.csv'
    else:
        csv_file = args.output

    # Reads the whole CSD
    csd_reader = EntryReader('CSD')
    

    # Alternatively run using a .gcd file of CSD refcodes
    # user_filepath = str(input("Enter filepath to CSD refcode list" '\n'))
    # csd_reader = EntryReader(user_filepath)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(['refcode', 'compound name', 'formula', 'Melting point', 'SMILES', 'bioactivity', 'solvent', 'molecular_weight', 'heavy_atom', 'deposition_date'])
        writer.writerow(['refcode', 'Melting point', 'SMILES'])
        for entry in csd_reader:
            mp = entry.melting_point
            if (mp is not None) and (not entry.is_organometallic) and (entry.molecule.smiles):  # if the entry has a melting point write selected data for entry into the csv
                compd = entry.chemical_name
                cname = compd.encode(encoding='utf-8')
                #writer.writerow((entry.identifier, cname, entry.formula, entry.melting_point, entry.molecule.smiles, entry.bioactivity, entry.solvent, entry.molecule.molecular_weight, len(entry.molecule.heavy_atoms), entry.deposition_date))
                writer.writerow((entry.identifier, entry.melting_point, entry.molecule.smiles))
            
            f.flush()

if __name__ == "__main__":
    import csv
    from ccdc.io import EntryReader
    import argparse
    main()