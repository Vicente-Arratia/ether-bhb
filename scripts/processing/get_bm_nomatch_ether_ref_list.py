'''
Get list of references for objects in Big Mac that do not have a match in the ETHER database.
'''
import pandas as pd

def process_references(data_col):
    # Flatten list of semicolon-separated references and strip whitespace
    return [r.strip() for refs in data[data_col] for r in refs.split(';')]

# Read input data
data = pd.read_csv('/home/ether/ether-bhb/out/crossmatch/bhbc-bigmac_nomatch_ether.csv')

# Create DataFrame with processed references
refs_df = pd.DataFrame({
    col: process_references(data_col) 
    for data_col, col in zip(['Paper(s)_BM', 'BibCode(s)_BM', 'DOI(s)_BM'], 
                            ['REF', 'BIBCODE', 'DOI'])
})

# Add column of frequency of unique identifiers
refs_df['FREQ'] = refs_df['REF'].value_counts()[refs_df['REF']].values

# Remove duplicates and sort by frequency
refs_df = (refs_df.drop_duplicates()
                  .sort_values(by='FREQ', ascending=False))

# Save results
refs_df.to_csv('/home/ether/ether-bhb/out/processing/bm_nomatch_ether_ref_list.csv', index=False)