import os
import time
import numpy as np
import pandas as pd
import astropy.units as u
from astroquery.alma import Alma
from astropy.coordinates import SkyCoord
from astroquery.exceptions import RemoteServiceError

alma = Alma()
alma.archive_url = 'https://almascience.eso.org'

if 'alma' not in os.listdir('/home/ether/ether-bhb/out/queries/eso'):
    os.mkdir('/home/ether/ether-bhb/out/queries/eso/alma')

# Default query mode:
query_mode = 'new'

# Import coordinates data file:
table1 = pd.read_csv('/home/ether/ether-bhb/out/crossmatch/dual_binary.csv', float_precision = 'round_trip', usecols = ['RA', 'DEC', 'Z'])
table1 = table1[table1['RA'].notna() & table1['DEC'].notna()]

if 'archive_qsl.csv' not in os.listdir('/home/ether/ether-bhb/out/queries/eso/alma'):
    print('\nNo queried sources list found.\n')
    # Create file that keeps track of already queried sources:
    table2 = pd.DataFrame()
else:
    # Ask to query empty sources only:
    query_mode = str(input("\nRedo empty queries? (y): "))
    
    # Read list of previously queried sources:
    table2 = pd.read_csv('/home/ether/ether-bhb/out/queries/eso/alma/archive/qsl.csv', float_precision = 'round_trip')
    
    print('\nCrossmatching...\n')
    if query_mode == 'y':
        try:
            empty_table_mask = table2['FLAG'] == 0 # Mask to keep only sources that returned empty tables
            # Crossmatch:
            coords1 = SkyCoord(ra = table1['RA'], dec = table1['DEC'], unit = 'deg')
            coords2 = SkyCoord(ra = table2['RA'][empty_table_mask], dec = table2['DEC'][empty_table_mask], unit = 'deg')
            idx, d2d, d3d = coords1.match_to_catalog_sky(coords2)
            xmatch_mask = d2d < 1*u.arcsec
            table1 = table1[xmatch_mask] # Sample of sources to query. Sources that returned empty tables only.
        except u.core.UnitTypeError:
            print('No sources to query.\n')
            raise SystemExit
    else: 
        query_mode = 'new'
        # Crossmatch:
        coords2 = SkyCoord(ra = table2['RA'], dec = table2['DEC'], unit = 'deg')
        coords1 = SkyCoord(ra = table1['RA'], dec = table1['DEC'], unit = 'deg')
        idx, d2d, d3d = coords1.match_to_catalog_sky(coords2)
        xmatch_mask = d2d < 1*u.arcsec
        table1 = table1[~xmatch_mask] # Sample of sources to query. New sources only.

if 'archive_query.csv' not in os.listdir('/home/ether/ether-bhb/out/queries/eso/alma'):
    combined = pd.DataFrame()
else:
    combined = pd.read_csv('/home/ether/ether-bhb/out/queries/eso/alma/archive/query.csv', float_precision = 'round_trip')

# Query:
print(f'{len(table1)} sources will be queried.\n')
for i, j in zip(table1.index[:], np.arange(len(table1))):
    # time.sleep(1)
    pos = SkyCoord(ra = table1['RA'][i], dec = table1['DEC'][i], unit = 'deg')
    try:
        if table1['Z'][i] < 0.05:
            query = alma.query_region(pos, 10*u.arcsec)
        else:
            query = alma.query_region(pos, 3*u.arcsec)
        if query is None:
            if query_mode == 'new':
                table2.loc[len(table2), ['RA', 'DEC', 'FLAG']] = [table1['RA'][i], table1['DEC'][i], 0]
        else:
            query = query.to_pandas()
            if query.empty:
                if query_mode == 'new':
                    table2.loc[len(table2), ['RA', 'DEC', 'FLAG']] = [table1['RA'][i], table1['DEC'][i], 0]
            else:
                if query_mode == 'new':
                    table2.loc[len(table2), ['RA', 'DEC', 'FLAG']] = [table1['RA'][i], table1['DEC'][i], 1]
                else:
                    table2.loc[table2[empty_table_mask].index[j], ['RA', 'DEC', 'FLAG']] = [table1['RA'][i], table1['DEC'][i], 1]

                # Process query:
                if query[pd.isna(query['pub_title']) == False].empty:
                    pass
                else:
                    # Remove unwanted authors:
                    for i_user_ids in ['neilnagar', 'joacoh99']:
                        query = query[query['pi_userid'] != i_user_ids]
                    if query.empty:
                        pass
                    else:
                        # Count the occurrences of publication titles:
                        ref_counts = query['pub_title'].value_counts()
                        if ref_counts.empty:
                            # Nothing to process, skip this iteration
                            pass
                        elif ref_counts.idxmax() == '':
                            # If the most common reference is empty, try to get the second most common reference
                            if len(ref_counts) > 1:
                                second_most_common = ref_counts.index[1]
                                p_query = query[query['pub_title'] == second_most_common].copy()
                            else:
                                # No second reference exists, so fall back to using the only available entry
                                p_query = query.copy()
                        else:
                            # Use the most common reference
                            p_query = query[query['pub_title'] == ref_counts.idxmax()].copy()

                        p_query['RA_MASTER'], p_query['DEC_MASTER'] = table1['RA'][i], table1['DEC'][i]

                        # Concatenate to full file:
                        combined.loc[len(combined), p_query.keys()] = p_query.iloc[0]
                        combined.to_csv('/home/ether/ether-bhb/out/queries/eso/alma/archive/query.csv', index = False)
    
    except RemoteServiceError: # If query fails then just flag it as if it returned an empty table
        if query_mode == 'new':
            table2.loc[len(table2), ['RA', 'DEC', 'FLAG']] = [table1['RA'][i], table1['DEC'][i], 0]
    
    table2.to_csv('/home/ether/ether-bhb/out/queries/eso/alma/archive/qsl.csv', index = False)

# Create frequency sorted list of references:
combined = combined.drop_duplicates()
fsl = pd.DataFrame({'REF': combined['pub_title'].str.slice(stop = 50).value_counts().index.to_list()
                    , 'FREQ': combined['pub_title'].str.slice(stop = 50).value_counts().values}) # fsl: Frequency sorted list
fsl.to_csv('/home/ether/ether-bhb/out/queries/eso/alma/archive/references.csv', index = False)