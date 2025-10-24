'''
Crossmatch BHB_Candidates to Big MAC, and the output of that to ETHER.
'''
import os
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

def bytes_to_string(DataFrame):
	'''
	Converts 'Bytes' objects inside Pandas DataFrames to proper strings.
	Use to avoid the inclution of b'' prexi in strings when converting from
	Astropy Table to DataFrame.

	Parameters
	----------
	DataFrame: Pandas DataFrame
		Source DataFrame.

	Returns
	-------
	df: Pandas DataFrame
		Corrected DataFrame.
	'''
	df = DataFrame.copy()
	str_df = df.select_dtypes([object])
	str_df = str_df.stack().str.decode('utf-8').unstack()
	for col in str_df:
		df[col] = str_df[col]
	return df

def csv_to_fits(csv_path):
    '''
    Converts a .csv file to a .fits file.

    Parameters
    ----------
    csv_path: str
        Path of the input .csv file.

    Returns
    -------
        .fits file of the input .csv.
    '''
    input_file = pd.read_csv(csv_path, float_precision='round_trip', dtype={'BNAME': str})
    
    # Convert string columns to ASCII, replacing non-ASCII characters
    for col in input_file.select_dtypes(include=['object']):
        input_file[col] = input_file[col].str.encode('ascii', errors='replace').str.decode('ascii')
    
    output_file = Table.from_pandas(input_file)
    output_file.write(csv_path[:-4]+'.fits', format='fits', overwrite=True)

nodat = 9876543

# Create directory to store output
if 'out' not in os.listdir('/home/ether/ether-bhb/'):
    os.mkdir('/home/ether/ether-bhb/out')
    os.mkdir('/home/ether/ether-bhb/out/crossmatch')
elif 'crossmatch' not in os.listdir('/home/ether/ether-bhb/out'):
    os.mkdir('/home/ether/ether-bhb/out/crossmatch')

# Read the data
ether_path = '/home/ether/ethersample/shared-files/ether-1p0.fits'
ether = bytes_to_string(Table.read(ether_path).to_pandas())
coord_mask = (ether['RA'] != nodat) & (ether['DEC'] != nodat) # Discard objects without coordinates
ether_coords = SkyCoord(ra = ether['RA'][coord_mask], dec = ether['DEC'][coord_mask], unit = 'deg')

bhbc_path = '/home/ether/ether-bhb/data/BHB_Candidates.fits'
bhbc = bytes_to_string(Table.read(bhbc_path).to_pandas())
bhbc_coords = SkyCoord(ra = bhbc['RA'], dec = bhbc['DEC'], unit = 'deg')

bigmac_path = '/home/ether/ether-bhb/data/BigMAC_maintable_DR0p9.csv'
bigmac = pd.read_csv(bigmac_path, na_values = [-99, '-99'])
bigmac['Sep(kpc)'] = bigmac['Sep(kpc)'].where(bigmac['Sep(kpc)'] > 0, nodat)
bigmac.fillna(nodat, inplace = True)
bigmac.drop_duplicates(inplace = True)
for i in bigmac.index:
    if bigmac.RA1[i][-1] == 's':
        bigmac.loc[i, 'RA1'] = bigmac.RA1[i][:-1]
bigmac_coords = SkyCoord(ra = bigmac['RA1'], dec = bigmac['Dec1'], unit = ('hourangle', 'deg'))

output = pd.DataFrame()

# Use different crossmatch radius for low and high redshift samples
for xmatch_radius in [3, 10]*u.arcsec:

    # Define redshift masks
    if xmatch_radius == 3*u.arcsec:
        bigmac_z_mask = (bigmac['z1'] > 0.05) | bigmac['z1'].isna()
        bhbc_z_mask = (bhbc['Z'] > 0.05) | bhbc['Z'].isna()
        ether_z_mask = (ether['Z'][coord_mask] > 0.05) | (ether['Z'][coord_mask] == nodat)
    else:
        bigmac_z_mask = (bigmac['z1'] <= 0.05) | bigmac['z1'].isna()
        bhbc_z_mask = (bhbc['Z'] <= 0.05) | bhbc['Z'].isna()
        ether_z_mask = (ether['Z'][coord_mask] <= 0.05) | (ether['Z'][coord_mask] == nodat)

    # Crossmatch BHBC to BigMAC
    idx, d2d, d3d = bhbc_coords[bhbc_z_mask].match_to_catalog_sky(bigmac_coords[bigmac_z_mask])
    xmatch = d2d < xmatch_radius

    bhbc_match = bhbc[bhbc_z_mask][xmatch]
    bhbc_nomatch = bhbc[bhbc_z_mask][~xmatch]

    bigmac_match = bigmac[bigmac_z_mask].iloc[idx[xmatch]].set_index(bhbc_match.index)
    bigmac_nomatch = bigmac[bigmac_z_mask].drop(bigmac[bigmac_z_mask].index[idx[xmatch]])
    
    # Concatenate both catalogs
    bhbc_bigmac = pd.concat([bhbc_match.rename(columns = lambda x: f'{x}_BHBC'), bigmac_match.rename(columns = lambda x: f'{x}_BM')], axis = 1)
    bhbc_bigmac = pd.concat([bhbc_bigmac, bhbc_nomatch.rename(columns = lambda x: f'{x}_BHBC')])
    bhbc_bigmac = pd.concat([bhbc_bigmac, bigmac_nomatch.rename(columns = lambda x: f'{x}_BM')]).reset_index(drop = True)

    # Create homogenized coordinates columns for crossmatch with ETHER
    mask = bhbc_bigmac.RA_BHBC.notna()
    bhbc_bigmac.loc[mask, ['RA_BHBC-BM', 'DEC_BHBC-BM']] = bhbc_bigmac.loc[mask, ['RA_BHBC', 'DEC_BHBC']].values

    mask = bhbc_bigmac['RA_BHBC-BM'].isna()
    to_deg = SkyCoord(ra = bhbc_bigmac.loc[mask, 'RA1_BM'].values, dec = bhbc_bigmac.loc[mask, 'Dec1_BM'].values, unit = ('hourangle', 'deg'))
    bhbc_bigmac.loc[mask, ['RA_BHBC-BM', 'DEC_BHBC-BM']] = np.column_stack((to_deg.ra.deg, to_deg.dec.deg))

    # Create homogenized redshift column
    mask = bhbc_bigmac['Z_BHBC'].notna()
    bhbc_bigmac.loc[mask, 'Z_BHBC-BM'] = bhbc_bigmac['Z_BHBC'][mask]

    mask = bhbc_bigmac['Z_BHBC-BM'].isna()
    bhbc_bigmac.loc[mask, 'Z_BHBC-BM'] = bhbc_bigmac['z1_BM'][mask]

    # Define redshift mask for concatenated catalogues
    if xmatch_radius == 3*u.arcsec:
        bhbc_bigmac_z_mask = (bhbc_bigmac['Z_BHBC-BM'] > 0.05) | bhbc_bigmac['Z_BHBC-BM'].isna()
    else:
        bhbc_bigmac_z_mask = (bhbc_bigmac['Z_BHBC-BM'] <= 0.05) | bhbc_bigmac['Z_BHBC-BM'].isna()

    # Crossmatch BHBC-BM to ETHER
    bhbc_bigmac_coords = SkyCoord(ra = bhbc_bigmac['RA_BHBC-BM'], dec = bhbc_bigmac['DEC_BHBC-BM'], unit = 'deg')
    idx, d2d, d3d = bhbc_bigmac_coords[bhbc_bigmac_z_mask].match_to_catalog_sky(ether_coords[ether_z_mask])
    xmatch = d2d < xmatch_radius

    bhbc_bigmac_match = bhbc_bigmac[bhbc_bigmac_z_mask][xmatch]
    bhbc_bigmac_nomatch = bhbc_bigmac[bhbc_bigmac_z_mask][~xmatch]

    ether_match = ether[coord_mask][ether_z_mask].iloc[idx[xmatch]].set_index(bhbc_bigmac_match.index)

    # Concatenate both catalogs
    ether_bhbc_bigmac = pd.concat([bhbc_bigmac_match, ether_match], axis = 1)
    ether_bhbc_bigmac = pd.concat([ether_bhbc_bigmac, bhbc_bigmac_nomatch])

    # Add result to output DataFrame
    output = pd.concat([output, ether_bhbc_bigmac])

# Save files
output.fillna(nodat, inplace = True)
output.drop_duplicates(inplace = True)
output.to_csv('/home/ether/ether-bhb/out/crossmatch/ether-bhbc-bigmac.csv', index = False)

csv_to_fits('/home/ether/ether-bhb/out/crossmatch/ether-bhbc-bigmac.csv')

bhbc_bigmac_nomatch.fillna(nodat, inplace = True)
bhbc_bigmac_nomatch.drop_duplicates(inplace = True)
bhbc_bigmac_nomatch.to_csv('/home/ether/ether-bhb/out/crossmatch/bhbc-bigmac_nomatch_ether.csv', index = False)