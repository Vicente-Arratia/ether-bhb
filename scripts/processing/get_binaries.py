'''
Homogenized sample of Dual and Binary AGN from crossmatch between ETHER, BHBC and Big MAC.
'''

import time
import numpy as np
import pandas as pd
from astropy.constants import c
from astropy.cosmology import Planck18 as cosmo

s = time.time()
# Functions
def get_dist(z):
    '''
    Calculate the angular diameter distance based on redshift using the Planck18 cosmology.

    Parameters:
    ----------
    z : float or array-like
        Redshift of the object.

    Returns:
    -------
    float or array-like
        Angular diameter distance in megaparsecs (Mpc).
    '''
    mask = (z*(c.value*1e-3)) > 720 # Only for objects with velocity > 720 km/s
    ld = cosmo.luminosity_distance(z.where(mask)).value # Luminosity distance in Mpc
    return ld/((1+z)**2) # Convert to angular diameter distance in Mpc

def get_period(sep, mbh, ratio, g = 4.5e-24):
    '''
    Calculate the orbital period of a binary system based on the separation, black hole mass of the primary component, and mass ratio, from Kepler's third law (i.e., "Wang equation").
    
    Parameters:
    ----------
    sep : float or array-like
        Separation in kiloparsecs (kpc).

    mbh : float or array-like
        Black hole mass of the primary component in log(solar masses).

    ratio : float or array-like
        Mass ratio of the binary system.

    g : float
        Gravitational constant in kpc^3 / (M_sun * yr^2).

    Returns:
    -------
    float or array-like
        Orbital period in years.
    '''
    return ((4*(np.pi**2)*(sep**3))/(g*(10**mbh)*(1+ratio)))**(1/2)

def get_sep(period, mbh, ratio, g = 4.5e-24):
    '''
    Calculate the separation between binary components based on the orbital period, black hole mass of the primary component, and mass ratio, from Kepler's third law (i.e., "Wang equation").
    
    Parameters:
    ----------
    period : float or array-like
        Orbital period in years.

    mbh : float or array-like
        Black hole mass of the primary component in log(solar masses).

    ratio : float or array-like
        Mass ratio of the binary system.

    g : float
        Gravitational constant in kpc^3 / (M_sun * yr^2).

    Returns:
    -------
    float or array-like
        Separation in kiloparsecs.
    '''
    return ((g*(period**2)*(10**mbh)*(1+ratio))/(4*(np.pi**2)))**(1/3)

def linear_to_angular(linear, z):
    """
    Linear2Angular separation based on the Ned Wright's Cosmology Calculator
    
    Parameters:
    ----------
    linear : float or array-like
        Linear separation between binary components in pc.
    
    z : float or array-like
        Redshift of the binary system.

    Returns:
    -------
    float or array-like
        Angular separation between the binary components in arcsec.
    """ 
    kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z).value
    pc_per_arcsec = kpc_per_arcmin * 1000 / 60
    angular_dist = linear / pc_per_arcsec
    return angular_dist

def angular_to_linear(angle, dist):
    '''
    Convert angular separation to linear separation.

    Parameters:
    ----------
    angle : float or array-like
        Angular separation in arcseconds.

    dist : float or array-like
        Angular diameter distance in megaparsecs (Mpc).

    Returns:
    -------
    float or array-like
        Linear separation in kiloparsecs (kpc).
    '''
    return ((angle*(dist*1e3))/206265) # Linear distance in kpc

# Constants
nodat = float(9876543)

# Load and preprocess data
df = pd.read_csv('/Users/vicentearratiacarrasco/Documents/work/g9/ether-bhb/out/crossmatch/ether-bhbc-bigmac.csv', na_values = [nodat, "b'9876543'", str(nodat), '9876543.0', '', ' ']) # File with both dual and binary AGN from Big MAC catalogue.
df = df[~df['TYPE_BHBC'].str.contains('Stats', na = False)] # Remove stats

# Create unified column of names
df['NAME_COMP'] = df['ONAME'].str.strip() # Add name from ONAME in ETHER
df['NAME_COMP'] = df['NAME_COMP'].fillna(df['Name1_BM'].str.strip()) # Add name from BM data
df['NAME_COMP'] = df['NAME_COMP'].fillna(df['NAME_BHBC'].str.strip()) # Add name from BHBC data

# Create unified column of coordinates
df['RA_COMP'] = df['RA'].fillna(df['RA_BHBC-BM']) # Add RA from BHBC data
df['DEC_COMP'] = df['DEC'].fillna(df['DEC_BHBC-BM']) # Add DEC from BHBC data

# Create unified column of redshift
df['Z_COMP'] = df['Z'].fillna(df['Z_BHBC']).fillna(df['z1_BM']) # Add redshift from ETHER > BHBC > BM data

# Calculate missing distances from redshift
df['DIST'] = df['DIST'].fillna(get_dist(df['Z_COMP']))

# Create unified column of total black hole mass
df['MBH_COMP'] = df['MBH'][df['MBHPUB'] != 50] # Add good quality MBH data from ETHER
df['MBH_COMP'] = df['MBH_COMP'].fillna(df['MBH'][(df['MBHPUB'] == 50) & (df['MBHLIM'] == 0)]) # Add good quality MBH data from ETHER
df['MBH_COMP'] = df['MBH_COMP'].fillna(df['MBHWISE_AGNCLEANED'][df['MBHPUB'] == 50]) # Add WISE cleaned MBH data from ETHER
df['MBH_COMP'] = df['MBH_COMP'].fillna(df['MBH_TOT_BHBC']) # Add total MBH from BHBC data

# Create unified column of literature angular/linear separation
df['SEP_LIT_ANG'] = df['Sep_BM']
df['SEP_LIT_LIN'] = df['Sep(kpc)_BM'].where(df['Sep(kpc)_BM'] > 0).fillna(angular_to_linear(df['SEP_LIT_ANG'], df['DIST'])).fillna(df['SEP_BHBC']*1e-3)

# Calculate missing ratios, primary black hole masses and periods
df['RATIO+FUDGE'] = 1/(df['RATIO_BHBC'].fillna(abs((10**df['MBH1_BHBC'])/((10**df['MBH_COMP']) - (10**df['MBH1_BHBC'])))).fillna(1)) 
df['MBH1+FUDGE'] = df['MBH1_BHBC'].fillna(np.log10((10 ** df['MBH_COMP'])/(1+df['RATIO+FUDGE'])))
df['PERIOD+FUDGE'] = df['PERIOD_BHBC'].fillna(get_period(df['SEP_LIT_LIN'].fillna(0.001), df['MBH1+FUDGE'], df['RATIO+FUDGE']))

# Calculate linear and angular separations
df['SEP_ETHER_LIN'] = get_sep(df['PERIOD+FUDGE'],
                              df['MBH1+FUDGE'],
                              df['RATIO+FUDGE']).fillna(0.001) # Calculate linear separation in kpc

df['SEP_ETHER_ANG'] = linear_to_angular(df['SEP_ETHER_LIN']*1e3,
                                        df['Z_COMP'].to_numpy()) # Calculate angular separation in arcsec

# Create unified columns of linear and angular separations
df['SEP_ALL_LIN'] = (df['SEP_LIT_LIN'].fillna(df['SEP_ETHER_LIN']))*1e3 # Stores separation in parsecs
df['SEP_ALL_ANG'] = (df['SEP_LIT_ANG'].fillna(df['SEP_ETHER_ANG']))*1e6 # Stores separation in microarcseconds

# Keep only binaries with redshift and period data
df = df[((df['SEP_ALL_LIN'] < 100) & (df['SEP_ALL_LIN'] > 0)) & df['Z'].notna() & df['PERIOD+FUDGE'].notna()]

# Final fixes
# Remove objects by name
df = df[df['NAME_COMP'] != 'NGC7674']

# Create new DataFrame with relevant columns only
new_df = pd.DataFrame(columns = ['NAME',
                                 'DIST', 'Z',
                                 'RA', 'DEC',
                                 'SEP_LIT_ANG', 'SEP_LIT_LIN',
                                 'SEP_ETHER_ANG', 'SEP_ETHER_LIN',
                                 'SEP_ALL_ANG', 'SEP_ALL_LIN',
                                 'MBH_TOT', 'MBH_1', 'MBH_RATIO',
                                 'FLUX230', 'FLUX230ERR', 'FLUX230LIM', 'RES230',
                                 'FLUXEHTVLBI', 'FREQEHTVLBI',
                                 'PERIOD'])

# Update columns
new_df['NAME'] = df['NAME_COMP']
new_df[['DIST', 'Z']] = df[['DIST', 'Z_COMP']]
new_df[['RA', 'DEC']] = df[['RA_COMP', 'DEC_COMP']]
new_df[['SEP_LIT_ANG', 'SEP_LIT_LIN']] = df[['SEP_LIT_ANG', 'SEP_LIT_LIN']]
new_df[['SEP_ETHER_ANG', 'SEP_ETHER_LIN']] = df[['SEP_ETHER_ANG', 'SEP_ETHER_LIN']]
new_df[['SEP_ALL_ANG', 'SEP_ALL_LIN']] = df[['SEP_ALL_ANG', 'SEP_ALL_LIN']]
new_df['MBH_TOT'] = df['MBH_COMP']
new_df['MBH_1'] = df['MBH1_BHBC']
new_df['MBH_RATIO'] = df['RATIO_BHBC']
new_df['FLUX230'] = df['FLUX230']
new_df['FLUX230ERR'] = df['FLUX230ERR']
new_df['FLUX230LIM'] = df['FLUX230LIM']
new_df['RES230'] = df['RES230']
new_df['FLUXEHTVLBI'] = df['FLUXEHTVLBI']
new_df['FREQEHTVLBI'] = df['FREQEHTVLBI']
new_df['PERIOD'] = df['PERIOD_BHBC']
new_df['PERIOD+FUDGE'] = df['PERIOD+FUDGE']

# Sort by distance in ascending order
new_df = new_df.sort_values(by = 'DIST', ascending = True)
new_df.drop_duplicates(inplace = True)

# Save CSV
new_df.to_csv('/Users/vicentearratiacarrasco/Documents/work/g9/ether-bhb/out/processing/binaries.csv', index = False)

f = time.time()
print(f'{round(f-s, 2)} s')
'''
Notes:
------

    Print columns names:
        df.keys().to_list()

    Separations from BHBC:
        df[df['SEP_BHBC'].notna()][['NAME_BHBC', 'SEP_BHBC', 'BIB/DOI_BHBC']]

    Non-periodic binaries from BM:
        
        df = pd.read_csv('/Users/vicentearratiacarrasco/Documents/work/g9/ether-bhb/data/BigMAC_maintable_DR0p9.csv')

        With known or unknown separation:
            df[(df['Sep(kpc)']*1e3 < 100) & ~df['Parsed Selection Method'].str.contains('Periodicity', na = False) & df['z1'].notna()]
        
        With known separation only:
            df[(df['Sep(kpc)'] > 0) & (df['Sep(kpc)']*1e3 < 100) & ~df['Parsed Selection Method'].str.contains('Periodicity', na = False) & df['z1'].notna()]

    BM separation == 1 pc (likely assumptions):
        df = pd.read_csv('/Users/vicentearratiacarrasco/Documents/work/g9/ether-bhb/data/BigMAC_maintable_DR0p9.csv')
        df[df['Sep(kpc)'] == 0.001]

    Number of objects with no period and no separation:
        mask = new_df['PERIOD'].isna() & new_df['SEP_LIT_LIN'].isna()
        len(new_df[mask])
'''