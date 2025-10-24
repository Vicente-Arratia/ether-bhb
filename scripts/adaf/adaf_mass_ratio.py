'''
Calculates ADAF SED model assuming different mass and X-ray flux ratios
'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import astropy.units as u
import ETHER_Functions as ether
from astropy.coordinates import SkyCoord
from concurrent.futures import ProcessPoolExecutor

# Function to process a single source
def process_source(i):
    # Define amount of values in the Gaussian distributions:
    gauss_length = 1000

    # Define source main parameters:
    lumx = master['LUMX'][i]*ratio  # X-ray luminosity
    lumxlo = master['LUMXLO'][i]*ratio  # X-ray luminosity lower limit
    lumxhi = master['LUMXHI'][i]*ratio  # X-ray luminosity upper limit
    
    mbh = 10**master['MBH'][i]*ratio  # Measured black hole mass in solar masses
    mbhlo = 10**master['MBHLO'][i]*ratio  # Measured black hole mass lower limit
    mbhhi = 10**master['MBHHI'][i]*ratio  # Measured black hole mass upper limit

    # Define gaussians for black hole mass:
    mbhlo_gauss = np.random.normal(mbh, abs(mbh - mbhlo), gauss_length)  # Lower half of final mbh gaussian distribution
    mbhhi_gauss = np.random.normal(mbh, abs(mbh - mbhhi), gauss_length)  # Upper half of final mbh gaussian distribution
    mbh_gauss = np.array([val for pair in zip(mbhlo_gauss[:int(gauss_length/2)], mbhhi_gauss[int(gauss_length/2):]) for val in pair])  # Interleaves the two random gaussians
    mbh_gauss[mbh_gauss < 0] = np.nan
    
    # Define gaussians for X-ray luminosity:
    lumxlo_gauss = np.random.normal(lumx, abs(lumx - lumxlo), gauss_length)  # Lower half of final X-ray Luminosity gaussian distribution
    lumxhi_gauss = np.random.normal(lumx, abs(lumx - lumxhi), gauss_length)  # Upper half of final X-ray Luminosity gaussian distribution
    lumx_gauss = np.array([val for pair in zip(lumxlo_gauss[:int(gauss_length/2)], lumxhi_gauss[int(gauss_length/2):]) for val in pair])  # Interleaves the two random gaussians
    lumx_gauss[lumx_gauss < 0] = np.nan

    # Define DataFrame with paired black hole mass and x-ray luminosity values, together with structure to save MC results:
    gauss_pairs = pd.DataFrame({'MBH': np.sort(mbh_gauss), 'LUMX': lumx_gauss})
    gauss_pairs = gauss_pairs[gauss_pairs['MBH'].notna() & gauss_pairs['LUMX'].notna()]
    mc_results = pd.DataFrame()

    # Observed-frame to rest-frame frequency conversions:
    rf_freq = {}
    for keyfreq in ether.freqvec + ['XRAY']:
        if keyfreq == 'XRAY':
            rf_freq[f'{keyfreq}'] = master['FREQXRAY'][i] * (1 + master['Z'][i])  # K-Correction
            rf_freq[f'{keyfreq}_mt_idx'] = ether.closest_idx(nu_vec, rf_freq[f'{keyfreq}'])  # Finds the index value of the closest frequency to the rest-frame frequency in the main nu vector
        else:
            rf_freq[f'{keyfreq}'] = keyfreq * 1e9 * (1 + master['Z'][i])
            rf_freq[f'{keyfreq}_mt_idx'] = ether.closest_idx(nu_vec, rf_freq[f'{keyfreq}'])

    # Iteration over pair of black hole mass and x-ray luminosity values:
    for j in gauss_pairs.index:
        # Save time by skipping already read file:
        template_mbh = ether.closest_idx(mbh_vec, gauss_pairs['MBH'][j])
        if j == gauss_pairs.index[0]:  # For the first pair of values
            mbh_template_label = mbh_vec[template_mbh]  # Find the MBH value of the closest template
            lup_table = pd.read_csv(f'/home/ether/etheranalysis/sed_modelling/adaf/templates/mbh_{mbh_template_label}.csv')  # Read the closest look up table
        else:  # For the remaining pairs
            if mbh_template_label == mbh_vec[template_mbh]:  # If the current template MBH value is the same as before, then use the previous template
                pass
            else:  # If not then read the new look up table
                mbh_template_label = mbh_vec[template_mbh]  # Replace the look up table MBH label with the new one
                lup_table = pd.read_csv(f'/home/ether/etheranalysis/sed_modelling/adaf/templates/mbh_{mbh_template_label}.csv')  # Read the closest look up table

        lup_table_xcol = lup_table[str(rf_freq['XRAY_mt_idx'])]  # Find the column corresponding to the rest-frame X-ray frequency
        lup_table_sline_idx = ether.closest_idx(lup_table_xcol, gauss_pairs['LUMX'][j])  # Find the index of the line closest to the measured X-ray Luminosity
        lup_table_sed = lup_table.iloc[[lup_table_sline_idx]]  # Finds the best-fit SED row for the source

        # Add resulting values to mc_results DataFrame:
        mc_results.loc[len(mc_results), 'LBOL'] = ether.get_lum_bol(nu_vec, lup_table_sed.T[lup_table_sline_idx].to_numpy() / nu_vec)  # Bolometric luminosity of best-fit SED
        mc_results.loc[len(mc_results), 'MDOT'] = mdot_vec.iloc[lup_table_sline_idx]  # Best-fit Eddington ratio value
        for keyfreq in ether.freqvec:
            mc_results.loc[len(mc_results) - 1, f'FLUX{keyfreq}'] = ether.nulnu_to_snu(lup_table_sed[str(rf_freq[f"{keyfreq}_mt_idx"])].values[0], master['LUMDIST'][i], keyfreq * 1e9)

    # Create a result dictionary to store values
    result = {
        'ONAME': master['ONAME'][i],
        'RA': master['RA'][i],
        'DEC': master['DEC'][i],
        'MBH': np.log10(mbh),
        'LUMX': lumx
    }
    
    # Fill result dictionary with final values:
    for k in mc_results.keys():
        result[k] = mc_results[k].median()  # Actual value
        result[f'{k}LO'] = np.sqrt(np.mean((mc_results[k][mc_results[k] < mc_results[k].median()]) ** 2))  # Lower limit. RMS of the first half of the sorted column
        result[f'{k}HI'] = np.sqrt(np.mean((mc_results[k][mc_results[k] > mc_results[k].median()]) ** 2))  # Upper limit. RMS of the upper half of the sorted column

    return result

# Create directory to store output files
if 'out' not in os.listdir('/home/ether/ether-bhb/'):
    os.mkdir('/home/ether/ether-bhb/out')
    os.mkdir('/home/ether/ether-bhb/out/adaf_mass_ratio')
elif 'adaf_mass_ratio' not in os.listdir('/home/ether/ether-bhb/out/'):
    os.mkdir('/home/ether/ether-bhb/out/adaf_mass_ratio')

# Read master:
master_original = pd.read_csv('/home/ether/ether-bhb/out/ether-fp-bigmac.csv', float_precision = 'round_trip')
master_original = master_original[:]

# Read in black hole mass, Eddington ratio and SED frequency range vectors:
mbh_vec = pd.read_csv('/home/ether/etheranalysis/sed_modelling/adaf/templates/mbh_vec.csv', float_precision = 'round_trip')['MBH']
mdot_vec = pd.read_csv('/home/ether/etheranalysis/sed_modelling/adaf/templates/mdot_vec.csv', float_precision = 'round_trip')['MDOT']
nu_vec = pd.read_csv('/home/ether/etheranalysis/sed_modelling/adaf/templates/nu_vec.csv', float_precision = 'round_trip')['FREQ']

ratios = np.append(np.unique(np.array([[1/x, (x-1)/x] for x in range(2, 12, 2)]).flatten()), 1)
for ratio in ratios:
    master = master_original.copy()
    # Crossmatch master to list of sources that already went through the simulation:
    if os.path.exists(f'/home/ether/ether-bhb/out/adaf_mass_ratio/ratio_{ratio:.3f}.csv'):
        output = pd.read_csv(f'/home/ether/ether-bhb/out/adaf_mass_ratio/ratio_{ratio:.3f}.csv', float_precision = 'round_trip')
        master_coords = SkyCoord(ra = master['RA'], dec = master['DEC'], unit = 'deg')
        adaf_coords = SkyCoord(ra = output['RA'], dec = output['DEC'], unit = 'deg')

        idx, d2d, d3d = master_coords.match_to_catalog_sky(adaf_coords)
        xmatch_mask = d2d < 1 * u.arcsec
        master = master[~xmatch_mask]  # Keep only sources that haven't been through the simulations
    else:
        output = pd.DataFrame()

    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Use ProcessPoolExecutor to process sources in parallel with all available CPU cores
    with ProcessPoolExecutor(max_workers = num_cores) as executor:
        results = list(tqdm(executor.map(process_source, master.index), total = len(master.index)))

    # Convert results back to DataFrame and save
    output = pd.concat([output, pd.DataFrame(results)])
    output.to_csv(f'/home/ether/ether-bhb/out/adaf_mass_ratio/ratio_{ratio:.3f}.csv', index=False)