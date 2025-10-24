import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Complete dual/binary sample
df = pd.read_csv('/Users/vicentearratiacarrasco/Documents/work/g9/ether-bhb/out/processing/binaries.csv')

# Global variables
min_flux = 10

# Define masks
gold = df['FLUXEHTVLBI'].notna() & (df['FLUX230'] > min_flux)
silver = df['FLUXEHTVLBI'].notna() | (df['FLUX230'] > min_flux)
other = ~gold & ~silver
no_sep_no_period = (np.round(df['SEP_ALL_LIN'], 10) == 1) & df['PERIOD'].isna()

for unit in ['ANG', 'LIN']:

    # Make figure
    fig, ax = plt.subplots(figsize = (12, 8))

    # Plot data
    for mask, color, label, zorder, alpha, marker in zip([gold, silver], ['gold', 'silver'], ['Gold', 'Silver'], [3, 2], [1, 0.7], ['^', 's']):
        sc = ax.scatter(df[mask]['Z'], df[mask][f'SEP_ALL_{unit}'], label = label, c = df[mask]['PERIOD+FUDGE'], norm = LogNorm(vmax = 20), edgecolor = 'k', linewidth = 0.5, zorder = zorder, s = 300, alpha = alpha, marker = marker)
        ax.scatter(df[mask & no_sep_no_period]['Z'], df[mask & no_sep_no_period][f'SEP_ALL_{unit}'], facecolor = 'None', edgecolor = 'r', linewidth = 1, zorder = zorder, s = 300, marker = marker)

    # Other objects
    ax.scatter(df[other]['Z'], df[other][f'SEP_ALL_{unit}'], label = 'Other', color = 'k', edgecolor = 'k', linewidth = 1, zorder = 0, s = 200, alpha = 0.5, marker = '.')
    ax.scatter(df[other & no_sep_no_period]['Z'], df[other & no_sep_no_period][f'SEP_ALL_{unit}'], facecolor = 'None', edgecolor = 'r', linewidth = 1, zorder = 0, s = 200, marker = '.')

    # Add orbital period colorbar
    cax = ax.inset_axes([0, 1.1, 1, 0.05])
    cbar = plt.colorbar(sc, cax = cax, orientation = 'horizontal')
    cbar.ax.set_title('Orbital period (Years)', fontsize = 20)
    cbar.ax.tick_params(labelsize = 20)

    # Resolution threshold
    if unit == 'ANG':
        plt.axhline(y = 10, color = 'grey', linestyle = '--', linewidth = 1, zorder = -4)
        ax.set_ylabel(r'Expected separation [$\mu$as]', fontsize = 20)
    else:
        ax.set_ylabel(r'Expected separation [pc]', fontsize = 20)
    
    # Other plot parameters
    ax.set_xlabel('Redshift (z)', fontsize = 20)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    ax.legend(loc = 'upper right', prop = {'size': 20}, ncol = 1)
    plt.tight_layout()

    plt.savefig(f'/Users/vicentearratiacarrasco/Documents/work/g9/ether-bhb/out/plots/binaries_sep_{unit.lower()}_vs_redshift_vs_period.png', dpi = 300)