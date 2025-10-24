import os
import pandas as pd
import astropy.units as u
from astroquery.eso import Eso

if 'out' not in os.listdir('/home/ether/ether-bhb'):
    os.mkdir('/home/ether/ether-bhb/out')
if 'queries' not in os.listdir('/home/ether/ether-bhb/out'):
    os.mkdir('/home/ether/ether-bhb/out/queries')
if 'eso' not in os.listdir('/home/ether/ether-bhb/out/queries'):
    os.mkdir('/home/ether/ether-bhb/out/queries/eso')
if 'muse' not in os.listdir('/home/ether/ether-bhb/out/queries/eso'):
    os.mkdir(f'/home/ether/ether-bhb/out/queries/eso/muse')

eso = Eso()

data = pd.read_csv('/home/ether/ether-bhb/out/crossmatch/muse_dual_binary.csv')
data = data[data['DIST'] < 200]

for i in data.index[:1]:
    obj_id = data['NAME'][i].replace(' ', '')
    obj_ra = data['RA'][i]
    obj_dec = data['DEC'][i]

    search_radius = 10
    table = eso.query_instrument('muse', column_filters = {'coord1':  obj_ra, 'coord2': obj_dec, 'box': search_radius*u.arcsec.to(u.deg), 'format': 'decimal'})
    # table = eso.query_surveys(surveys = 'MUSE-DEEP', column_filters = {'target': obj_id, 'coord1':  obj_ra, 'coord2': obj_dec, 'box': search_radius*u.arcsec.to(u.deg), 'format': 'decimal'})
    
    # Gradually increase size of search box up to a specified maximum amount of arcseconds
    # while (table is None) & (search_radius < 30):
    #     search_radius += 5 # Step size
    #     table = eso.query_instrument('muse', column_filters = {'coord1':  obj_ra, 'coord2': obj_dec, 'box': search_radius*u.arcsec.to(u.deg), 'format': 'decimal'})

    if table is None:
        print(f'\nNo MUSE data products found for object "{obj_id}" within {search_radius} arcseconds from input coordinates.\n')
    else:
        if obj_id not in os.listdir('/home/ether/ether-bhb/out/queries/eso/muse'):
            os.mkdir(f'/home/ether/ether-bhb/out/queries/eso/muse/{obj_id}')

        print(f'\nSource: {obj_id}')
        print(f'Search radius: {search_radius} arcseconds')
        print(f'Number of MUSE data products found: {len(table)}\n')
        print(f'{table}\n')

        eso.retrieve_data(list(table['DP.ID']), destination = f'/home/ether/ether-bhb/out/queries/eso/muse/{obj_id}')