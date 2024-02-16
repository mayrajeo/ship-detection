from fastcore.script import *
import requests
import json

@call_parse
def main(tile_id:str, # Tile id to query, for example 34VEM
         start_date:str, # Start date for query, format yyyy-mm-dd
         end_date:str, # End date for query, format yyyy-mm-dd
         cloud_cover:int=20, # Maximum cloud cover percentage, default 20
         outpath:str='.' # Where to save the resulting product id list, default '.'
         ):
    
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=(startswith(Name,'S2') "
    instrument = "and (Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'instrumentShortName' and att/OData.CSC.StringAttribute/Value eq 'MSI') "
    product = "and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI1C') "
    tile = f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'tileId' and att/OData.CSC.StringAttribute/Value eq '{tile_id}'))) "
    cloud = f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover}.00) "
    start = f"and ContentDate/Start ge {start_date}T00:00:00.000Z "
    end = f"and ContentDate/Start lt {end_date}T23:59:59.000Z"
    fin = "&$orderby=ContentDate/Start desc&$expand=Attributes&$count=True&$top=1000&$skip=0"
    resp = requests.get(base_url + instrument + product + tile + cloud + start + end + fin)
    resp_data = json.loads(resp.text)

    products = sorted([r['Name'] for r in resp_data['value']])
    uniq_timesteps = sorted({'_'.join(p.split('_')[:3]) for p in products})
    final_products = []
    for t in uniq_timesteps:
        matches = [p for p in products if t in p]
        if len(matches) == 1: final_products.extend(matches)
        else:
            if len([m for m in matches if 'N0500' in m]) > 0:
                # If there are products with the latest processing baseline use only them
                final_products.extend([m for m in matches if 'N0500' in m])
            else: # for some reason there are duplicates, add all of them
                final_products.extend(matches)

    with open(f'{outpath}/{tile_id}_tileids.txt', 'w') as f:
        for p in final_products: 
            print(p, file=f)
