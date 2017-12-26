from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib
import scriptine
import os
import math, re, MySQLdb, types, urllib2, datetime, random, time, unicodedata, optparse
from datetime import timedelta
import pandas as pd
begmonth=datetime.datetime.now()
begmonth=begmonth.replace(day=1)
date=begmonth.strftime("%Y-%m-%d %H:%M:%S")
print(date)

# In[2]:

debug=False
delay=120

try:
    # For Python 3.0 and later
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.parse import urlencode
except ImportError:
    # Fall back to Python 2's urllib2 and urllib
    from urllib2 import HTTPError
    from urllib import quote
    from urllib import urlencode


# In[3]:


CLIENT_ID = 'mrsvBGDT1QDCWnCyW2E6rg'
CLIENT_SECRET = "ZNhzA0Lj1t52FuLLphqASaBz3V08dFkfWqeiDXCkozLg6e2nNFAKJ66hxEaXNxIi"


# In[4]:


# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.
TOKEN_PATH = '/oauth2/token'
GRANT_TYPE = 'client_credentials'


# In[5]:


# Defaults for our simple example.
DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = ' Boston, MA'
SEARCH_LIMIT = 20


# In[6]:


def obtain_bearer_token(host, path):
    """Given a bearer token, send a GET request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        str: OAuth bearer token, obtained using client_id and client_secret.
    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    assert CLIENT_ID, "Please supply your client_id."
    assert CLIENT_SECRET, "Please supply your client_secret."
    data = urlencode({
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': GRANT_TYPE,
    })
    headers = {
        'content-type': 'application/x-www-form-urlencoded',
    }
    response = requests.request('POST', url, data=data, headers=headers)
    bearer_token = response.json()['access_token']
    return bearer_token


# In[7]:


def request(host, path, bearer_token, url_params=None):
    """Given a bearer token, send a GET request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        bearer_token (str): OAuth bearer token, obtained using client_id and client_secret.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % bearer_token,
    }

    if debug:
      print(u'Querying {0} ...'.format(url))

    try:
      response = requests.request('GET', url, headers=headers, params=url_params)
    except:
      time.sleep(delay)
      print ('time.sleep(delay) A')
            
    return response.json()


# In[8]:


def search(bearer_token, term, location):
    """Query the Search API by a search term and location.
    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.
    Returns:
        dict: The JSON response from the request.
    """

    url_params = {
        'term': term.replace(" ", '+'),
        'location': location.replace(" ", '+'),
        'limit': SEARCH_LIMIT
    }
    return request(API_HOST, SEARCH_PATH, bearer_token, url_params=url_params)


# In[9]:


def get_business(bearer_token, business_id):
    """Query the Business API by a business ID.
    Args:
        business_id (str): The ID of the business to query.
    Returns:
        dict: The JSON response from the request.
    """
    business_path = BUSINESS_PATH + business_id

    return request(API_HOST, business_path, bearer_token)


# In[10]:


#f = open("yelp_test.json", "w")


# In[11]:


def query_api(term, location):
    """Queries the API by the input values from the user.
    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
    """
    bearer_token = obtain_bearer_token(API_HOST, TOKEN_PATH)

    response = search(bearer_token, term, location)

    businesses = response.get('businesses')

    if not businesses:
        print(u'No businesses for {0} in {1} found.'.format(term, location))
        return

    business_id = businesses[0]['id']

    if debug:
	print(u"{0} businesses found, querying business info "         'for the top result "{1}" ...'.format(
           len(businesses), business_id))
    
    try:
	response = get_business(bearer_token, business_id)
    except:
	time.sleep(delay)
	print ('time.sleep(delay) A')
   
    #f.write(str(response))
  
    if debug:
	print(u'Result for business "{0}" found:'.format(business_id))
	
    try:
	pprint.pprint(response, indent=2)
    except:
	time.sleep(delay)
	print ('time.sleep(delay) A')


# In[12]:


z= ["02108", "02109", "02110", "02111", "02113", "02114", "02115", "02116", "02118", "02119", "02120", "02121", "02122", "02124", "02125", "02126", "02127", "02128", "02129", "02130", "02131", "02132", "02134", "02135", "02136", "02151", "02152", "02163", "02199", "02203", "02210", "02215", "02467", "02138", "02139", "02140", "02141", "02142"]


# In[13]:


def main():
   for x in z :   
        try:
           result = query_api("food", x)
        except HTTPError as error:
            sys.exit(
            'Encountered HTTP error {0} on {1}:\n {2}\nAbort program.'.format(
                error.code,
                error.url,
                error.read(),
        )
      ) 

if __name__ == '__main__':
 main()
# scriptine.run()


# In[14]:


# f.close()


# Storing yelp data in sqllite database

# In[15]:


import sqlite3
yelp_db_nw = 'yelp_db_nw.sqlite' 
conn = sqlite3.connect(yelp_db_nw) 
c = conn.cursor()


# In[16]:


for row in conn.execute("pragma table_info('yelp_reviews')").fetchall():
    print (row)


# In[17]:


import pandas as pd
data_nwf = pd.read_csv("yelp_data.csv", encoding ='latin-1')
data_nwf.head()


# In[18]:

'''
data_nwf.to_sql('yelp_reviews',           # Name of the table.
             con=conn,                    # The handle to the file that is set up.
             if_exists='replace',         # Overwrite, append, or fail.
             index=False)                 # Add index as column.
'''


# In[19]:


# for row in conn.execute("pragma table_info('yelp_reviews')").fetchall():
#    print (row)


# Now use the latitude and longitude obtained from yelp api to query uber api
# Also, the minimum distance defined for a profitable ride is 1 mile. This is a straight line distance calculated through geopy api

# In[20]:


# pip install uber-rides
from uber_rides.session import Session
from uber_rides.client import UberRidesClient
from datetime import datetime

session = Session(server_token='Uvu3eEPnLtPKCbTU7KrCko5jo1ua4CVgYAqd0JfO')
client = UberRidesClient(session)


# In[21]:


def find_first():
    df = pd.read_sql('SELECT coordinates__latitude, coordinates__longitude FROM yelp_reviews LIMIT 1', con=conn)
    df.head()
    return df


# In[22]:


def find_second():
    df_2 = pd.read_sql('SELECT yelp_id, coordinates__latitude, coordinates__longitude FROM yelp_reviews where yelp_id IN (SELECT yelp_id FROM yelp_reviews ORDER BY RANDOM() LIMIT 1)', con = conn)
    df_2.head()
    return df_2

#east zone time

import datetime

class EST5EDT(datetime.tzinfo):

    def utcoffset(self, dt):
        return datetime.timedelta(hours=-5) + self.dst(dt)

    def dst(self, dt):
        d = datetime.datetime(dt.year, 3, 8)        #2nd Sunday in March
        self.dston = d + datetime.timedelta(days=6-d.weekday())
        d = datetime.datetime(dt.year, 11, 1)       #1st Sunday in Nov
        self.dstoff = d + datetime.timedelta(days=6-d.weekday())
        if self.dston <= dt.replace(tzinfo=None) < self.dstoff:
            return datetime.timedelta(hours=1)
        else:
            return datetime.timedelta(0)

    def tzname(self, dt):
        return 'EST5EDT'


# In[23]:


def actual_second():
    df_1 = find_first()
    df_2 = find_second()
    from geopy.distance import vincenty
    start_loc = (df_1['coordinates__latitude'][0], df_1['coordinates__longitude'][0])
    print(start_loc)
    end_loc = (df_2['coordinates__latitude'][0], df_2['coordinates__longitude'][0])
    print(end_loc)
    
    distance = vincenty(start_loc, end_loc).miles
    if(distance > 1):
        print(distance)
        response = client.get_price_estimates(
        start_latitude= df_1['coordinates__latitude'][0],
        start_longitude= df_1['coordinates__longitude'][0],
        end_latitude=  df_2['coordinates__latitude'][0],
        end_longitude= df_2['coordinates__longitude'][0],
        seat_count=2
        )
#       time = datetime.now()
        print(time)
        prices = response.json.get("prices")
        print(type("prices"))
	dt = datetime.datetime.now(tz=EST5EDT()) 
        
        for price in prices:
            price["time"]= dt.strftime('%H:%M:%S')
            price["Date-time"] = dt.strftime('%Y-%m-%d %H:%M:%S')
            price["start_latitude"] = df_1['coordinates__latitude'][0]
            price["start_longitude"] = df_1['coordinates__longitude'][0]
            price["end_latitude"] = df_2['coordinates__latitude'][0]
            price["end_longitude"] = df_2['coordinates__longitude'][0]
#        f = open("uber.json", "a")
	filename = os.path.join(os.getcwd(), 'uber.json')	
#	f = open(filename, "a")
#        f.write(str(time))
#        f.write(str(prices))
#        f.close()
        df = pd.DataFrame(prices)
        df = df.append(df)
        out = df.to_json(orient='records')
        with open(filename, 'a') as outfile:
            json.dump(out, outfile)
            outfile.close()
        for p in prices:
            print(p)
    else:
        actual_second() 
        
        
        
        
    


# In[24]:


actual_second()


# In[25]:


df_3 = pd.read_sql('SELECT count(*) FROM yelp_reviews', con =conn)
df_3.head()


# In[26]:


    
'''    
    for p in prices:
        print(p)
'''          
    #print ("%s :: Distance %s Low  %s  High %s" % (p['localized_display_name'],p['distance'],p['low_estimate'],p['high_estimate']))    
    #  print ("%s Distance %.3f Low  %.2f  High %.2f" % (p['localized_display_name'],p['distance'],p['low_estimate'],p['high_estimate']))  
  


# Load uber data into SQL db

# In[27]:


import json, os
def load_uber(j):
    p=os.path.join("", j)
    print (p)
    with open(p, 'rU') as f:
      data = [json.loads(row) for row in f]
    return data


# In[28]:


# import sqlite3
# Will create uber_db.sqlite if it doesn't exist.
uber_db = 'uber_db.sqlite' 
conn = sqlite3.connect(uber_db) 
c = conn.cursor()


# In[29]:


review_u = pd.read_csv("uber.csv", encoding ='latin-1')
review_u.head()


# In[30]:


review_u.to_sql('uber_data',              # Name of the table.
             con=conn,                    # The handle to the file that is set up.
             if_exists='replace',         # Overwrite, append, or fail.
             index=False) 


# In[31]:



for row in conn.execute("pragma table_info('uber_data')").fetchall():
    print (row)


# In[32]:


df = pd.read_sql('SELECT * FROM uber_data', con=conn)
df.head()


# In[33]:


df = pd.read_sql('SELECT * FROM uber_data', con=conn)
#df.headisplay()
