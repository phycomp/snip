from lxml.etree import HTMLParser, HTML
from pandas import DataFrame

cntnt=open('polypRslt.html').read()
df=DataFrame(columns=['mrn1', 'mrn2', 'report'])
for ele in tree.findall('.//div[@name="srchRslt"]'): 
    v=ele.values() 
    xdiv=ele.xpath('div')[0]
    df.mrn1, df.mrn2, df.report=v[2], v[3], xdiv.text 
df.to_csv('/tmp/newPolyp.csv') 
