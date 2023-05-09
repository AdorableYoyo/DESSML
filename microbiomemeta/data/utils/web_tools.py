from urllib.request import urlopen, Request
from urllib.parse import quote, urlencode
from urllib.error import HTTPError

import xmltodict


def CIRconvert(ids):
    """ Convert chemical identities to SMILES using CADD Group Chemoinformatics Tools.

    Args:
        ids (str): chemical identities.

    Return (str):
        Return the SMILES string if the identity is found. Otherwise, return None.
    """
    try:
        url = "http://cactus.nci.nih.gov/chemical/structure/" + quote(ids) + "/smiles"
        ans = urlopen(url).read().decode("utf8")
        return ans
    except HTTPError:
        return None


def GIEconvert(ids):
    """ Convert Gene IDs to UniProtKB entry IDs.

    Args:
        ids (list): list of gene identities, the identities must be strings.

    Return (str):
        Gene id to entry id map in tab separated format.
    """
    url = "https://www.uniprot.org/uploadlists/"

    params = {"from": "GENENAME", "to": "ID", "format": "tab", "query": " ".join(ids)}

    data = urlencode(params)
    data = data.encode("utf-8")
    req = Request(url, data)
    with urlopen(req) as f:
        response = f.read()
    return response.decode("utf-8")


def uniprot_query(params):
    """ Retrieve data from UniProt with their REST API.

    Args:
        params (str): queries. A example:
          params = {
            'from': 'ACC+ID',
            'to': 'ENSEMBL_ID',
            'format': 'tab',
            'query': 'P40925 P40926 O43175 Q9UM73 P97793'
          }

    Return:
        response (str): response from UniProt as utf-8 encoded string.
    """
    url = "https://www.uniprot.org/uniprot/"

    data = urlencode(params)
    data = data.encode("utf-8")
    req = Request(url, data)
    with urlopen(req) as f:
        response = f.read()
    return response.decode("utf-8")


def pfam_query(uniprotid):
    """ Get search result from Pfam with UniProt ID.

    Args:
        uniprotid (str): protein's UniProt ID.

    Return:
        res_dict (dict): the response from Pfam is in XML format. The XML is then
          converted to dictionary with xmltodict tool.
    """
    url = "http://pfam.xfam.org/protein/{}?output=xml"
    response = urlopen(url.format(uniprotid))
    xml = response.read().decode()
    res_dict = xmltodict.parse(xml)
    return res_dict
