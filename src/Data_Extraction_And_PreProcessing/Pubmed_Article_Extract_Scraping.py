"""
Given the Article ID, this code retrieves the Abstract, Title, MesH terms and PT Tags from PubMed by web scraping
Input:
All_Clincial_Hedges_Articles.csv
Output:
Final_Retrieved_Clinical_Hedges_PT.csv
"""

from bs4 import BeautifulSoup
import requests

import os
import csv

class PubMedObject(object):
    soup = None
    url = None

    # pmid is a PubMed ID
    # url is the url of the PubMed web page
    # search_term is the string used in the search box on the PubMed website
    def __init__(self, pmid=None, url='', search_term=''):
        if pmid:
            pmid = pmid.strip()
            url = "http://www.ncbi.nlm.nih.gov/pubmed/%s" % pmid
        if search_term:
            url = "http://www.ncbi.nlm.nih.gov/pubmed/?term=%s" % search_term
        page = requests.get(url).text
        self.soup = BeautifulSoup(page, "html.parser")

        # set the url to be the fixed one with the PubMedID instead of the search_term
        if search_term:
            try:
                url = "http://www.ncbi.nlm.nih.gov/pubmed/%s" % self.soup.find("dl",class_="rprtid").find("dd").text
            except AttributeError as e:  # NoneType has no find method
                print("Error on search_term=%s" % search_term)
        self.url = url

    def get_title(self):
        title = self.soup.find(class_="abstract").find("h1")
        return title.text if title is not None else None

    def get_abstract(self):
        abstract = self.soup.find(class_="abstract").find(class_="abstr")
        return abstract.text[8:] if abstract is not None else None

    def get_PT(self):
        find = self.soup.find(class_="abstract").find(class_="other_content")
        find = find.find(class_="morecit") if find is not None else None
        find = find.find('h4', text='Publication types') if find is not None else None
        PT = find.findNext('ul') if find is not None else None
        return list(sorted(list(PT.stripped_strings))) if PT is not None else None

    def get_MesH(self):
        find = self.soup.find(class_="abstract").find(class_="other_content")
        find = find.find(class_="morecit") if find is not None else None
        find = find.find('h4', text='MeSH terms') if find is not None else None
        MesH = find.findNext('ul') if find is not None else None
        return list(sorted(list(MesH.stripped_strings))) if MesH is not None else None


path_to_data = os.path.abspath("../Dataset")
all_articles = []

with open(os.path.join(path_to_data, "All_Clincial_Hedges_Articles.csv")) as fp:
    reader = csv.reader(fp)
    for row in reader:
        all_articles.append(row[0])

file_all_PT = open(os.path.join(path_to_data, 'Final_Retrieved_Clinical_Hedges_PT.csv'), 'w', newline='')
writer_all_PT = csv.writer(file_all_PT)


def PubMed_Extract(writer, id_list):
    for i, pmid in enumerate(id_list):
        try:
            pubMed = PubMedObject(pmid=pmid)
            title = pubMed.get_title()
            abstract = pubMed.get_abstract()
            PT = pubMed.get_PT()
            MesH = pubMed.get_MesH()
            writer.writerow([pmid, title, ' '.join(abstract.split('\n')) if abstract is not None else None,
                             ', '.join(PT) if PT is not None else None, ', '.join(MesH) if MesH is not None else None])
        except Exception as e:
            print("ID {} raised Exception: ".format(i), e)
        print("Done with {}".format(i))


PubMed_Extract(writer_all_PT, all_articles)
file_all_PT.close()
