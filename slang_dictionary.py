# -*- coding: utf-8 -*-
"""COMM557_slang dictionary.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fDcYxJsR3qP_Pfr3xtDYdYjrYCKNYNIS
"""

!pip install requests
!pip install html5lib
!pip install bs4

import requests
URL = "https://www.webopedia.com/reference/twitter-dictionary-guide/"
r = requests.get(URL)
print(r.content)

from bs4 import BeautifulSoup
soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib
table = soup.find('div', attrs = {'id':'article_main_column'})

lst = []
for y in table.find_all("p",class_="p1"):
      ft = y.getText()
      if (ft.find(':') != -1):
        ft = ft.split(':')
        for i in ft:
          lst.append(i)

slang = {lst[i]:lst[i+1] for i in range (0, len(lst),2)}

del slang['Some of the Twitter lingo describes the collection of people who use the service, while other abbreviations are used to describe specific functions and features of the service itself (like the Twitter abbreviation “RT”). There are also a number of terms which are essentially Twitter shorthand']

print(slang)