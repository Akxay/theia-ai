from bs4 import BeautifulSoup
import urllib.request
import re
import os

directory = 'data_msan'

if not os.path.exists(directory):
    os.makedirs(directory)


url = 'https://www.usfca.edu/arts-sciences/graduate-programs/data-science/our-students'  # noqa

response = urllib.request.urlopen(url)
html = response.read()

# images
soup = BeautifulSoup(html, "lxml")
x = []
for element in soup.findAll("p"):
    x.append(element)

headshot_links = []
names = []

for element in x:
    element = str(element).split('src="')
    if len(element) > 1:
        # get the link
        element = element[1]
        element = element.split(" title")
        headshot_links.append(element[0][:-1])

        # get the names from the link
        element = element[0].split('msan-student-')[1]
        element = element.split('.jpg')
        names.append(element[0])

for i in range(len(headshot_links)):
    try:
        f = open(directory+'/'+str(names[i]) + '.jpg', 'wb')
        f.write(urllib.request.urlopen(headshot_links[i]).read())
        f.close()
    except Exception:
        print("Unpredictable error:", headshot_links[i])
