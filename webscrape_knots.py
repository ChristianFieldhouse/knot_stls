"""script used to grab all the Arc Presentations from katlas.org"""
import requests
from bs4 import BeautifulSoup
import numpy as np

knots = []

for i in range(1, 166):
    print(i)
    page = requests.get(f"http://katlas.org/wiki/10_{i}")
    soup = BeautifulSoup(page.text, "html.parser")

    AP = soup.html.find_all("table")[5].find_all("td")[-1].text
    
    if AP[:2] != "[{":
        AP = soup.html.find_all("table")[6].find_all("td")[-1].text

    AP = np.array(eval(AP.replace("{", "(").replace("}", ")")))

    # initial 'dip'
    points = [(0, AP[0, 0], 1), (0, AP[0, 0], 0), (0, AP[0, 1], 0), (0, AP[0, 1], 1)]
    for i in range(len(AP)):
        next_row = np.argmax(AP[:, 0] == points[-1][1])
        points += [
            (next_row, points[-1][1], 1),
            (next_row, points[-1][1], 0),
            (next_row, AP[next_row, 1], 0), 
            (next_row, AP[next_row, 1], 1),
        ]
    knots.append((AP, points))

just_knots = [k[1] for k in knots]
print(just_knots)
