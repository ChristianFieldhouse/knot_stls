"""script used to grab all the Arc Presentations from katlas.org"""
import requests
from bs4 import BeautifulSoup
import numpy as np

knots = []

knot_counts = [1, 0, 0, 1, 1, 2, 3, 7, 21, 49, 165]

for i in range(8):
    for k in range(1, knot_counts[i] + 1):
        print(i, k)
        page = requests.get(f"http://katlas.org/wiki/{i}_{k}")
        soup = BeautifulSoup(page.text, "html.parser")

        AP = soup.html.find_all("table")[5].find_all("td")[-1].text
        
        if AP[:2] != "[{":
            AP = soup.html.find_all("table")[6].find_all("td")[-1].text
        if AP[:2] != "[{":
            AP = soup.html.find_all("table")[7].find_all("td")[-1].text

        AP = np.array(eval(AP.replace("{", "(").replace("}", ")")))

        # initial 'dip'
        points = [(0, AP[0, 0], 1), (0, AP[0, 0], 0), (0, AP[0, 1], 0), (0, AP[0, 1], 1)]
        for _ in range(len(AP)):
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
