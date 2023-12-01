import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# 1.	Käytetään CSV tai JSON formaattia tietokannan datasta (exportataan se sieltä)
# 2.	Visualisoidaan data 3D kuvaan DONE
# 3.	Arvotaan 6 pistettä välille 1000-2000 ja funktio joka tulostaa nämä arvotut pisteet raakadatan päälle 3D kuvaan.
# 4.	Aliohjelma jolla voi mitata kaksi 3D-avaruuden pisteen välinen etäisyys ja todistetaan aliohjelman toimivuus (tulostetaan lasku?)
# 5.	Käy läpi kaikki tietokannasta luetut 3D avaruuden pisteet ja mittaa etäisyydet kaikkiin 6 satunnaiseen pisteeseen ja valitaan ”voittajaksi” (mitä meinaa?) se piste johon etäisyys on pienin.
# 6.	Tehdään tietoraenne mihin tallennetaan kunkin keskipisteen voittajat. Suoritetaan kohta 4 uudelleen siten, että nyt voittajien data tallennetaan tuohon tietorakenteeseen
# 7.	Lasketaan uudet keskipisteet edellisessä kohdassa kerätyn datan perusteella. Jos joku keskipiste ei voittanut yhtään 3D pistettä itselleen, niin sen uusi arvo määritellään arpomalla uusi alkuarvo.
# 8.	Integroidaan kohdat kokonaisuudeksi joka toistetaan kunnes algoritmin antama tulos on riittävän tarkka (eli keskipiteet eivät enää liiku tai kokonaisuus toistetaan X kertaa)
# 9.	Lopuksi tulostusfunktio jolla saadaan tulostettua algoritmin opettamat 6 kesipistettä c-kieliseen taulukkoon kmeans.h tiedostoon.

# Vaihe 1: Käytetään JSON-formaattia tietokannan datasta
def load_data_from_json(file_name):
    # Ladataan data JSON-tiedostosta
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

# Vaihe 2: Muunnetaan data numpy-taulukoksi
def convert_data_to_np_array(data):
    # Muunnetaan data numpy-taulukoksi
    return np.array([[d['sensorvalue_b'], d['sensorvalue_c'], d['sensorvalue_d']] for d in data])

# Vaihe 3: Arvotaan 6 pistettä välille 1300-1800
def arvo_satunnaispisteet(n, low, high):
    # Arvotaan 6 pistettä annetulla välillä
    return np.random.uniform(low, high, (n, 3))

# Vaihe 4: Aliohjelma kahden 3D-avaruuden pisteen etäisyyden mittaamiseen
def laske_3D_etaisyys(point1, point2):
    # Laskee etäisyyden kahden pisteen välillä
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Vaihe 5: Etsitään lähin keskipiste jokaiselle datan pisteelle
def lahimmat_keskipisteet(data, centers):
    # Etsitään lähin keskipiste jokaiselle datan pisteelle
    assignments = []
    for point in data:
        distances = [laske_3D_etaisyys(point, center) for center in centers]
        closest_center = np.argmin(distances)
        assignments.append(closest_center)
    return assignments

# Vaihe 6: Tallennetaan keskipisteiden "voittajat"
def uudet_keskipisteet(data, assignments, n_centers):
    # Lasketaan uudet keskipisteet ja päivitetään keskipisteiden ryhmät
    new_centers = []
    for i in range(n_centers):
        points = data[np.array(assignments) == i]
        if len(points) > 0:
            new_centers.append(np.mean(points, axis=0))
        else:
            new_centers.append(arvo_satunnaispisteet(1, 1300, 1800)[0]) # Huom: Alkuperäinen välillä 1500-1800, muutettu ohjeiden mukaisesti välille 1000-2000 t sauli
    return new_centers

# Vaihe 7: Integroidaan kohdat kokonaisuudeksi
def kmeans(data, n_centers, n_iterations):
    # Suoritetaan KMeans-algoritmi
    centers = arvo_satunnaispisteet(n_centers, 1300, 1800) # Huom: Alkuperäinen välillä 1500-1800, muutettu ohjeiden mukaisesti välille 1000-2000 t sauli
    for _ in range(n_iterations):
        assignments = lahimmat_keskipisteet(data, centers)
        centers = uudet_keskipisteet(data, assignments, n_centers)
    return centers, assignments


# Vaihe 8: Visualisointi
def visualisointi(data, centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Piirretään kaikki datapisteet
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', marker='o')

    # Piirretään KMeans-keskipisteet, käytetään eri väriä
    centers = np.array(centers)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=100)

    ax.set_xlabel('Sensorvalue B')
    ax.set_ylabel('Sensorvalue C')
    ax.set_zlabel('Sensorvalue D')
    plt.show()
   
# 9. Toinen visualisointi 
def visualisointi2(data, centers, assignments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Piirretään datapisteet
    for i in range(len(centers)):
        points = data[np.array(assignments) == i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    # Piirretään keskipisteet, käytetään eri väriä
    centers = np.array(centers)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=100)

    ax.set_xlabel('Sensorvalue B')
    ax.set_ylabel('Sensorvalue C')
    ax.set_zlabel('Sensorvalue D')
    plt.show()
    
# Pääohjelman suoritus
file_name = 'updated_data.json'
raw_data = load_data_from_json(file_name)
data = convert_data_to_np_array(raw_data)

# Käynnistetään KMeans 6 keskipisteellä ja toistetaan 10 kertaa
centers, assignments = kmeans(data, 6, 10)
visualisointi(data, centers)
visualisointi2(data, centers, assignments)


# Vaihe 9: Tulostetaan lopulliset keskipisteet
print("Lopulliset keskipisteet:")
for center in centers:
    print(center)

# Tallennetaan keskipisteet tiedostoon (esimerkki)
np.savetxt('kmeans_centers.txt', centers)
