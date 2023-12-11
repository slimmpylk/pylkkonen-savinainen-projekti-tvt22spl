import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# 1.	Käytetään CSV tai JSON formaattia tietokannan datasta (exportataan se sieltä)
# 2.	Visualisoidaan data 3D kuvaan DONE
# 3.	Arvotaan 6 pistettä realistiselle välille ja funktio joka tulostaa nämä arvotut pisteet raakadatan päälle 3D kuvaan.
# 4.	Aliohjelma jolla voi mitata kaksi 3D-avaruuden pisteen välinen etäisyys ja todistetaan aliohjelman toimivuus (tulostetaan lasku?)
# 5.	Käy läpi kaikki tietokannasta luetut 3D avaruuden pisteet ja mittaa etäisyydet kaikkiin 6 satunnaiseen pisteeseen ja valitaan ”voittajaksi” (mitä meinaa?) se piste johon etäisyys on pienin.
# 6.	Tehdään tietoraenne mihin tallennetaan kunkin keskipisteen lahimmatDatapisteet. Suoritetaan kohta 4 uudelleen siten, että nyt voittajien data tallennetaan tuohon tietorakenteeseen
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
    return np.random.uniform(low, high, (n, 3)) # taulukon alimmasta arvosta ylimpään range

# Vaihe 4: Aliohjelma kahden 3D-avaruuden pisteen etäisyyden mittaamiseen
def laske_3D_etaisyys(point1, point2):
    # Laskee etäisyyden kahden pisteen välillä
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Vaihe 5: Etsitään lähin keskipiste jokaiselle datan pisteelle
def lahimmat_keskipisteet(data, centers): #<- huom centers keskipisteet tuodaan tänne
    # Etsitään lähin keskipiste jokaiselle datan pisteelle
    klusterisijoitukset = [] # lista, joka sisältää mihin keskipisteeseen kunkin datapisteen etäisyys on lyhin eli voittaja
    for point in data: # point = x,y,z ym.
        distances = [laske_3D_etaisyys(point, center) for center in centers] # laskee etäisyyden 3. vaiheen funktiota käyttäen
        closest_center = np.argmin(distances) # etsii lähimmän klusterikeskuksen argmin eli pienin arvo funktiota käyttäen
        klusterisijoitukset.append(closest_center) # lisää lähimmän klusterikeskuksen klusterisijoitukset-listaan
    print(klusterisijoitukset)
    return klusterisijoitukset

# Vaihe 6: Tallennetaan iteroidut keskipisteiden "lahimmatDatapisteet"
def uudet_keskipisteet(data, klusterisijoitukset, n_centers):
    lahimmatDatapisteet = np.zeros((n_centers, data.shape[1]))
    klusterienMaara = np.zeros(n_centers, dtype=int)

    # Lisätään datapisteet niiden klustereihin
    for point, klusterisijoitus in zip(data, klusterisijoitukset):
        lahimmatDatapisteet[klusterisijoitus] += point  # Lisää datapiste klusterin summaan (keskipisteiden summat)
        klusterienMaara[klusterisijoitus] += 1  # Kasvata klusterin datapistemäärää

    # Laske uudet keskipisteet
    for i in range(n_centers):
        if klusterienMaara[i] > 0:
            lahimmatDatapisteet[i] /= klusterienMaara[i]  # Päivitä keskipiste keskiarvona
        else:
            # Uusi keskipiste satunnaisesti, jos klusterissa ei ole datapisteitä
            lahimmatDatapisteet[i] = arvo_satunnaispisteet(1, 1300, 1800)[0] # arpoo uuden pisteen jos jokin kuudesta pisteestä ei ole voittaja
    
    # ei siis käytetä kumulatiivista summaa
    return lahimmatDatapisteet


# Vaihe 6: Tallennetaan keskipisteiden "lahimmatDatapisteet": uudet_keskipisteet-funktio suorittaa tämän vaiheen. Tässä funktiossa käydään läpi kaikki keskipisteet ja kerätään kunkin 
# keskipisteen "lahimmatDatapisteet" eli niille lähimmät arvotut datapisteet klusterisijoitukset-listan perusteella. 
# Sitten lasketaan kunkin keskipisteen voittamien pisteiden keskiarvo, joka määrittää uuden sijainnin 
# kyseiselle keskipisteelle. Jos jokin keskipiste ei voita yhtään datapistettä, se korvataan uudella satunnaisesti arvotulla pisteellä.

# Vaihe 7 & 8: Integroidaan kohdat kokonaisuudeksi
def kmeans(data, n_centers, n_iterations):
    # Suoritetaan KMeans-algoritmi
    centers = arvo_satunnaispisteet(n_centers, 1300, 1800) 
    for _ in range(n_iterations):
        klusterisijoitukset = lahimmat_keskipisteet(data, centers)
        centers = uudet_keskipisteet(data, klusterisijoitukset, n_centers) #centers = treenatut keskipisteet
    
    return centers, klusterisijoitukset

# Vaihe 7 & 8: Integroidaan kohdat kokonaisuudeksi
# kmeans-funktio yhdistää edelliset vaiheet toistuvaksi prosessiksi. Aluksi se luo satunnaiset alkuarvot keskipisteille ja suorittaa sitten toistuvasti vaiheita 5 ja 6, 
# päivittäen keskipisteet jokaisella iteraatiolla. Tämä toistetaan, kunnes algoritmi on suoritettu määritelty määrä iteraatioita (n_iterations). 
# Tämä prosessi vastaa vaatimusta, että algoritmi toistetaan kunnes se on riittävän tarkka - tosin tässä toteutuksessa tarkkuutta ei arvioida muuten kuin iteraatioiden määrällä.

# Vaihe 8: Visualisointi
def visualisointi(data, centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Piirrä kaikki datapisteet
    data_points = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', marker='o', label='Datapisteet')

    # Piirrä KMeans-klusterikeskukset
    centers = np.array(centers)
    cluster_centers = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=100, label='Centroid keskipisteet')

    # Lisää selite kuvaajaan
    ax.legend()

    ax.set_xlabel('Sensoriarvo B')
    ax.set_ylabel('Sensoriarvo C')
    ax.set_zlabel('Sensoriarvo D')
    plt.show()

# 9. Toinen visualisointi 
def visualisointi2(data, centers, klusterisijoitukset):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Piirrä kunkin klusterin datapisteet
    for i in range(len(centers)):
        points = data[np.array(klusterisijoitukset) == i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f'Klusteri {i+1} Pisteet')

    # Piirrä klusterikeskukset
    centers = np.array(centers)
    cluster_centers = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=100, label='Centroid keskipisteet')

    # Lisää selite kuvaajaan
    ax.legend()

    ax.set_xlabel('Sensoriarvo B')
    ax.set_ylabel('Sensoriarvo C')
    ax.set_zlabel('Sensoriarvo D')
    plt.show()


    
# Pääohjelman suoritus
file_name = 'updated_data.json'
raw_data = load_data_from_json(file_name)
data = convert_data_to_np_array(raw_data)

# Tulosta alun datapisteet
print("Alun datapisteet:")
print(data[:5])  # Muuta 5 siihen, kuinka monta pistettä haluat tulostaa

# Laske ja tulosta kumulatiiviset summat
# cumulative_sums = np.cumsum(data, axis=0)
# print("Kumulatiiviset summat:")
# print(cumulative_sums)

# Käynnistetään KMeans 6 keskipisteellä ja toistetaan vaikka 600 kertaa
centers, klusterisijoitukset = kmeans(data, 6, 600)
visualisointi(data, centers)
visualisointi2(data, centers, klusterisijoitukset)


# Vaihe 9: Tulostetaan lopulliset keskipisteet
print("Lopulliset keskipisteet:")
for center in centers:
    print(center)
    
# Tallennetaan treenatut keskipisteet C-kielen taulukkomuotoon
def tallenna_c_taulukkoon(centers, tiedosto_nimi):
    with open(tiedosto_nimi, 'w') as tiedosto:
        tiedosto.write('float centers[{}][3] = {{\n'.format(len(centers)))
        for center in centers:
            tiedosto.write('    {{{:.6f}, {:.6f}, {:.6f}}},\n'.format(center[0], center[1], center[2]))
        tiedosto.write('};\n')

# Kutsu funktiota tallentamaan keskipisteet
tallenna_c_taulukkoon(centers, 'kmeans_centers.h')

