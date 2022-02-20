# Projektarbeit Benedikt Hollmann - Anime Face GAN

Erstellen von Anime Gesichtern unter Nutzung von unterschiedlichen Implementierungen eines Generative Adversial Networks (GAN)

- Deep Convolution GAN (mit Label Smoothing und parallelen Layern im Generator)
- Wasserstein Deep Convolution GAN (mit parallelen Layern im Generator)
- Wasserstein Deep Convolution GAN mit Gradient Penalty (vereinfachte Layerstruktu mit Label Smoothing)

Dozent: Philipp Koch  
Immatrikulationsnummer: 446024  
Name: Benedikt Hollmann  

# Datengrundlage

Als Grundlage dient ein Datensatz von 63632 Anime Gesichtern (https://github.com/bchao1/Anime-Face-Dataset). 

# Datenanalyse

Der Autor der Datengrundlage hat erwähnt, dass innerhalb der Datengesamtheit einzelne Bildinformationen fehlerhaft sind. Um weiterhin eine qualitative Datengrundlage zu gewährleisten, wird anhand des originalen Datasets eine bereinigte Kopie erstellt. Die bereinigte Kopie besteht aus Bildern, welche eine valide Datengröße besitzen und sich als .png oder .jpg fehlerfrei öffnen lassen.

Im Schritt der Datenbereinigung wurde analysiert, welche Bildhöhen und Bildbreitenverteilung die Bilddaten aus dem Dataset besitzen. 


Analyse Bildbreite             |  Analyse Bildhöhe
:-------------------------:|:-------------------------:
![Bildbreite](https://user-images.githubusercontent.com/56730144/154847414-6cba4481-c48d-4722-8a47-1120ba3aaf1a.png) |  ![Bildhöhe](https://user-images.githubusercontent.com/56730144/154847423-bc7b2e9c-8445-42aa-b524-23a27628468e.png)
  
Eine Analyse der Relation zwischen Breite und Höhe zeigt auf, dass alle Datensätze jeweils identische Seitenverhältnisse besitzen und ein Cropping somit durchgeführt werden kann ohne dass Bilddaten auf Grund von Formatierungen wegfallen.

![grafik](https://user-images.githubusercontent.com/56730144/154847428-204debbc-bd9e-47e8-9f7b-72aee5b3642d.png)

# Änderung der Gewichtsinitialisierung

Entgegen der innerhalb des orignalem GAN-Paper genannten initialen Gewichtsverteilung nach dem Schema:
```
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

wird eine Abwandlung diesem für alle betrachteten GAN-Modelle verwendet. Verwendet wird die He-Initialisierung (auch Kaiming-He Initialisierung genannt), welche im Gegensatz zum originalem Initialisierungsverfahren folgende Vorteile bietet:

- Schnelleres Konvergieren der Ergebnisse
- Probleme bezüglich Exploding/Vanishing - Gradients wird behoben, was insbesondere im Umfeld von GAN's hilfreich erscheint.

Ermöglicht wird dies durch die Dämpfung des Initialisierungsinputs, da Schwankungen innerhalb der originalen Initialisierung exponentiell vergrößert werden.
Siehe hierzu (https://arxiv.org/abs/1502.01852v1)

```
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu') 
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

# Weitere Anpassungen
## Parallel Deep Convolution Layer
Innerhalb des DC-WGAN und des Wasserstein DC-WGAN wurde in Nähe des Inputlayers eine parallele Conv2D Struktur im Generator aufgebaut. Dies hat den Hintergrund, dass ein Equilibrium zwischen dem Generator-Loss und dem Wasserstein-Loss stattfinden muss damit das Modell in der nicht-Wasserstein Struktur gut konvergiert.

- Parallel Layer 1 = KernelSize (1,2)
- Parallel Layer 2 = KernelSize (2,1)

Die Idee ist es, dass durch die unterschiedliche Form der Kernel die jeweiligen Layer unterschiedliche Feinstrukturen herausarbeiten (z.B. Haare, Augen) und eigenständig die Merkmale erlernen. Die gemeinsamen Merkmale werden mittels Concatenate-Layer wieder zusammengetan und in den folgenden Hidden Layern weiter verarbeitet. Das initiale DC-WGAN besaß keine ausreichende Leistungsfähigkeit, um Bilder welche vom Diskriminator als Original bewertet werden zu generieren.   

## Label Smoothing
Die innerhalb der Modelle verwendeten Aktivierungsfunktionen tangieren dazu, Bilder mit dem Label 0 als Fake und Bilder mit dem Label 1 als Real anzusehen. Es kann dazu kommen, dass die Modelle anfangen zu übergeneralisieren und stets die reale Zahl 1 als Garant einer Aktivierung füe reale Bilder nutzen. Um diese Übergeneralisierung zu reduzieren, wird das Label Smoothing Verfahren eingesetzt. Hierbei werde reale Label als 0.9 deklariert. Werden Aktivierungen >0.9 durchgeführt, resultiert eine Abweichung gegenüber des optimalen Wertes wodurch eine Art penalty angewendet wird.

## Experimentelles Gradient Penalty Smoothing
Die Überlegung des Gradient Penalty Smoothing ist analog der Idee das Label Smoothing. Es wird mittels der Manipulation der Gradiententensoren eine Verlagerung der Gradienten von realen Bildern von 1 auf 0.9 avisiert.

Innerhalb der Wasserstein Gradientenberechnung für den Diskriminator
![grafik](https://user-images.githubusercontent.com/56730144/154849031-9e512a33-7048-408a-b9db-0d6887f74d02.png)  

wird die Kostenfunktion des Diskriminators in Relation zur Erkennungsrate von Bildern der Klassen "Fake" oder "Original" durchgeführt.

![grafik](https://user-images.githubusercontent.com/56730144/154849318-03c24177-42b6-4396-a016-35edcb6cfdd8.png)  
Hierbei ist der maximal zulässige Wert *max* mit 1 angegeben.

Durch eine Anpassung der Funktion hinzu  
![grafik](https://user-images.githubusercontent.com/56730144/154849636-c0f251e4-d61d-4081-998f-da072825a381.png)
wird innerhalb des Wasserstein DC-GAN mit Gradient Penalty eine zusätzliche Regularisierung probiert.




## Misc

- Experimentelle Ermittlung von Dropout-Layer Positionen und Werten
- Integration von BatchNormalization
- 
