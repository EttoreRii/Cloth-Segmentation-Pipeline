# Modulo di Visione Artificiale per la Manipolazione Robotica di Capi d'Abbigliamento

Questo repository contiene il codice e la documentazione del **Modulo di Visione Artificiale**, il cui obiettivo è identificare le parti salienti di un maglioncino (*colletto, polsini e bordo inferiore*) ed estrarne le coordinate spaziali 3D necessarie per la manipolazione tramite un robot (es. Universal Robots UR5).

Il sistema integra un modello di segmentazione ad istanze real-time con un algoritmo di post-processing geometrico per la proiezione 3D dei punti di interesse a partire da dati RGB-D.

---

## 📌 Indice
- [Dataset e Annotazione](#-dataset-e-annotazione)
- [Architettura del Modello](#-architettura-del-modello)
- [Valutazione delle Performance](#-valutazione-delle-performance)
- [Algoritmo di Estrazione delle Coordinate 3D](#-algoritmo-di-estrazione-delle-coordinate-3d)
- [Formato dei Dati in Uscita](#-formato-dei-dati-in-uscita)

---

## 📊 Dataset e Annotazione

Per l'addestramento del modello è stato creato un dataset ad hoc utilizzando immagini RGB e mappe di profondità acquisite tramite una telecamera **Intel RealSense D435i**. Il dataset comprende sia immagini scattate in un ambiente controllato di laboratorio, sia immagini provenienti da fonti online per aumentare la variabilità e la robustezza complessiva del modello.

L'etichettatura delle immagini è stata effettuata manualmente con il software **CVAT (Computer Vision Annotation Tool)**, definendo i poligoni per tre classi di interesse:
* 🧥 **Collar**: La linea del colletto del maglioncino.
* 🧤 **Cuff**: I polsini delle maniche.
* 🧵 **Hem**: Il bordo inferiore (fondomaglia).

I poligoni di annotazione sono stati convertiti nel formato richiesto da **Ultralytics YOLOv8-seg** e il dataset finale è stato suddiviso in set di **Training (80%)** e **Validation (20%)**.

---

## 🧠 Architettura del Modello

Il cuore del sistema si basa su **YOLOv8-seg Nano (YOLOv8n-seg)**, la versione più leggera e veloce della famiglia Ultralytics dedicata alla segmentazione delle istanze.

La scelta è motivata dalla necessità di operare in **tempo reale** su hardware embedded con risorse computazionali e di memoria limitate, come la **NVIDIA Jetson Orin Nano**. L'architettura YOLOv8n-seg garantisce un ottimo compromesso tra frame rate (FPS) e accuratezza (mAP).

### Caratteristiche principali dell'architettura:
* **Backbone:** Una versione ottimizzata di *CSPDarknet53* per l'efficace estrazione delle feature.
* **Neck:** Utilizzo di *PANet (Path Aggregation Network)* per migliorare la fusione delle feature a diverse scale geometriche.
* **Head:** *Decoupled Head* per la classificazione e la regressione dei bounding box, integrata con un ramo dedicato alla generazione delle maschere di segmentazione tramite prototipi.

### Dettagli di Addestramento:
* **Epoche:** 30
* **Risoluzione di input:** 640x640 pixel
* **Ottimizzatore:** SGD (Stochastic Gradient Descent)
* **Funzione di Loss:** Combinazione di *box loss*, *class loss* e *mask loss*.

---

## 📈 Valutazione delle Performance

I risultati ottenuti al termine dell'addestramento sul validation set sono riassunti nella tabella seguente:

| Metrica | Precision | Recall | mAP@50 |
| :--- | :---: | :---: | :---: |
| **Box (Detection)** | 85.4% | 85.0% | **90.3%** |
| **Mask (Segmentation)** | 65.4% | 60.7% | **56.8%** |

### Analisi dei Risultati
L'elevato valore di **mAP@50 per i box (90.3%)** indica un'ottima capacità del modello di localizzare correttamente le macro-zone di interesse. La metrica **mAP@50 per le maschere (56.8%)** risulta inferiore a causa della complessità intrinseca della segmentazione di elementi flessibili e sottili (come cuciture e bordi dell'abbigliamento), ma si dimostra comunque ampiamente adeguata per l'estrazione della linea media necessaria alla successiva manipolazione robotica.

---

## 📐 Algoritmo di Estrazione delle Coordinate 3D

Una volta ottenute le maschere binarie dalle predizioni di YOLOv8-seg, viene applicata una pipeline di post-processing geometrico strutturata in 4 passaggi:

1. **Filtraggio delle Maschere:** Se vengono rilevate più istanze della stessa classe, viene mantenuta solo quella con il punteggio di confidenza più elevato. Viene inoltre applicato un filtraggio basato su *Intersection over Union (IoU)* per eliminare sovrapposizioni spurie.
2. **Y-Averaging (Linea Media):** Per ogni coordinata $x$ lungo la maschera, viene calcolata la media dei valori $y$ dei pixel appartenenti alla maschera. Questo permette di individuare lo "scheletro" o la linea centrale dell'elemento (es. il centro della curva del colletto), ignorando lo spessore del bordo o del tessuto.
3. **Campionamento dei Punti:**
   * **Collar & Hem:** Vengono estratti $N$ punti equidistanti lungo l'asse $x$ per mappare accuratamente la curvatura della cucitura.
   * **Cuff:** Vengono estratti unicamente i punti estremi (inizio e fine) per definire l'orientamento vettoriale e la larghezza del polsino.
4. **Proiezione 3D:** Per ogni punto $(x, y)$ 2D campionato dall'immagine RGB, viene recuperato il rispettivo valore di profondità $z$ dal frame *depth* allineato della RealSense. Sfruttando i parametri intrinseci della telecamera (focal length e optical center), le coordinate pixel vengono trasformate in coordinate tridimensionali $(X, Y, Z)$ espresse nel sistema di riferimento della telecamera e, tramite matrice di calibrazione *eye-in-hand*, in quello della base del robot.

---

## 💾 Formato dei Dati in Uscita

Le coordinate 3D finali vengono serializzate e salvate in un file in formato **JSON**, rendendole immediatamente leggibili da parte del modulo di pianificazione delle traiettorie del robot.

Ogni elemento contiene l'elenco dei punti espressi in metri:

```json
{
  "timestamp": 1718812345.67,
  "features": {
    "collar": [
      {"x": 0.123, "y": 0.456, "z": 0.789},
      {"x": 0.145, "y": 0.462, "z": 0.785}
    ],
    "cuff_left": [
      {"x": -0.210, "y": 0.350, "z": 0.810},
      {"x": -0.180, "y": 0.340, "z": 0.805}
    ],
    "hem": [
      {"x": -0.050, "y": 0.600, "z": 0.750}
    ]
  }
}