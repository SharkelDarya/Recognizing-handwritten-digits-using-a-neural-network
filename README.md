# Recognizing-handwritten-digits-using-a-neural-network
Trening sieci neuronowej do rozpoznawania odręcznych cyfr.

<p align="center">
      <img src="https://i.ibb.co/XzVZpdF/digit.png" alt="Project Logo" width="512">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Engine-PyCharm%2023-B7F352" alt="Engine">
</p>

## About

Projekt rozwiązuje problem klasyfikacji cyfr przy użyciu zbioru danych MNIST. Zbiór danych MNIST zawiera obrazy odręcznie napisanych cyfr od 0 do 9. Projekt wykorzystuje głęboką sieć neuronową zbudowaną przy użyciu biblioteki Keras do trenowania modelu rozpoznawania i klasyfikowania cyfr na obrazach.
## Documentation

### Libraries
**-** **`NumPy`**, **`matplotlib`**, **`sklearn`**, **`keras`**

### Przygotowanie danych
- Używany jest zbiór danych MNIST, który jest podzielony na zestawy treningowe i testowe.
- Piksele obrazu są normalizowane do zakresu od 0 do 1.
  
### Tworzenie modelu sieci neuronowej
- Tworzonшу modelг sekwencyjny z wieloma warstwami.
- Zastosowano kategoryczną entropię krzyżową jako funkcję straty.
  
### Trening modelowy
- Model jest trenowany na danych treningowych przez 20 epok przy użyciu optymalizatora Adam.
  
### Ocena wydajności
- Ocena modelu na danych testowych i wyprowadzania dokładności klasyfikacji.
  
### Analiza krzywych uczenia
- Wykresy dokładności i funkcji strat w epokach treningowych.
  
### Uzyskiwanie prognoz i analizowanie metryk
- Używanie modelu do przewidywania klas na danych testowych.
- Obliczanie precyzji i wycofanie.
- Krzywa ROC
  
### Wizualizacja wyników
- Wyświetlanie niektórych obrazów testowych oraz odpowiadające im prawdziwe etykiety.
## Developers

- Darya Sharkel (https://github.com/SharkelDarya)

