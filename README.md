# Fys-Stk4155
Hei hei Thomas, da er det å brette opp skjorta og dra opp buksa.
Ettersom Tiril forlot oss så e me nå ikke holdt tebake lenger.

Som den berømte Erna Solberg ein gong sa:
"It's not the fart that kills, it's the smell"

# Prosjekt 1

[Prosjekt1](https://compphysics.github.io/MachineLearning/doc/Projects/2021/Project1/pdf/Project1.pdf)
![Nerd](https://th.bing.com/th/id/OIP.2S1Ssmvx65EgrXPwrIwwewHaHa?w=200&h=200&c=7&r=0&o=5&pid=1.7)

## Opg.1

1. Lag et datasett som består av x og y.
    * Verdiene på x og y skal være mellom 0 og 1
2. Få disse til å funke på FrankeFunksjon(x,y)
    * f(x,y) = FrankeFunksjon(x,y)
    * Legg til normaldistribuert stokastisk støy til denne
3. Lag egen kode for å utføre en "Standard Least Square Regression Analysis"
    * Bruk enten Matrix Inversion, Singular Value Decomposition(fra numpy) eller kode fra før(uke 1 og 2).
    * Bruk polynomial i x og y opp til 5 orden.
4. Finn "Confidence interval" av parameterne \beta ved å beregna varaiansen.
5. Evaluer "Mean Square Error"(MSE) og R^2
6. Koden må inkludere en skalerings metode, et eks på dette er å ta minus gjennomsnittet.
    * Her må vi forklare hvorfor vi velger den valgte meoden.
7. Split dataen i trenings/test sett
    * 70/30 fordeling
    * Bruk egen kode fra før eller Scikit-learn