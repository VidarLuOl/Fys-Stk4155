# Fys-Stk4155
Hei hei Thomas, da er det å brette opp skjorta og dra opp buksa.
Ettersom Tiril forlot oss så e me nå ikke holdt tebake lenger.

Som den berømte Erna Solberg ein gong sa:
"It's not the fart that kills, it's the smell"

# Prosjekt 1

[Prosjekt1](https://compphysics.github.io/MachineLearning/doc/Projects/2021/Project1/pdf/Project1.pdf)

[Overleaf](https://www.overleaf.com/project/613f69c77de05db1b34e766b)

![Nerd](https://www.overleaf.com/project/613f69c77de05db1b34e766b)

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

## Litt om scaling
The Normalizer scales each data point such that the feature vector has a euclidean length of one. In other words, it projects a data point on the circle (or sphere in the case of higher dimensions) with a radius of 1. This means every data point is scaled by a different number (by the inverse of it’s length). This normalization is often used when only the direction (or angle) of the data matters, not the length of the feature vector.

The RobustScaler works similarly to the StandardScaler in that it ensures statistical properties for each feature that guarantee that they are on the same scale. However, the RobustScaler uses the median and quartiles, instead of mean and variance. This makes the RobustScaler ignore data points that are very different from the rest (like measurement errors). These odd data points are also called outliers, and might often lead to trouble for other scaling techniques.

## Hva er bootstrap?
The independent bootstrap works like this:

1. Draw with replacement n numbers for the observed variables x=(x1,x2,⋯,xn).
2. Define a vector x∗ containing the values which were drawn from x.
3. Using the vector x∗ compute βˆ∗ by evaluating βˆ under the observations x∗.
4. Repeat this process k times