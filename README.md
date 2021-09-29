# Fys-Stk4155
Hei hei Thomas, da er det å brette opp skjorta og dra opp buksa.
Ettersom Tiril forlot oss så e me nå ikke holdt tebake lenger.

Som den berømte Erna Solberg ein gong sa:
"It's not the fart that kills, it's the smell"

# Prosjekt 1

[Prosjekt1](https://compphysics.github.io/MachineLearning/doc/Projects/2021/Project1/pdf/Project1.pdf)

[Overleaf](https://www.overleaf.com/project/613f69c77de05db1b34e766b)

![Nerd](https://th.bing.com/th/id/OIP.hSuMIGsyCZIsc5Fw56D16QHaKo?w=117&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7)



## Litt om scaling
The Normalizer scales each data point such that the feature vector has a euclidean length of one. In other words, it projects a data point on the circle (or sphere in the case of higher dimensions) with a radius of 1. This means every data point is scaled by a different number (by the inverse of it’s length). This normalization is often used when only the direction (or angle) of the data matters, not the length of the feature vector.

The RobustScaler works similarly to the StandardScaler in that it ensures statistical properties for each feature that guarantee that they are on the same scale. However, the RobustScaler uses the median and quartiles, instead of mean and variance. This makes the RobustScaler ignore data points that are very different from the rest (like measurement errors). These odd data points are also called outliers, and might often lead to trouble for other scaling techniques.

## Hva er bootstrap?
The independent bootstrap works like this:

1. Draw with replacement n numbers for the observed variables x=(x1,x2,⋯,xn).
2. Define a vector x∗ containing the values which were drawn from x.
3. Using the vector x∗ compute βˆ∗ by evaluating βˆ under the observations x∗.
4. Repeat this process k times