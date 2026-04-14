## მიმოხილვა
House Prices — Advanced Regression Techniques - მთავარი მიზანია 79 feature ს მიხედვით დავადგინოთ სახლის დაახლოებითუი ფასი. შეფასების კრიტერიუმია RMSLE (ფესვი ლოგარითმების სხვაობიდან)

## ჩემი მიდგომა
1. cleaning - outlier ების ამოცნობა. აღმოჩნდა რომ ორი სახლი მონაცემებთან შედარებით გაცილებით იაფად გაიყიდა, ასევე ამოვშალე ის სვეტები რომლებშიც 70% ზე მეტი დაკარგული იყო.
2. Feature engineering - სვეტები რომლებიც ხარისხს აღნიშნავდა ჩავანაცვლე შესაბამისი რიცხვებით (რაც მაღალი ხარისხი უფრო მაღალი რიცხვი), კატეგორიული ცვლადები რომელთაც აქვთ მაქსიმუმ 5 უნიკალური მნიშვნელობა OHE, 5 ზე მეტი WOE, ბოლოს ახალი ცვლადების დამატება (მაგალითად ,სახლის ასასკი).
3. Feature selection  - RFE , Correlation filter
4. Training - Ridge,Lasso, DecisionTree, XGBoost სხვადასხვა ჰიპერპარამეტრებით.
5. MLFlow - ყველა ექსპერიმენტი დარეგისტრირებულია.

## რეპოზიტორიის სტრუქტურა

```
ML_Assignment_1/
├── model_experiment.ipynb   ←  ძირითადი ფაილი სადაც ხდებოდა  
├── model_inference.ipynb    ←  საუკეთესო მოდელის ჩამოტვირთვა და საბმიშენის შექმნა
├── training.py              ← მეთოდი რომელიც ატრენინგებს და შედეგს ტვირტავს dagshub ზე
├── preprocessor.py          ← პრეპროცესორები feature engineering თვის
├── mapping.py               ← ხარისხის განმსაძღვრელი სვეტები თავიანთი მაპინგით
├── plots/
├── submission.csv           
└── Readme.md
```


---

## Cleaning

### Outlier-ების ამოღება

`GrLivArea > 4000` და `SalePrice < 300,000` კომბინაციის მქონე 2 ჩანაწერი ამოვიღე. ძალიან დიდი სახლი ძალიან დაბალ ფასად.

### High-Missing სვეტების წაშლა

Train split-ზე დავთვალე missing rate. სვეტები, სადაც 70%-ზე მეტი მნიშვნელობა გამოტოვებულია, სრულად ავიღე 



### NA მნიშვნელობების შევსება (`NAFiller`)

- **რიცხვითი სვეტები** — მედიანა 
- **კატეგორიული სვეტები** — მოდა 

---

## Feature Engineering

### QualityEncoder — ორდინალური კოდირება

კონკრეტული სვეტები  ტექსტური მნიშვნელობებით (Ex/Gd/TA/Fa/Po/NA) — გადავიყვანე 5/4/3/2/1/0 სკალაზე. 

### FeatureAdder — ახალი ნიშნები

| ახალი ნიშანი | ფორმულა | დასაბუთება |
|---|---|---|
| `TotalSF` | TotalBsmtSF + 1stFlrSF + 2ndFlrSF | სრული ფართი სართულების ჩათვლით |
| `TotalBathrooms` | FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath |  აბაზანების საერთო რაოდენობა  |
| `HouseAge` | YrSold - YearBuilt | სახლის ასაკი გაყიდვის მომენტში |
| `RemodAge` | YrSold - YearRemodAdd | ბოლო რემონტიდან დასული დრო |
| `WasRemodeled` | YearBuilt ≠ YearRemodAdd | binary — გარემონტდა თუ არა |
| `QualityXArea` | OverallQual × GrLivArea | ხარისხი × ფართობი |
| `HasPool`, `HasGarage`, `Has2ndFloor`, `HasFireplace` |


### WOEEncoder — Weight of Evidence

კატეგორიული სვეტები (5-ზე მეტი მნიშვნელობა) WOE. 

### OneHotEncoderSafe — One-Hot კოდირება

დარჩენილი კატეგორიული სვეტები (≤5 unique) get_dummies-ით. 

---

## Feature Selection

### 1. RFE (Recursive Feature Elimination)

Ridge-ზე დაყრდნობით ვარჩევ 30 საუკეთესო ნიშანს. RFE თანდათან ყველაზე სუსტ ნიშანს ამოაგდებს.

**30 vs 50 feature-ის შედარება:**
- 30 feature: val_rmsle ≈ 0.0093 
- 50 feature: val_rmsle ≈ 0.0090

→ 50 feature-ი ოპტიმალურია.

### 2. კორელაციის ფილტრი (threshold = 0.85)

RFE-ს შემდეგ ავიღე ის წყვილები, სადაც Pearson კორელაცია > 0.85. წყვილიდან ვტოვებ იმ ნიშანს, რომელსაც target-თან ძლიერი კორელაცია აქვს.

**threshold-ის ეფექტი:**
- 0.95 →  overfitting რისკი
- 0.85 → ოპტიმალური ბალანსი
- 0.50 →  underfitting

---

## Training და ექსპერიმენტები

### Ridge Regression — alpha ვარიაცია

```
alpha=0.0001  → train_rmsle≈0.11, val_rmsle≈0.14  
alpha=1       → train≈0.12, val≈0.13               
alpha=100     → train≈0.13, val≈0.13               
alpha=10000   → train≈0.18, val≈0.18               
```

Alpha=0,001-დან 100-მდე ოპტიმალური დიაპაზონია.

### Decision Tree — depth ვარიაცია

| max_depth | train_rmsle | val_rmsle | დასკვნა |
|---|---|---|---|
| 3 | 0.19 | 0.21 | underfitting — ხე ძალიან მარტივია |
| 4 | 0.15 | 0.18 | გაუმჯობესება |
| 6 | 0.12 | 0.17 | კარგი |
| 8 | 0.09 | 0.18 | მცირე overfitting |
| None | 0.00 | 0.28 | overfitting |

### XGBoost — hyperparameter grid

| კონფიგურაცია | train_rmsle | val_rmsle | overfit |
|---|---|---|---|
| n=300, d=3, lr=0.05 | 0.108 | 0.128 | 0.020 |
| n=500, d=3, lr=0.05 | 0.097 | 0.126 | 0.029 |
| n=400, d=4, lr=0.15 | 0.089 | 0.131 | 0.042 |
| n=600, d=6, lr=0.03 | 0.074 | 0.133 | 0.059 |
| n=700, d=3, lr=0.10 | 0.093 | 0.127 | 0.034 |

### საბოლოო მოდელის შერჩევა

შერჩევის კრიტერიუმი: **ყველაზე დაბალი `val_rmsle` 

გამარჯვებული: Ridge alpha=0.0001    kaggle score=0.1412


---

## MLflow Tracking

ექსპერიმენტები: [dagshub.com/lukaLomadze/ML_Assignment_1](https://dagshub.com/lukaLomadze/ML_Assignment_1)

### ჩაწერილი მეტრიკები

| მეტრიკა | აღწერა |
|---|---|
| `train_rmsle` |
| `val_rmsle` | 
| `cv_rmsle_mean` | 5-fold cross-validation RMSLE საშუალო |
| `cv_rmsle_std` | cross-validation სტანდარტული გადახრა |
| `overfit` | val_rmsle − train_rmsle (დადებითი → overfitting) |

