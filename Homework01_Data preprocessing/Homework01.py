import pandas as pd #lab1
import seaborn as sns #lab2
import matplotlib.pyplot as plt
from pathlib import Path #lab4


#lab1
df = pd.read_csv(r'C:\Users\ketar\ML-CPE\Homework01_Data preprocessing\Titanic-Dataset.csv') #Titanic-Dataset.csv pushไปปลอดภัยไหมอาจารบอกด้วยพอดีคอมมันหาไฟล์ไม่เจอ
print("Shape:", df.shape)
print(df.dtypes)
print(df.describe(include='all'))
print("Missing per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Survived distribution:\n", df['Survived'].value_counts(normalize=True))


#lab2
# Histogram
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Age distribution')
plt.show()

# Correlation heatmap
num_cols = df.select_dtypes(include=['int64','float64']).columns
corr = df[num_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()


#lab3
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop_duplicates()
df['Name'] = df['Name'].str.strip()
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df['Pclass'] = df['Pclass'].astype('category')
df['Fare'].describe()
df['Fare'].median(), df['Fare'].mean()
# IQR capping
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df['Fare_clipped'] = df['Fare'].clip(lower, upper)


#lab4
base = Path(__file__).parent
input_csv = base / 'Titanic-Dataset.csv'   # ปรับพาธถ้าจำเป็น
out_encoded = base / 'titanic_preprocessed_encoded.csv'

df = pd.read_csv(input_csv)

df['Age'] = pd.to_numeric(df.get('Age'), errors='coerce').fillna(df['Age'].median())
df['Fare'] = pd.to_numeric(df.get('Fare'), errors='coerce').fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0]) if 'Embarked' in df.columns else df.get('Embarked')
df['FamilySize'] = df.get('SibSp', 0).fillna(0).astype(int) + df.get('Parch', 0).fillna(0).astype(int) + 1
df['HasCabin'] = df['Cabin'].notnull().astype(int) if 'Cabin' in df.columns else 0

if 'Name' in df.columns:
    df['Title'] = df['Name'].astype(str).str.extract(r',\s*([^\.]+)\.').fillna('Unknown')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    # รวม title ที่น้อยเป็น 'Rare'
    vc = df['Title'].value_counts()
    rare = vc[vc < 10].index
    df['Title'] = df['Title'].replace(list(rare), 'Rare')

# --- Encoding ---
# Label encode Sex (binary)
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male':0, 'female':1}).fillna(0).astype(int)
else:
    df['Sex'] = 0

# One-hot encode (เฉพาะคอลัมน์ที่มี)
to_dummy = [c for c in ['Embarked','Pclass','Title'] if c in df.columns]
if to_dummy:
    df = pd.get_dummies(df, columns=to_dummy, drop_first=True)

# ลบคอลัมน์ไม่จำเป็นก่อนบันทึก
for c in ['PassengerId','Name','Ticket','Cabin']:
    if c in df.columns:
        df = df.drop(columns=c)

# บันทึกไฟล์ที่ผ่านการเข้ารหัสแล้ว
df.to_csv(out_encoded, index=False)
print("Saved encoded CSV to:", out_encoded)
