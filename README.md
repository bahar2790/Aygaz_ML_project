## Projenin Kaggle Üzerindeki Sayfası

Projemi Kaggle üzerinde görmek için şu bağlantıya tıklayabilirsiniz:  
[Kaggle Proje Linki](https://www.kaggle.com/code/baharakay/ml-project)




Makine Öğrenmesi ile Revenue Tahmini

Proje Açıklaması

Bu proje, belirli bir veri kümesi üzerinden makine öğrenmesi yöntemlerini kullanarak revenue (gelir) tahmini yapmak amacıyla geliştirilmiştir. Proje boyunca çeşitli veri işleme, model oluşturma ve değerlendirme adımları uygulanmıştır.

Kullanılan Algoritmalar:
Denetimli Öğrenme: Lineer Regresyon, Random Forest
Denetimsiz Öğrenme: K-Means, PCA (Temel Bileşen Analizi)

İçindekiler

Veri Analizi
Model Eğitimi
Tahmin ve Sonuçlar
Sonuç Değerlendirmesi
Veri Analizi

Projenin ilk aşamasında, veri seti üzerinde analizler yapılmıştır. Bu aşamada:

Eksik veriler tespit edilmiş ve gerekli ön işlemler uygulanmıştır.
Verilerin dağılımı ve özellikler arası ilişkiler görselleştirilmiştir.
Model Eğitimi

Makine öğrenmesi modellerinin eğitimi için şu adımlar izlenmiştir:

Veriler eğitim ve test setlerine ayrılmıştır.
Linear Regression, Random Forest ve diğer algoritmalar uygulanmıştır.
Model performansı değerlendirilmiş ve en iyi sonuç veren model seçilmiştir.

Model Eğitimi ve Tahminler

Bu aşamada, hem denetimli hem de denetimsiz öğrenme yöntemleriyle model eğitimi yapılmıştır. İki tür modelin de performansları analiz edilmiştir.


2. Denetimsiz Öğrenme - K-Means:
from sklearn.cluster import KMeans

Eksik Verilerin Düşürülmesi: Eksik satırlar silinir.

df = df.dropna()
Özelliklerin Ölçeklendirilmesi: Veriler standartlaştırılır, böylece farklı ölçeklerdeki özellikler eşit ağırlığa sahip olur.


X_scaled = scaler.fit_transform(X)
K-Means Modeli: Veriler 5 kümeye ayrılır.


kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
Küme Etiketlerinin Eklenmesi: Her veri noktası, ait olduğu kümeye göre etiketlenir.


df['cluster'] = kmeans.labels_
Ortalama Revenue Hesaplama: Her küme için ortalama "revenue" hesaplanır ve veriye eklenir.


df['predicted_revenue'] = df['cluster'].map(revenue_by_cluster)
Sonuçların Görselleştirilmesi: Bütçe ve revenue değerleri küme etiketlerine göre görselleştirilir.
Bu kod, denetimsiz öğrenme ile verileri kümelere ayırır ve her küme için ortalama revenue tahmin eder.

Çıktı Yorumları:
Bu kod parçası, K-Means algoritmasını kullanarak kümelere ayırma işlemi yapar. 
Ardından, her küme için revenue ortalamalarını alıp veri noktalarına tahmini "revenue" değeri olarak atar. Sonuç olarak:

Tahmini Revenue ve Gerçek Revenue Değerleri:
![image](https://github.com/user-attachments/assets/2a2553bc-6d34-4b24-ab5d-764985dbb3be)

Bu çıktı, denetimsiz öğrenme yöntemlerinden biri olan K-Means ile elde edilen tahmin sonuçlarıdır.
K-Means doğrudan tahmin amaçlı bir algoritma olmadığından, revenue'yu kümelere ayırarak ortalama değer üzerinden bir tahmin üretir. 
Ancak bu yaklaşım, denetimli öğrenme yöntemleri (örneğin, lineer regresyon veya random forest) kadar kesin ve doğruluğu yüksek bir tahmin sağlamayabilir. 
Denetimsiz öğrenmede bu tür tahminler, genellikle daha genel kalır.

Sonuç:
Bu kod, denetimsiz öğrenme algoritması olan K-Means ile tahminler üretmiştir.
Ancak, revenue gibi belirli bir bağımlı değişkenin tahmin edilmesi gereken durumlarda denetimli öğrenme yöntemleri (örneğin, lineer regresyon) genellikle daha uygun ve daha doğru sonuçlar verir
3. Denetimli Öğrenme - Lineer Regresyon Modeli:
Kod Adımları:
Veri Hazırlığı:
X = df.drop('revenue', axis=1)
y = df['revenue']
Veri çerçevesinde "revenue" bağımlı değişken olarak ayrılıyor ve bağımsız değişkenler (diğer tüm sütunlar) X değişkeninde tutuluyor.
Veri Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Veriler eğitim ve test setlerine bölünüyor (%80 eğitim, %20 test). Bu adım modelin doğruluğunu test etmek için kullanılıyor.
Ridge Regresyon Modelinin Eğitilmesi:
model = Ridge(alpha=1)
model.fit(X_train, y_train)
Ridge Regresyon modeli oluşturuluyor ve alpha=1 parametresi ile eğitiliyor. Ridge Regresyon, aşırı öğrenmeyi (overfitting) azaltmak için kullanılan bir regresyon türüdür ve L2 regularizasyonu uygular.
Alpha parametresi, regularizasyon miktarını kontrol eder.
Tahminler:
y_pred = model.predict(X_test)
Eğitim seti ile eğitilen model, test seti üzerinde tahminler yapar. y_pred değişkeni, tahmin edilen revenue değerlerini içerir.
Performans Metrikleri:
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
Mean Squared Error (MSE): Tahmin edilen ve gerçek değerler arasındaki ortalama karesel hatadır. Düşük değerler modelin daha iyi olduğunu gösterir.
R² Score: Modelin bağımlı değişkeni ne kadar iyi açıkladığını gösteren bir metriktir. 1’e ne kadar yakınsa model o kadar iyi performans gösterir. Burada 0.735 gibi oldukça iyi bir değer elde edilmiş.
Özelliklerin Önem Dereceleri:
feature_importances = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
Bu kısımda, Ridge Regresyon modelinin her bir bağımsız değişkene verdiği katsayılar (koefisiyentler) hesaplanmış. Yüksek katsayılara sahip değişkenler model için daha önemli kabul edilir.
Tahmin Sonuçlarının Gösterilmesi:
results = pd.DataFrame({
    'Actual Revenue': y_test,
    'Predicted Revenue': y_pred
}).reset_index(drop=True)
Çıktı Yorumları:
Mean Squared Error (MSE): 0.000112889402135993
Bu değer, modelin hatasının oldukça düşük olduğunu gösteriyor. MSE ne kadar küçükse, modelin tahminlerinin gerçek değerlere o kadar yakın olduğu anlamına gelir. 0.0001 gibi küçük bir değer, tahminlerde iyi bir doğruluk sağlandığını gösteriyor.
R² Score: 0.7356119471903002
R² skorunun 1'e yakın olması, modelin bağımlı değişkenin (revenue) %73,5'ini açıklayabildiğini gösteriyor. Bu, özellikle Ridge Regresyon gibi regularizasyonlu bir model için oldukça iyi bir sonuçtur.
Özelliklerin Önem Dereceleri:


vote_count           3.427486e-01
budget               2.041043e-01
popularity           9.352957e-02
vote_count ve budget özellikleri en yüksek katsayılara sahip, yani revenue tahmininde en önemli özelliklerdir. Bu, daha fazla oya sahip filmlerin ve yüksek bütçeli filmlerin daha fazla gelir sağlama eğiliminde olduğunu gösteriyor.
Diğer önemli bir özellik ise popularity. Bu da filmin ne kadar popüler olduğunun revenue tahminine etkisi olduğunu gösterir.
Daha düşük katsayılara sahip özellikler ise modelde çok az etkiye sahiptir. Örneğin, release_year ve runtime gibi özellikler neredeyse hiç etkiye sahip değildir.
Tahmin Sonuçları:


![image](https://github.com/user-attachments/assets/3b54f80a-8d2e-45d4-99e7-2a27ec8ac4d3)

İlk 5 satırda, gerçek revenue değerlerinin 0 olduğu görülüyor, bu filmler muhtemelen düşük performanslı ya da hiç gelir elde edememiş filmler.
Model de buna yakın tahminlerde bulunmuş, ancak çok küçük negatif tahminler üretiyor (örneğin, -0.001035 gibi). Bu farklar modeldeki küçük sapmalardan kaynaklanabilir.



