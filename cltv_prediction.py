##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması


##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

# muhasebe kayıtları genelde fatura özelinde düzenlenir. faturalar => ana odak
##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import \
    MinMaxScaler  # lifetime value hesaplandıktan sonra 0,1 - 0,100 gibi değerler arasına çekmek istersem sklearn içerisindeki MinMaxScaler() metodunu kullanırız.


# modelleri kurarken kullandığımız değişkenlerin dağılımları sonuçları direkt etkileyebilir.
# değişkenleri oluşturduktan sonra aykırı değerlere dokunmamız gerekiyor.
# boxplot() veya IQR olarak geçen bir yöntem aracılığıyla önce aykırı değerleri tespit edeceğiz.
# aykırı değerleri baskılama yöntemi ile belirlemiş olduğumuz aykırı değerleri belirli bir eşik değeriyle değiştireceğiz. bunun için 2 fonksiyonumuz var.
# silmeyeceğiz, baskılayacağız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)  # çeyrek değerler hesaplanacak
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1  # çeyrek değerlerin farkı hesaplanacak
    up_limit = quartile3 + 1.5 * interquantile_range  # 3. çeyrek değerin 1.5 IQR üstü => üst değer ( 3. çeyrek + 1.5 * fark )
    # ( 3.çeyrekten ve 3 ve 1. çeyrek değerinin farkından 1.5 birim fazla olan değerler, aykırıdır.)
    low_limit = quartile1 - 1.5 * interquantile_range  # 1. çeyrek değerinin 1.5 IQR altındaki  => alt değer ( 1.çeyrek - 1.5 * fark )
    return low_limit, up_limit


# bu fonksiyonun görevi kendisine girilen değişken için eşik değer belirlemektir.
# aykırı değer nedir? Bir değişkenin genel dağılımının çok dışında olan değerlerdir. yaş => 300 olamaz. bu bir aykırı değer. veri setinden kaldırılması gerekir.
# quantile fonksiyonu çeyreklik hesaplamak için kullanılır.
# çeyreklik hesaplamak nedir? => değişkeni k - b'ye sıralar , yüzdelik olarak karşılık gelen değeri bir değişkenin çeyrek değeridir.
# neden %1 ve %99 yaptık ? => değişebilir projeden projeye

########
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# bu fonksiyon aykırı değer baskılama yöntemi olarak kullanabileceğimiz bir fonksiyon.
# bu fonksiyonu bir dataframe ve değişken ile çağırdığımızda outlier_thresholds fonk. çağıracak.
# aykırı değerler nedir? bu aykırı değerlere karşılık belirlenmesi gereken üst ve alt limit nedir?
#     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
# ilgili değişkenlerde üst sınırda olanlar varsa bunların yerine üst limiti ata

#########################
# Verinin Okunması
#########################

df_ = pd.read_excel("online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

# her bir ürün için toplam ne kadar ödendiğini bulmak için  => quantity * price
# her bir kullanıcıya göre groupby a aldıktan sonra price toplamını alıp bir fatura başına ne kadar bedel ödendiği bilgisine erişebilirim

#########################
# Veri Ön İşleme
#########################
# daha önce aykırı değerlere odaklanmadık ama modelleme için şart

df.dropna(inplace=True)  # eksik değerleri veri setinden sildim
# min değeri - lerde çıkıyor bunun sebebi iade faturalarının olması
df = df[~df["Invoice"].str.contains("C", na=False)]  # Invoice de içinde C olmayanları ~ tilda getiricek.

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")  # quantity için eşik değerleri hesapla
replace_with_thresholds(df, "Price")  # price için eşik değerleri hesapla

df[['Quantity', 'Price', 'Customer ID']].describe().T  # invoicedate gelmesin diye
# mean ve std birbirinden cook uzaksa aykırılık var


df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)  # analiz tarihini max günün 2 gün sonrasını al

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde) ( kendi son - ilk tarihi )
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1) satın alma sıklığı
# monetary: satın alma başına ortalama kazanç ! burada ortalama alacağız RFM gibi toplamı değil ! average order value

cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,  # recency
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],  # müşteri yaşı
     'Invoice': lambda Invoice: Invoice.nunique(),  # eşsiz kaç fatura - frequency -
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})  # monetary

# üstteki label'ı kaldır
cltv_df.columns = cltv_df.columns.droplevel(0)
# < lambda_ 0> ... gibi görünüyor yeniden isimlendir
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]  # satın alma başına ort. kazanç

cltv_df.describe().T

# frequency 1 den büyük olacak şekilde oluştur. describe'da frequency => min en küçük 2 olmalı

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# recency ve T haftalık olmalı , günlük değil

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
cltv_df.describe().T

##############################################################
# 2. BG-NBD Modelinin Kurulması
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)
# bir model nesnesi oluşturacağım, bu nesne aracılığıyla sen fit metodunu kullanarak frequency, recency ve T değerlerini girdiğinde bu modeli kurar
# penalizer_coef=0.001 => bu modelin parametrelerinin bulunması aşamasında katsayılara uygulanacak olan ceza katsayısıdır
# bg nbd modeli bize en çok olabilirlik yöntemi ile beta ve gamma dağılımlarının parametrelerini bulmakta ve bir tahmin yapabilmemiz için ilgili
# modeli oluşturmaktadır.

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

# con_... => bgf nin bir fonksiyonu
# t: 1 => 1 haftalık tahmin yap

# bir hafta içinde beklediğimiz satın almaları hesaplayıp bunu cltv_df ' e ekle

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# buradaki predict ve conti_... olan fonk. aynı işlemi görür fakat gamma'da predict kullanılmaz!! o yüzden cont_.. daha iyi

################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################
# t => 4 bir aylık tahmin

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# 1 ay içerisinde ne kadar satın alma olabilecek?

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()  # bir aylık periyotta şirketimizin beklediği satış sayısı budur

################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

# burada öncelikle basit bir grafik üzerinden tahmin sonuçlarının başarısı değerlendirilebilir.
plot_period_transactions(bgf)
plt.show()

# gerçek değerler maviler , tahmin edilenler turuncular ( modelin tahmin ettiği )


# bg/nbd => satın alma sayısını modeller

##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])  # model kur bilgisi

ggf.conditional_expected_average_profit(cltv_df['frequency'],  # toplam işlem sayısı
                                        cltv_df['monetary']).head(10)  # işlem başına ort değeri

# azalan istersek

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)
# bütün müşterilerin içerisinde beklenen karı, expected average ı, ortalama karı getirmiş oldu

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

# potansiyel müşteri değerlerini de yakalama kabiliyeti var

# elimizde, expected transaction değerleri var,1 haftalık , 1 aylık , 3 aylık , ortalama beklenen karlılık değerleri müşteri özelinde hesaplandı.

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()
# cltv fonksiyonu gammagamma ve bgnbd modelini ver,frequ,recency,yaş, monetary göster, zaman periyodu ver, ( time = aylık ) , (T ve recency) girdiğin değerler haftalıksa frekans ver W
# zaman içerisinde sattığın ürünlerde çeşitli indirimler yapabilirsin. bunu da göz önünde bulundur. => discount_rate = 0.01

# daha önce cltv_df oluşturmuştuk , cltv_df ile cltv leri ifade eden veri setini birleştirmemiz gerekiyor.

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")  # cltv_df'e göre birleştir
cltv_final.sort_values(by="clv", ascending=False).head(10)

# clv ne ? return ettiği df'deki isimlendirme
# inceleyelim en üsttekini
# bu kişinin 3 aylık periyotta yapması beklenen satış sayısı 14 birim.

# örneğin 1257 id ye sahip müşteriye bak,
# gözlemlerde en fazla satın alma buradan beklenirken ( expected_purc_3_month) , karlılığı düşük olduğu ( expected_avg_ profit ) için en yukarda değil.!
# karlılığın düşük olduğunu monetaryden de bakabilirsin. bu kişi yüksek frekansta alışveriş yapmış ama düşük hacimli yapmış


# recency'si kendi içinde bu kadar yüksek olan bu müşteriler nasıl cltv'de en büyük değeri vadediyor?
# senin için düzenli olan bir müşterinin kendi içinde recency değeri arttıkça, müşterinin satın alma olasılığı yaklaşıyordur
# bu veri seti özelinde, recency ve T değeri birbirine yakında yüksek CLTV değerleri elde edildi.
# ilk 5 e baktığımda frequency değeri düşük olan müşteri var. monetary yüksek mi ? evet. o zaman bu potansiyel müşteri anlamına gelir.
# hepsine bakıp yorumlamak önemli
