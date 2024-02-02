###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

#############################################################
# 1. İş Problemi (Business Problem)
#############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesidfcğp
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # virgülden sonra 3 rakam

df_ = pd.read_excel("/Users/melisacevik/PycharmProjects/CRM-Analytics/online_retail_II.xlsx",
                    sheet_name='Year 2009-2010')
df = df_.copy()
df.head()

df.shape
df.isnull().sum()

# eşsiz ürün sayısı?
df["Description"].nunique()
# bu eşsiz ürünler faturalarda kaçar defa eklendi
df["Description"].value_counts().head()

# en çok sipariş edilen ürün
df.groupby("Description").agg({"Quantity": "sum"}).head()

df.groupby("Description").agg({"Quantity": "sum"}).sort_values(by="Quantity", ascending=False)

# Invoice çoklama yaptı cünkü bir ürün birden fazla faturada bulunabilir

# Toplam fatura sayısı
df["Invoice"].nunique()

# Fatura başına toplam kaç para kazanılmıştır?

# ürün başına kazanılan para
df["TotalPrice"] = df["Quantity"] * df["Price"]  # her gözlem değeri için

df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()  # faturadaki ürün başına kazanılan paranın toplamı

###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################

df.shape
df.isnull().sum()
df.dropna(inplace=True)  # eksik değerleri kaldıralım.
df.isnull().sum()

# invoice 'de başında C olan ifadeler iadeleri ifade ediyor, iadeleri veri setinden çıkaralım. veri setini bozuyorlar.

# içinde c olanları seç
df[df["Invoice"].str.contains("C", na=False)].head()
# içinde c olmayanları seç
df = df[~df["Invoice"].str.contains("C", na=False)]

###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################
# Recency - Frequency - Monetary
# her bir müşteri özelinde R-F-M değerlerini hesaplayacağız
# Recency = analizin yapıldığı tarih - müşterinin son satın alma tarihi
# Frequency = müşterinin toplam satın alması
# Monetary = müşterinin toplam parasal değer

df.head()

# 2011 veri seti olduğu için bugünün değerini alamayız
# ama analizin yapıldığı son tarihine 2 gün ekleyip analiz yapıldığı tarih diye kabul ederiz. Bu tarih üzerinden Recency hesaplarız

df["InvoiceDate"].max()

today_date = dt.datetime(2010, 12, 11)
type(today_date)

# Bütün müşterilere göre groupby()'a alıp, R,F,M hesaplayıp,
# Recency => today_dateden groupby()'a alıp her bir müşterinin max tarihini bulcaz. today_date'den çıkarıp Recency bulacağız.
# Frequency => Customer ID'ye göre groupby() a alıp her bir müşterinin eşsiz fatura sayısına gidip kaç işlem yapmış bakacağız.
# Monetary => Customer ID'ye göre groupby() a alıp Total_Price ların sum'ını alırsak bu durumda her bir müşterinin toplam kaç para bıraktığını buluruz.

rfm = df.groupby('Customer ID').agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     "Invoice": lambda Invoice: Invoice.nunique(),
                                     "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.head()
rfm.columns = ["recency", "frequency", "monetary"]  # isimlerini değiştirdik

rfm.describe().T
# gözlemlediğimizde monetary değeri 0 geliyor bunu silmemiz gerekiyor.

rfm = rfm[rfm["monetary"] > 0]  # monetary 0 dan sonraki en küçüğü almıs oldu

## Müşteri segmentasyonu yaptık.

rfm.shape

###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
# qcut => küçükten büyüğe sıralar, belirli parçalara göre (q: 5) böler.
# recency değeri küçük olan yüksek puandır yani 5

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
# value error verdi ( oluşturulan aralıklarda unique değerler yer almamaktadır )
# 0-20 arasında 1 var 20-40 arasında da 1 var, 40-60da da 1 var.
# ( çok fazla aynı değerler olduğu için farklı labellara koyamadı o yüzden rank method first dedik ilk gördüğünü atacak o labela
# qcut zaten küçükten büyüğe sıralıyor belirli parçalara bölüyor, t-g gibi.

# bu değerlerin üzerinden skor değişkeni oluşturmamız gerekiyor. Yani R ve F değerlerini bir araya getireceğiz.
# monetary'i gözlemlemek için hesapladık. 2 boyutlu olan görselde RFM skorlarına gitmek için R ve F değerleri yeterli.


rfm["RFM_SCORE"] = rfm['recency_score'].astype(str) + rfm["frequency_score"].astype(
    str)  # yan yana topladı "1" + "1" = 11 olur

rfm.describe().T  # string tipte olduğu için gelmediler. label oluştururken de string tipteydi. o yüzden sayısal değişken gibi analiz edilmedi.

# champion class =Z Recency 5, Frequency'i 5 olandı.

# rfm skorları 55 olan müşteriler
rfm[rfm["RFM_SCORE"] == "55"]

# daha az değerli müşteriler
rfm[rfm["RFM_SCORE"] == "11"]

###############################################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################
# regex (regular expression)
# R de 5 F'de 5 gördüğünde champion yaz gibi bir kullanım yapacağız.

# RFM isimlendirmesi
# birinci elemanında 1-2 ve 2.elemanında 1-2 görürsen hibernating sınıfında
# r'[1-2][1-2]': 'hibernating',

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# yazacağımız kodla pattern match(yapı yakalama) 'i sağlar.
# iki değer var 11-55 gibi

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)  # birleştirilen skorlar seg_map i
# rfm_score'daki değerleri örneğin "25" replace et, değiştir

# ben bu segmentleri oluşturdum ne yapacağım?
# öncelikle bu oluşturulan stringlerin bir analizini yapmak lazım.
# bizim bu sınıflarımız var, bu sınıfların özellikleri bunlar
# bu sınıflardaki kişilerin Recency ortalamaları şudur, Frequency ortalamaları ne? gibi temel bilgilere ulaşacağız.


rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
# metrikleri segmente göre grupladım , SKORLARA GÖRE DEĞİL. Bunların ortalamalarını alarak karşılaştırma yapacağım.
# ŞU AN GERÇEK DEĞERLERDEYİZ !

# örneğin at_Risk , toplamda 611 kişi var. recency ortalaması 152 ( 152 gündür yoklar ) , frequency 3 ...
# champions => diğerlerine göre de bir değerlendirme yapmam lazım => recency 7.119 evet aralarındaki en küçük. doğru
# new_customers ile championsu ayıran => frequency değeri( toplam satın alma )
# need_attention ilgi görmezse bu churn'dür!


# need_attention sınıfına ait bilgileri almak istersek seçmeliyiz.

rfm[rfm["segment"] == "cant_loose"]
rfm[rfm["segment"] == "new_customer"].index  # müşterilerin idleri

# yeni dataframe oluşturalım.

new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

# id'lerde float var kurtul

new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

# vereceğimiz format bu değil , excel veya csv formatı ile dışarı çıkarmamız lazım.

new_df.to_csv("new_customers.csv")
rfm.to_csv("rfm.csv")


# çalışma dizinine git  > CRM- Analytics > Reload from Disk


###############################################################
# 7. Tüm Sürecin Fonksiyonlaştırılması
###############################################################

def create_rfm(dataframe, csv=False):
    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm


df = df_.copy()

rfm_new = create_rfm(df)
