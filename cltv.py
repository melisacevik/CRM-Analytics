############################################
# CUSTOMER LIFETIME VALUE (Müşteri Yaşam Boyu Değeri)
############################################

# 1. Veri Hazırlama
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Segmentlerin Oluşturulması
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması

##################################################
# 1. Veri Hazırlama
##################################################

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


# amacımız : cltv değerlerini her bir müşteri için hesaplamak ve daha sonra segmentasyon yapmak
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()

# önce eksik değerlerime bakarım onları veri setimden çıkarıp öyle analiz etmem gerekir.
df.isnull().sum()

# önce Invoice içindeki C olanlardan ( canceled faturalardan kurtulalım )
df = df[~df["Invoice"].str.contains('C', na=False)]  # ~ (tilda) olmazsa C olanları getirir
df.describe().T

# Quantitylerde - değerler var kurtul
df = df[df["Quantity"] > 0]
# eksik değerleri uçur
df.dropna(inplace=True)
df.isnull().sum()

# Her ürün için fiyat görüyorum miktar görüyorum ama ne kadar ödendiğini göremiyorum
# her zaman total price'a ihtiyaç duyacağım ( işlem başına ne kadar para bıraktığı )
df["TotalPrice"] = df["Quantity"] * df["Price"]

# şimdi transaction datası formatından customer lifetime value clvn'nin hesaplanması için gerekli olan metriklere çevirmemiz gerekmektedir.

cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

# bu işlemle her bir müşterini:
# - eşsiz faturalarını( invoice )
# - bu işlemlerde kaç birim satın almış (quantity)
# - o faturada ödediği miktarı(TotalPrice) görebilirim

# invoice => her müşterinin eşsiz kaç tane faturası olduğunu görürüm ( total transaction için )
# quantity => gözlem için hesapladık
# öncelik invoice, total price
# key'e değişken ismi, o değişkeni alıp value'u uygular

# değişken isimlerimi değiştireceğim
cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
# 12347.00000                  2                     828                1323.32000
# bu müşteri          2 kere fatura kesilmiş , 828 birim ürün almış , bu kadar ödeme yapılmış

# rfm analizi ile benzer , toplam işlem sayısı => frequency , total price => monetary

##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################

cltv_c.head()

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################
# her bir müşteri için işlem yapma sayısı / müşteri sayısı yapılacak.

cltv_c.head()
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / (cltv_c.shape[0])

# cltv_c.shape[0]  =>  # kaç eşsiz müşteri ( customer id ye göre group by işlemi yapmıştık )

##################################################
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
##################################################

cltv_c.head()

cltv_c[cltv_c["total_transaction"] > 1].shape[0]  # 1 den fazla işlemi olan bütün müşteriler
cltv_c.shape[0]  # bütün müşteriler

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################
# 0.10 olmak zorunda değil.

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

# büyükten küçüğe sırala

cltv_c.sort_values("cltv", ascending=False).head()

# neden bu müşteri en yukarıda tekrar analiz et! cltv_c.describe().T diyerek kontrol et

##################################################
# 8. Segmentlerin Oluşturulması
##################################################

# müşteri grubuyla ilgilenmeyi tercih etmem gerekirse odaklanabilirim. ( cltv hesaplandı )
cltv_c.sort_values("cltv", ascending=False).tail()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])  # qcut küçükten büyüğe sıralama yapar

cltv_c.sort_values("segment", ascending=False)

# ama mantıklı mı?
# segment oluştururken 3 segmentte yapabiliriz , fark çoksa segmenti arttırabiliriz.
cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_csv("cltv.csv")


##################################################
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması
##################################################

def create_cltv_c(dataframe, profit=0.10):
    # Veriyi hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
    # avg_order_value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']
    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]
    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    # profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit
    # Customer Value
    cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])
    # Customer Lifetime Value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']
    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c


df = df_.copy()

clv = create_cltv_c(df)
