# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)

Bu proje, bir e-ticaret şirketinin müşterilerini RFM (Recency, Frequency, Monetary) metrikleri kullanarak segmentlere ayırma sürecini içerir.

## 1. İş Problemi (Business Problem)

Bir e-ticaret şirketi, müşterilerini segmentlere ayırarak pazarlama stratejileri belirlemek istemektedir.

## 2. Veriyi Anlama (Data Understanding)

Kullanılan veri seti [Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) adını taşımaktadır ve İngiltere merkezli online bir satış mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içermektedir.

Veri setindeki önemli değişkenler şunlardır:
- InvoiceNo: Fatura numarası
- StockCode: Ürün kodu
- Description: Ürün ismi
- Quantity: Ürün adedi
- InvoiceDate: Fatura tarihi ve zamanı
- UnitPrice: Ürün fiyatı (Sterlin cinsinden)
- CustomerID: Eşsiz müşteri numarası
- Country: Müşterinin yaşadığı ülke

## 3. Veri Hazırlama (Data Preparation)

Veri setindeki eksik değerler kaldırılmış ve iade işlemleri veri setinden çıkarılmıştır.

## 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)

Müşteri segmentasyonu için gerekli olan Recency, Frequency ve Monetary metrikleri hesaplanmıştır.

## 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)

RFM metrikleri üzerinden her bir müşteriye ait RFM skorları oluşturulmuştur.

## 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)

Oluşturulan RFM skorlarına göre müşteriler belirli segmentlere ayrılmış ve her bir segmentin analizi yapılmıştır.

## 7. Tüm Sürecin Fonksiyonlaştırılması

Projenin tüm süreçlerini içeren bir fonksiyon oluşturulmuştur.

## Kullanım

Proje kodlarını kullanarak müşteri segmentasyonu analizini gerçekleştirebilir ve elde ettiğiniz sonuçları değerlendirebilirsiniz.
