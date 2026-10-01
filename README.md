# Kestirimci Bakım: Turbofan Motor RUL Tahmini

NASA C-MAPSS (FD001) verisiyle motorların kalan faydalı ömrünü (RUL) tahmin eden proje.

## Problem
Motor sensör verisinden, motor arızalanmadan kaç çevrim kaldığını tahmin etmek.

## Veri
NASA C-MAPSS FD001: 100 eğitim motoru, 100 test motoru, 21 sensör.
Sabit kalan sensörler ve operasyon ayarları çıkarıldı, 14 sensör kullanıldı.
RUL etiketi 125 çevrimde kesildi. Eğitim ve test motorları zaten ayrı olduğundan veri sızıntısı yok.

## Yöntem
- Ölçekleme: MinMaxScaler (sadece eğitim verisine fit edildi)
- Baseline: Random Forest, XGBoost (tek satır girdi)
- Derin öğrenme: 2 katmanlı LSTM, 30 çevrimlik pencere
- Test: her motorun son çevrimi

## Sonuçlar

| Model | Test RMSE | NASA skoru |
|---|---|---|
| Random Forest | 17.18 | 917.2 |
| XGBoost | 16.71 | 808.4 |
| LSTM | 13.11 | 275.5 |

RMSE ve NASA skoru için düşük değer iyidir. LSTM değerleri son epoch'a aittir, en iyi epoch seçilmemiştir.

## Yorumlanabilirlik
XGBoost üzerinde SHAP analizi yapıldı (reports/shap_summary.png).
En etkili sensörler: s11, s9, s4, s12, s14.

## Sınırlılıklar
- Sadece FD001 kullanıldı (tek çalışma koşulu, tek arıza türü).
- Test seti 100 motor olduğu için epoch'lar arası RMSE oynaklığı yüksek.
- SHAP sadece XGBoost için yapıldı, LSTM için yapılmadı.
- Hiperparametre ayarı yapılmadı, tek bir rastgele tohum kullanıldı.

## Çalıştırma
1. pip install -r requirements.txt
2. Verileri data/raw/ klasörüne koy
3. python src/train_baseline.py
4. python src/train_lstm.py
5. python src/explain.py
